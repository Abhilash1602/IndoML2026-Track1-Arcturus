import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from speechbrain.lobes.models.beats import BEATs
from torch.distributions import Beta

from src.dataset import CATEGORY_LIST


def _load_atst_frame_backbone(ckpt_path):
    """Load the frozen ATST-Frame backbone (fschmid56/PretrainedSED, MIT,
    checkpoint ATST-F_strong_1.pt -- AudioSet-Strong frame-level-SED-finetuned,
    vendored under src/atst_frame/). Only the `model.atst.*` keys are present in
    the checkpoint (the mel front end has no learnable parameters); the
    `strong_head`/`weak_head` classification heads are dropped since we only
    want the 768-dim frame-level sequence embeddings for fusion, not AudioSet's
    447-class predictions. Verified directly against the downloaded checkpoint's
    actual state_dict keys, not assumed from the source repo's own wrapper code."""
    from src.atst_frame.atstf_wrapper import ATSTWrapper
    wrapper = ATSTWrapper()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    backbone_sd = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.atst.")}
    missing, unexpected = wrapper.load_state_dict(backbone_sd, strict=False)
    assert not unexpected, f"Unexpected keys loading ATST-Frame backbone: {unexpected}"
    assert not [m for m in missing if m.startswith("atst.")], f"Missing ATST-Frame backbone keys: {missing}"
    return wrapper


class FrequencyDynamicConv2d(nn.Module):
    """Best-effort port of Dynamic_conv2d (github.com/frednam93/FDY-SED, pool_dim='freq'
    default, verified directly from the reference source): num_kernels parallel Conv2d
    basis kernels combined via time-varying attention weights, where the attention itself
    is computed from frequency-pooled context. Tensor layout here is (B, C, Freq, T) —
    pools over Freq (dim=2), attention varies over T (dim=3)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_kernels=4, attention_channels=32):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            for _ in range(num_kernels)
        ])
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_channels, num_kernels, kernel_size=1),
        )

    def forward(self, x):
        pooled = x.mean(dim=2)  # (B, C_in, T) -- average over frequency
        attn_weights = torch.softmax(self.attention(pooled), dim=1)  # (B, K, T)
        outputs = torch.stack([conv(x) for conv in self.convs], dim=1)  # (B, K, C_out, Freq', T')
        attn_weights = attn_weights.unsqueeze(2).unsqueeze(2)  # (B, K, 1, 1, T)
        return (outputs * attn_weights).sum(dim=1)


class ResidualBlock(nn.Module):
    """Session 6 ablation: FrequencyDynamicConv2d (FDY-Conv) was tried here and
    correlated with a real Codabench regression (Combined 1.0899->1.0597, Event F1
    0.4265->0.3821 for the baseline alone, unrelated to Mean Teacher since Phase 0
    never touches the distillation code) concentrated in Synthetic-domain F1
    (0.4537->0.3813) with Dice rising slightly -- the "broader but less precise"
    signature. FDY-Conv's time-varying softmax attention over its K parallel
    kernels is a plausible mechanism for exactly this: it blends kernel outputs
    per time step, which can blur sharp onset/offset transitions in a way that
    specifically hurts collar-based Event F1 without hurting region-overlap Dice.
    Reverted to plain Conv2d pending a clean ablation to confirm."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + self.shortcut(x))


def _build_synthetic_ir_bank(n_irs=50, ir_len_sec=0.5, sr=16000, seed=42):
    """Session 9, engineering-hack plan Priority B: a small synthetic parametric
    impulse-response bank standing in for Device-Impulse-Response (DIR)
    augmentation, the winning DCASE-2024 recipe's single most Natural-domain-
    relevant augmentation (simulates real recording-device/room coloration) --
    no real IR dataset exists in this project and fetching one carries real
    setup/licensing risk, so each IR here is built from a few randomly
    delayed/decaying "early reflection" spikes plus an exponentially-decaying
    colored-noise tail with a randomized RT60-like decay constant. A lower-
    fidelity proxy for real device coloration, but zero downloads/licensing
    risk and captures the time-domain (phase/comb-filtering) structure a
    frequency-domain-only augmentation (FilterAugment) cannot."""
    generator = torch.Generator().manual_seed(seed)
    ir_len = int(ir_len_sec * sr)
    irs = torch.zeros(n_irs, ir_len)
    t = torch.arange(ir_len, dtype=torch.float32)
    for i in range(n_irs):
        ir = torch.zeros(ir_len)
        ir[0] = 1.0  # direct path
        n_reflections = int(torch.randint(2, 6, (1,), generator=generator).item())
        for _ in range(n_reflections):
            delay = int(torch.randint(1, max(2, int(0.05 * sr)), (1,), generator=generator).item())
            gain = 0.3 * float(torch.rand(1, generator=generator))
            if delay < ir_len:
                ir[delay] += gain
        rt60 = 0.1 + 0.4 * float(torch.rand(1, generator=generator))  # 0.1-0.5s decay
        decay_envelope = torch.exp(-t / (rt60 * sr / 6.91))  # ~-60dB at rt60
        tail_noise = torch.randn(ir_len, generator=generator) * decay_envelope * 0.05
        ir = ir + tail_noise
        irs[i] = ir / (ir.abs().max() + 1e-8)
    return irs


class DualBranchSED(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # persistent=False: deterministic/regenerable from a fixed seed, keep it
        # out of the state_dict so older checkpoints stay loadable unchanged.
        self.register_buffer("_ir_bank", _build_synthetic_ir_bank(sr=config.SR), persistent=False)

        c0, c1, c2, c3 = getattr(config, "CNN_CH", (32, 32, 64, 128))
        self._cnn_out_ch = c3
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0),
            nn.GELU(),
        )
        self.res1 = nn.Sequential(ResidualBlock(c0, c1), nn.MaxPool2d((2, 1)))
        self.res2 = nn.Sequential(ResidualBlock(c1, c2), nn.MaxPool2d((2, 1)))
        self.res3 = nn.Sequential(ResidualBlock(c2, c3), nn.MaxPool2d((2, 1)))

        # Session 7d: frozen BEATs branch, fused alongside (not replacing) the CNN.
        # DISABLED via Config.USE_BEATS=False -- a real Codabench submission regressed
        # (combined 1.0982->1.0669, Synthetic F1 0.4426->0.4003), the same
        # Synthetic-domain failure as FDY-Conv and Freq-MixStyle, despite the best
        # local score this project has ever produced. Kept behind the flag: BEATs'
        # patch embedding was verified (speechbrain/lobes/models/beats.py) to yield
        # ~496 frames for a 10s/1000-frame clip (~2x coarser than our 10ms hop, not
        # the ~16x once feared), and the fusion design matches DG-SED and other DCASE
        # SOTA systems -- the technique isn't wrong, it just doesn't help here.
        self.use_beats = config.USE_BEATS
        if self.use_beats:
            self.beats = BEATs(ckp_path=config.BEATS_CKPT_PATH, freeze=True)
            for param in self.beats.parameters():
                param.requires_grad = False
            self.beats.eval()

        # WavLM fusion branch (Session 8 stage-1 smoke test): same fusion shape as the
        # BEATs branch above (both output 768-dim, concatenated before temporal_proj),
        # but WavLM's pretraining is speech-specific (denoising + masked prediction),
        # unlike BEATs' general-audio-event pretraining -- a different mechanism that
        # the earlier frozen-BEATs regression doesn't rule out. Stage 1 keeps it fully
        # frozen (requires_grad=False) matching the org's own baseline's documented
        # reasoning: fine-tuning a large pretrained encoder end-to-end from step 0
        # overfits and underperforms a from-scratch model, hence their two-stage
        # (freeze-then-unfreeze-at-low-LR) recipe. `unfreeze_wavlm()` below flips this
        # for a later stage-2 attempt without rebuilding the model.
        self.use_wavlm = getattr(config, "USE_WAVLM", False)
        if self.use_wavlm:
            from transformers import AutoModel
            self.wavlm = AutoModel.from_pretrained(getattr(config, "WAVLM_NAME", "microsoft/wavlm-base-plus"))
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm.eval()

        # ATST-Frame fusion branch (Session 9 smoke test): same 768-dim fusion
        # shape as BEATs/WavLM above, but pretrained specifically for frame-level
        # SED on AudioSet-Strong (fschmid56/PretrainedSED's ATST-F_strong_1.pt) --
        # a supervised task much closer to ours than BEATs'/WavLM's generic
        # self-supervised pretraining. Frozen-only: the analogous "unfreeze at low
        # LR" idea was tried for WavLM (Stage 2) and decisively failed to beat the
        # real-ensemble baseline across 4 epochs with no improving trend, so no
        # unfreeze_atst_frame() is implemented here.
        self.use_atst_frame = getattr(config, "USE_ATST_FRAME", False)
        if self.use_atst_frame:
            self.atst_frame = _load_atst_frame_backbone(
                getattr(config, "ATST_FRAME_CKPT_PATH", "./checkpoints/atst_frame/ATST-F_strong_1.pt"))
            for param in self.atst_frame.parameters():
                param.requires_grad = False
            self.atst_frame.eval()

        extra_dim = 768 if (self.use_beats or self.use_wavlm or self.use_atst_frame) else 0
        self.temporal_proj = nn.Linear(self._cnn_out_ch * 10 + extra_dim, config.CONF_DIM)
        self.conformer = torchaudio.models.Conformer(
            input_dim=config.CONF_DIM,
            num_heads=config.CONF_HEADS,
            ffn_dim=config.CONF_DIM * 4,
            num_layers=config.CONF_LAYERS,
            depthwise_conv_kernel_size=31,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

        # Session 9 plan, Part 1: multi-task 7-category auxiliary head, same shape
        # as the primary classifier, sharing the same trunk. Off by default
        # (Config.USE_CATEGORY_HEAD) -- when off, forward() returns exactly the
        # same (B,T) tensor as before, so every existing script/checkpoint is
        # completely unaffected. See src/dataset.py's _build_targets for the
        # (T,7) multi-hot targets this head is trained against, and
        # temp/inspect_category_variance.py for the gate-check that validated
        # the 7 categories carry genuine, non-degenerate per-frame signal.
        self.use_category_head = getattr(config, "USE_CATEGORY_HEAD", False)
        if self.use_category_head:
            self.category_head = nn.Sequential(
                nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, len(CATEGORY_LIST)),
            )

        # Session 9 plan, Part 2 Stage 2A: onset/offset auxiliary head (OOL,
        # arXiv:2601.04178), same shape/sharing pattern as category_head above.
        # 2 output channels (onset, offset), trained against src/dataset.py's
        # _build_targets single-frame boundary impulses. Off by default
        # (Config.USE_BOUNDARY_HEAD) -- zero inference-path change when off,
        # per the plan's explicit Stage 2A scoping (decode pipeline untouched
        # at this stage; this only tests whether a sharp-boundary auxiliary
        # task improves the shared trunk enough to move Viterbi-decoded F1).
        self.use_boundary_head = getattr(config, "USE_BOUNDARY_HEAD", False)
        if self.use_boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2),
            )

    def _freq_mixstyle(self, x, p=0.5, alpha=0.6, eps=1e-6):
        """Freq-MixStyle (DG-SED, Xiao et al. 2024/2025, arXiv:2407.03654): mix each
        instance's per-frequency mean/std (computed over time, i.e. frequency-wise
        instance-norm statistics) with a randomly-paired instance's, at training time
        only. Zero learnable parameters -- this is an augmentation, not a module.
        Their own t-SNE analysis on real DCASE features found the frequency axis of
        the mel-spectrogram carries most of the domain-relevant information (more
        than the channel axis), and mixing it measurably improved cross-domain
        generalization in their own experiments (DESED-vs-MAESTRO; ours is
        Natural-vs-Synthetic Vaani test clips).

        Session 7 ablation: on a real Codabench submission, the plain-Conv2d
        baseline WITH this enabled scored strictly worse on every metric than the
        same baseline without it (Combined 1.0441 vs 1.0670, Event F1 0.3889 vs
        0.4078, and -- the one it was specifically meant to help -- Synthetic F1
        0.3960 vs 0.4265). Not validated as beneficial on this dataset/architecture;
        currently disabled in forward() pending further isolation. Left implemented
        (not deleted) since it's a real, cheap, literature-grounded technique worth
        revisiting if the underlying domain-gap problem resurfaces after other
        levers are exhausted -- this specific negative result doesn't prove the
        general technique is wrong, only that it didn't help here."""
        if not self.training or x.size(0) < 2 or torch.rand(1).item() > p:
            return x
        mu = x.mean(dim=3, keepdim=True)  # (B, C, F, 1) -- per-frequency mean over time
        var = x.var(dim=3, keepdim=True, unbiased=False)
        sigma = (var + eps).sqrt()
        x_norm = (x - mu) / sigma
        perm = torch.randperm(x.size(0), device=x.device)
        lam = Beta(alpha, alpha).sample((x.size(0), 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sigma_mix = lam * sigma + (1 - lam) * sigma[perm]
        return x_norm * sigma_mix + mu_mix

    def _filter_augment(self, x, num_bands=(2, 5), db_range=(-6.0, 6.0)):
        """FilterAugment (Nam, Ko, Lee & Park 2022, 'FilterAugment: An acoustic
        environmental data augmentation method'): splits the frequency axis into a
        random number of bands and applies a random per-band gain in dB, linearly
        interpolated across the frequency axis for a smooth frequency-response-style
        perturbation -- simulating the kind of recording/microphone frequency-
        response variation real deployments see. Different from Freq-MixStyle
        (which mixes statistics between two real instances); this doesn't require a
        second instance to work from, and is part of virtually every FDY-Conv/DCASE
        SOTA recipe's augmentation stack alongside mixup. Training-only, zero
        learnable parameters, one random curve per instance in the batch."""
        if not self.training:
            return x
        batch, _, freq_bins, _ = x.shape
        n_bands = torch.randint(num_bands[0], num_bands[1] + 1, (1,)).item()
        control_points = torch.empty(batch, 1, n_bands, device=x.device).uniform_(*db_range)
        gain_db = F.interpolate(control_points, size=freq_bins, mode="linear", align_corners=True)
        return x + gain_db.unsqueeze(-1)  # (B,1,F,1) -- additive in dB == multiplicative in amplitude

    def _dir_augment(self, audio, p=0.5):
        """Device-Impulse-Response (DIR) convolution, Session 9 engineering-hack
        plan Priority B: convolve each instance with a randomly-selected impulse
        response from self._ir_bank (see _build_synthetic_ir_bank), simulating
        room/microphone coloration -- the single highest-leverage augmentation
        the winning DCASE-2024 recipe uses that this project was missing,
        specifically because it targets real-recording-condition variation
        (the Natural-domain gap), not just a frequency-response gain curve like
        FilterAugment. Applied to the raw waveform, training-only, p=0.5,
        peak-renormalized after so it colors the signal without just changing
        its loudness."""
        if not self.training or torch.rand(1).item() > p:
            return audio
        batch, length = audio.shape
        idx = torch.randint(0, self._ir_bank.size(0), (batch,), device=audio.device)
        irs = self._ir_bank.to(audio.device)[idx]  # (B, ir_len)
        kernel = irs.flip(-1).unsqueeze(1)  # (B, 1, ir_len) -- flip for true (not cross-)correlation
        x = audio.reshape(1, batch, length)
        out = F.conv1d(x, kernel, padding=kernel.size(-1) - 1, groups=batch)
        out = out.reshape(batch, -1)[:, :length]
        orig_peak = audio.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        new_peak = out.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        return out * (orig_peak / new_peak)

    def _freq_warp(self, x, p=0.5, warp_range=(0.9, 1.1)):
        """Frequency-Warping, Session 9 engineering-hack plan Priority B: a
        random frequency-axis stretch/squeeze (VTLN-style), implemented by
        resizing the mel axis then resizing back -- genuinely new relative to
        FilterAugment (a per-band gain curve, not a geometric warp) and
        Freq-MixStyle (an instance-statistics mix, not a warp). Training-only,
        zero learnable parameters."""
        if not self.training or torch.rand(1).item() > p:
            return x
        n_mels = x.size(2)
        warp_factor = float(torch.empty(1).uniform_(*warp_range))
        warped_n_mels = max(1, int(round(n_mels * warp_factor)))
        x = F.interpolate(x, size=(warped_n_mels, x.size(3)), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(n_mels, x.size(3)), mode="bilinear", align_corners=False)
        return x

    def _spec_augment(self, x):
        if not self.training:
            return x
        x = x.clone()
        batch, n_mels, frames = x.size(0), x.size(2), x.size(3)
        for b in range(batch):
            fill_val = x[b].min()  # dB scale: silence is very negative, not 0.0 — mask with this sample's own floor
            for _ in range(torch.randint(0, 3, ()).item()):  # up to 2 frequency blocks
                width = torch.randint(1, 9, ()).item()  # max width 8
                start = torch.randint(0, max(1, n_mels - width + 1), ()).item()
                x[b, :, start:start + width, :] = fill_val
            for _ in range(torch.randint(0, 3, ()).item()):  # up to 2 temporal blocks
                width = torch.randint(1, 21, ()).item()  # max width 20
                start = torch.randint(0, max(1, frames - width + 1), ()).item()
                x[b, :, :, start:start + width] = fill_val
        return x

    def train(self, mode=True):
        """Keep a frozen encoder branch (BEATs or WavLM, stage 1) in eval mode
        regardless of the parent's mode -- otherwise model.train() would also flip
        on its own internal dropout, adding noise to what's supposed to be a
        stable, frozen semantic-context source. Once `unfreeze_wavlm()` has been
        called for stage 2, WavLM follows the parent's train/eval mode normally."""
        super().train(mode)
        if self.use_beats:
            self.beats.eval()
        if self.use_wavlm and not getattr(self, "_wavlm_unfrozen", False):
            self.wavlm.eval()
        if self.use_atst_frame and not getattr(self, "_atst_frame_unfrozen", False):
            self.atst_frame.eval()
        return self

    def unfreeze_atst_frame(self):
        """Stage 2: unfreeze ATST-Frame for end-to-end fine-tuning. Mirrors
        unfreeze_wavlm() exactly. Caller is responsible for using layer-wise LR
        decay (via self.atst_frame.separate_params()) and a consistency-loss-
        regularized training loop -- unfreezing with a plain supervised loss
        was tried for WavLM and failed decisively (see STRATEGY.md's WavLM
        Stage 2 section and the CPJKU-recipe research explaining why)."""
        if not self.use_atst_frame:
            return
        self._atst_frame_unfrozen = True
        for param in self.atst_frame.parameters():
            param.requires_grad = True

    def unfreeze_wavlm(self):
        """Stage 2: unfreeze WavLM for end-to-end fine-tuning at a low LR (the
        org baseline's own recipe -- LR_ENC=5e-5, well below the head's LR).
        Caller is responsible for using a separate, much lower LR for these params
        (e.g. via a second param group), and for excluding self.wavlm's own
        internal dropout concerns are not addressed here beyond letting it now
        follow model.train()/eval() normally."""
        if not self.use_wavlm:
            return
        self._wavlm_unfrozen = True
        for param in self.wavlm.parameters():
            param.requires_grad = True

    def _cnn_frames(self, x):
        """Augmentation + CNN branch at 10 ms: returns ((B, T, C*10) features, the raw
        waveform after DIR) -- shared by forward() and FusionSED."""
        if x.dim() == 3:
            x = x.squeeze(1)
        # DIR augmentation on the raw waveform, before the CNN branch's own mel AND
        # before any fusion encoder sees it -- both should experience the same
        # simulated recording-condition change, not just the CNN branch.
        if getattr(self.config, "USE_DIR", True):
            x = self._dir_augment(x)
        raw_audio = x  # keep the raw waveform around -- BEATs does its own preprocessing
        target_frames = x.shape[-1] // self.config.HOP_LENGTH

        mel = self.amplitude_to_db(self.mel_transform(x))
        mel = mel[..., :target_frames]
        x = mel.unsqueeze(1)  # (B, 1, 80, T)
        # Session 9: Freq-MixStyle's Session 7 isolated-ablation regression doesn't
        # necessarily hold combined with the rest of the augmentation stack (the
        # winning DCASE-2024 recipe never runs it alone either) -- re-enabled behind
        # Config.USE_FREQ_MIXSTYLE, validated via a smoke test before trusting it.
        if getattr(self.config, "USE_FREQ_MIXSTYLE", False):
            x = self._freq_mixstyle(x)
        x = self._filter_augment(x)
        if getattr(self.config, "USE_FREQ_WARP", True):
            x = self._freq_warp(x)
        x = self._spec_augment(x)  # masking last, per canonical SpecAugment convention

        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)  # (B, 128, 10, T)

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, 128, 10)
        x = x.view(x.size(0), x.size(1), -1)  # (B, T, 1280)
        return x, raw_audio

    def forward(self, x):
        x, raw_audio = self._cnn_frames(x)

        if self.use_beats:
            # Upsample BEATs' coarser frame rate (~496 frames/10s vs. our 1000) to
            # the CNN branch's exact frame count via linear interpolation, then
            # concatenate as extra per-frame context -- the CNN branch still
            # supplies the fine-grained features feeding the classifier. Disabled by
            # default (Config.USE_BEATS); see the __init__ note for why.
            with torch.no_grad():
                wav_lens = torch.ones(raw_audio.size(0), device=raw_audio.device)
                beats_features = self.beats(raw_audio, wav_lens)[0]  # (B, T_beats, 768)
            beats_features = F.interpolate(
                beats_features.transpose(1, 2), size=x.size(1), mode="linear", align_corners=True
            ).transpose(1, 2)  # (B, T, 768)
            x = torch.cat([x, beats_features], dim=-1)  # (B, T, 1280+768)

        if self.use_wavlm:
            # Same fusion shape as BEATs above (WavLM-base+ also outputs 768-dim,
            # ~50Hz frame rate -- upsampled the same way). Stage 1: frozen, no_grad.
            # Stage 2 (after unfreeze_wavlm()): gradients flow, no_grad is skipped.
            ctx = torch.no_grad() if not getattr(self, "_wavlm_unfrozen", False) else torch.enable_grad()
            with ctx:
                wavlm_features = self.wavlm(raw_audio).last_hidden_state  # (B, T_wavlm, 768)
            wavlm_features = F.interpolate(
                wavlm_features.transpose(1, 2), size=x.size(1), mode="linear", align_corners=True
            ).transpose(1, 2)  # (B, T, 768)
            x = torch.cat([x, wavlm_features], dim=-1)  # (B, T, 1280+768)

        if self.use_atst_frame:
            # Frozen ATST-Frame branch: its own mel front end (hop_length=160,
            # matching Config.HOP_LENGTH, but independently computed -- n_mels=64
            # and different normalization from the CNN branch's own 80-mel front
            # end, so it must NOT reuse self.mel_transform/self.amplitude_to_db
            # above). Native rate ~25Hz (40ms/patch) is coarser than our 100Hz CNN
            # rate, same situation as WavLM's ~50Hz -- upsampled identically.
            with torch.no_grad():
                atst_mel = self.atst_frame.mel_forward(raw_audio)  # zero learnable params, always no_grad
            ctx = torch.no_grad() if not getattr(self, "_atst_frame_unfrozen", False) else torch.enable_grad()
            with ctx:
                atst_features = self.atst_frame(atst_mel)  # (B, T_atst, 768)
            atst_features = F.interpolate(
                atst_features.transpose(1, 2), size=x.size(1), mode="linear", align_corners=True
            ).transpose(1, 2)  # (B, T, 768)
            x = torch.cat([x, atst_features], dim=-1)  # (B, T, 1280+768)

        x = self.temporal_proj(x)  # (B, T, CONF_DIM)
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)  # (B, T, CONF_DIM)
        main_logits = self.classifier(x).squeeze(-1)  # (B, T)
        extras = []
        if self.use_category_head:
            extras.append(self.category_head(x))  # (B,T,7)
        if self.use_boundary_head:
            extras.append(self.boundary_head(x))  # (B,T,2)
        if extras:
            return (main_logits, *extras)
        return main_logits


class StandaloneATSTSED(nn.Module):
    """3.5-week re-plan, Must-do #2: a genuinely INDEPENDENT ATST-Frame-backbone
    model -- not a frozen-feature branch fused into DualBranchSED's shared
    CNN+Conformer trunk, the pattern every prior BEATs/WavLM/ATST-Frame fusion
    attempt used. That shared-trunk design is the diagnosed reason none of
    those additions ever moved the ensemble despite strong standalone scores:
    their predictions all pass through the same trunk's own biases, so their
    errors correlate with each other and with every other ensemble member
    built on that trunk (see STRATEGY.md's "same-architecture-family ensemble
    diversity ceiling" finding). This model has its OWN temporal head, OWN
    classifier, and gets its OWN independently-tuned Viterbi decode -- there
    is no shared trunk left for its errors to correlate through.

    Reuses _load_atst_frame_backbone/ATSTWrapper verbatim (same checkpoint,
    same mel front end, same separate_params() for LLRD unfreezing) and the
    same temporal_proj -> Conformer -> classifier shape DualBranchSED uses
    downstream of its trunk -- just fed directly from ATST-Frame's raw
    768-dim output instead of a CNN-concatenated feature. Frozen by default;
    call unfreeze_atst_frame() for a Stage-2-style LLRD+consistency-loss
    fine-tune, mirroring DualBranchSED's own unfreeze_atst_frame() exactly."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.atst_frame = _load_atst_frame_backbone(
            getattr(config, "ATST_FRAME_CKPT_PATH", "./checkpoints/atst_frame/ATST-F_strong_1.pt"))
        for param in self.atst_frame.parameters():
            param.requires_grad = False
        self.atst_frame.eval()
        self._atst_frame_unfrozen = False

        self.temporal_proj = nn.Linear(768, config.CONF_DIM)
        self.conformer = torchaudio.models.Conformer(
            input_dim=config.CONF_DIM,
            num_heads=config.CONF_HEADS,
            ffn_dim=config.CONF_DIM * 4,
            num_layers=config.CONF_LAYERS,
            depthwise_conv_kernel_size=31,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def train(self, mode=True):
        """Mirrors DualBranchSED.train(): keep the frozen ATST-Frame backbone
        in eval mode regardless of the parent's mode, until unfreeze_atst_
        frame() has been called."""
        super().train(mode)
        if not self._atst_frame_unfrozen:
            self.atst_frame.eval()
        return self

    def unfreeze_atst_frame(self):
        """Stage 2 (this standalone model's own): unfreeze ATST-Frame for
        end-to-end fine-tuning. Mirrors DualBranchSED.unfreeze_atst_frame()
        exactly. Caller is responsible for layer-wise LR decay (via
        self.atst_frame.separate_params()) and consistency-loss
        regularization -- unfreezing with a plain supervised loss already
        failed decisively for WavLM (see STRATEGY.md)."""
        self._atst_frame_unfrozen = True
        for param in self.atst_frame.parameters():
            param.requires_grad = True

    def forward(self, raw_audio):
        target_frames = self.config.target_samples // self.config.HOP_LENGTH
        with torch.no_grad():
            atst_mel = self.atst_frame.mel_forward(raw_audio)  # zero learnable params, always no_grad
        ctx = torch.no_grad() if not self._atst_frame_unfrozen else torch.enable_grad()
        with ctx:
            atst_features = self.atst_frame(atst_mel)  # (B, T_atst~25Hz, 768)
        atst_features = F.interpolate(
            atst_features.transpose(1, 2), size=target_frames, mode="linear", align_corners=True
        ).transpose(1, 2)  # (B, target_frames, 768)

        x = self.temporal_proj(atst_features)  # (B, T, CONF_DIM)
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)  # (B, T, CONF_DIM)
        return self.classifier(x).squeeze(-1)  # (B, T)


def _init_dual_head(model, config):
    """Sep 29 two-convention model. Natural references are human judgements (tighter
    edges); synthetic references are exact mix times (full acoustic extent). With
    Config.DUAL_HEAD, `classifier` learns the human convention (Gold/Natural data only)
    and `classifier_syn` the synthetic one; Config.OUTPUT_HEAD picks what forward()
    returns: "nat" (default), "syn", or "both" ((B, T, 2) = [nat, syn]). Off by default,
    so every existing checkpoint and call site is unchanged."""
    model.dual_head = getattr(config, "DUAL_HEAD", False)
    model.output_head = getattr(config, "OUTPUT_HEAD", "nat")
    if model.dual_head:
        model.classifier_syn = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
    # Phase C per-category model: 7 category logits next to the any-noise head.
    # OUTPUT_HEAD="anycat" returns (B, T, 8) = [any-noise, 7 categories].
    model.cat_head = getattr(config, "CAT_HEAD", False)
    if model.cat_head:
        model.classifier_cat = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 7),
        )


def _dual_output(model, x):
    nat = model.classifier(x).squeeze(-1)
    if getattr(model, "cat_head", False) and model.output_head == "anycat":
        return torch.cat([nat.unsqueeze(-1), model.classifier_cat(x)], dim=-1)
    if not getattr(model, "dual_head", False) or model.output_head == "nat":
        return nat
    syn = model.classifier_syn(x).squeeze(-1)
    return syn if model.output_head == "syn" else torch.stack([nat, syn], dim=-1)


def load_dual_warm_start(model, state_dict):
    """Load a single-head (or dual) checkpoint into a DUAL_HEAD model. A missing
    synthetic head starts as a copy of the checkpoint's classifier. Returns True if
    the synthetic head was initialised that way."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("classifier_syn.") for k in missing), missing
    if missing:
        model.classifier_syn.load_state_dict(model.classifier.state_dict())
    return bool(missing)


class StandaloneDashengSED(nn.Module):
    """3.5-week re-plan: standalone model using Dasheng (mispeech/dasheng-base,
    Apache-2.0, huggingface.co/mispeech/dasheng-base) as its backbone instead
    of ATST-Frame -- a genuinely different pretraining lineage (Dasheng is a
    1.2B-parameter-family MAE-style audio transformer pretrained on 272k
    hours of general audio; ATST-Frame is pretrained on AudioSet-Strong for
    frame-level SED specifically). This matters because StandaloneATSTSED,
    despite having no shared trainable trunk with any existing ensemble
    member, still failed the ensemble-swap test -- most likely because it
    shares the exact same frozen ATST-Frame checkpoint already used twice in
    the ensemble (atst_stage1/atst_stage2), so errors rooted in that shared
    representation still correlated (see STRATEGY.md). Dasheng has never
    been used anywhere in this project's ensemble, so this failure mode
    doesn't apply here.

    Feasibility confirmed directly (not assumed): `mispeech/dasheng-base`
    loads via `transformers.AutoModel.from_pretrained(..., trust_remote_
    code=True)`, 85.45M params (same class as ATST-Frame's 85.42M), produces
    genuine frame-level `hidden_states` (768-dim, ~25Hz for 10s audio --
    (B, 250, 768)), its own feature extractor's hop_size=160/sampling_rate=
    16000 match this project's Config.HOP_LENGTH/SR exactly, and a batch of
    8 uses only ~382MB GPU memory. Same temporal_proj -> Conformer ->
    classifier downstream shape as StandaloneATSTSED, for a fair, isolated
    architecture-family comparison.

    The mel front end is instantiated once here (own MelSpectrogram/
    AmplitudeToDB modules, zero learnable params) rather than calling
    DashengFeatureExtractor's __call__ every forward pass (which rebuilds
    those transforms from scratch each time) -- functionally identical
    output, matching DashengFeatureExtractor's exact parameters
    (n_fft=512, win_length=512, hop_length=160, n_mels=64, f_min=0,
    f_max=8000, center=True), confirmed by direct inspection of its source."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        from transformers import AutoModel
        # DASHENG_NAME selects the size (Oct 10: dasheng-0.6B / 1.2B); default keeps every
        # existing base checkpoint loading unchanged.
        name = os.environ.get("DASHENG_NAME", "mispeech/dasheng-base")
        self.dasheng = AutoModel.from_pretrained(name, trust_remote_code=True)
        self._dasheng_depth = 1 + max(int(m.group(1)) for m in (
            re.search(r"blocks\.(\d+)\.", n) for n, _ in self.dasheng.named_parameters()) if m)
        dasheng_dim = self.dasheng.encoder.norm.weight.shape[0]
        for param in self.dasheng.parameters():
            param.requires_grad = False
        self.dasheng.eval()
        self._dasheng_unfrozen = False

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SR, n_fft=512, win_length=512, hop_length=160,
            f_min=0, f_max=8000, n_mels=64, center=True,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=120)

        # CONTEXT=mae (Oct 12): the Conformer head is replaced by src/context_mae.ContextNet at the
        # backbone's native 25 Hz, pre-trained by masked reconstruction on in-domain audio
        # (CONTEXT_INIT; CONTEXT_RANDOM=1 keeps only its feature statistics = the control).
        self.context_kind = os.environ.get("CONTEXT", "conformer")
        if self.context_kind == "mae":
            from src.context_mae import ContextNet
            self.context = ContextNet(d_in=dasheng_dim)
            self.register_buffer("ctx_mu", torch.zeros(dasheng_dim))
            self.register_buffer("ctx_sd", torch.ones(dasheng_dim))
            init = os.environ.get("CONTEXT_INIT")
            if init:
                ck = torch.load(init, map_location="cpu", weights_only=True)
                self.ctx_mu.copy_(ck["mu"]), self.ctx_sd.copy_(ck["sd"])
                if os.environ.get("CONTEXT_RANDOM") != "1":
                    self.context.load_state_dict(ck["state"])
            head_dim = self.context.d
        else:
            self.temporal_proj = nn.Linear(dasheng_dim, config.CONF_DIM)
            self.conformer = torchaudio.models.Conformer(
                input_dim=config.CONF_DIM,
                num_heads=config.CONF_HEADS,
                ffn_dim=config.CONF_DIM * 4,
                num_layers=config.CONF_LAYERS,
                depthwise_conv_kernel_size=31,
            )
            head_dim = config.CONF_DIM
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
        _init_dual_head(self, config)

    def train(self, mode=True):
        super().train(mode)
        if not self._dasheng_unfrozen:
            self.dasheng.eval()
        return self

    def unfreeze_dasheng(self):
        """Mirrors unfreeze_atst_frame()/unfreeze_wavlm() exactly. Caller is
        responsible for a low-LR/LLRD schedule and consistency-loss
        regularization if this is ever used -- unfreezing with a plain
        supervised loss already failed decisively for WavLM."""
        self._dasheng_unfrozen = True
        for param in self.dasheng.parameters():
            param.requires_grad = True

    def separate_params(self):
        """Layer-wise-LR-decay groups for the Dasheng backbone, mirroring
        ATSTWrapper.separate_params() exactly: 12 groups, ordered DEEPEST
        FIRST (group 0 = block 11 + the final encoder.norm, highest LR
        under LLRD; group 11 = embeddings/positional/init_bn + block 0,
        lowest LR). Raises on any unassigned parameter so an architecture
        change can never silently drop params out of the optimizer."""
        import re
        groups = [[] for _ in range(12)]
        for name, param in self.dasheng.named_parameters():
            match = re.search(r"blocks\.(\d+)\.", name)
            if match:
                # 12 groups whatever the depth (base: identity; 0.6B: 32 blocks -> 12 groups)
                groups[int(match.group(1)) * 12 // self._dasheng_depth].append(param)
            elif name.startswith("encoder.norm."):
                groups[11].append(param)
            elif name.startswith(("encoder.time_pos_embed", "encoder.freq_pos_embed",
                                  "encoder.init_bn.", "encoder.patch_embed.")):
                groups[0].append(param)
            else:
                raise ValueError(f"Unassigned Dasheng parameter for LLRD: {name}")
        return list(reversed(groups))

    def mel_forward(self, raw_audio):
        mel = self.mel_transform(raw_audio)  # (B, 64, T)
        return self.amplitude_to_db(mel)

    def frame_features(self, raw_audio):
        """Backbone features on the 10 ms grid, (B, target_frames, D), before the head."""
        target_frames = self.config.target_samples // self.config.HOP_LENGTH
        with torch.no_grad():
            dasheng_mel = self.mel_forward(raw_audio)  # zero learnable params, always no_grad
        ctx = torch.no_grad() if not self._dasheng_unfrozen else torch.enable_grad()
        with ctx:
            dasheng_features = self.dasheng(dasheng_mel).hidden_states  # (B, T_dasheng~25Hz, 768)
        return F.interpolate(
            dasheng_features.transpose(1, 2), size=target_frames, mode="linear", align_corners=True
        ).transpose(1, 2)  # (B, target_frames, 768)

    def forward(self, raw_audio):
        if self.context_kind == "mae":
            with torch.no_grad():
                mel = self.mel_forward(raw_audio)
            ctx = torch.no_grad() if not self._dasheng_unfrozen else torch.enable_grad()
            with ctx:
                f = self.dasheng(mel).hidden_states  # (B, T25, D)
            h, _ = self.context((f - self.ctx_mu) / self.ctx_sd)  # (B, T25, d)
            x = tokens_to_frames(h, 4, self.config.target_samples // self.config.HOP_LENGTH)
            return _dual_output(self, x)
        x = self.temporal_proj(self.frame_features(raw_audio))  # (B, T, CONF_DIM)
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)  # (B, T, CONF_DIM)
        return _dual_output(self, x)  # (B, T), or (B, T, 2) with OUTPUT_HEAD="both"




class StandaloneWhisperSED(nn.Module):
    """Win-strategy Lane B (Oct 10): a speech-ASR encoder as the backbone -- a lineage none of
    our general-audio members share. Vaani noise events are non-speech sounds inside speech;
    Whisper's encoder keeps the background-sound type (Whisper-AT, Interspeech 2023) and has
    20 ms frames (vs 40 ms for Dasheng/ATST).

    WHISPER_NAME (default openai/whisper-large-v3); encoder only, frozen, run in fp16/bf16.
    The 30 s positional table is cut to the clip's 10 s (500 positions) so a 10 s clip is not
    padded to 30 s (3x compute); log-mel follows WhisperFeatureExtractor exactly (n_fft 400,
    hop 160, log10, max-8 floor, (x+4)/4), computed on the GPU."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        from transformers import WhisperFeatureExtractor, WhisperModel
        name = os.environ.get("WHISPER_NAME", "openai/whisper-large-v3")
        self.whisper = WhisperModel.from_pretrained(name, torch_dtype=torch.float16).encoder
        n_frames = config.target_samples // config.HOP_LENGTH  # 1000 mel frames = 10 s
        n_pos = n_frames // 2
        self.whisper.embed_positions.weight.data = self.whisper.embed_positions.weight.data[:n_pos].clone()
        self.whisper.embed_positions.num_embeddings = n_pos
        self.whisper.config.max_source_positions = n_pos
        self._n_mel_frames = n_frames
        for param in self.whisper.parameters():
            param.requires_grad = False
        self.whisper.eval()
        fe = WhisperFeatureExtractor.from_pretrained(name)
        self.register_buffer("mel_filters", torch.tensor(fe.mel_filters, dtype=torch.float32), persistent=False)
        self.register_buffer("stft_window", torch.hann_window(400), persistent=False)
        dim = self.whisper.config.d_model
        self.temporal_proj = nn.Linear(dim, config.CONF_DIM)
        self.conformer = torchaudio.models.Conformer(
            input_dim=config.CONF_DIM, num_heads=config.CONF_HEADS, ffn_dim=config.CONF_DIM * 4,
            num_layers=config.CONF_LAYERS, depthwise_conv_kernel_size=31,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
        _init_dual_head(self, config)

    def train(self, mode=True):
        super().train(mode)
        self.whisper.eval()
        return self

    def state_dict(self, *args, **kwargs):
        # the frozen 635M encoder is reloaded from the hub; checkpoints keep only the trainable part
        sd = super().state_dict(*args, **kwargs)
        return type(sd)((k, v) for k, v in sd.items() if not k.startswith(("whisper.", "branch.whisper.")))

    def load_state_dict(self, state_dict, strict=True):
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("whisper.")}
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith("whisper.")]
        if strict and (missing or unexpected):
            raise RuntimeError(f"StandaloneWhisperSED load: missing {missing}, unexpected {unexpected}")
        return missing, unexpected

    def mel_forward(self, raw_audio):
        stft = torch.stft(raw_audio.float(), 400, 160, window=self.stft_window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2  # (B, 201, T)
        mel = (self.mel_filters.T @ mag).clamp(min=1e-10).log10()  # (B, n_mels, T)
        mel = torch.maximum(mel, mel.amax(dim=(1, 2), keepdim=True) - 8.0)
        mel = (mel + 4.0) / 4.0
        n = self._n_mel_frames
        return F.pad(mel, (0, max(0, n - mel.size(-1))))[..., :n]

    def frame_features(self, raw_audio):
        """(B, target_frames, d_model) on the 10 ms grid: 20 ms tokens, token t centred on frames 2t..2t+1."""
        target_frames = self.config.target_samples // self.config.HOP_LENGTH
        with torch.no_grad():
            mel = self.mel_forward(raw_audio).to(torch.float16)
            feats = self.whisper(mel).last_hidden_state.float()  # (B, 500, d_model)
        return tokens_to_frames(feats, 2, target_frames)

    def forward(self, raw_audio):
        x = self.temporal_proj(self.frame_features(raw_audio))
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)
        return _dual_output(self, x)


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank update (B @ A) * alpha / r; B starts at zero, so the
    wrapped layer is exactly the pretrained one at initialisation."""

    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * (1.0 / base.in_features ** 0.5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


class StandaloneW2vBertSED(nn.Module):
    """Lane B (Oct 11): w2v-BERT 2.0 (facebook/w2v-bert-2.0, MIT; 580M Conformer, SSL on 4.5M h of
    speech in 143 languages incl. Indic) with LoRA (rank LORA_RANK on q/k/v/out of every layer),
    the setup of the non-verbal (breath/laugh) start/end detector of arXiv:2508.05385. The rest of
    the encoder is frozen. Front end = SeamlessM4TFeatureExtractor reproduced on the GPU: Kaldi
    fbank (80 mel, 25/10 ms, povey, preemphasis 0.97) on 2^15-scaled audio, per-utterance mean/var
    normalisation per bin, stride-2 stacking -> 160-d at 20 ms."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        from transformers import Wav2Vec2BertModel
        self.w2v = Wav2Vec2BertModel.from_pretrained(os.environ.get("W2VBERT_NAME", "facebook/w2v-bert-2.0"))
        for param in self.w2v.parameters():
            param.requires_grad = False
        r = int(os.environ.get("LORA_RANK", "16"))
        for layer in self.w2v.encoder.layers:
            att = layer.self_attn
            for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
                setattr(att, name, LoRALinear(getattr(att, name), r=r, alpha=2 * r))
        dim = self.w2v.config.hidden_size
        self.temporal_proj = nn.Linear(dim, config.CONF_DIM)
        self.conformer = torchaudio.models.Conformer(
            input_dim=config.CONF_DIM, num_heads=config.CONF_HEADS, ffn_dim=config.CONF_DIM * 4,
            num_layers=config.CONF_LAYERS, depthwise_conv_kernel_size=31,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
        _init_dual_head(self, config)

    def mel_forward(self, raw_audio):
        feats = []
        for wav in raw_audio.float():
            fb = torchaudio.compliance.kaldi.fbank(
                (wav * 32768.0).unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0.0,
                window_type="povey", preemphasis_coefficient=0.97, remove_dc_offset=True,
                frame_length=25.0, frame_shift=10.0, use_energy=False)  # (T, 80)
            fb = (fb - fb.mean(0, keepdim=True)) / torch.sqrt(fb.var(0, keepdim=True, unbiased=True) + 1e-7)
            t = fb.size(0) // 2 * 2
            feats.append(fb[:t].reshape(t // 2, 160))
        return torch.stack(feats)  # (B, T/2, 160)

    def frame_features(self, raw_audio):
        """(B, target_frames, 1024) on the 10 ms grid; token t (20 ms) covers fbank frames 2t, 2t+1."""
        target_frames = self.config.target_samples // self.config.HOP_LENGTH
        with torch.no_grad():
            x = self.mel_forward(raw_audio)
        feats = self.w2v(input_features=x).last_hidden_state.float()
        return tokens_to_frames(feats, 2, target_frames)

    def forward(self, raw_audio):
        x = self.temporal_proj(self.frame_features(raw_audio))
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)
        return _dual_output(self, x)

def beats_tokens_to_frames(feats, n_freq, freq_proj, target_frames, patch_frames=16):
    """BEATs emits 16x16 fbank patches flattened time-major: token index = t * n_freq + f
    (speechbrain beats.py extract_features: (B, 768, T, F) -> reshape(B, 768, -1)).
    Interpolating that sequence as if it were time gives 160 ms true resolution with the
    8 frequency bands sweeping across each block. Instead: regroup to (B, T, F*D), project
    back to D, and place token t at its real centre, fbank frame patch_frames * t +
    patch_frames / 2 (10 ms shift), linearly interpolating onto the 10 ms grid."""
    b, n, d = feats.shape
    t_steps = n // n_freq
    assert t_steps * n_freq == n, f"BEATs token count {n} not divisible by {n_freq} frequency patches"
    x = freq_proj(feats.reshape(b, t_steps, n_freq * d))  # (B, T, D)
    return tokens_to_frames(x, patch_frames, target_frames)


def tokens_to_frames(x, frames_per_token, target_frames):
    """Linearly interpolate (B, T, D) tokens onto the 10 ms grid with token t centred on
    frame frames_per_token * t + (frames_per_token - 1) / 2 -- where a patch of
    frames_per_token mel frames actually sits -- instead of align_corners stretching,
    which shifts boundaries by up to ~frames_per_token / 2 frames near the clip edges."""
    t_steps = x.size(1)
    pos = (torch.arange(target_frames, device=x.device, dtype=x.dtype)
           - (frames_per_token - 1) / 2) / frames_per_token
    pos = pos.clamp(0, t_steps - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=t_steps - 1)
    w = (pos - lo.to(pos.dtype)).view(1, -1, 1)
    return x[:, lo] * (1 - w) + x[:, hi] * w  # (B, target_frames, D)


class StandaloneBEATsSED(nn.Module):
    """Third backbone family for the unfrozen + LLRD + Mean Teacher + ICT
    ("Stage 2") recipe -- the only recipe that has added real ensemble value
    this season (ATST S2 +0.0096 real, Dasheng S2 +0.0101 real). BEATs
    (iter3+ AS2M, local checkpoint) is a lineage never used in the current
    ensemble. The early frozen-fusion BEATs attempt regressed, but frozen
    Dasheng also failed the ensemble test while its Stage 2 passed, so the
    frozen result does not predict the Stage 2 one.

    Same temporal_proj -> Conformer -> classifier head as StandaloneATSTSED/
    StandaloneDashengSED. BEATs does its own fbank front end on raw 16 kHz
    audio and outputs (B, ~496, 768) for 10 s (~50 Hz), upsampled to the
    project's 100 Hz frame grid. Built with speechbrain's freeze=False so its
    forward never forces no_grad; freezing is controlled here, like Dasheng.

    Config.BEATS_FREQ_RESHAPE=True fixes the time axis: the ~496 tokens are really
    62 time steps x 8 frequency patches (160 ms per step), see beats_tokens_to_frames.
    Default False keeps older checkpoints loadable (they were trained on the
    flattened-interpolation features)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.beats = BEATs(ckp_path=getattr(config, "BEATS_CKPT_PATH",
                                            "./checkpoints/beats/BEATs_iter3_plus_AS2M.pt"), freeze=False)
        for param in self.beats.parameters():
            param.requires_grad = False
        self.beats.eval()
        self._beats_unfrozen = False
        self.freq_reshape = getattr(config, "BEATS_FREQ_RESHAPE", False)
        if self.freq_reshape:
            self.n_freq_patches = 128 // 16  # 128-bin fbank / 16-bin patches
            self.freq_proj = nn.Linear(768 * self.n_freq_patches, 768)

        self.temporal_proj = nn.Linear(768, config.CONF_DIM)
        self.conformer = torchaudio.models.Conformer(
            input_dim=config.CONF_DIM,
            num_heads=config.CONF_HEADS,
            ffn_dim=config.CONF_DIM * 4,
            num_layers=config.CONF_LAYERS,
            depthwise_conv_kernel_size=31,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
        _init_dual_head(self, config)

    def train(self, mode=True):
        super().train(mode)
        if not self._beats_unfrozen:
            self.beats.eval()
        return self

    def unfreeze_beats(self):
        self._beats_unfrozen = True
        for param in self.beats.parameters():
            param.requires_grad = True

    def separate_params(self):
        """12 LLRD groups, deepest first, mirroring ATSTWrapper/Dasheng.
        `encoder.layer_norm` runs BEFORE the layers (layer_norm_first=False,
        verified in speechbrain's beats.py), so it joins the input-side group
        with the patch embedding / positional conv / pre-projection norm."""
        import re
        groups = [[] for _ in range(12)]
        for name, param in self.beats.named_parameters():
            match = re.search(r"encoder\.layers\.(\d+)\.", name)
            if match:
                groups[int(match.group(1))].append(param)
            elif name.startswith(("post_extract_proj.", "patch_embedding.", "encoder.pos_conv.",
                                  "encoder.layer_norm.", "layer_norm.")):
                groups[0].append(param)
            else:
                raise ValueError(f"Unassigned BEATs parameter for LLRD: {name}")
        return list(reversed(groups))

    def frame_features(self, raw_audio):
        """Backbone features on the 10 ms grid, (B, target_frames, D), before the head."""
        target_frames = self.config.target_samples // self.config.HOP_LENGTH
        wav_lens = torch.ones(raw_audio.size(0), device=raw_audio.device)
        ctx = torch.no_grad() if not self._beats_unfrozen else torch.enable_grad()
        with ctx:
            feats = self.beats(raw_audio, wav_lens)[0]  # (B, ~496, 768)
        if self.freq_reshape:
            return beats_tokens_to_frames(feats, self.n_freq_patches, self.freq_proj, target_frames)
        return F.interpolate(
            feats.transpose(1, 2), size=target_frames, mode="linear", align_corners=True
        ).transpose(1, 2)

    def forward(self, raw_audio):
        x = self.temporal_proj(self.frame_features(raw_audio))
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)
        return _dual_output(self, x)


class StandaloneStrongSED(nn.Module):
    """Plan Step 2: AudioSet-Strong pretrained transformer (PretrainedSED, vendored in
    src/pretrained_sed, loaded via src/strong_backbones.build_strong) + the same
    temporal_proj -> Conformer -> classifier head as the other Standalone models.

    Unlike our SSL/clip-level backbones, these were fine-tuned on ~100k AudioSet clips
    with human onset/offset labels (447 classes incl. breathing, vehicles, animals,
    baby cry, alarms, music), so their frame features already encode boundaries.

    Backbone output is (B, 250, D) at 40 ms (4 mel frames per token), placed at the
    true token centres on the 10 ms grid (tokens_to_frames). Config.STRONG_BACKBONE:
    "ATST-F" | "BEATs" | "ASIT" | "fpasst" | "M2D"; Config.STRONG_STAGE: "strong" |
    "weak" | "ssl" (same architecture, for controlled comparisons). Frozen by default;
    unfreeze_backbone() + separate_params() (12 groups, deepest first) for Stage 2."""

    FRAMES_PER_TOKEN = 4

    def __init__(self, config):
        super().__init__()
        from src.strong_backbones import build_strong
        self.config = config
        self.backbone = build_strong(getattr(config, "STRONG_BACKBONE", "ATST-F"), head_type=None,
                                     stage=getattr(config, "STRONG_STAGE", "strong"))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_unfrozen = False

        self.temporal_proj = nn.Linear(self.backbone.embed_dim, config.CONF_DIM)
        self.conformer = torchaudio.models.Conformer(
            input_dim=config.CONF_DIM,
            num_heads=config.CONF_HEADS,
            ffn_dim=config.CONF_DIM * 4,
            num_layers=config.CONF_LAYERS,
            depthwise_conv_kernel_size=31,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.CONF_DIM, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        if not self._backbone_unfrozen:
            self.backbone.eval()
        return self

    def unfreeze_backbone(self):
        self._backbone_unfrozen = True
        for param in self.backbone.parameters():
            param.requires_grad = True

    def separate_params(self):
        return self.backbone.model.separate_params()

    def backbone_parameters(self):
        return self.backbone.model.parameters()

    def frame_features(self, raw_audio):
        """Backbone features on the 10 ms grid, (B, target_frames, D), before the head."""
        target_frames = self.config.target_samples // self.config.HOP_LENGTH
        ctx = torch.no_grad() if not self._backbone_unfrozen else torch.enable_grad()
        with ctx:
            feats = self.backbone(self.backbone.mel_forward(raw_audio))  # (B, 250, D)
        return tokens_to_frames(feats, self.FRAMES_PER_TOKEN, target_frames)

    def forward(self, raw_audio):
        x = self.temporal_proj(self.frame_features(raw_audio))
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)
        return self.classifier(x).squeeze(-1)



class FusionSED(DualBranchSED):
    """Plan Step 3: the load-bearing atst_stage2 structure -- DualBranchSED's genuine
    10 ms CNN branch concatenated with an (optionally unfrozen) pretrained backbone
    branch -- generalised to the Dasheng / BEATs / AudioSet-Strong backbones, which
    today exist only as Standalone models whose only time evidence is 40-160 ms
    tokens upsampled to 10 ms. The CNN supplies real 10 ms evidence for offsets and
    short events (77% of Natural near misses are offset errors; <0.3 s events have
    recall 0.21).

    Config.FUSION_BRANCH: "dasheng" | "beats" | "strong" (the matching Standalone
    class supplies frame_features(); its own head is dropped). The branch can be
    warm-started from an already fine-tuned Standalone Stage-2 checkpoint with
    load_branch_from(). Branch frozen by default; unfreeze_branch() +
    separate_params() (12 LLRD groups, deepest first) for Stage 2."""

    def __init__(self, config):
        for flag in ("USE_BEATS", "USE_WAVLM", "USE_ATST_FRAME"):
            assert not getattr(config, flag, False), f"FusionSED carries its own branch; {flag} must be off"
        super().__init__(config)
        kind = getattr(config, "FUSION_BRANCH", "dasheng")
        cls = {"dasheng": StandaloneDashengSED, "beats": StandaloneBEATsSED, "strong": StandaloneStrongSED,
               "whisper": StandaloneWhisperSED, "w2vbert": StandaloneW2vBertSED}[kind]
        self.fusion_kind = kind
        self.branch = cls(config)
        del self.branch.temporal_proj, self.branch.conformer, self.branch.classifier  # head unused here
        branch_dim = {"strong": lambda: self.branch.backbone.embed_dim,
                      "whisper": lambda: self.branch.whisper.config.d_model,
                      "w2vbert": lambda: self.branch.w2v.config.hidden_size,
                      "dasheng": lambda: self.branch.dasheng.encoder.norm.weight.shape[0]}.get(kind, lambda: 768)()
        self.temporal_proj = nn.Linear(self._cnn_out_ch * 10 + branch_dim, config.CONF_DIM)

    _HEAD_PREFIXES = ("temporal_proj.", "conformer.", "classifier.")

    def load_branch_from(self, standalone_state_dict):
        """Load a Standalone{Dasheng,BEATs,Strong}SED state_dict into the branch (its head
        keys are dropped). Strict on everything else."""
        sd = {k: v for k, v in standalone_state_dict.items() if not k.startswith(self._HEAD_PREFIXES)}
        self.branch.load_state_dict(sd, strict=True)

    def _branch_backbone(self):
        return {"dasheng": lambda: self.branch.dasheng, "beats": lambda: self.branch.beats,
                "strong": lambda: self.branch.backbone.model,
                "whisper": lambda: self.branch.whisper,
                "w2vbert": lambda: self.branch.w2v}[self.fusion_kind]()

    def branch_parameters(self):
        return self._branch_backbone().parameters()

    def unfreeze_branch(self):
        method = {"dasheng": "unfreeze_dasheng", "beats": "unfreeze_beats", "strong": "unfreeze_backbone"}
        getattr(self.branch, method[self.fusion_kind])()

    def separate_params(self):
        return self.branch.separate_params()

    def train(self, mode=True):
        super().train(mode)
        self.branch.train(mode)  # the Standalone class keeps a frozen backbone in eval
        return self

    def forward(self, x):
        x, raw_audio = self._cnn_frames(x)  # (B, T, C*10)
        feats = self.branch.frame_features(raw_audio)  # (B, target_frames, D)
        x = torch.cat([x, feats[:, :x.size(1)]], dim=-1)
        x = self.temporal_proj(x)
        lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)
        return self.classifier(x).squeeze(-1)
