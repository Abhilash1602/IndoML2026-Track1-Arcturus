import os

import torch


def select_device():
    requested = os.environ.get("VAANI_DEVICE")
    if requested:
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    free_memory = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return f"cuda:{max(range(len(free_memory)), key=free_memory.__getitem__)}"


class Config:
    SR = 16000
    HOP_LENGTH = 160
    MAX_DURATION = 10.0
    N_MELS = 80
    N_FFT = 1024
    CONF_DIM = 144
    CONF_HEADS = 4
    CONF_LAYERS = 4
    BATCH_SIZE = 16
    # Session 7c: scaled up from 20. Every DCASE SOTA recipe behind the techniques
    # tried this session (FDY-Conv, FilterAugment, mixup) trains for 200-300 epochs;
    # our per-epoch time is now ~2-3min with no GPU contention, so there's real
    # headroom to actually match that training volume instead of ~20 lean epochs.
    # EARLY_STOP_PATIENCE raised alongside it since mixup/FilterAugment add
    # per-epoch noise that a patience of 4 could mistake for a real plateau.
    EPOCHS = 60
    LR = 1e-3
    DISTILLED_LR = float(os.environ.get("DISTILLED_LR", "3e-4"))  # env override: Gold fine-tunes
    LOSS_BCE_W = 1.0
    LOSS_DICE_W = 0.5
    LOSS_BND_W = 0.2
    # OWBCE (arXiv:2403.13254): per-frame BCE weight 1 + height * (sine-window
    # response around each onset/offset in the target mask). 0.0 = off, which
    # keeps SEDLoss byte-identical to the unweighted BCE. HALF_WIDTH is in
    # 10ms frames (window spans 2*HALF_WIDTH+1 frames).
    OWBCE_HEIGHT = 0.0
    OWBCE_HALF_WIDTH = 10
    MEAN_TEACHER_EMA_DECAY = 0.999
    MEAN_TEACHER_CONSISTENCY_WEIGHT = 1.0
    MEAN_TEACHER_RAMPUP_EPOCHS = 5
    EARLY_STOP_PATIENCE = 8
    NUM_WORKERS = 0
    DEVICE = select_device()
    # Session 7d: frozen BEATs branch fused alongside the CNN (see DualBranchSED).
    # DISABLED (USE_BEATS=False) -- a real Codabench submission showed it regressed
    # combined 1.0982->1.0669, concentrated in Synthetic F1 (0.4426->0.4003), the
    # same failure signature as FDY-Conv and Freq-MixStyle, despite the best local
    # score this project has ever produced. Flag + code kept for a possible revisit.
    USE_BEATS = False
    # Downloaded from the Bencr/beats-checkpoints HF dataset mirror of Microsoft's
    # official BEATs_iter3_plus_AS2M.pt (github.com/microsoft/unilm/tree/master/beats).
    BEATS_CKPT_PATH = "./checkpoints/beats/BEATs_iter3_plus_AS2M.pt"
    # StandaloneBEATsSED: regroup the 62 time x 8 frequency patch tokens before upsampling
    # (fixes the flattened-sequence interpolation). False keeps old checkpoints loadable.
    BEATS_FREQ_RESHAPE = os.environ.get("BEATS_FREQ_RESHAPE", "0") == "1"
    # StandaloneStrongSED (plan Step 2): PretrainedSED backbone and its pretraining stage.
    STRONG_BACKBONE = os.environ.get("STRONG_BACKBONE", "ATST-F")
    STRONG_STAGE = os.environ.get("STRONG_STAGE", "strong")
    # FusionSED (plan Step 3): which backbone joins the 10 ms CNN branch.
    FUSION_BRANCH = os.environ.get("FUSION_BRANCH", "dasheng")
    # Two-convention model (Sep 29): a second classifier head for the synthetic exact-mix
    # boundary convention; OUTPUT_HEAD selects nat | syn | both at forward time.
    DUAL_HEAD = os.environ.get("DUAL_HEAD", "0") == "1"
    # Phase C per-category model: 7 category logits beside the any-noise head (OUTPUT_HEAD="anycat").
    CAT_HEAD = os.environ.get("CAT_HEAD", "0") == "1"
    OUTPUT_HEAD = os.environ.get("OUTPUT_HEAD", "nat")
    # Session 8: frozen/staged WavLM branch (see DualBranchSED), same fusion shape
    # as BEATs above but speech-specific pretraining (denoising + masked
    # prediction) rather than BEATs' general-audio-event pretraining -- a
    # different mechanism the earlier frozen-BEATs regression doesn't rule out.
    # Mutually exclusive with USE_BEATS (both fuse a 768-dim branch the same way).
    USE_WAVLM = False
    WAVLM_NAME = "microsoft/wavlm-base-plus"

    # Session 9: frozen ATST-Frame branch (see DualBranchSED), a third 768-dim
    # fusion option alongside BEATs/WavLM above. Unlike BEATs (general-audio SSL)
    # and WavLM (generic speech SSL), ATST-Frame's ATST-F_strong_1.pt checkpoint is
    # already fine-tuned on AudioSet-Strong for frame-level SED -- a supervised
    # task closely matching our own, and its native ~25Hz (40ms hop) frame rate is
    # upsampled to the CNN branch's rate the same way WavLM's ~50Hz is. Mutually
    # exclusive with USE_BEATS/USE_WAVLM in practice (all three fuse a 768-dim
    # branch the same way). Frozen-only: the analogous "unfreeze at low LR" idea
    # was tried for WavLM (Stage 2) and decisively failed, so no
    # unfreeze_atst_frame() is planned.
    USE_ATST_FRAME = False
    # MIT-licensed checkpoint from fschmid56/PretrainedSED (AudioSet-Strong
    # fine-tuned ATST-Frame backbone), stable GitHub Releases URL, no auth needed.
    ATST_FRAME_CKPT_URL = (
        "https://github.com/fschmid56/PretrainedSED/releases/download/"
        "v0.0.1/ATST-F_strong_1.pt"
    )
    ATST_FRAME_CKPT_PATH = "./checkpoints/atst_frame/ATST-F_strong_1.pt"

    # Session 9: ATST-Frame Stage 2 (unfreeze + layer-wise LR decay + Mean
    # Teacher/ICT consistency regularization) -- the CPJKU DCASE-2024-winning
    # recipe's actual Stage 2, not the purely-supervised unfreeze that failed
    # for WavLM. ATST_BASE_LR/LLRD_DECAY feed the 12 LLRD param groups built
    # from ATSTWrapper.separate_params() (deepest layer first): group i gets
    # lr = ATST_BASE_LR * (LLRD_DECAY ** i). ICT_ALPHA matches apply_mixup's
    # own default alpha, reused for the interpolation-consistency loss.
    ATST_BASE_LR = float(os.environ.get("ATST_BASE_LR", "1e-4"))  # env override: Gold fine-tunes
    LLRD_DECAY = 0.5
    ICT_ALPHA = 0.2
    USE_FREQ_MIXSTYLE = False  # Session 9: re-enable behind a flag once validated inside the full augmentation stack
    # Session 9: independently toggleable so DIR can be isolated from Freq-Warp/
    # Freq-MixStyle -- the combined-stack smoke test showed no clear signal
    # either way in 6 epochs, and DIR specifically (real recording-condition
    # simulation) is the one flagged as most relevant to the Natural-domain gap.
    USE_DIR = True
    USE_FREQ_WARP = True

    # Session 9 plan, Part 1: multi-task 7-category auxiliary head. Gates both
    # src/dataset.py's _build_targets (whether it also builds a (T,7) cat_mask)
    # and DualBranchSED's optional category_head -- off by default so every
    # existing script/checkpoint is completely unaffected. CATEGORY_LOSS_WEIGHT
    # kept small/auxiliary, analogous to the existing LOSS_BND_W=0.2.
    USE_CATEGORY_HEAD = False
    CATEGORY_LOSS_WEIGHT = 0.2

    # Session 9 plan, Part 2 Stage 2A: onset/offset auxiliary head (OOL,
    # arXiv:2601.04178). Gates both src/dataset.py's _build_targets (whether
    # it also builds a (T,2) single-frame onset/offset boundary_mask) and
    # DualBranchSED's optional boundary_head -- off by default, same safe
    # pattern as USE_CATEGORY_HEAD above. Decode pipeline (viterbi_decode)
    # is completely untouched by this flag -- Stage 2A is auxiliary-loss-only.
    USE_BOUNDARY_HEAD = False
    BOUNDARY_LOSS_WEIGHT = 0.2

    @property
    def target_samples(self):
        return int(self.MAX_DURATION * self.SR)