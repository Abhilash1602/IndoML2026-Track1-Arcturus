import io
import json
import os

import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from datasets import Audio, load_dataset
from torch.utils.data import Dataset

# 1. Pull token securely from your server's OS environment variables
def load_full_dataset():
    return load_dataset(
        "ARTPARK-IISc/Vaani-Noise-Event-Dataset",
        split="train", token=os.environ.get("HF_TOKEN"), cache_dir="./data/hf_cache"
    )


def pad_or_trim(audio, target_len):
    return F.pad(audio, (0, target_len - audio.numel())) if audio.numel() < target_len else audio[:target_len]


# Session 9 plan, Part 1: the 7 real noise categories in NoiseSubCategoryTimeStamp's
# `category` field, verified by directly loading validationMetadata.json and the
# cached HF dataset (see temp/inspect_category_variance.py) -- exactly matches the
# org baseline's "8 channel = any-noise + 7 categories" design. Gate-checked for
# genuine per-frame variance on both Gold and Silver before being trusted here
# (temp/inspect_category_variance.py's STRATEGY.md-documented pass), avoiding the
# exact collapse-to-trivial-always-true bug that killed the earlier
# NoiseCategory-as-boolean attempt (train.py's fit_distilled docstring).
CATEGORY_LIST = ["human_non_speech", "vehicle_traffic", "animal", "baby_child",
                 "phone_signal_alarm", "singing_music", "appliance_machine"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}


def _write_event_targets(mask, cat_mask, boundary_mask, start_sec, end_sec, category, total_frames, config):
    """Session 9 plan, Part 2 Stage 2A full-mix run: the per-event target-
    writing math factored out of _build_targets so SyntheticSEDDataset (whose
    events use `onset`/`offset` keys, not `start`/`end`) can share the
    identical single-frame-impulse convention rather than duplicating it.
    Mutates `mask`/`cat_mask`/`boundary_mask` in place; `cat_mask`/
    `boundary_mask` may be None to skip that target.

    Boundary targets are single-frame binary impulses at the exact onset/
    offset frame, per arXiv:2601.04178's own stated convention ("one at
    frames where an onset or offset occurs, and zero otherwise") -- confirmed
    from the paper's text, not assumed. An event whose offset frame falls at
    or past `total_frames` (runs off the end of the clip) contributes no
    offset impulse, since there is no valid frame to place it on."""
    raw_end = int(float(end_sec) * config.SR / config.HOP_LENGTH)
    start = min(max(int(float(start_sec) * config.SR / config.HOP_LENGTH), 0), total_frames - 1)
    end = min(max(raw_end, 0), total_frames - 1)
    if start < end:
        mask[start:end] = 1.0
        if cat_mask is not None:
            idx = CATEGORY_TO_IDX.get(category)
            if idx is not None:
                cat_mask[start:end, idx] = 1.0
        if boundary_mask is not None:
            boundary_mask[start, 0] = 1.0
            if raw_end < total_frames:
                boundary_mask[end, 1] = 1.0


def _build_targets(events, total_frames, config):
    """Shared label-construction helper for VaaniDataset/OfficialValidationDataset
    (previously identical, duplicated logic). Returns (mask, cat_mask,
    boundary_mask) where `mask` is the existing (total_frames,) any-noise
    binary target (unchanged behavior), `cat_mask` is a (total_frames, 7)
    multi-hot per-category target (Session 9 plan Part 1, gated by
    Config.USE_CATEGORY_HEAD), and `boundary_mask` is a (total_frames, 2)
    [onset, offset] target (Session 9 plan Part 2 Stage 2A, gated by
    Config.USE_BOUNDARY_HEAD). Both aux targets are None when their flag is
    off, so every existing caller that unpacks (audio, mask, clip_id) is
    unaffected. Deliberately NOT folded into an 8-channel primary mask: the
    scored submission format is class-agnostic/binary-only, and mixing shapes
    risks breaking the wide surface of code (Viterbi decode, ensembling,
    generate_submission.py) that assumes a plain (B,T) primary output."""
    mask = torch.zeros(total_frames, dtype=torch.float32)
    use_category_head = getattr(config, "USE_CATEGORY_HEAD", False)
    use_boundary_head = getattr(config, "USE_BOUNDARY_HEAD", False)
    cat_mask = torch.zeros(total_frames, len(CATEGORY_LIST), dtype=torch.float32) if use_category_head else None
    boundary_mask = torch.zeros(total_frames, 2, dtype=torch.float32) if use_boundary_head else None
    for event in events or []:
        if "start" not in event or "end" not in event:
            continue
        _write_event_targets(mask, cat_mask, boundary_mask, event["start"], event["end"],
                              event.get("category"), total_frames, config)
    return mask, cat_mask, boundary_mask


class SyntheticSEDDataset(Dataset):
    """Strong-labelled synthetic clips from src/synthesis.py (Silver-clip backgrounds +
    Gold-extracted loud events at controlled SNR). Same (audio, mask, clip_id) 3-tuple
    interface as VaaniDataset so it drops into the existing training loops. Built to
    fix the current model's underfitting of the loud/diverse event categories
    (16-23% recall) and to provide a Synthetic-domain-predictive validation signal --
    the eval set's synthetic subset is itself "clean speech + added noise"."""

    def __init__(self, root, config, add_noise=True):
        self.root, self.config, self.add_noise = root, config, add_noise
        with open(os.path.join(root, "labels.jsonl"), encoding="utf-8") as fh:
            self.rows = [json.loads(line) for line in fh]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = os.path.join(self.root, "audio", f"{row['clip_id']}.wav")
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            audio = torch.as_tensor(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=torch.float32)
            if sr != self.config.SR:
                audio = torchaudio.functional.resample(audio, sr, self.config.SR)
        except Exception:
            audio = torch.zeros(self.config.target_samples, dtype=torch.float32)
        audio = pad_or_trim(audio, self.config.target_samples)

        total_frames = self.config.target_samples // self.config.HOP_LENGTH
        mask = torch.zeros(total_frames, dtype=torch.float32)
        use_boundary_head = getattr(self.config, "USE_BOUNDARY_HEAD", False)
        boundary_mask = torch.zeros(total_frames, 2, dtype=torch.float32) if use_boundary_head else None
        for event in row["events"]:
            # Synthetic events use onset/offset keys (not start/end like
            # NoiseSubCategoryTimeStamp), so _write_event_targets is called
            # directly here rather than via _build_targets.
            _write_event_targets(mask, None, boundary_mask, event["onset"], event["offset"],
                                  None, total_frames, self.config)

        if self.add_noise and torch.rand(1).item() < 0.5:
            level = torch.empty(1).uniform_(0.001, 0.015).item()
            audio = torch.clamp(audio + torch.randn_like(audio) * level, -1.0, 1.0)
        if boundary_mask is not None:
            # Session 9 plan, Part 2 Stage 2A full-mix run: synthetic clips
            # have exact, real per-event timing (they're synthesized) -- same
            # trust tier as Gold, always valid.
            return audio, mask, row["clip_id"], boundary_mask, torch.tensor(1.0, dtype=torch.float32)
        return audio, mask, row["clip_id"]


class VaaniDataset(Dataset):
    def __init__(self, hf_dataset, hf_dataset_no_decode, config):
        self.dataset, self.dataset_no_decode, self.config = hf_dataset, hf_dataset_no_decode, config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            row = self.dataset[idx]
            audio = torch.as_tensor(row["audio"]["array"], dtype=torch.float32)
        except Exception:
            audio, row = torch.zeros(self.config.target_samples), self.dataset_no_decode[idx]
        audio = pad_or_trim(audio, self.config.target_samples)
        total_frames = self.config.target_samples // self.config.HOP_LENGTH
        mask, cat_mask, boundary_mask = _build_targets(
            row.get("NoiseSubCategoryTimeStamp"), total_frames, self.config)
        if torch.rand(1).item() < 0.5:
            noise_level = torch.empty(1).uniform_(0.001, 0.015).item()
            noise = torch.randn_like(audio) * noise_level
            audio = torch.clamp(audio + noise, -1.0, 1.0)
        clip_id = row.get("imageFileName", f"clip_{idx}")
        extras = []
        if cat_mask is not None:
            extras.append(cat_mask)
        if boundary_mask is not None:
            # Session 9 plan, Part 2 Stage 2A full-mix run: Gold has real
            # per-event timestamps -- always a trustworthy boundary target,
            # even for a genuinely zero-event clip. See boundary_valid's use
            # in SilverTrainingWrapper below for the untrustworthy case.
            extras.append(boundary_mask)
            extras.append(torch.tensor(1.0, dtype=torch.float32))
        return (audio, mask, clip_id, *extras)


class OfficialValidationDataset(Dataset):
    """The real, organizer-provided held-out validation set (Codabench Files tab,
    `validation.zip` -> data/official_validation_raw/validation/). Same label schema
    as the HF training data (NoiseSubCategoryTimeStamp with start/end), plus an
    explicit `syntheticData` bool flag -- the first real ground truth we've had for
    which of Natural/Synthetic each row belongs to, matching the true eval split
    exactly (1806 natural + 695 synthetic, verified against the folder file counts).
    Unlike our homegrown gold_val (a held-out slice of Gold train data) and synth_val
    (our own from-scratch synthesis), this is real organizer data in the true eval
    distribution -- use it to calibrate/replace those proxies, not just supplement
    them."""

    def __init__(self, root, config, subset=None):
        self.root, self.config = root, config
        with open(os.path.join(root, "validationMetadata.json"), encoding="utf-8") as fh:
            rows = json.load(fh)
        if subset == "natural":
            rows = [r for r in rows if not r.get("syntheticData")]
        elif subset == "synthetic":
            rows = [r for r in rows if r.get("syntheticData")]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _audio_path(self, row):
        folder = "syntheticNoiseAudio" if row.get("syntheticData") else "naturalNoisyAudio"
        return os.path.join(self.root, folder, row["segmentFileName"])

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            wav, sr = sf.read(self._audio_path(row), dtype="float32", always_2d=False)
            audio = torch.as_tensor(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=torch.float32)
            if sr != self.config.SR:
                audio = torchaudio.functional.resample(audio, sr, self.config.SR)
        except Exception:
            audio = torch.zeros(self.config.target_samples, dtype=torch.float32)
        audio = pad_or_trim(audio, self.config.target_samples)

        total_frames = self.config.target_samples // self.config.HOP_LENGTH
        mask, cat_mask, boundary_mask = _build_targets(
            row.get("NoiseSubCategoryTimeStamp"), total_frames, self.config)
        extras = [t for t in (cat_mask, boundary_mask) if t is not None]
        return (audio, mask, row["segmentFileName"], *extras)


class BronzeInferenceDataset(Dataset):
    """Audio-only loader for the self-distilled pool: both Silver (unverified_timestamps)
    and Bronze (no_timestamps) rows are pseudo-labeled via the model's own Viterbi-decoded
    predictions (Run A recipe — Session 5 found training on Silver's real-but-unverified
    timestamps directly underperformed this on real Codabench submissions, despite
    scoring better on local Gold-only validation)."""

    def __init__(self, hf_dataset, config):
        self.hf_dataset, self.config = hf_dataset, config
        qualities = hf_dataset["annotationQuality"]
        self.indices = [
            i for i, quality in enumerate(qualities)
            if quality in {"unverified_timestamps", "no_timestamps"}
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_index = self.indices[idx]
        item = self.hf_dataset[row_index]
        # 3.5-week re-plan, Must-do #1: the HF schema has neither `filename`
        # nor `audio_id` (confirmed by direct inspection), and `imageFileName`
        # is NOT unique (90,637 rows -> only 44,139 distinct values, up to 53
        # rows share one image -- multiple people/recordings describe the
        # same source image) so it cannot be used as a clip_id without
        # silently colliding different clips together. The absolute row
        # index into `full_ds` (this dataset's `hf_dataset`) IS unique and
        # deterministic (same cached arrow files -> same row order every
        # load), and temp/build_transcript_tag_labels.py's output is keyed
        # by this same absolute index for that reason -- do not switch this
        # back to imageFileName.
        clip_id = f"row_{row_index}"
        has_tag = 1.0 if item.get("NoiseCategory") else 0.0
        try:
            wav, sample_rate = sf.read(io.BytesIO(item["audio"]["bytes"]))
            wav = torch.as_tensor(wav, dtype=torch.float32)
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            if sample_rate != self.config.SR:
                wav = torchaudio.functional.resample(wav, sample_rate, self.config.SR)
            wav = pad_or_trim(wav, self.config.target_samples)
        except Exception:
            wav = torch.zeros(self.config.target_samples, dtype=torch.float32)
        return wav, clip_id, torch.tensor(has_tag, dtype=torch.float32)


class SilverTrainingWrapper(Dataset):
    def __init__(self, inference_dataset, pseudo_labels_dict, audio_shape, mask_shape, use_boundary_head=False):
        self.base_dataset, self.pseudo_labels = inference_dataset, pseudo_labels_dict
        self.audio_shape, self.mask_shape = audio_shape, mask_shape
        # Session 9 plan, Part 2 Stage 2A full-mix run: Silver/Bronze only
        # ever have a flattened Viterbi pseudo-label frame mask (no per-event
        # start/end list at all), so there is no way to build a trustworthy
        # single-frame onset/offset impulse here -- unlike VaaniDataset/
        # SyntheticSEDDataset, this always emits a zero placeholder plus
        # boundary_valid=0.0, so the training loop excludes these samples
        # from the boundary loss entirely rather than training on noise.
        self.use_boundary_head = use_boundary_head

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        wav, clip_id, _ = self.base_dataset[idx]
        mask = self.pseudo_labels.get(clip_id, torch.zeros(self.mask_shape))
        mask = torch.as_tensor(mask, dtype=torch.float32).flatten()
        target_frames = self.mask_shape[0]
        if mask.numel() < target_frames:
            mask = F.pad(mask, (0, target_frames - mask.numel()))
        else:
            mask = mask[:target_frames]
        audio = wav.view(self.audio_shape)
        if torch.rand(1).item() < 0.5:
            noise_level = torch.empty(1).uniform_(0.001, 0.015).item()
            noise = torch.randn_like(audio) * noise_level
            audio = torch.clamp(audio + noise, -1.0, 1.0)
        if self.use_boundary_head:
            boundary_mask = torch.zeros(target_frames, 2, dtype=torch.float32)
            return audio, mask, clip_id, boundary_mask, torch.tensor(0.0, dtype=torch.float32)
        return audio, mask, clip_id