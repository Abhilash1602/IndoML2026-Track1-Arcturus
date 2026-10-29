"""Session 9 plan, Part 2 Stage 2A full-mix run, Part 1 Step 5: pre-flight
gate before spending any GPU time on the full self-distillation-mix run.
Confirms SyntheticSEDDataset and SilverTrainingWrapper both correctly
extend their tuple arity under Config.USE_BOUNDARY_HEAD (matching
VaaniDataset, already verified in temp/verify_boundary_head.py), AND that a
real ConcatDataset + DataLoader (default collate) across all three
heterogeneous dataset types produces a single uniformly-shaped batch with no
error -- the literal regression test for the tuple-arity-mismatch collate
bug this whole extension exists to prevent.

Usage: python -m temp.verify_boundary_head_full_mix
"""
import json
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.config import Config
from src.dataset import SilverTrainingWrapper, SyntheticSEDDataset

ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok &= cond


class _FakeSilverInferenceDataset(Dataset):
    """Minimal stand-in for BronzeInferenceDataset's (wav, clip_id, has_tag)
    interface -- SilverTrainingWrapper only ever calls __getitem__ on this,
    so a real HF-backed dataset isn't needed for this collate/shape test."""

    def __init__(self, n, samples):
        self.n, self.samples = n, samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.zeros(self.samples), f"silver_{idx}", torch.tensor(0.0)


with tempfile.TemporaryDirectory() as tmpdir:
    audio_dir = os.path.join(tmpdir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    sr = 16000
    for i in range(2):
        sf.write(os.path.join(audio_dir, f"synth_{i}.wav"), np.zeros(sr * 2, dtype=np.float32), sr)
    with open(os.path.join(tmpdir, "labels.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"clip_id": "synth_0", "events": [{"onset": 0.3, "offset": 0.8}]}) + "\n")
        fh.write(json.dumps({"clip_id": "synth_1", "events": [{"onset": 1.0, "offset": 1.4}]}) + "\n")

    # 1. SyntheticSEDDataset: 3-tuple off, 5-tuple on, correct boundary_mask/valid.
    config_off = Config()
    config_off.MAX_DURATION = 2.0
    synth_off = SyntheticSEDDataset(tmpdir, config_off, add_noise=False)
    item_off = synth_off[0]
    check("SyntheticSEDDataset returns a 3-tuple when USE_BOUNDARY_HEAD is off", len(item_off) == 3)

    config_on = Config()
    config_on.MAX_DURATION = 2.0
    config_on.USE_BOUNDARY_HEAD = True
    synth_on = SyntheticSEDDataset(tmpdir, config_on, add_noise=False)
    item_on = synth_on[0]
    check("SyntheticSEDDataset returns a 5-tuple when USE_BOUNDARY_HEAD is on", len(item_on) == 5)
    _, mask_on, clip_id_on, boundary_mask_on, boundary_valid_on = item_on
    total_frames = config_on.target_samples // config_on.HOP_LENGTH
    onset_frame = int(0.3 * config_on.SR / config_on.HOP_LENGTH)
    offset_frame = int(0.8 * config_on.SR / config_on.HOP_LENGTH)
    check("SyntheticSEDDataset boundary_mask has shape (T,2)", tuple(boundary_mask_on.shape) == (total_frames, 2))
    check("SyntheticSEDDataset onset/offset impulses at the right frames (onset/offset key convention)",
          boundary_mask_on[onset_frame, 0].item() == 1.0 and boundary_mask_on[offset_frame, 1].item() == 1.0)
    check("SyntheticSEDDataset boundary_valid is 1.0 (real, trustworthy synthesis-time timing)",
          boundary_valid_on.item() == 1.0)
    _, mask_off, _ = item_off
    check("SyntheticSEDDataset primary mask unaffected by USE_BOUNDARY_HEAD", torch.equal(mask_off, mask_on))

    # 2. SilverTrainingWrapper: 3-tuple off, 5-tuple on, all-zero placeholder + valid=0.
    fake_base = _FakeSilverInferenceDataset(3, config_on.target_samples)
    pseudo_labels = {f"silver_{i}": torch.zeros(total_frames) for i in range(3)}
    silver_off = SilverTrainingWrapper(
        fake_base, pseudo_labels, audio_shape=(config_off.target_samples,), mask_shape=(total_frames,))
    check("SilverTrainingWrapper returns a 3-tuple by default (backward compat)", len(silver_off[0]) == 3)

    silver_on = SilverTrainingWrapper(
        fake_base, pseudo_labels, audio_shape=(config_on.target_samples,), mask_shape=(total_frames,),
        use_boundary_head=True)
    item_silver_on = silver_on[0]
    check("SilverTrainingWrapper returns a 5-tuple when use_boundary_head=True", len(item_silver_on) == 5)
    _, _, _, silver_boundary_mask, silver_boundary_valid = item_silver_on
    check("SilverTrainingWrapper boundary_mask is an all-zero placeholder",
          silver_boundary_mask.abs().sum().item() == 0.0)
    check("SilverTrainingWrapper boundary_valid is 0.0 (no trustworthy per-event timing available)",
          silver_boundary_valid.item() == 0.0)

    # 3. The actual collate smoke test: ConcatDataset of a Gold-shaped fake +
    # the real SyntheticSEDDataset + the real SilverTrainingWrapper, all under
    # USE_BOUNDARY_HEAD=True, run through a real DataLoader with default
    # collate -- confirms no tuple-length mismatch crash across all three.
    class _FakeGoldDataset(Dataset):
        """Mimics VaaniDataset's exact 5-tuple shape under USE_BOUNDARY_HEAD
        without needing a real HF dataset -- VaaniDataset's own per-item
        correctness is already covered by temp/verify_boundary_head.py."""

        def __init__(self, n, total_frames, samples):
            self.n, self.total_frames, self.samples = n, total_frames, samples

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return (torch.zeros(self.samples), torch.zeros(self.total_frames), f"gold_{idx}",
                    torch.zeros(self.total_frames, 2), torch.tensor(1.0))

    gold_like = _FakeGoldDataset(4, total_frames, config_on.target_samples)
    combined = ConcatDataset([gold_like, silver_on, synth_on])
    loader = DataLoader(combined, batch_size=4, shuffle=True)
    try:
        batch = next(iter(loader))
        collate_ok = len(batch) == 5 and batch[3].shape[1:] == (total_frames, 2) and batch[4].shape == (4,)
        check("ConcatDataset+DataLoader collate across Gold/Silver/Synthetic succeeds with uniform 5-tuple shape",
              collate_ok)
    except Exception as exc:
        check(f"ConcatDataset+DataLoader collate succeeds (raised: {exc!r})", False)

print()
print("ALL CHECKS PASSED -- safe to proceed to the full-mix training run"
      if ok else "AT LEAST ONE CHECK FAILED -- do not proceed to training")
