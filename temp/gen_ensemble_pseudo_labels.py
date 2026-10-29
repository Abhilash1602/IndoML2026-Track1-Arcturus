"""Iteration-2 pseudo-labels: relabel the 79,526-clip Silver+Bronze self-
distillation pool with the current best REAL ensemble (Codabench 1.5427:
dasheng_s2 + phaseh + wavlm_full1 + atst_stage2, logit-averaged) instead of
the stale single-model Phase F teacher (real 1.4727) every self-distilled
member has been trained on. Mirrors the DCASE-2024 winner's (CPJKU)
iteration 2: pseudo-labels from an ensemble of the iteration-1 models.

Same Viterbi convention as train.py's generate_pseudo_labels (0.95/0.95) so
the teacher is the only thing that changes. Also labels with Phase F in the
same pass and reports agreement, so the size of the change is measured
before any retrain is spent on it.

Output: data/ensemble_pseudo_labels_v1.npz with `clip_ids` (row_<idx>, the
BronzeInferenceDataset convention), `hard` (uint8 masks), `soft` (float16
ensemble probabilities), `phasef_hard` (uint8, old labels, for comparison).
Load with temp.gen_ensemble_pseudo_labels.load_pseudo_labels().

Usage: python -m temp.gen_ensemble_pseudo_labels
"""
import multiprocessing as mp

import numpy as np
import torch
from datasets import Audio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED, StandaloneDashengSED
from src.config import Config
from src.dataset import BronzeInferenceDataset
from src.utils import viterbi_decode
from train import load_full_dataset

OUT = "./data/ensemble_pseudo_labels_v1.npz"
BIG = dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)
ENSEMBLE = {
    "dasheng_s2": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2.pth", {}),
    "phaseh": (DualBranchSED, "./checkpoints/dualbranch_phaseh.pth", BIG),
    "wavlm_full1": (DualBranchSED, "./checkpoints/dualbranch_wavlm_stage1_full.pth", dict(USE_WAVLM=True)),
    "atst_stage2": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2.pth", dict(USE_ATST_FRAME=True)),
}
PHASEF = (DualBranchSED, "./checkpoints/dualbranch_phasef.pth", {})
PS, PE = 0.95, 0.95  # identical to train.py's generate_pseudo_labels


def load(cls, path, overrides):
    config = Config()
    for k, v in overrides.items():
        setattr(config, k, v)
    model = cls(config).to(config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True))
    return model.eval()


def _viterbi(row):
    return viterbi_decode(row.astype(np.float32), p_stay_silence=PS, p_stay_event=PE).astype(np.uint8)


def load_pseudo_labels(path=OUT, kind="hard"):
    # Read each array ONCE: indexing data[kind] inside the loop would
    # re-decompress the whole compressed array per row (79,526x).
    data = np.load(path)
    clip_ids, labels = data["clip_ids"], data[kind].astype(np.float32)
    return {str(cid): labels[i] for i, cid in enumerate(clip_ids)}


def main():
    config = Config()
    n_frames = config.target_samples // config.HOP_LENGTH
    full_ds = load_full_dataset().cast_column("audio", Audio(decode=False))
    ds = BronzeInferenceDataset(full_ds, config)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
    ens = [load(*spec) for spec in ENSEMBLE.values()]
    phasef = load(*PHASEF)

    soft = np.zeros((len(ds), n_frames), dtype=np.float16)
    old_soft = np.zeros((len(ds), n_frames), dtype=np.float16)
    clip_ids = []
    pos = 0
    with torch.no_grad():
        for audio, ids, _ in tqdm(loader, desc="ensemble + Phase F labeling"):
            audio = audio.to(config.DEVICE)
            outs = [m(audio) for m in ens]
            t = min(min(o.shape[1] for o in outs), n_frames)
            probs = torch.sigmoid(torch.stack([o[:, :t] for o in outs]).mean(0)).cpu().numpy()
            old = torch.sigmoid(phasef(audio))[:, :t].cpu().numpy()
            b = probs.shape[0]
            soft[pos:pos + b, :t] = probs
            old_soft[pos:pos + b, :t] = old
            clip_ids.extend(ids)
            pos += b
    assert pos == len(ds) and len(set(clip_ids)) == len(clip_ids), "clip_id coverage/uniqueness failure"

    print("Viterbi-decoding both label sets...", flush=True)
    with mp.get_context("fork").Pool(96) as pool:
        hard = np.stack(pool.map(_viterbi, soft, chunksize=256))
        phasef_hard = np.stack(pool.map(_viterbi, old_soft, chunksize=256))

    agree = (hard == phasef_hard).mean()
    inter = np.logical_and(hard, phasef_hard).sum()
    union = np.logical_or(hard, phasef_hard).sum()
    changed_clips = (np.abs(hard.astype(int) - phasef_hard).sum(1) > 10).mean()
    print(f"\nclips={len(ds)}  frames/clip={n_frames}", flush=True)
    print(f"positive-frame rate: ensemble={hard.mean():.4f}  Phase F={phasef_hard.mean():.4f}", flush=True)
    print(f"frame agreement={agree:.4f}  positive-frame IoU={inter / max(union, 1):.4f}  "
          f"clips differing by >10 frames={changed_clips:.3f}", flush=True)
    np.savez_compressed(OUT, clip_ids=np.array(clip_ids), hard=hard, soft=soft, phasef_hard=phasef_hard)
    print(f"Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
