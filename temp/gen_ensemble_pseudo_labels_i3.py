"""Iteration-3 pseudo-labels: relabel the 79,526-clip Silver+Bronze pool with
the submission_v4 ensemble (real Codabench 1.5545, best real Natural F1 0.4195:
atst_stage2 + dasheng_owB + dasheng_owB_t + beats_s2 + beats_owA, logit-averaged).
Same Viterbi convention (0.95/0.95) as i2, so only the labeller changes; agreement
is reported against the i2 labels (data/ensemble_pseudo_labels_v1.npz).

Output: data/ensemble_pseudo_labels_v2.npz (clip_ids, hard, soft, prev_hard).
Load with temp.gen_ensemble_pseudo_labels.load_pseudo_labels(path=OUT_I3).

Usage: python -m temp.gen_ensemble_pseudo_labels_i3
"""
import multiprocessing as mp

import numpy as np
import torch
from datasets import Audio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED, StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.dataset import BronzeInferenceDataset
from src.utils import viterbi_decode
from train import load_full_dataset

OUT = "./data/ensemble_pseudo_labels_v2.npz"
PREV = "./data/ensemble_pseudo_labels_v1.npz"
BIG = dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)
ENSEMBLE = {
    "atst_stage2": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2.pth", dict(USE_ATST_FRAME=True)),
    "dasheng_owB": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth", {}),
    "dasheng_owB_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceB_teacher.pth", {}),
    "beats_s2": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2.pth", {}),
    "beats_owA": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceA.pth", {}),
}
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

    soft = np.zeros((len(ds), n_frames), dtype=np.float16)
    clip_ids = []
    pos = 0
    with torch.no_grad():
        for audio, ids, _ in tqdm(loader, desc="v4 ensemble labeling"):
            audio = audio.to(config.DEVICE)
            outs = [m(audio) for m in ens]
            t = min(min(o.shape[1] for o in outs), n_frames)
            probs = torch.sigmoid(torch.stack([o[:, :t] for o in outs]).mean(0)).cpu().numpy()
            b = probs.shape[0]
            soft[pos:pos + b, :t] = probs
            clip_ids.extend(ids)
            pos += b
    assert pos == len(ds) and len(set(clip_ids)) == len(clip_ids), "clip_id coverage/uniqueness failure"

    print("Viterbi-decoding...", flush=True)
    with mp.get_context("fork").Pool(96) as pool:
        hard = np.stack(pool.map(_viterbi, soft, chunksize=256))
    prev = np.load(PREV)
    assert list(prev["clip_ids"]) == list(clip_ids), "clip order differs from the i2 labels"
    phasef_hard = prev["hard"]  # i2 labels (name kept for the comparison code below)

    agree = (hard == phasef_hard).mean()
    inter = np.logical_and(hard, phasef_hard).sum()
    union = np.logical_or(hard, phasef_hard).sum()
    changed_clips = (np.abs(hard.astype(int) - phasef_hard).sum(1) > 10).mean()
    print(f"\nclips={len(ds)}  frames/clip={n_frames}", flush=True)
    print(f"positive-frame rate: i3={hard.mean():.4f}  i2={phasef_hard.mean():.4f}", flush=True)
    print(f"frame agreement={agree:.4f}  positive-frame IoU={inter / max(union, 1):.4f}  "
          f"clips differing by >10 frames={changed_clips:.3f}", flush=True)
    np.savez_compressed(OUT, clip_ids=np.array(clip_ids), hard=hard, soft=soft, prev_hard=phasef_hard)
    print(f"Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
