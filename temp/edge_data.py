"""Edge-model data (plan Step 1).

1. Cache the 80-band 10 ms log-mel + any-noise mask of the Gold TRAIN split (make_gold_loaders,
   seed 42; 8,888 clips) to scratchpad/edge_gold_train.npz -- same front end and layout as
   temp/cache_refiner_data.py, no member logits needed.
2. The error model: on the Gold HELD-OUT split (refiner_gold_heldout.npz, 2,223 clips no v4 member
   trained on), decode the v4 ensemble with the v5 pipeline (Viterbi 0.9802/0.9489 + gated edge
   correction) and record (onset error, offset error) for every short reference event
   (0.1-0.8 s) overlapped by exactly one prediction that overlaps no other reference.
   Saved to scratchpad/edge_errors.npy. Official validation is never touched.

Usage: python -m temp.edge_data
"""
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.edge_correction import correct_events, load_table
from temp.train_refiner import refs_of, viterbi_events
from temp.train_refiner_val import S
from train import make_gold_loaders

FRAMES = 1000
MIN_DUR, MAX_DUR = 0.1, 0.8


def cache_gold_train():
    path = f"{S}/edge_gold_train.npz"
    if os.path.exists(path):
        return
    config = Config()
    _, train_loader, _ = make_gold_loaders(config)
    ds = train_loader.dataset
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS)
    mel_tf = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
        torchaudio.transforms.AmplitudeToDB()).to(config.DEVICE)
    mel_out = np.zeros((len(ds), 80, FRAMES), np.float16)
    mask_out = np.zeros((len(ds), FRAMES), np.uint8)
    pos = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="gold_train mel"):
            audio, mask = batch[0].to(config.DEVICE), batch[1]
            b = audio.shape[0]
            mel = mel_tf(audio)[..., :FRAMES]
            mel_out[pos:pos + b, :, :mel.shape[-1]] = mel.cpu().numpy()
            m = mask[:, :FRAMES].numpy() > 0.5
            mask_out[pos:pos + b, :m.shape[1]] = m
            pos += b
    assert pos == len(ds)
    np.savez(path, mel=mel_out, mask=mask_out)
    print(f"gold_train: {len(ds)} clips, positive frame rate {mask_out.mean():.3f}", flush=True)


def overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def error_model():
    d = np.load(f"{S}/refiner_gold_heldout.npz")
    prob = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
    refs = refs_of(d["mask"].astype(np.float32))
    table = load_table("./checkpoints/edge_correction_v5.json")
    preds = [correct_events(e, p, table) for e, p in zip(viterbi_events(prob), prob)]
    errs = []
    for g, p in zip(refs, preds):
        for r in g:
            if not (MIN_DUR <= r[1] - r[0] <= MAX_DUR) or r[0] <= 0.0 or r[1] >= FRAMES * 0.01 - 0.01:
                continue
            hits = [q for q in p if overlap(q, r)]
            if len(hits) == 1 and sum(overlap(hits[0], r2) for r2 in g) == 1:
                errs.append((hits[0][0] - r[0], hits[0][1] - r[1]))
    errs = np.array(errs, np.float32)
    np.save(f"{S}/edge_errors.npy", errs)
    tol_ok = np.mean([(abs(a) <= 0.05) and (abs(b) <= 0.05) for a, b in errs])
    print(f"error model: {len(errs)} short Gold-heldout events | onset median {np.median(errs[:, 0]):+.3f} "
          f"MAE {np.abs(errs[:, 0]).mean():.3f} | offset median {np.median(errs[:, 1]):+.3f} "
          f"MAE {np.abs(errs[:, 1]).mean():.3f} | both within 50 ms {tol_ok:.1%}", flush=True)


if __name__ == "__main__":
    error_model()
    cache_gold_train()
