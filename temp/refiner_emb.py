"""Refiner variant input: the 5 v4 members' 144-d frame embeddings (25 Hz), reduced to 32 PCA
dimensions (fitted once on the official-validation embeddings and saved, so test clips are
projected identically), standardised, and repeated x4 onto the 10 ms grid -> (32, 1000) per clip.
"""
import json
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA

from temp.train_refiner_val import S

PCA_PATH = "./checkpoints/refiner_emb_pca.pkl"
DIM = 32


def _fit():
    rng = np.random.default_rng(0)
    rows = []
    for sub in ("natural", "synthetic"):
        e = np.load(f"{S}/verifier/official_{sub}_emb.npy", mmap_mode="r")  # (N, 5, 250, 144)
        idx = rng.choice(len(e), min(len(e), 600), replace=False)
        x = np.asarray(e[np.sort(idx)], np.float32).transpose(0, 2, 1, 3).reshape(-1, 5 * 144)
        rows.append(x[rng.choice(len(x), 60000, replace=False)])
    X = np.concatenate(rows)
    pca = PCA(DIM, random_state=0).fit(X)
    z = pca.transform(X)
    pickle.dump({"pca": pca, "mu": z.mean(0), "sd": z.std(0) + 1e-6}, open(PCA_PATH, "wb"))


_P = None


def project(emb):
    """emb (5, T25, 144) -> (32, 4 * T25) float32 on the 10 ms grid."""
    global _P
    if _P is None:
        if not os.path.exists(PCA_PATH):
            _fit()
        _P = pickle.load(open(PCA_PATH, "rb"))
    p = _P
    x = np.asarray(emb, np.float32).transpose(1, 0, 2).reshape(emb.shape[1], -1)
    z = (p["pca"].transform(x) - p["mu"]) / p["sd"]
    return np.repeat(z.T, 4, axis=1).astype(np.float32)


def extra_all_data(order="train_all"):
    """(N, 32, 1000) aligned with the caller's clip order:
    "crossfit"  -- temp.crossfit_refiner.all_data: subset outer, part inner;
    "train_all" -- temp.train_refiner_val.train_all: part outer, subset inner."""
    split = json.load(open("./data/val_split_s0.json"))
    blocks = {}
    for sub in ("natural", "synthetic"):
        ids = json.load(open(f"{S}/verifier/official_{sub}_ids.json"))
        e = np.load(f"{S}/verifier/official_{sub}_emb.npy", mmap_mode="r")
        pos = {n: i for i, n in enumerate(ids)}
        ref_ids = [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]]
        for part in ("train70", "hold30"):
            keep = set(split[part][sub])
            blocks[(part, sub)] = [project(e[pos[n]])[:, :1000] for n in ref_ids if n in keep]
    keys = ([(p, s) for s in ("natural", "synthetic") for p in ("train70", "hold30")] if order == "crossfit"
            else [(p, s) for p in ("train70", "hold30") for s in ("natural", "synthetic")])
    return np.stack([x for k in keys for x in blocks[k]])
