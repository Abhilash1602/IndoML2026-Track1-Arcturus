"""v7 follow-up: out-of-fold refiner predictions on ALL 2,501 validation clips.

Real results showed hold30 (542 Natural clips) under-estimated the refiner's Natural Dice cost
(hold30 -0.004..-0.008 vs real -0.018..-0.021). A 5-fold cross-fit gives 3.3x more clips to
measure the two fixes on: (1) averaging refiners over seeds, (2) falling back to the base events
where the refiner empties a clip or shrinks its coverage.

Folds are stratified by domain (seed 0). For each (fold, seed) the refiner (no VAD channel) is
trained on the other folds (15% of them as the early-stopping/tuning slice, seed-specific) and
predicts the held fold. Output: scratchpad/crossfit/f{fold}_s{seed}.npz (idx, presence, d_on, d_off).

Usage: CF_JOBS="0:0,1:0" VAANI_DEVICE=cuda:0 python -m temp.crossfit_refiner
"""
import os

import numpy as np
import torch

from temp.train_refiner_val import S, load_all, predict, train_one

K = 5
# CF_GOLD=1: Gold held-out (2,223 clips no v4 member trained on) is added to every fold's training
# set -- ~2x refiner data; held-out folds and outputs stay validation-only.
CF_GOLD = os.environ.get("CF_GOLD") == "1"
# CF_TAG: architecture variant (REFINER_WIDTH / REFINER_GRU / REFINER_ARCH / REFINER_EXTRA_SRC=emb)
CF_TAG = os.environ.get("CF_TAG", "")
OUT = os.path.join(S, f"crossfit_{CF_TAG}" if CF_TAG else ("crossfit_gold" if CF_GOLD else "crossfit"))
EMB = os.environ.get("REFINER_EXTRA_SRC") == "emb"


def all_data():
    data = load_all()
    parts = [(p, s) for s in ("natural", "synthetic") for p in ("train70", "hold30")]
    L, M, Y, V = (np.concatenate([data[k][j] for k in parts]) for j in range(4))
    nat = np.concatenate([np.full(len(data[k][0]), k[1] == "natural") for k in parts])
    return L, M, Y, V, nat


def folds(nat):
    rng = np.random.default_rng(0)
    fold = np.zeros(len(nat), int)
    for dom in (True, False):
        idx = np.where(nat == dom)[0]
        fold[rng.permutation(idx)] = np.arange(len(idx)) % K
    return fold


def main():
    os.makedirs(OUT, exist_ok=True)
    L, M, Y, V, nat = all_data()
    n_extra = 0
    if EMB:
        from temp.refiner_emb import extra_all_data
        V, n_extra = extra_all_data("crossfit"), 32
        assert len(V) == len(L)
    fold = folds(nat)
    n_val = len(L)
    if CF_GOLD:
        g = np.load(f"{S}/refiner_gold_heldout.npz")
        L = np.concatenate([L, g["logits"].astype(np.float32)])
        M = np.concatenate([M, g["mel"].astype(np.float32)])
        Y = np.concatenate([Y, g["mask"].astype(np.float32)])
        V = np.concatenate([V, np.zeros((len(g["mask"]),) + V.shape[1:], np.float32)])
        fold = np.r_[fold, np.full(len(g["mask"]), -1)]  # never held out
        print(f"+ Gold held-out: {len(g['mask'])} clips in every fold's training set", flush=True)
    for job in os.environ["CF_JOBS"].split(","):
        f, s = map(int, job.split(":"))
        path = f"{OUT}/f{f}_s{s}.npz"
        if os.path.exists(path):
            continue
        tr = np.where(fold != f)[0]
        perm = np.random.default_rng(100 + s).permutation(tr)
        tune, train = perm[: int(0.15 * len(tr))], perm[int(0.15 * len(tr)):]
        print(f"fold {f} seed {s}: train {len(train)} tune {len(tune)} held {int((fold == f).sum())}", flush=True)
        model, kind = train_one(n_extra, L, M, Y, V, tune, train, seed=s)
        held = np.where(fold == f)[0]
        assert held.max() < n_val
        pred = predict(model, L[held], M[held], V[held])
        np.savez(path, idx=held, kind=kind, **pred)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
