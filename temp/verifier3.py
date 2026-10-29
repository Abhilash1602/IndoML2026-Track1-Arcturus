"""Oct 14: verifier v3 = verifier v1's event features + per-event opinions of the three clean new
lineages (w2v-BERT, in-domain-pretrained-context Dasheng, Dasheng-0.6B; work/verifier3/, trained on
Gold-train + synthetic only) + member disagreement. Same data, labels (pure-FP event) and grouped
5-fold OOF as temp/verifier.py; v1 and v3 are compared on identical folds.

Pre-registered: v3 OOF AUC >= v1 + 0.01, and replacing v1's drop scores by v3's (threshold 0.35,
events with mean ensemble prob >= 0.70 never dropped) gains >= +0.001 pooled on validation
Natural + Synthetic (base = v4 Viterbi + correction, gated events as in v1).

Usage: CUDA_VISIBLE_DEVICES=1 python -m temp.verifier3          (local evaluation)
       CUDA_VISIBLE_DEVICES=1 python -m temp.verifier3 fit      (fit on everything -> checkpoints/verifier3.pkl)
"""
import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import temp.verifier as vf
from temp.train_refiner import f1_only, per_clip

NEW = ("w2v", "mae", "d06")
SETS = ("gold_heldout", "official_natural", "official_synthetic")


def new_features(L4, Lnew, a, b):
    """L4 (5, T) v4 logits, Lnew (3, T) new-model logits -> per-event extra features."""
    T = L4.shape[1]
    ia, ib = int(a * 100), max(int(b * 100), int(a * 100) + 1)
    ia, ib = min(ia, T - 1), min(max(ib, ia + 1), T)
    ma, mb = max(0, ia - 20), min(T, ib + 20)
    pn = 1 / (1 + np.exp(-Lnew.astype(np.float32)))
    p4 = 1 / (1 + np.exp(-L4.astype(np.float32)))
    f = []
    for m in range(len(pn)):
        inside = pn[m, ia:ib]
        margin = np.r_[pn[m, ma:ia], pn[m, ib:mb]]
        f += [inside.mean(), inside.max(), margin.mean() if len(margin) else 0.0, (inside > 0.5).mean()]
    all_in = np.r_[p4[:, ia:ib].mean(1), pn[:, ia:ib].mean(1)]
    f += [all_in.std(), pn[:, ia:ib].mean(), all_in.min()]
    return f


def build(sets, new):
    ev_X, ev_E, ev_y, ev_key = vf.build_training(sets)[:4]
    extra = []
    by = {D["name"]: D for D in sets}
    for name, i, k in ev_key:
        D = by[name]
        a, b = D["base"][i][k]
        extra.append(new_features(D["L"][i], np.stack([new[name][m][i] for m in NEW]), a, b))
    return ev_X, ev_E, np.array(extra, np.float32), ev_y, ev_key


def main():
    sets = [vf.load_set(n) for n in SETS]
    new = {n: {m: np.load(f"./work/verifier3/{m}_{n}.npy") for m in NEW} for n in SETS}
    ev_X, ev_E, ev_N, ev_y, ev_key = build(sets, new)
    pca = PCA(vf.N_PCA, random_state=0).fit(ev_E)
    X1 = np.c_[ev_X, pca.transform(ev_E)]
    X3 = np.c_[X1, ev_N]
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        m3 = vf.gbm().fit(X3, ev_y)
        pickle.dump(dict(pca=pca, model=m3, new=NEW), open("./checkpoints/verifier3.pkl", "wb"))
        print(f"fitted verifier3 on {len(ev_y)} events -> checkpoints/verifier3.pkl", flush=True)
        return
    o1, o3 = vf.oof_scores(X1, ev_y, ev_key), vf.oof_scores(X3, ev_y, ev_key)
    print(f"events {len(ev_y)} (pure-FP rate {ev_y.mean():.3f}) | OOF AUC v1 {roc_auc_score(ev_y, o1):.4f} "
          f"v3 {roc_auc_score(ev_y, o3):.4f}", flush=True)
    by = {D["name"]: D for D in sets}
    score = {}
    for tag, o in (("v1", o1), ("v3", o3)):
        drop = {key for key, s in zip(ev_key, o) if s > 0.35}
        rows_n, rows_s = [], []
        for D in sets:
            if D["name"] == "gold_heldout":
                continue
            evs = []
            for i, ev in enumerate(D["base"]):
                P = D["P"][i]
                conf = lambda e: float(P[int(e[0] * 100):max(int(e[1] * 100), int(e[0] * 100) + 1)].mean())
                evs.append([e for k, e in enumerate(ev) if (D["name"], i, k) not in drop or conf(e) >= 0.70])
            r = per_clip(D["refs"], evs)
            (rows_n if D["name"] == "official_natural" else rows_s).append(r)
        n, s = np.concatenate(rows_n), np.concatenate(rows_s)
        a = np.r_[n, s]
        score[tag] = f1_only(a) + a[:, 3].mean()
        print(f"[{tag}] drops {len(drop)} | validation pooled {score[tag]:.4f} | Nat F1 {f1_only(n):.4f} Dice {n[:, 3].mean():.4f} "
              f"| Syn F1 {f1_only(s):.4f} Dice {s[:, 3].mean():.4f}", flush=True)
    auc_gain = roc_auc_score(ev_y, o3) - roc_auc_score(ev_y, o1)
    gain = score["v3"] - score["v1"]
    print(f"GATE: {'PASS' if auc_gain >= 0.01 and gain >= 0.001 else 'FAIL'} (AUC {auc_gain:+.4f}, pooled {gain:+.4f})", flush=True)


if __name__ == "__main__":
    main()
