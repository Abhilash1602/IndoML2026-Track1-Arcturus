"""Oct 14, idea #2 step 3: verifier v1 + weak-label tagger evidence (temp/train_tagger.py outputs; trained only
on clip tags, Gold held-out excluded, so its outputs are out-of-sample here). Per event: for each of 8 channels
(any + 7 categories) the mean and max frame probability inside the event (25 Hz), the strongest category's
inside-minus-margin contrast, and the 8 clip-level probabilities. Same events, labels and grouped 5-fold OOF
as verifier v1 (temp/verifier.py). Pre-registered: AUC >= v1 + 0.01 and pooled >= v1 + 0.002 (drops at 0.35,
0.70 guard), synthetic within +/-0.0005.

Usage: python -m temp.verifier_tags          (evaluation)
       python -m temp.verifier_tags fit      (-> checkpoints/verifier_tags.pkl)
"""
import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import temp.verifier as vf
from temp.train_refiner import f1_only, per_clip

D = "./work/tags"
SETS = ("gold_heldout", "official_natural", "official_synthetic")


def tag_features(F, C, a, b):
    """F (250, 8) frame probs at 25 Hz, C (8,) clip probs -> per-event features."""
    n = F.shape[0]
    ia, ib = min(int(a * 25), n - 1), min(max(int(np.ceil(b * 25)), int(a * 25) + 1), n)
    ma, mb = max(0, ia - 5), min(n, ib + 5)
    ins = F[ia:ib].astype(np.float32)
    mar = np.r_[F[ma:ia], F[ib:mb]].astype(np.float32)
    mean_in, max_in = ins.mean(0), ins.max(0)
    mean_mar = mar.mean(0) if len(mar) else mean_in
    top = int(np.argmax(mean_in[1:])) + 1
    return list(mean_in) + list(max_in) + [mean_in[top] - mean_mar[top], mean_in[0] - mean_mar[0]] + list(C)


def main():
    sets = [vf.load_set(n) for n in SETS]
    tags = {n: (np.load(f"{D}/{n}_tagframe.npy"), np.load(f"{D}/{n}_tagclip.npy")) for n in SETS}
    ev_X, ev_E, ev_y, ev_key = vf.build_training(sets)[:4]
    by = {s["name"]: s for s in sets}
    ev_T = np.array([tag_features(tags[n][0][i], tags[n][1][i], *by[n]["base"][i][k]) for n, i, k in ev_key], np.float32)
    pca = PCA(vf.N_PCA, random_state=0).fit(ev_E)
    X1 = np.c_[ev_X, pca.transform(ev_E)]
    X2 = np.c_[X1, ev_T]
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        pickle.dump(dict(pca=pca, model=vf.gbm().fit(X2, ev_y)), open("./checkpoints/verifier_tags.pkl", "wb"))
        print(f"fitted verifier_tags on {len(ev_y)} events", flush=True)
        return
    o1, o2 = vf.oof_scores(X1, ev_y, ev_key), vf.oof_scores(X2, ev_y, ev_key)
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    res = {}
    for tag, o in (("v1", o1), ("v1+tags", o2)):
        drop = {key for key, s in zip(ev_key, o) if s > 0.35}
        rows = {}
        for s in sets[1:]:
            evs = []
            for i, ev in enumerate(s["base"]):
                P = s["P"][i]
                conf = lambda e: float(P[int(e[0] * 100):max(int(e[1] * 100), int(e[0] * 100) + 1)].mean())
                evs.append([e for k, e in enumerate(ev) if (s["name"], i, k) not in drop or conf(e) >= 0.70])
            rows[s["name"]] = per_clip(s["refs"], evs)
        n, sy = rows["official_natural"], rows["official_synthetic"]
        res[tag] = (roc_auc_score(ev_y, o), comb(np.r_[n, sy]), f1_only(sy), sy[:, 3].mean())
        print(f"[{tag:8s}] AUC {res[tag][0]:.4f} | drops {len(drop)} | pooled {res[tag][1]:.4f} | Nat F1 {f1_only(n):.4f} "
              f"Dice {n[:, 3].mean():.4f} | Syn F1 {f1_only(sy):.4f} Dice {sy[:, 3].mean():.4f}", flush=True)
    a, b = res["v1"], res["v1+tags"]
    ok = b[0] - a[0] >= 0.01 and b[1] - a[1] >= 0.002 and abs(b[2] - a[2]) <= 0.0005 and abs(b[3] - a[3]) <= 0.0005
    print(f"GATE: {'PASS' if ok else 'FAIL'} (AUC {b[0] - a[0]:+.4f}, pooled {b[1] - a[1]:+.4f})", flush=True)


if __name__ == "__main__":
    main()
