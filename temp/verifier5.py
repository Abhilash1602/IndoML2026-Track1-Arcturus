"""Oct 14: verifier v5 = the pure-false-positive classifier trained on ~3x the events, using
cross-fitted features (work/folds/, temp/cache_fold_feats.py: two Dasheng lanes, every Gold and
validation clip scored by models that never saw it).

Training events: the v5 decode (Viterbi + correction on the mean of the 2 OOF lanes) of all 11,111
Gold + 2,501 validation clips, gated as verifier v1; label = event overlaps no reference.
Features: temp/fp_filter.features on the 2 OOF lanes (L, P, mel energy E).
Evaluation = exactly verifier v1's events (v4-base decode on Gold held-out + validation), with v5
features computed on those events from the OOF logits; grouped 5-fold by clip (the same folds as the
fold models), so every scored event's clip is unseen by both the fold models and the GBM.
Variants: v1 (temp/verifier.py OOF), v5, avg = (v1 + v5) / 2.
Pre-registered: best of v5 / avg has AUC >= v1 + 0.01 and pooled >= v1 + 0.002 (drops at 0.35,
0.70 guard), synthetic within +/-0.0005.

Usage: python -m temp.verifier5          (evaluation)
       python -m temp.verifier5 fit      (fit v5 on all events -> checkpoints/verifier5.pkl)
"""
import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import temp.fp_filter as fpf
import temp.verifier as vf
from src.edge_correction import correct_events, load_table
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from train import gold_fold_of, val_fold_of

F = "./work/folds"
T5 = load_table("./checkpoints/edge_correction_v5.json")
VAL = ("official_natural", "official_synthetic")


def load_fold_sets():
    import json
    out = {}
    for s in ("gold",) + VAL:
        L = np.load(f"{F}/{s}_logits.npy").astype(np.float32)
        out[s] = dict(L=L, P=1 / (1 + np.exp(-L.mean(1))), E=np.load(f"{F}/{s}_E.npy").astype(np.float32),
                      refs=refs_of(np.load(f"{F}/{s}_mask.npy").astype(np.float32)))
    out["gold"]["fold"] = gold_fold_of(len(out["gold"]["L"]))
    for s in VAL:
        ids = [str(x) for x in np.load(f"./work/refiner_{s}.npz")["ids"]]
        out[s]["fold"] = np.array([val_fold_of(c) for c in ids])
    out["gold"]["heldout"] = np.load(f"{F}/gold_heldout_idx.npy")
    return out


def v5_events(D):
    X, y, fold = [], [], []
    for s, d in D.items():
        vit = viterbi_events(d["P"])
        gate = fpf.gate_of(d["P"], vit)
        for i, (ev, g) in enumerate(zip(vit, gate)):
            if not g:
                continue
            ev = correct_events(ev, d["P"][i], T5)
            for k in range(len(ev)):
                X.append(fpf.features(d["L"][i], d["P"][i], d["E"][i], ev, k))
                y.append(0 if any(fpf.overlap(ev[k], r) for r in d["refs"][i]) else 1)
                fold.append(d["fold"][i])
    return np.array(X, np.float32), np.array(y), np.array(fold)


def main():
    D = load_fold_sets()
    X5, y5, f5 = v5_events(D)
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        pickle.dump(dict(model=vf.gbm().fit(X5, y5)), open("./checkpoints/verifier5.pkl", "wb"))
        print(f"fitted verifier5 on {len(y5)} events ({y5.mean():.3f} pure FP)", flush=True)
        return
    # v1's evaluation events and OOF scores
    sets = [vf.load_set(n) for n in ("gold_heldout",) + VAL]
    ev_X, ev_E, ev_y, ev_key = vf.build_training(sets)[:4]
    pca = PCA(vf.N_PCA, random_state=0).fit(ev_E)
    o1 = vf.oof_scores(np.c_[ev_X, pca.transform(ev_E)], ev_y, ev_key)
    # v5 features on v1's events (gold_heldout rows mapped into the Gold-all arrays)
    gh = D["gold"]["heldout"]
    by = {s["name"]: s for s in sets}
    Xe, fe = [], []
    for name, i, k in ev_key:
        ev = by[name]["base"][i]
        if name == "gold_heldout":
            d, j = D["gold"], gh[i]
        else:
            d, j = D[name], i
        Xe.append(fpf.features(d["L"][j], d["P"][j], d["E"][j], ev, k))
        fe.append(d["fold"][j])
    Xe, fe = np.array(Xe, np.float32), np.array(fe)
    o5 = np.zeros(len(ev_y))
    for k in range(5):
        m = vf.gbm().fit(X5[f5 != k], y5[f5 != k])
        o5[fe == k] = m.predict_proba(Xe[fe == k])[:, 1]
    print(f"v5 training events {len(y5)} (pure FP {y5.mean():.3f}) vs v1 {len(ev_y)}", flush=True)
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    res = {}
    for tag, o in (("v1", o1), ("v5", o5), ("avg", (o1 + o5) / 2)):
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
        print(f"[{tag:3s}] AUC {res[tag][0]:.4f} | drops {len(drop)} | pooled {res[tag][1]:.4f} | Nat F1 {f1_only(n):.4f} "
              f"Dice {n[:, 3].mean():.4f} | Syn F1 {f1_only(sy):.4f} Dice {sy[:, 3].mean():.4f}", flush=True)
    for tag in ("v5", "avg"):
        ok = (res[tag][0] - res["v1"][0] >= 0.01 and res[tag][1] - res["v1"][1] >= 0.002
              and abs(res[tag][2] - res["v1"][2]) <= 0.0005 and abs(res[tag][3] - res["v1"][3]) <= 0.0005)
        print(f"GATE {tag}: {'PASS' if ok else 'FAIL'} (AUC {res[tag][0] - res['v1'][0]:+.4f}, pooled {res[tag][1] - res['v1'][1]:+.4f})",
              flush=True)


if __name__ == "__main__":
    main()
