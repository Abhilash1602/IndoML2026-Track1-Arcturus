"""Oct 14, idea #1: train the drop verifier on the events it is applied to. The final pipeline drops
from *recipe* events (ungated blend 0.7 with refiner presence -> Viterbi + correction -> refiner-vote
merges/drops), but verifier v1 was trained on the old v4-base decode events. Here, on validation with
cross-fitted refiners (work/crossfit), recipe-analog events are built for every clip, and three verifiers
(same v1 features: temp/verifier.event_rows) are compared on them, 5-fold by clip:
  A  trained on v4-base events (Gold held-out + other-fold validation)            = how v1 is trained
  B  trained on recipe events of the other folds
  C  trained on both
Drops: unguarded recipe events (mean blended prob < 0.70) with score > thr. Pre-registered: B or C >= A
+ 0.002 pooled (validation Natural + Synthetic), synthetic F1 and Dice within +/-0.0005 of A.

Usage: python -m temp.verifier_recipe            (evaluation)
       python -m temp.verifier_recipe fit B|C    (fit on everything -> checkpoints/verifier_recipe.pkl)
"""
import json
import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA

import temp.verifier as vf
from src.edge_correction import correct_events
from temp.crossfit_refiner import all_data, folds
from temp.eval_crossfit_refiner import clip, durations
from temp.eval_recipe_refiners import T, presence, recipe, seeds
from temp.fp_filter import overlap
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

W = 0.7


def mapping():
    split = json.load(open("./data/val_split_s0.json"))
    out = []
    for sub in ("natural", "synthetic"):
        ids = [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]]
        for part in ("train70", "hold30"):
            keep = set(split[part][sub])
            out += [(f"official_{sub}", j) for j, n in enumerate(ids) if n in keep]
    return out


def main():
    L, _, Y, _, nat = all_data()
    N, sec = len(L), durations()
    mp = mapping()
    assert len(mp) == N
    fold = folds(nat)
    refs = refs_of(Y)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    P = 1 / (1 + np.exp(-(W * L.mean(1) + (1 - W) * lg(presence("crossfit", N)))))
    base = [clip(correct_events(e, p, T), s) for e, p, s in zip(viterbi_events(P), P, sec)]
    Rs = seeds("crossfit", N, P, sec)
    rec = [recipe(b, Rs, i) for i, b in enumerate(base)]
    sets = {n: vf.load_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")}
    # recipe events + v1 features
    rX, rE, ry, rclip = [], [], [], []
    for i, ev in enumerate(rec):
        name, j = mp[i]
        for k, (f, e) in enumerate(vf.event_rows(sets[name], j, ev)):
            rX.append(f); rE.append(e); ry.append(0 if any(overlap(ev[k], r) for r in refs[i]) else 1); rclip.append(i)
    rX, rE, ry, rclip = np.array(rX, np.float32), np.array(rE, np.float32), np.array(ry), np.array(rclip)
    # v4-base events (how v1 is trained): Gold held-out (always in training) + validation by fold
    bX, bE, by, bkey = vf.build_training(list(sets.values()))[:4]
    inv = {v: i for i, v in enumerate(mp)}
    bfold = np.array([-1 if n == "gold_heldout" else fold[inv[(n, j)]] for n, j, _ in bkey])
    pca = PCA(vf.N_PCA, random_state=0).fit(np.r_[bE, rE])
    RX, BX = np.c_[rX, pca.transform(rE)], np.c_[bX, pca.transform(bE)]
    if len(sys.argv) > 2 and sys.argv[1] == "fit":
        Xf, yf = (RX, ry) if sys.argv[2] == "B" else (np.r_[BX, RX], np.r_[by, ry])
        pickle.dump(dict(pca=pca, model=vf.gbm().fit(Xf, yf), kind=sys.argv[2]), open("./checkpoints/verifier_recipe.pkl", "wb"))
        print(f"fitted verifier_recipe {sys.argv[2]} on {len(yf)} events", flush=True)
        return
    rf = fold[rclip]
    scores = {k: np.zeros(len(ry)) for k in "ABC"}
    for f in range(5):
        te = rf == f
        trA = bfold != f
        scores["A"][te] = vf.gbm().fit(BX[trA], by[trA]).predict_proba(RX[te])[:, 1]
        scores["B"][te] = vf.gbm().fit(RX[~te], ry[~te]).predict_proba(RX[te])[:, 1]
        scores["C"][te] = vf.gbm().fit(np.r_[BX[trA], RX[~te]], np.r_[by[trA], ry[~te]]).predict_proba(RX[te])[:, 1]
    from sklearn.metrics import roc_auc_score
    conf = np.array([P[i][int(a * 100):max(int(b * 100), int(a * 100) + 1)].mean() for i, ev in enumerate(rec) for a, b in ev])
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    r0 = per_clip(refs, rec)
    print(f"recipe events {len(ry)} (pure FP {ry.mean():.3f}, unguarded {np.mean(conf < 0.7):.3f}) | no extra drops: pooled {comb(r0):.4f}",
          flush=True)
    res = {}
    for thr in (0.25, 0.35, 0.5):
        for k, s in scores.items():
            drop = set(np.where((s > thr) & (conf < 0.7))[0])
            out, c = [], 0
            for i, ev in enumerate(rec):
                out.append([e for e, kk in zip(ev, range(c, c + len(ev))) if kk not in drop]); c += len(ev)
            r = per_clip(refs, out)
            res[(k, thr)] = (comb(r), f1_only(r[~nat]), r[~nat, 3].mean())
            print(f"[{k}] thr {thr}: AUC {roc_auc_score(ry, s):.4f} drops {len(drop):4d} | pooled {comb(r):.4f} | Nat F1 {f1_only(r[nat]):.4f} "
                  f"Dice {r[nat, 3].mean():.4f} | Syn F1 {f1_only(r[~nat]):.4f} Dice {r[~nat, 3].mean():.4f}", flush=True)
    for k in "BC":
        best = max((t for t in (0.25, 0.35, 0.5)), key=lambda t: res[(k, t)][0] - res[("A", t)][0])
        d = res[(k, best)][0] - res[("A", best)][0]
        syn_ok = abs(res[(k, best)][1] - res[("A", best)][1]) <= 0.0005 and abs(res[(k, best)][2] - res[("A", best)][2]) <= 0.0005
        print(f"GATE {k}: {'PASS' if d >= 0.002 and syn_ok else 'FAIL'} (thr {best}: {d:+.4f} vs A, synthetic ok {syn_ok})", flush=True)


if __name__ == "__main__":
    main()
