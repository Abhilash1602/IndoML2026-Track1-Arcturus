"""Oct 14, idea #3: decide per CLIP, not per event. Today's real probes showed that dropping every
unguarded event gains Event F1 (+0.0056) but empties clips and costs Dice; the right choice depends on clip
context. On validation (cross-fitted refiners; the recipe analog of the final pipeline):
  c0 = recipe events after verifier drops (OOF v1-style verifier, threshold 0.35, 0.70 guard)  [current]
  c1 = c0 without any unguarded event                                                          [aggressive]
A gradient-boosted regressor (5-fold by clip) predicts each clip's change in the pooled score
(micro-F1 linearised at the current totals + Dice / n_clips) from switching c0 -> c1, using only
features available at test time; a clip switches when the prediction is > 0.
Pre-registered: selection >= c0 + 0.002 pooled, synthetic F1 and Dice within +/-0.0005.

Usage: python -m temp.clip_selector
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

import temp.verifier as vf
from src.edge_correction import correct_events
from temp.crossfit_refiner import all_data, folds
from temp.eval_crossfit_refiner import clip, durations
from temp.eval_recipe_refiners import T, presence, recipe, seeds
from temp.fp_filter import overlap
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.verifier_recipe import W, mapping


def main():
    L, _, Y, _, nat = all_data()
    N, sec = len(L), durations()
    mp, fold, refs = mapping(), folds(nat), refs_of(Y)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    P = 1 / (1 + np.exp(-(W * L.mean(1) + (1 - W) * lg(presence("crossfit", N)))))
    base = [clip(correct_events(e, p, T), s) for e, p, s in zip(viterbi_events(P), P, sec)]
    Rs = seeds("crossfit", N, P, sec)
    rec = [recipe(b, Rs, i) for i, b in enumerate(base)]
    sets = {n: vf.load_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")}
    bX, bE, by, bkey = vf.build_training(list(sets.values()))[:4]
    inv = {v: i for i, v in enumerate(mp)}
    bfold = np.array([-1 if n == "gold_heldout" else fold[inv[(n, j)]] for n, j, _ in bkey])
    rows, rE, rclip = [], [], []
    for i, ev in enumerate(rec):
        name, j = mp[i]
        for f, e in vf.event_rows(sets[name], j, ev):
            rows.append(f); rE.append(e); rclip.append(i)
    rows, rE, rclip = np.array(rows, np.float32), np.array(rE, np.float32), np.array(rclip)
    pca = PCA(vf.N_PCA, random_state=0).fit(np.r_[bE, rE])
    RX, BX = np.c_[rows, pca.transform(rE)], np.c_[bX, pca.transform(bE)]
    score = np.zeros(len(rclip))
    for f in range(5):
        te = fold[rclip] == f
        score[te] = vf.gbm().fit(BX[bfold != f], by[bfold != f]).predict_proba(RX[te])[:, 1]
    conf = np.array([P[i][int(a * 100):max(int(b * 100), int(a * 100) + 1)].mean() for i, ev in enumerate(rec) for a, b in ev])
    c0, c1, feats, k = [], [], [], 0
    for i, ev in enumerate(rec):
        s, cf = score[k:k + len(ev)], conf[k:k + len(ev)]
        k += len(ev)
        keep0 = [(e, si, ci) for e, si, ci in zip(ev, s, cf) if not (si > 0.35 and ci < 0.7)]
        c0.append([e for e, _, _ in keep0])
        c1.append([e for e, _, ci in keep0 if ci >= 0.7])
        ung = [(si, ci, e[1] - e[0]) for e, si, ci in keep0 if ci < 0.7]
        gua = [ci for _, _, ci in keep0 if ci >= 0.7]
        feats.append([len(keep0), len(ung), len(gua), sec[i], P[i].mean(), P[i].max(),
                      max([u[0] for u in ung], default=0), min([u[0] for u in ung], default=0),
                      min([u[1] for u in ung], default=1), np.mean([u[1] for u in ung]) if ung else 1,
                      sum(u[2] for u in ung), sum(e[1] - e[0] for e in c0[-1]), float(len(gua) == 0)])
    feats = np.array(feats, np.float32)
    r0, r1 = per_clip(refs, c0), per_clip(refs, c1)
    TP, FP, FN = r0[:, 0].sum(), r0[:, 1].sum(), r0[:, 2].sum()
    Dn = 2 * TP + FP + FN
    dF_dTP, dF_dE = 2 * (FP + FN) / Dn ** 2, -2 * TP / Dn ** 2
    target = dF_dTP * (r1[:, 0] - r0[:, 0]) + dF_dE * ((r1[:, 1] + r1[:, 2]) - (r0[:, 1] + r0[:, 2])) + (r1[:, 3] - r0[:, 3]) / N
    pred = np.zeros(N)
    for f in range(5):
        te = fold == f
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=30, random_state=0)
        pred[te] = m.fit(feats[~te], target[~te]).predict(feats[te])
    sel = [b if p > 0 else a for a, b, p in zip(c0, c1, pred)]
    keep_rule = [b if b else a for a, b in zip(c0, c1)]  # the real-probed "never empty a clip" rule
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    rng = np.random.default_rng(0)
    ni, si = np.where(nat)[0], np.where(~nat)[0]
    boots = [np.concatenate([rng.choice(ni, len(ni)), rng.choice(si, len(si))]) for _ in range(1000)]
    for name, ev in (("c0 (current)", c0), ("c1 (drop all unguarded)", c1), ("keep rule", keep_rule), ("learned selection", sel)):
        r = per_clip(refs, ev)
        d = np.array([comb(r[ix]) - comb(r0[ix]) for ix in boots])
        print(f"[{name:24s}] pooled {comb(r):.4f} ({comb(r) - comb(r0):+.4f}, P>0 {np.mean(d > 0):.3f}) | Nat F1 {f1_only(r[nat]):.4f} "
              f"Dice {r[nat, 3].mean():.4f} | Syn F1 {f1_only(r[~nat]):.4f} Dice {r[~nat, 3].mean():.4f} | clips switched "
              f"{sum(1 for a, b in zip(c0, ev) if a != b)}", flush=True)
    rs = per_clip(refs, sel)
    gain = comb(rs) - comb(r0)
    syn_ok = abs(f1_only(rs[~nat]) - f1_only(r0[~nat])) <= 0.0005 and abs(rs[~nat, 3].mean() - r0[~nat, 3].mean()) <= 0.0005
    print(f"GATE: {'PASS' if gain >= 0.002 and syn_ok else 'FAIL'} (learned selection {gain:+.4f}, synthetic ok {syn_ok})", flush=True)


if __name__ == "__main__":
    main()
