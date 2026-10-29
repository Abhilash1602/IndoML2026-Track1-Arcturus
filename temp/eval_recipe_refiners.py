"""Compare refiner sets on the local current-recipe analog (validation, out-of-fold).

Recipe (tracks the real leaderboard for merges): on Natural-gated clips of the v4-ensemble v5
decode, merge an adjacent pair when a majority of the refiner seeds have one event spanning both,
then drop events that no seed supports. Reported for each cross-fit directory (e.g. the original
validation-only refiners vs the ones that also train on Gold held-out), plus the merge-only and
drop-only parts, with a paired bootstrap of the pooled difference between directories.

Usage: python -m temp.eval_recipe_refiners crossfit crossfit_gold
"""
import glob
import sys

import numpy as np

from src.edge_correction import correct_events, edge_contrast, load_table
from src.refiner import decode_proposals
from temp.crossfit_refiner import all_data
from temp.eval_crossfit_refiner import clip, durations
from temp.fp_filter import overlap
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

T = load_table("./checkpoints/edge_correction_v5.json")


def presence(out_dir, N):
    P = np.zeros((3, N, 1000), np.float32)
    for s in range(3):
        for f in glob.glob(f"{S}/{out_dir}/f*_s{s}.npz"):
            d = np.load(f)
            P[s, d["idx"]] = d["presence"]
    return P.mean(0)


def seeds(out_dir, N, bp, sec):
    R = []
    for s in range(3):
        files = sorted(glob.glob(f"{S}/{out_dir}/f*_s{s}.npz"))
        assert len(files) == 5, (out_dir, s, files)
        P = {k: np.zeros((N, 1000), np.float32) for k in ("presence", "d_on", "d_off")}
        for f in files:
            d = np.load(f)
            for k in P:
                P[k][d["idx"]] = d[k]
        R.append([clip(correct_events(decode_proposals(P["presence"][i], P["d_on"][i], P["d_off"][i], .5), bp[i], T), sec[i])
                  for i in range(N)])
    return R


def recipe(b, Rs, i, merges=True, drops=True):
    K = len(Rs)
    pv = [sum(any(overlap(b[k], f) and overlap(b[k + 1], f) for f in R[i]) for R in Rs) for k in range(len(b) - 1)]
    sup = [sum(any(overlap(e, f) for f in R[i]) for R in Rs) for e in b]
    m, sp = [], []
    for k, e in enumerate(b):
        if merges and m and pv[k - 1] * 2 > K:
            m[-1] = (m[-1][0], max(m[-1][1], e[1])); sp[-1] = K
        else:
            m.append(e); sp.append(sup[k])
    return [e for e, s in zip(m, sp) if not drops or s > 0]


def main():
    dirs = sys.argv[1:]
    L, _, Y, _, nat = all_data()
    N, sec = len(L), durations()
    bp = 1 / (1 + np.exp(-L.mean(1)))
    refs = refs_of(Y)
    vit = viterbi_events(bp)
    base = [clip(correct_events(e, p, T), s) for e, p, s in zip(vit, bp, sec)]
    gate = np.array([True if not e else np.mean([edge_contrast(p, a, b) for a, b in e]) < .5 for e, p in zip(vit, bp)])
    r0 = per_clip(refs, base)
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    rng = np.random.default_rng(0)
    ni, si = np.where(nat)[0], np.where(~nat)[0]
    boots = [np.concatenate([rng.choice(ni, len(ni)), rng.choice(si, len(si))]) for _ in range(1000)]
    full, blend = {}, {}
    Lm = L.mean(1)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    for d in dirs:
        Rs = seeds(d, N, bp, sec)
        for name, kw in (("merges + drops", {}), ("merges only", {"drops": False}), ("drops only", {"merges": False})):
            out = [recipe(b, Rs, i, **kw) if gate[i] else b for i, b in enumerate(base)]
            r = per_clip(refs, out)
            if name == "merges + drops":
                full[d] = r
            print(f"[{d:14s}] {name:15s} natF1 {f1_only(r[nat]) - f1_only(r0[nat]):+.4f} natDice "
                  f"{r[nat, 3].mean() - r0[nat, 3].mean():+.4f} | pooled {comb(r) - comb(r0):+.4f}", flush=True)
        # current best (v11g_w75): gated blend w = 0.75 with these refiners' presence, then the recipe
        P = bp.copy()
        Pb = 1 / (1 + np.exp(-(0.75 * Lm + 0.25 * lg(presence(d, N)))))
        P[gate] = Pb[gate]
        vit2 = viterbi_events(P)
        base2 = [clip(correct_events(e, p, T), s) for e, p, s in zip(vit2, P, sec)]
        gate2 = [True if not e else np.mean([edge_contrast(p, a, c) for a, c in e]) < .5 for e, p in zip(vit2, P)]
        Rs2 = seeds(d, N, P, sec)
        r = per_clip(refs, [recipe(b, Rs2, i) if gate2[i] else b for i, b in enumerate(base2)])
        blend[d] = r
        print(f"[{d:14s}] {'gated blend .75 + recipe':15s} natF1 {f1_only(r[nat]) - f1_only(r0[nat]):+.4f} natDice "
              f"{r[nat, 3].mean() - r0[nat, 3].mean():+.4f} | pooled {comb(r) - comb(r0):+.4f}", flush=True)
    for k, res in (("merges + drops", full), ("gated blend .75 + recipe", blend)):
        for d in dirs[1:]:
            a, b = res[dirs[0]], res[d]
            dd = np.array([comb(b[ix]) - comb(a[ix]) for ix in boots])
            print(f"{d} - {dirs[0]} ({k}): {comb(b) - comb(a):+.4f} (5%={np.percentile(dd, 5):+.4f}, P>0={np.mean(dd > 0):.3f})",
                  flush=True)


if __name__ == "__main__":
    main()
