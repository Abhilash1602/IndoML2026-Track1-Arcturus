"""Oct 12: the fair cSEBB re-check left open in the plan (Step 1b). Differences from temp/eval_csebb.py:
  * tuned for ONE global decoder on Natural + Synthetic together (5-fold cross-fit over the
    validation clips; Gold held-out always in the tuning part), and the Viterbi baseline is
    re-tuned per fold too, so both decoders are out-of-sample;
  * step filters down to 0.08 s (59% of Natural events are 0.15-0.6 s);
  * variants: (a) cSEBB, (b) cSEBB + v5 edge correction, (c) hybrid drop: Viterbi+corr events
    dropped when every overlapping cSEBB box has confidence < tau, (d) Dice-locked cuts: cSEBB
    boundaries inside Viterbi+corr events split them (frames unchanged).
Per-clip rows for every configuration are computed once; CV only selects among them.

Usage: python -m temp.eval_csebb2
"""
import itertools

import numpy as np

from src.edge_correction import correct_events, load_table
from temp.eval_csebb import candidates, scores_of
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

T = load_table("./checkpoints/edge_correction_v5.json")
STEPS = (0.08, 0.12, 0.16, 0.24, 0.32, 0.48)
ABSREL = [(0.2, 1e6), (0.3, 1e6), (1.0, 1e6), (1.0, 1.5), (1.0, 2.0), (0.3, 1.5)]
THR = (0.4, 0.5, 0.6, 0.7, 0.8)
VIT = [(ps, pe) for ps in (0.97, 0.9802, 0.99) for pe in (0.93, 0.9489, 0.97)]


def main():
    P, R, dom = [], [], []
    for name, dm in (("gold_heldout", "gold"), ("official_natural", "nat"), ("official_synthetic", "syn")):
        d = np.load(f"{S}/refiner_{name}.npz")
        p = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
        P.append(p); R += refs_of(d["mask"]); dom += [dm] * len(p)
    P = np.concatenate(P)
    dom = np.array(dom)
    rows = {}
    for ps, pe in VIT:
        ev = [correct_events(e, p, T) for e, p in zip(viterbi_events(P, ps, pe), P)]
        rows[("vit", ps, pe)] = per_clip(R, ev)
        if (ps, pe) == (0.9802, 0.9489):
            base_ev = ev
    print("viterbi rows done", flush=True)
    sc = scores_of(P)
    for step, (a, r) in itertools.product(STEPS, ABSREL):
        c = candidates(P, step, a, r, sc)
        for thr in THR:
            det = [[(x, y) for x, y, cf in e if cf > thr] for e in c]
            rows[("csebb", step, a, r, thr)] = per_clip(R, det)
            rows[("csebb+corr", step, a, r, thr)] = per_clip(R, [correct_events(e, p, T) for e, p in zip(det, P)])
        for tau in (0.3, 0.4, 0.5):
            keep = [[e for e in be if max([cf for x, y, cf in cc if min(y, e[1]) > max(x, e[0])], default=0) >= tau]
                    for be, cc in zip(base_ev, c)]
            rows[("hybrid_drop", step, a, r, tau)] = per_clip(R, keep)
        for cmin in (0.5, 0.7):
            cut = []
            for be, cc in zip(base_ev, c):
                out = []
                for e0, e1 in be:
                    pts = sorted({round(v, 2) for x, y, cf in cc if cf > cmin for v in (x, y) if e0 + 0.03 < v < e1 - 0.03})
                    bounds = [e0] + pts + [e1]
                    out += [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
                cut.append(out)
            rows[("dice_locked_cut", step, a, r, cmin)] = per_clip(R, cut)
        print(f"cSEBB step {step} abs {a} rel {r} done ({len(rows)} configs)", flush=True)

    val = np.where(dom != "gold")[0]
    gold = np.where(dom == "gold")[0]
    fold = np.random.default_rng(0).integers(0, 5, len(val))
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    out = {}
    for fam in ("vit", "csebb", "csebb+corr", "hybrid_drop", "dice_locked_cut"):
        keys = [k for k in rows if k[0] == fam]
        oof = np.zeros((len(val), 4))
        chosen = []
        for f in range(5):
            tune = np.r_[gold, val[fold != f]]
            k = max(keys, key=lambda k: comb(rows[k][tune]))
            chosen.append(k)
            oof[fold == f] = rows[k][val[fold == f]]
        out[fam] = oof
        nat = dom[val] == "nat"
        print(f"[{fam:16s}] OOF pooled {comb(oof):.4f} | Nat F1 {f1_only(oof[nat]):.4f} Dice {oof[nat, 3].mean():.4f} | "
              f"Syn F1 {f1_only(oof[~nat]):.4f} Dice {oof[~nat, 3].mean():.4f} | chosen {chosen[0][1:]}", flush=True)
    rng = np.random.default_rng(1)
    for fam in ("csebb", "csebb+corr", "hybrid_drop", "dice_locked_cut"):
        d = [comb(out[fam][s]) - comb(out["vit"][s]) for s in (rng.integers(0, len(val), len(val)) for _ in range(1000))]
        print(f"  {fam} vs tuned Viterbi+corr: {comb(out[fam]) - comb(out['vit']):+.4f} (P>0 {np.mean(np.array(d) > 0):.3f})", flush=True)


if __name__ == "__main__":
    main()
