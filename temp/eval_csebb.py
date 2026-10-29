"""Fair test of change-point Sound Event Bounding Boxes (cSEBBs, Ebbers et al., Interspeech 2024;
MERL `sebbs`) as the event constructor, against our Viterbi + gated edge correction.

The three earlier SEBB attempts discarded the SEBB confidence (no detection threshold), so every
background segment became an event; they were never a valid test. Here:
  * candidates: CSEBBsPredictor(step_filter_length, merge_threshold_abs, merge_threshold_rel)
    .predict() on the ensemble's frame probability -> (onset, offset, confidence) boxes;
  * detection: keep boxes with confidence > threshold;
  * tuning: grid over the 4 hyper-parameters on Gold HELD-OUT (the v4 ensemble never trained on
    it), maximising our official metric (micro event F1 with max(0.2*dur, 0.05) on onset AND
    offset + macro Dice) -- not sed_scores_eval's collar F1;
  * evaluation: scored ONCE on the official validation set (Natural and Synthetic), vs the
    Viterbi + correction decode of the same probabilities, with a paired bootstrap.
cSEBB needs no domain gate: one decoder for every clip (the organisers prefer a single general
approach).

Usage: python -m temp.eval_csebb
"""
import itertools

import numpy as np
import sed_scores_eval
from sebbs.csebbs import CSEBBsPredictor

from src.edge_correction import correct_events, load_table
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

T = load_table("./checkpoints/edge_correction_v5.json")
STEPS = (0.16, 0.24, 0.32, 0.48, 0.64)
ABS = (0.1, 0.2, 0.3, 1.0)
REL = (1.5, 2.0, 3.0, 1e6)
THR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def load(name):
    d = np.load(f"{S}/refiner_{name}.npz")
    return 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1))), refs_of(d["mask"].astype(np.float32))


def scores_of(P):
    ts = np.arange(P.shape[1] + 1) * 0.01
    return {f"c{i}": sed_scores_eval.io.create_score_dataframe(p.reshape(-1, 1).astype(np.float64), timestamps=ts,
                                                               event_classes=["noise"]) for i, p in enumerate(P)}


def candidates(P, step, a, r, scores=None):
    pred = CSEBBsPredictor(step_filter_length=step, merge_threshold_abs=a, merge_threshold_rel=r, sound_classes=["noise"])
    out = pred.predict(scores if scores is not None else scores_of(P))
    return [[(float(on), float(off), float(c)) for on, off, _, c in out[f"c{i}"]] for i in range(len(P))]


def detect(cands, thr):
    return [[(a, b) for a, b, c in e if c > thr] for e in cands]


comb = lambda r: f1_only(r) + r[:, 3].mean()


def main():
    G, Gr = load("gold_heldout")
    gs = scores_of(G)
    rv = per_clip(Gr, [correct_events(e, p, T) for e, p in zip(viterbi_events(G), G)])
    results = []
    for step, a, r in itertools.product(STEPS, ABS, REL):
        c = candidates(G, step, a, r, gs)
        for thr in THR:
            rr = per_clip(Gr, detect(c, thr))
            results.append((comb(rr), f1_only(rr), rr[:, 3].mean(), step, a, r, thr))
    results.sort(reverse=True)
    print(f"Gold held-out: Viterbi+corr F1 {f1_only(rv):.4f} Dice {rv[:, 3].mean():.4f} comb {comb(rv):.4f}", flush=True)
    for x in results[:5]:
        print(f"  cSEBB comb {x[0]:.4f} (F1 {x[1]:.4f} Dice {x[2]:.4f}) step {x[3]} abs {x[4]} rel {x[5]} thr {x[6]}", flush=True)
    _, _, _, step, a, r, thr = results[0]
    rows = {}
    for sub in ("official_natural", "official_synthetic"):
        P, R = load(sub)
        rows[sub] = (per_clip(R, [correct_events(e, p, T) for e, p in zip(viterbi_events(P), P)]),
                     per_clip(R, detect(candidates(P, step, a, r), thr)))
        v, s = rows[sub]
        print(f"VALIDATION {sub}: Viterbi+corr F1 {f1_only(v):.4f} Dice {v[:, 3].mean():.4f} | cSEBB F1 {f1_only(s):.4f} "
              f"Dice {s[:, 3].mean():.4f}", flush=True)
    (nv, ns), (sv, ss) = rows["official_natural"], rows["official_synthetic"]
    c2 = lambda n, s: f1_only(np.r_[n, s]) + np.r_[n, s][:, 3].mean()
    rng = np.random.default_rng(0)
    d = [c2(ns[i], ss[j]) - c2(nv[i], sv[j]) for i, j in
         ((rng.integers(0, len(nv), len(nv)), rng.integers(0, len(sv), len(sv))) for _ in range(1000))]
    print(f"VALIDATION pooled: Viterbi+corr {c2(nv, sv):.4f} | cSEBB {c2(ns, ss):.4f} ({c2(ns, ss) - c2(nv, sv):+.4f}, "
          f"P>0={np.mean(np.array(d) > 0):.3f})", flush=True)


if __name__ == "__main__":
    main()
