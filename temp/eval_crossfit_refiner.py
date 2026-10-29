"""Score the v7 refiner fixes on out-of-fold predictions for all 2,501 validation clips
(temp/crossfit_refiner.py). Base = the v5 pipeline on the v4 ensemble (never trained on
validation): Viterbi 0.9802/0.9489 + gated edge correction. Refiner events follow
release/v7_inference.refiner_events (proposals thr 0.5 -> correction -> clip -> drop < 0.02 s),
used only on Natural-looking clips (the v7 gate). Candidates, fixed before looking:
  single seed s0/s1/s2      -- what v7/v7b did (calibration against the real -0.018..-0.021 Dice)
  avg                       -- refiner outputs averaged over 3 seeds
  X + fb_empty              -- base events where the refiner returns nothing
  X + fb_r                  -- base events where refiner coverage < r * base coverage (r 0.8/0.9/1.0)
Paired bootstrap (stratified by domain) against the base.

Usage: python -m temp.eval_crossfit_refiner
"""
import glob
import json
import os

import numpy as np

from src.edge_correction import correct_events, edge_contrast, load_table
from src.refiner import decode_proposals
from temp.crossfit_refiner import OUT, all_data
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

TABLE = load_table("./checkpoints/edge_correction_v5.json")
ROOT = "./data/official_validation_raw/validation"


def durations():
    split = json.load(open("./data/val_split_s0.json"))
    meta = {r["segmentFileName"]: float(r["duration"]) for r in json.load(open(f"{ROOT}/validationMetadata.json", encoding="utf-8"))}
    out = []
    for sub in ("natural", "synthetic"):
        ids = [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]]
        for part in ("train70", "hold30"):
            keep = set(split[part][sub])
            out += [min(meta[n], 10.0) for n in ids if n in keep]  # same order as all_data()
    return np.array(out)


def clip(ev, sec):
    ev = [(a, min(b, sec)) for a, b in ev]
    return [(round(a, 3), round(b, 3)) for a, b in ev if b - a >= 0.02]


def cover(ev):
    return sum(b - a for a, b in ev)


def main():
    L, _, Y, _, nat = all_data()
    sec = durations()
    assert len(sec) == len(L)
    N = len(L)
    base_p = 1 / (1 + np.exp(-L.mean(1)))
    refs = refs_of(Y)
    base_vit = viterbi_events(base_p)
    base = [clip(correct_events(e, p, TABLE), s) for e, p, s in zip(base_vit, base_p, sec)]
    gate = np.array([True if not e else np.mean([edge_contrast(p, a, b) for a, b in e]) < 0.5 for e, p in zip(base_vit, base_p)])

    preds, kinds = {}, []
    for s in range(3):
        files = sorted(glob.glob(f"{OUT}/f*_s{s}.npz"))
        assert len(files) == 5, (s, files)
        P = {k: np.zeros((N, 1000), np.float32) for k in ("presence", "d_on", "d_off")}
        for f in files:
            d = np.load(f)
            kinds.append(str(d["kind"]))  # epoch chosen on the best of 3 decodes; v7 always decodes proposals
            for k in P:
                P[k][d["idx"]] = d[k]
        preds[f"s{s}"] = P
    print(f"best decode on tune slices: { {k: kinds.count(k) for k in set(kinds)} }", flush=True)
    preds["avg"] = {k: np.mean([preds[f"s{s}"][k] for s in range(3)], 0) for k in ("presence", "d_on", "d_off")}

    def refined(P):
        return [clip(correct_events(decode_proposals(P["presence"][i], P["d_on"][i], P["d_off"][i], 0.5), base_p[i], TABLE), sec[i])
                for i in range(N)]

    cands = {"base (v5 pipeline)": base}
    for name in ("s0", "s1", "s2", "avg"):
        R = refined(preds[name])
        cands[name] = [r if g else b for r, b, g in zip(R, base, gate)]
        if name in ("s0", "avg"):
            cands[f"{name} + fb_empty"] = [r if g and (r or not b) else b for r, b, g in zip(R, base, gate)]
            for rr in (0.8, 0.9, 1.0):
                cands[f"{name} + fb_{rr}"] = [r if g and r and cover(r) >= rr * cover(b) else (r if g and not b else b)
                                            for r, b, g in zip(R, base, gate)]
    rows = {k: per_clip(refs, v) for k, v in cands.items()}
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    rng = np.random.default_rng(0)
    ni, si = np.where(nat)[0], np.where(~nat)[0]
    boots = [np.concatenate([rng.choice(ni, len(ni)), rng.choice(si, len(si))]) for _ in range(2000)]
    b0 = rows["base (v5 pipeline)"]
    c0 = np.array([comb(b0[b]) for b in boots])
    print(f"{N} clips ({nat.sum()} Natural); gate routes {gate[nat].mean():.1%} of Natural, {gate[~nat].mean():.1%} of synthetic", flush=True)
    for k, r in rows.items():
        d = np.array([comb(r[b]) for b in boots]) - c0
        changed = np.mean([a != b for a, b in zip(cands[k], base)])
        print(f"[{k:18s}] comb={comb(r):.4f} d={d.mean():+.4f} (5%={np.percentile(d, 5):+.4f} P>0={np.mean(d > 0):.3f}) | "
              f"natF1 {f1_only(r[nat]):.4f} ({f1_only(r[nat]) - f1_only(b0[nat]):+.4f}) natDice {r[nat, 3].mean():.4f} "
              f"({r[nat, 3].mean() - b0[nat, 3].mean():+.4f}) | syn {f1_only(r[~nat]):.4f}/{r[~nat, 3].mean():.4f} | "
              f"changed {changed:.1%} empty {sum(not e for e in cands[k])}", flush=True)


if __name__ == "__main__":
    main()
