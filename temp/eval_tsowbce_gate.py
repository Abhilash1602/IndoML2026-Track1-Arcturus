"""Oct 11 gate: tolerance-shaped OWBCE (VW_* members) vs the same round with the fixed OWBCE
window (VB_* members). Both rounds start from the train70 VT teachers with identical data, so the
only difference is the loss. Scored on hold30 only, as 10-member v6 analogs (v4 + 5 round members,
logit mean, fixed decode + gated correction), with a paired clip bootstrap.

Pre-registered: pass if v4+VW >= v4+VB + 0.004 pooled, P(>0) >= 0.90, and Natural F1 up.
Single-member rows (VB vs VW per family) are printed for the mechanism.

Usage: python -m temp.eval_tsowbce_gate [suffix]   (suffix "_teacher" default, or "" for students)
"""
import json
import os
import sys

import numpy as np

from src.config import Config
from temp.eval_dual_gated import S, V4, f1, pooled, refs_of, rows
from temp.eval_holdout import FT
from temp.eval_new_members import cached
from train import official_val_subset


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_teacher"
    hold = json.load(open("./data/val_split_s0.json"))["hold30"]
    config = Config()
    ds = {sub: official_val_subset(config, sub, "hold30") for sub in ("natural", "synthetic")}
    base, refs = {}, {}
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        pos = {str(c): k for k, c in enumerate(d["ids"])}
        order = [ds[sub].dataset.rows[i]["segmentFileName"] for i in ds[sub].indices]
        assert set(order) == set(hold[sub])
        sel = [pos[c] for c in order]
        base[sub], refs[sub] = d["logits"].astype(np.float32)[sel], refs_of(d["mask"][sel])
    v4 = [{sub: base[sub][:, k] for sub in base} for k in range(len(V4))]
    A, B = os.environ.get("GATE_A", "VW"), os.environ.get("GATE_B", "VB")  # candidate round vs reference round
    rounds = {}
    for pfx in (B, A):
        rounds[pfx] = {}
        for m, (cls, stem, ov) in FT.items():
            path = stem.format(P=pfx) + suffix + ".pth"
            if os.path.exists(path):
                rounds[pfx][m] = cached(f"{pfx}_{m}{suffix}", cls, path, ov, ds)
            else:
                print(f"missing {path}", flush=True)

    def score(ms, correct=True):
        return tuple(rows(refs[sub], 1 / (1 + np.exp(-np.mean([m[sub] for m in ms], axis=0))), correct)
                     for sub in ("natural", "synthetic"))

    for m in V4:
        for pfx in (B, A):
            if m in rounds[pfx]:
                nat, syn = score([rounds[pfx][m]], correct=False)
                print(f"  single {pfx}_{m:14s}: natF1 {f1(nat):.4f} natDice {nat[:, 3].mean():.4f} synF1 {f1(syn):.4f} "
                      f"pooled {pooled(nat, syn):.4f}", flush=True)
    fam = [m for m in V4 if m in rounds[B] and m in rounds[A]]
    res = {f"v4+{B}": score(v4 + [rounds[B][m] for m in fam]), f"v4+{A}": score(v4 + [rounds[A][m] for m in fam]),
           f"{B} only": score([rounds[B][m] for m in fam]), f"{A} only": score([rounds[A][m] for m in fam])}
    rng = np.random.default_rng(0)
    nn_, ns = len(res[f"v4+{B}"][0]), len(res[f"v4+{B}"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(2000)]
    for a, b in ((f"v4+{A}", f"v4+{B}"), (f"{A} only", f"{B} only")):
        bb = np.array([pooled(*res[a], i, j) - pooled(*res[b], i, j) for i, j in boots])
        for k in (a, b):
            nat, syn = res[k]
            print(f"[{k:8s}] pooled {pooled(nat, syn):.4f} | natF1 {f1(nat):.4f} natDice {nat[:, 3].mean():.4f} "
                  f"synF1 {f1(syn):.4f} synDice {syn[:, 3].mean():.4f}", flush=True)
        d = pooled(*res[a]) - pooled(*res[b])
        print(f"  {a} vs {b}: {d:+.4f} (5%={np.percentile(bb, 5):+.4f}, P>0={np.mean(bb > 0):.3f}) on {len(fam)} families", flush=True)
    d = pooled(*res[f"v4+{A}"]) - pooled(*res[f"v4+{B}"])
    p = float(np.mean(np.array([pooled(*res[f"v4+{A}"], i, j) - pooled(*res[f"v4+{B}"], i, j) for i, j in boots]) > 0))
    up = f1(res[f"v4+{A}"][0]) > f1(res[f"v4+{B}"][0])
    print(f"GATE: {'PASS' if d >= 0.004 and p >= 0.90 and up else 'FAIL'} (delta {d:+.4f}, P {p:.3f}, natF1 up {up})", flush=True)


if __name__ == "__main__":
    main()
