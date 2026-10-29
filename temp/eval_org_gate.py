"""Oct 12 gate for the organisers'-baseline member (work/hold30_logits/ORG.npz, temp/train_org_baseline.py).
hold30 only. Base = v4 + VT (the 10-member v6 analog). Candidates: base + ORG (11), base + ORG x2, and
base with each VT member replaced by ORG (10). Pre-registered pass: best candidate >= base + 0.004 pooled,
P >= 0.90, Natural F1 up (reported with the selection caveat when it is the best of several)."""
import json

import numpy as np

from src.config import Config
from temp.eval_dual_gated import S, V4, f1, pooled, refs_of, rows
from train import official_val_subset


def main():
    hold = json.load(open("./data/val_split_s0.json"))["hold30"]
    ds = {s: official_val_subset(Config(), s, "hold30") for s in ("natural", "synthetic")}
    base, refs = {}, {}
    for s in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{s}.npz")
        pos = {str(c): k for k, c in enumerate(d["ids"])}
        sel = [pos[ds[s].dataset.rows[i]["segmentFileName"]] for i in ds[s].indices]
        base[s], refs[s] = d["logits"].astype(np.float32)[sel], refs_of(d["mask"][sel])
    v4 = [{s: base[s][:, k] for s in base} for k in range(5)]
    vt = [dict(np.load(f"work/hold30_logits/VT_{m}.npz")) for m in V4]
    org = dict(np.load("work/hold30_logits/ORG.npz"))
    sc = lambda ms: tuple(rows(refs[s], 1 / (1 + np.exp(-np.mean([m[s] for m in ms], 0))), True) for s in ("natural", "synthetic"))
    nat, syn = (rows(refs[s], 1 / (1 + np.exp(-org[s])), False) for s in ("natural", "synthetic"))
    print(f"single ORG: natF1 {f1(nat):.4f} natDice {nat[:, 3].mean():.4f} synF1 {f1(syn):.4f} synDice {syn[:, 3].mean():.4f} "
          f"pooled {pooled(nat, syn):.4f}", flush=True)
    cands = {"base": v4 + vt, "base+ORG": v4 + vt + [org], "base+ORGx2": v4 + vt + [org, org]}
    for k, m in enumerate(V4):
        cands[f"base[{m}->ORG]"] = v4 + [x for j, x in enumerate(vt) if j != k] + [org]
    res = {k: sc(v) for k, v in cands.items()}
    rng = np.random.default_rng(0)
    nn_, ns = len(res["base"][0]), len(res["base"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(2000)]
    for k, (n, s) in res.items():
        p = np.mean([pooled(n, s, i, j) - pooled(*res["base"], i, j) > 0 for i, j in boots]) if k != "base" else 0
        print(f"[{k:28s}] pooled {pooled(n, s):.4f} ({pooled(n, s) - pooled(*res['base']):+.4f}, P>0 {p:.3f}) | "
              f"natF1 {f1(n):.4f} natDice {n[:, 3].mean():.4f} synF1 {f1(s):.4f} synDice {s[:, 3].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
