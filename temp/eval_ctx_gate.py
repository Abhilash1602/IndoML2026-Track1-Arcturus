"""Oct 12 gate: does in-domain masked-reconstruction pre-training of the context network help?
Members VC_mae (pre-trained context) and VC_rand (same, random context = control), both fine-tuned
with the student recipe on train70 from the VT dasheng_owB backbone. Each REPLACES VT_dasheng_owB in
the 10-member v6 analog (v4 + VT), so the ensemble size is fixed. hold30 only, paired bootstrap.
Pre-registered: (v4+VT with VC_mae) >= (v4+VT with VC_rand) + 0.004 pooled, P >= 0.90, Natural F1 up.
"""
import json
import os

import numpy as np

from src.architecture import StandaloneDashengSED
from src.config import Config
from temp.eval_dual_gated import S, V4, f1, pooled, refs_of, rows
from temp.eval_new_members import cached
from train import official_val_subset


def main():
    hold = json.load(open("./data/val_split_s0.json"))["hold30"]
    ds = {sub: official_val_subset(Config(), sub, "hold30") for sub in ("natural", "synthetic")}
    base, refs = {}, {}
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        pos = {str(c): k for k, c in enumerate(d["ids"])}
        order = [ds[sub].dataset.rows[i]["segmentFileName"] for i in ds[sub].indices]
        assert set(order) == set(hold[sub])
        base[sub], refs[sub] = d["logits"].astype(np.float32)[[pos[c] for c in order]], refs_of(d["mask"][[pos[c] for c in order]])
    v4 = [{s: base[s][:, k] for s in base} for k in range(5)]
    vt = {m: dict(np.load(f"work/hold30_logits/VT_{m}.npz")) for m in V4}
    os.environ["CONTEXT"] = "mae"
    new = {k: cached(f"VC_{k}", StandaloneDashengSED, f"./checkpoints/standalone_dasheng_stage2_i2_owbceVC_{k}_teacher.pth", {}, ds)
           for k in ("mae", "rand")}

    def score(ms):
        return tuple(rows(refs[s], 1 / (1 + np.exp(-np.mean([m[s] for m in ms], 0))), True) for s in ("natural", "synthetic"))

    for name, m in [("VT_dasheng_owB", vt["dasheng_owB"]), ("VC_rand", new["rand"]), ("VC_mae", new["mae"])]:
        nat, syn = (rows(refs[s], 1 / (1 + np.exp(-m[s])), False) for s in ("natural", "synthetic"))
        print(f"  single {name:16s}: natF1 {f1(nat):.4f} natDice {nat[:, 3].mean():.4f} synF1 {f1(syn):.4f} pooled {pooled(nat, syn):.4f}", flush=True)
    others = [vt[m] for m in V4 if m != "dasheng_owB"]
    res = {"v4+VT": score(v4 + list(vt.values())), "v4+VT[owB->rand]": score(v4 + others + [new["rand"]]),
           "v4+VT[owB->mae]": score(v4 + others + [new["mae"]])}
    rng = np.random.default_rng(0)
    nn_, ns = len(res["v4+VT"][0]), len(res["v4+VT"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(2000)]
    for k, (nat, syn) in res.items():
        print(f"[{k:18s}] pooled {pooled(nat, syn):.4f} | natF1 {f1(nat):.4f} natDice {nat[:, 3].mean():.4f} synF1 {f1(syn):.4f} "
              f"synDice {syn[:, 3].mean():.4f}", flush=True)
    a, b = res["v4+VT[owB->mae]"], res["v4+VT[owB->rand]"]
    d = pooled(*a) - pooled(*b)
    p = float(np.mean([pooled(*a, i, j) - pooled(*b, i, j) > 0 for i, j in boots]))
    up = f1(a[0]) > f1(b[0])
    d0 = pooled(*a) - pooled(*res["v4+VT"])
    print(f"mae vs rand (ensemble): {d:+.4f} P {p:.3f}; mae vs current VT ensemble {d0:+.4f}", flush=True)
    print(f"GATE: {'PASS' if d >= 0.004 and p >= 0.90 and up else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
