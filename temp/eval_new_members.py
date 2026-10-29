"""Win-strategy gate (Oct 12): do the new backbone members (Lane A Dasheng-0.6B, Lane B
Whisper + 10 ms CNN) add to our ensemble? Measured ONLY on hold30 (data/val_split_s0.json),
which neither the new members (VAL_TRAIN=train70) nor the VT members have trained on.

Base = the local analog of the real 10-member v6 ensemble: the 5 v4 members (never saw
validation; cached logits) + the 5 VT members (VA's recipe on train70, EMA teachers).
Candidates = base + each new member, and base + all new members (logit mean, fixed decode
0.9802/0.9489, + gated edge correction as in v6). Single-model rows are printed too.
Paired clip bootstrap vs base. Pre-registered gate: >= +0.006 pooled, P(>0) >= 0.95.

New members via env NEW_MEMBERS="name=kind:path,..." with kind dasheng06 | whisperfusion.
Hold30 logits are cached in work/hold30_logits/<name>.npz.

Usage: NEW_MEMBERS="d06=dasheng06:./checkpoints/...teacher.pth,wh=whisperfusion:./checkpoints/fusion_whisperVT_stage1.pth" \
       python -m temp.eval_new_members
"""
import os

import numpy as np

from src.architecture import FusionSED, StandaloneDashengSED
from temp.eval_dual_gated import S, V4, f1, pooled, rows
from temp.eval_holdout import FT, hold_logits
from train import official_val_subset
from src.config import Config

CACHE = "./work/hold30_logits"
KINDS = {"dasheng06": (StandaloneDashengSED, {}), "whisperfusion": (FusionSED, dict(FUSION_BRANCH="whisper")),
         "w2vbertfusion": (FusionSED, dict(FUSION_BRANCH="w2vbert"))}


def cached(name, cls, path, ov, ds):
    f = f"{CACHE}/{name}.npz"
    if os.path.exists(f) and os.path.getmtime(f) > os.path.getmtime(path):
        z = np.load(f)
        return {sub: z[sub] for sub in ("natural", "synthetic")}
    out = hold_logits(cls, path, ov, ds)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(f, **out)
    return out


def main():
    import json
    from temp.eval_dual_gated import refs_of
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
    members = {f"v4_{m}": {sub: base[sub][:, k] for sub in base} for k, m in enumerate(V4)}
    for m, (cls, stem, ov) in FT.items():
        path = stem.format(P="VT") + "_teacher.pth"
        members[f"VT_{m}"] = cached(f"VT_{m}", cls, path, ov, ds)
    new = {}
    for spec in filter(None, os.environ.get("NEW_MEMBERS", "").split(",")):
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        if not os.path.exists(path):
            print(f"missing {path}", flush=True)
            continue
        if kind == "dasheng06":
            os.environ["DASHENG_NAME"] = "mispeech/dasheng-0.6B"
        cls, ov = KINDS[kind]
        new[name] = cached(name, cls, path, ov, ds)
        os.environ.pop("DASHENG_NAME", None)

    def score(ms, correct=True):
        return tuple(rows(refs[sub], 1 / (1 + np.exp(-np.mean([m[sub] for m in ms], axis=0))), correct)
                     for sub in ("natural", "synthetic"))

    for name, m in {**{k: v for k, v in members.items() if k.startswith("VT_")}, **new}.items():
        nat, syn = score([m], correct=False)
        print(f"  single {name:22s}: natF1 {f1(nat):.4f} natDice {nat[:, 3].mean():.4f} | synF1 {f1(syn):.4f} "
              f"synDice {syn[:, 3].mean():.4f} | pooled {pooled(nat, syn):.4f}", flush=True)
    base_ms = list(members.values())
    cands = {"BASE (v4 + VT, 10)": base_ms}
    for name, m in new.items():
        cands[f"BASE + {name}"] = base_ms + [m]
        cands[f"BASE + {name} x2"] = base_ms + [m, m]
        cands[f"VT5 + {name}"] = [v for k, v in members.items() if k.startswith("VT_")] + [m]
    if len(new) > 1:
        cands["BASE + all new"] = base_ms + list(new.values())
    cands["VT5 only"] = [v for k, v in members.items() if k.startswith("VT_")]
    res = {k: score(v) for k, v in cands.items()}
    rng = np.random.default_rng(0)
    nn_, ns = len(res["VT5 only"][0]), len(res["VT5 only"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(2000)]
    b0 = np.array([pooled(*res["BASE (v4 + VT, 10)"], i, j) for i, j in boots])
    for k, (nat, syn) in res.items():
        bb = np.array([pooled(nat, syn, i, j) for i, j in boots]) - b0
        print(f"[{k:28s}] pooled {pooled(nat, syn):.4f} ({pooled(nat, syn) - pooled(*res['BASE (v4 + VT, 10)']):+.4f}, "
              f"5%={np.percentile(bb, 5):+.4f}, P>0={np.mean(bb > 0):.3f}) | natF1 {f1(nat):.4f} natDice {nat[:, 3].mean():.4f} "
              f"synF1 {f1(syn):.4f} synDice {syn[:, 3].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
