"""Plan Phase V1 decision: does adding validation train70 to fine-tuning help, measured
ONLY on hold30 (data/val_split_s0.json), which no model has trained on?

Pre-registered candidates (decode fixed at 0.9802/0.9489, logit-mean ensembles):
  v4, v5 (= v4 + gated edge correction)
  FT5        -- the 5 v4 members fine-tuned with train70 (EMA teachers), no correction
  FT5 +corr  -- same + correction
  UNION      -- v4 + FT5 (10 members), and UNION +corr
Students are reported as a secondary check. Paired clip bootstrap vs v5 on hold30.
Gate to commit (plan V1): >= +0.006 pooled vs v5, P(>0) >= 0.95, gain visible in Natural F1.

Usage: python -m temp.eval_holdout
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.architecture import DualBranchSED, StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from temp.eval_dual_gated import S, V4, f1, pooled, refs_of, rows
from train import official_val_subset

FT = {  # v4 member -> (class, fine-tuned checkpoint stem with {P} = tag prefix, config overrides)
    "atst_stage2": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2_owbce{P}_atst", dict(USE_ATST_FRAME=True)),
    "dasheng_owB": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbce{P}_owB", {}),
    "beats_s2": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbce{P}_bS2", {}),
    "dasheng_owB_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbce{P}_owBt", {}),
    "beats_owA": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbce{P}_bA", {}),
}
# Model sets to compare, "PREFIX:suffix" (suffix "" = best student, "_teacher", "_last", "_last_teacher").
KINDS = [k.split(":") for k in os.environ.get("HOLD_KINDS", "VT:_teacher,VT:").split(",")]
DECISION_OUT = os.environ.get("HOLD_DECISION", "v1_decision.json")


@torch.no_grad()
def hold_logits(cls, path, overrides, ds):
    config = Config()
    for k, v in overrides.items():
        setattr(config, k, v)
    model = cls(config)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = model.to(config.DEVICE).eval()
    out = {}
    for sub, d in ds.items():
        loader = DataLoader(d, batch_size=16, num_workers=config.NUM_WORKERS)
        out[sub] = np.concatenate([model(a.to(config.DEVICE)).float()[:, :1000].cpu().numpy() for a, _, _ in loader])
    del model
    torch.cuda.empty_cache()
    return out


def main():
    hold = json.load(open("./data/val_split_s0.json"))["hold30"]
    config = Config()
    ds = {sub: official_val_subset(config, sub, "hold30") for sub in ("natural", "synthetic")}
    base = {}
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        ids = [str(x) for x in d["ids"]]
        order = [ds[sub].dataset.rows[i]["segmentFileName"] for i in ds[sub].indices]
        assert set(order) == set(hold[sub])
        pos = {c: k for k, c in enumerate(ids)}
        sel = [pos[c] for c in order]
        base[sub] = (d["logits"].astype(np.float32)[sel], refs_of(d["mask"][sel]))
    ft = {}
    for pfx, suffix in KINDS:
        which = f"{pfx}{suffix or '_student'}"
        for m, (cls, stem, ov) in FT.items():
            path = stem.format(P=pfx) + suffix + ".pth"
            if os.path.exists(path):
                ft[(m, which)] = hold_logits(cls, path, ov, ds)
            else:
                print(f"missing {path}", flush=True)

    def score(members, correct=False):
        res = {}
        for sub in ("natural", "synthetic"):
            prob = 1 / (1 + np.exp(-np.mean([mb[sub] for mb in members], axis=0)))
            res[sub] = rows(base[sub][1], prob, correct)
        return res["natural"], res["synthetic"]

    v4m = [{sub: base[sub][0][:, k] for sub in base} for k in range(len(V4))]
    results = {"v4": score(v4m), "v5": score(v4m, True)}
    for which in [f"{p}{s or '_student'}" for p, s in KINDS]:
        fm = [ft[(m, which)] for m in V4 if (m, which) in ft]
        if len(fm) < len(V4):
            print(f"{which}: only {len(fm)} of {len(V4)} fine-tuned members available", flush=True)
        if not fm:
            continue
        results[f"FT{len(fm)}_{which}"] = score(fm)
        results[f"FT{len(fm)}_{which} +corr"] = score(fm, True)
        results[f"UNION_{which}"] = score(v4m + fm)
        results[f"UNION_{which} +corr"] = score(v4m + fm, True)
        for m in V4:
            if (m, which) in ft:
                for sub in ("natural", "synthetic"):
                    a = rows(base[sub][1], 1 / (1 + np.exp(-ft[(m, which)][sub])))
                    b = rows(base[sub][1], 1 / (1 + np.exp(-base[sub][0][:, V4.index(m)])))
                    print(f"  {m}{which} on hold30 {sub}: F1 {f1(b):.4f} -> {f1(a):.4f}, Dice {b[:, 3].mean():.4f} -> {a[:, 3].mean():.4f}",
                          flush=True)
    rng = np.random.default_rng(0)
    nn_, ns = len(results["v4"][0]), len(results["v4"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(2000)]
    b5 = np.array([pooled(*results["v5"], i, j) for i, j in boots])
    summary = {}
    for k, (nat, syn) in results.items():
        bb = np.array([pooled(nat, syn, i, j) for i, j in boots]) - b5
        summary[k] = dict(delta=float(pooled(nat, syn) - pooled(*results["v5"])), p=float(np.mean(bb > 0)),
                          nat_f1=float(f1(nat)), syn_f1=float(f1(syn)), syn_dice=float(syn[:, 3].mean()))
        print(f"[{k:30s}] pooled={pooled(nat, syn):.4f} vs v5 {pooled(nat, syn) - pooled(*results['v5']):+.4f} "
              f"(5%={np.percentile(bb, 5):+.4f}, P>0={np.mean(bb > 0):.3f}) | natF1={f1(nat):.4f} natDice={nat[:, 3].mean():.4f} "
              f"synF1={f1(syn):.4f} synDice={syn[:, 3].mean():.4f}", flush=True)
    # Pre-registered decision (plan V1): EMA-teacher candidates only; gate = +0.006 pooled vs v5,
    # P(>0) >= 0.95 and a Natural F1 gain; among passing candidates take the highest pooled.
    v5n = summary["v5"]["nat_f1"]
    passing = {k: v for k, v in summary.items() if k.endswith(("teacher", "teacher +corr")) and v["delta"] >= 0.006
               and v["p"] >= 0.95 and v["nat_f1"] > v5n}
    choice = max(passing, key=lambda k: passing[k]["delta"]) if passing else None
    decision = {"commit": choice is not None, "choice": choice,
                "union": bool(choice and choice.startswith("UNION")), "correct": bool(choice and choice.endswith("+corr")),
                "summary": summary}
    json.dump(decision, open(os.path.join(S, DECISION_OUT), "w"), indent=1)
    print(f"\nDECISION: {'COMMIT -> ' + choice if choice else 'NO COMMIT (gate not met), v5 stays'}", flush=True)


if __name__ == "__main__":
    main()
