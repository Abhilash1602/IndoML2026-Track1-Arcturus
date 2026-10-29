"""Trunk-level test of the two-convention hypothesis: Natural-specialist models
(v4's Dasheng/BEATs members fine-tuned with NO synthetic data, SYNTH_FRAC=0) serve the
clips the v4 edge-contrast gate calls Natural; every other clip keeps plain v4.

Pre-registered routes for gated-Natural clips (specialists = EMA-teacher checkpoints):
  R1 -- atst_stage2 (v4's only member not specialised) + the 4 specialists
  R2 -- v4's 5 members + the 4 specialists
each also with the v5 gated edge correction, and with true-domain (ORACLE) routing as
an upper bound. Paired clip bootstrap vs v4 and vs v5 on the official validation set.

Usage: python -m temp.eval_specialist_routed
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.architecture import StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from temp.eval_dual_gated import S, V4, VAL_ROOT, events, f1, pooled, refs_of, rows
from src.edge_correction import edge_contrast

SPECIALISTS = {  # tag -> (class, checkpoint)
    "NS_owB": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceNS_owB_teacher.pth"),
    "NS_owBt": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceNS_owBt_teacher.pth"),
    "NS_bA": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceNS_bA_teacher.pth"),
    "NS_bS2": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceNS_bS2_teacher.pth"),
}


@torch.no_grad()
def logits_of(cls, path):
    config = Config()
    model = cls(config)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = model.to(config.DEVICE).eval()
    out = {}
    for sub in ("natural", "synthetic"):
        loader = DataLoader(OfficialValidationDataset(VAL_ROOT, config, subset=sub), batch_size=16,
                            num_workers=config.NUM_WORKERS)
        out[sub] = np.concatenate([model(a.to(config.DEVICE)).float()[:, :1000].cpu().numpy() for a, _, _ in loader])
    del model
    torch.cuda.empty_cache()
    return out


def main():
    base, gate = {}, {}
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        L = d["logits"].astype(np.float32)
        base[sub] = (L, refs_of(d["mask"]))
        prob = 1 / (1 + np.exp(-L.mean(1)))
        gate[sub] = np.array([True if not (ev := events(p)) else np.mean([edge_contrast(p, a, b) for a, b in ev]) < 0.5
                              for p in prob])
    spec = {k: logits_of(*v) for k, v in SPECIALISTS.items()}
    for k, x in spec.items():
        r = rows(base["natural"][1], 1 / (1 + np.exp(-x["natural"])))
        rs = rows(base["synthetic"][1], 1 / (1 + np.exp(-x["synthetic"])))
        print(f"  specialist {k} alone: natural F1={f1(r):.4f} Dice={r[:, 3].mean():.4f} | "
              f"synthetic F1={f1(rs):.4f} Dice={rs[:, 3].mean():.4f}", flush=True)

    def score(route=None, oracle=False, correct=False):
        res = {}
        for sub in ("natural", "synthetic"):
            L, refs = base[sub]
            v4 = L.mean(1)
            if route is None:
                logit = v4
            else:
                nat_members = [L[:, V4.index("atst_stage2")]] if route == "R1" else [L[:, k] for k in range(len(V4))]
                nat_members += [spec[k][sub] for k in SPECIALISTS]
                routed = np.mean(nat_members, axis=0)
                use = np.full(len(L), sub == "natural") if oracle else gate[sub]
                logit = np.where(use[:, None], routed, v4)
            res[sub] = rows(refs, 1 / (1 + np.exp(-logit)), correct)
        return res["natural"], res["synthetic"]

    results = {"v4": score(), "v5": score(correct=True)}
    for r in ("R1", "R2"):
        results[r] = score(r)
        results[f"{r} +corr"] = score(r, correct=True)
        results[f"{r} ORACLE"] = score(r, oracle=True)
    rng = np.random.default_rng(0)
    nn_, ns = len(results["v4"][0]), len(results["v4"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(1000)]
    b4 = np.array([pooled(*results["v4"], i, j) for i, j in boots])
    b5 = np.array([pooled(*results["v5"], i, j) for i, j in boots])
    for k, (nat, syn) in results.items():
        bb = np.array([pooled(nat, syn, i, j) for i, j in boots])
        print(f"[{k:10s}] pooled={pooled(nat, syn):.4f} | vs v4 {pooled(nat, syn) - pooled(*results['v4']):+.4f} "
              f"P>0={np.mean(bb > b4):.3f} | vs v5 {pooled(nat, syn) - pooled(*results['v5']):+.4f} "
              f"P>0={np.mean(bb > b5):.3f} | natF1={f1(nat):.4f} natDice={nat[:, 3].mean():.4f} "
              f"synF1={f1(syn):.4f} synDice={syn[:, 3].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
