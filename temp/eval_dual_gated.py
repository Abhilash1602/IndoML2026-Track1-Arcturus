"""Evaluate two-convention (dual-head) checkpoints inside the v4 ensemble on the
official validation set, with deployable routing:

  gate: a clip is Natural-style if the v4 ensemble's decoded events have mean edge
        contrast < 0.5 (src/edge_correction.edge_contrast; the threshold is the
        a-priori midpoint from our own data, never fitted on validation; clips with no
        predicted event count as Natural). Validation domain flags are used only to
        report the gate's accuracy and the oracle-routing upper bound.

Per dual checkpoint (student/teacher of each DUAL_TAG given on the command line):
  add6   -- v4's 5 members + the routed dual model (logit mean)
  swap   -- v4 with the routed dual model replacing the member it was warm-started from
  oracle -- add6 with true-domain routing (upper bound, not deployable)
each also with the v5 gated edge correction on top. Paired clip bootstrap vs v4.

Usage: python -m temp.eval_dual_gated dOwB:dasheng_owB bA:beats_owA ...  (tag:replaced_member)
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.architecture import StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.edge_correction import correct_events, edge_contrast, load_table
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, match_events

S = "./work"
V4 = ("atst_stage2", "dasheng_owB", "beats_s2", "dasheng_owB_t", "beats_owA")  # order in refiner_*.npz
VAL_ROOT = "./data/official_validation_raw/validation"
PS, PE, H = 0.9802, 0.9489, 0.01
TABLE = load_table("./checkpoints/edge_correction_v5.json")


def events(prob):
    return [(e["onset"], e["offset"]) for e in extract_events_viterbi(
        viterbi_decode(prob, p_stay_silence=PS, p_stay_event=PE), H)]


def refs_of(mask):
    return [[(e["onset"], e["offset"]) for e in extract_events_from_mask(m.astype(np.int32), H)] for m in mask]


def rows(refs, probs, correct=False):
    out = []
    for g, p in zip(refs, probs):
        ev = events(p)
        if correct:
            ev = correct_events(ev, p, TABLE)
        tp, fp, fn = match_events(g, ev)
        out.append((tp, fp, fn, macro_segment_dice([g], [ev], hop_sec=H)))
    return np.array(out, float)


def f1(a):
    return 2 * a[:, 0].sum() / max(2 * a[:, 0].sum() + a[:, 1].sum() + a[:, 2].sum(), 1)


def pooled(nat, syn, i=None, j=None):
    x = np.concatenate([nat if i is None else nat[i], syn if j is None else syn[j]])
    return f1(x) + x[:, 3].mean()


@torch.no_grad()
def dual_logits(path, kind):
    config = Config()
    config.DUAL_HEAD, config.OUTPUT_HEAD = True, "both"
    model = {"dasheng": StandaloneDashengSED, "beats": StandaloneBEATsSED}[kind](config)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = model.to(config.DEVICE).eval()
    out = {}
    for sub in ("natural", "synthetic"):
        loader = DataLoader(OfficialValidationDataset(VAL_ROOT, config, subset=sub), batch_size=16,
                            num_workers=config.NUM_WORKERS)
        out[sub] = np.concatenate([model(a.to(config.DEVICE)).float()[:, :1000].cpu().numpy() for a, _, _ in loader])
    del model
    torch.cuda.empty_cache()
    return out  # sub -> (N, 1000, 2)


def main():
    specs = [a.split(":") for a in sys.argv[1:]]
    base = {}
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        base[sub] = (d["logits"].astype(np.float32), refs_of(d["mask"]))
    gate = {}
    for sub in ("natural", "synthetic"):
        L, _ = base[sub]
        prob = 1 / (1 + np.exp(-L.mean(1)))
        g = []
        for p in prob:
            ev = events(p)
            g.append(True if not ev else np.mean([edge_contrast(p, a, b) for a, b in ev]) < 0.5)
        gate[sub] = np.array(g)
    print(f"gate accuracy: natural clips routed natural {gate['natural'].mean():.1%}, "
          f"synthetic clips routed synthetic {(~gate['synthetic']).mean():.1%}", flush=True)

    def score(extra=None, replace=None, route="gate", correct=False):
        res = {}
        for sub in ("natural", "synthetic"):
            L, refs = base[sub]
            members = [L[:, k] for k, n in enumerate(V4) if n != replace]
            if extra is not None:
                x = extra[sub]
                use_nat = gate[sub] if route == "gate" else np.full(len(x), sub == "natural")
                members.append(np.where(use_nat[:, None], x[..., 0], x[..., 1]))
            prob = 1 / (1 + np.exp(-np.mean(members, axis=0)))
            res[sub] = rows(refs, prob, correct)
        return res["natural"], res["synthetic"]

    results = {"v4": score(), "v5 (v4 + correction)": score(correct=True)}
    for tag, replaced in specs:
        kind = "dasheng" if replaced.startswith("dasheng") else "beats"
        for which in ("", "_teacher"):
            path = f"./checkpoints/dual_{tag}{which}.pth"
            if not os.path.exists(path):
                continue
            x = dual_logits(path, kind)
            name = f"{tag}{which or '_student'}"
            results[f"{name} add6"] = score(x)
            results[f"{name} add6 +corr"] = score(x, correct=True)
            results[f"{name} swap"] = score(x, replace=replaced)
            results[f"{name} swap +corr"] = score(x, replace=replaced, correct=True)
            results[f"{name} add6 ORACLE"] = score(x, route="oracle")
            for sub in ("natural", "synthetic"):
                for h, hn in ((0, "nat-head"), (1, "syn-head")):
                    r = rows(base[sub][1], 1 / (1 + np.exp(-x[sub][..., h])))
                    print(f"  {name} alone {hn} on {sub}: F1={f1(r):.4f} Dice={r[:, 3].mean():.4f}", flush=True)
    rng = np.random.default_rng(0)
    nn_, ns = len(results["v4"][0]), len(results["v4"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(1000)]
    b0 = np.array([pooled(*results["v4"], i, j) for i, j in boots])
    for k, (nat, syn) in results.items():
        d = np.array([pooled(nat, syn, i, j) for i, j in boots]) - b0
        print(f"[{k:28s}] pooled={pooled(nat, syn):.4f} d={pooled(nat, syn) - pooled(*results['v4']):+.4f} "
              f"5%={np.percentile(d, 5):+.4f} P>0={np.mean(d > 0):.3f} | natF1={f1(nat):.4f} natDice={nat[:, 3].mean():.4f} "
              f"synF1={f1(syn):.4f} synDice={syn[:, 3].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
