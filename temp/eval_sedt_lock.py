"""Gate G2: the Dice-locked SEDT scored ONCE on the full official validation set (2,501 clips;
neither the v4 members nor SEDT ever trained on it).

Base = v5 pipeline on the v4 ensemble (Viterbi 0.9802/0.9489 + gated edge correction).
Candidate = base events on Natural-gated clips re-cut at SEDT event edges (lock_partition),
with the checkpoint and c_min fixed beforehand on Gold held-out:
  PRIMARY  -- the seed with the best Gold held-out locked gain, its stored c_min;
  secondary (reported, not decided on) -- the other seed, and both seeds' cuts pooled.
Dice must be identical (asserted per clip). Domain-stratified paired bootstrap vs base.
Pass: pooled >= +0.006, P(>0) >= 0.95, Natural Dice identical, synthetic unchanged.

Usage: python -m temp.eval_sedt_lock
"""
import glob
import json

import numpy as np
import torch

from src.edge_correction import correct_events, edge_contrast, load_table
from src.event_lock import frames_equal, lock_partition
from src.sedt import SEDT, select_events
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S
from temp.train_sedt import predict_all

DEVICE = "cuda:0"
TABLE = load_table("./checkpoints/edge_correction_v5.json")


def load_val():
    out = {}
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        ids = [str(x) for x in d["ids"]]
        assert ids == json.load(open(f"{S}/sedt/official_{sub}_ids.json")), sub
        P = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
        vit = viterbi_events(P)
        base = [correct_events(e, p, TABLE) for e, p in zip(vit, P)]
        gate = [True if not e else np.mean([edge_contrast(p, a, b) for a, b in e]) < 0.5 for e, p in zip(vit, P)]
        tok = torch.from_numpy(np.load(f"{S}/sedt/official_{sub}_tok.npy")).to(DEVICE)
        mel = torch.from_numpy(np.load(f"{S}/sedt/official_{sub}_mel.npy")).to(DEVICE)
        out[sub] = dict(refs=refs_of(d["mask"].astype(np.float32)), base=base, gate=gate, tok=tok, mel=mel)
    return out


def main():
    V = load_val()
    cks = {}
    for path in sorted(p for p in glob.glob("./checkpoints/sedt_gold*_s*.pth") if not p.endswith("_alone.pth")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = SEDT(**ck.get("kwargs", {})).to(DEVICE)
        model.load_state_dict(ck["state"])
        cks[path] = (ck, {sub: predict_all(model, v["tok"], v["mel"]) for sub, v in V.items()})
        print(f"{path}: epoch {ck['epoch']} c_min {ck['c_min']} Gold held-out gain {ck['gain']:+.4f} "
              f"standalone F1 {ck['standalone_F1']:.4f}", flush=True)
    primary = max(cks, key=lambda p: cks[p][0]["gain"])

    def locked(preds_list, c_min):
        res = {}
        for sub, v in V.items():
            evs = []
            for i, (ev, g) in enumerate(zip(v["base"], v["gate"])):
                if not g:
                    evs.append(ev)
                    continue
                sel = [(a, b) for preds in preds_list for a, b, c in preds[sub][i] if c >= c_min]
                new = lock_partition(ev, sel)
                assert frames_equal(new, ev)
                evs.append(new)
            res[sub] = evs
        return res

    cands = {"base (v5 pipeline)": {sub: v["base"] for sub, v in V.items()}}
    cands[f"PRIMARY {primary}"] = locked([cks[primary][1]], cks[primary][0]["c_min"])
    for p in cks:
        if p != primary:
            cands[f"secondary {p}"] = locked([cks[p][1]], cks[p][0]["c_min"])
    if len(cks) > 1:
        cands["secondary pooled seeds"] = locked([c[1] for c in cks.values()], cks[primary][0]["c_min"])
    for p, (ck, preds) in cks.items():  # standalone SEDT (diagnostic)
        for sub, v in V.items():
            r = per_clip(v["refs"], [select_events(e, 0.5) for e in preds[sub]])
            print(f"standalone SEDT {p} on validation {sub}: F1 {f1_only(r):.4f} Dice {r[:, 3].mean():.4f}", flush=True)

    rows = {k: {sub: per_clip(V[sub]["refs"], c[sub]) for sub in V} for k, c in cands.items()}
    comb = lambda n, s: f1_only(np.r_[n, s]) + np.r_[n, s][:, 3].mean()
    b = rows["base (v5 pipeline)"]
    rng = np.random.default_rng(0)
    nn_, ns = len(b["natural"]), len(b["synthetic"])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(2000)]
    c0 = np.array([comb(b["natural"][i], b["synthetic"][j]) for i, j in boots])
    for k, r in rows.items():
        assert np.array_equal(r["natural"][:, 3], b["natural"][:, 3]), "Dice changed"
        d = np.array([comb(r["natural"][i], r["synthetic"][j]) for i, j in boots]) - c0
        print(f"G2 [{k:44s}] combined {comb(r['natural'], r['synthetic']):.4f} ({d.mean():+.4f}, 5%={np.percentile(d, 5):+.4f}, "
              f"P>0={np.mean(d > 0):.3f}) | natF1 {f1_only(r['natural']):.4f} natDice {r['natural'][:, 3].mean():.4f} | "
              f"syn {f1_only(r['synthetic']):.4f}/{r['synthetic'][:, 3].mean():.4f} | "
              f"events nat {sum(map(len, cands[k]['natural']))}", flush=True)


if __name__ == "__main__":
    main()
