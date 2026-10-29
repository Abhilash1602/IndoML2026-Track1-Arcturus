"""S2 follow-up: does the in-domain boundary refiner (trained on validation train70, inputs =
v4 members + log-mel) add to the v6 recipe (v4 + V1 VT teachers + edge correction)?
Pre-registered combinations, scored once on hold30 with a paired bootstrap vs the v6 recipe:
  R    -- refiner proposals alone (+corr)
  C1   -- proposals with presence = mean(refiner presence, v6-recipe probability), refiner distances
  C2   -- v6-recipe Viterbi segmentation, edges moved to the refiner's regressed onset/offset
each with and without the gated edge correction.

Usage: python -m temp.eval_refiner_v6
"""
import json

import numpy as np

from src.config import Config
from src.edge_correction import correct_events, load_table
from src.refiner import decode_proposals, refine_segments
from temp.eval_dual_gated import S, V4
from temp.eval_holdout import FT, hold_logits
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from train import official_val_subset

TABLE = load_table("./checkpoints/edge_correction_v5.json")


def main():
    config = Config()
    split = json.load(open("./data/val_split_s0.json"))["hold30"]
    pred = dict(np.load(f"{S}/refiner_val_hold30_x0.npz"))
    v4p, v6p, refs, nat = [], [], [], []
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        ids = [str(x) for x in d["ids"]]
        keep = [k for k, n in enumerate(ids) if n in set(split[sub])]  # same order as the refiner run
        names = [ids[k] for k in keep]
        ds = official_val_subset(config, sub, "hold30")
        ds_order = [ds.dataset.rows[i]["segmentFileName"] for i in ds.indices]
        pos = {n: j for j, n in enumerate(ds_order)}
        vt = [hold_logits(cls, stem.format(P="VT") + "_teacher.pth", ov, {sub: ds})[sub] for cls, stem, ov in FT.values()]
        vt = [x[[pos[n] for n in names]] for x in vt]
        L4 = d["logits"][keep].astype(np.float32)
        v4p.append(1 / (1 + np.exp(-L4.mean(1))))
        v6p.append(1 / (1 + np.exp(-np.mean([L4[:, k] for k in range(len(V4))] + vt, axis=0))))
        refs += refs_of(d["mask"][keep].astype(np.float32))
        nat += [sub == "natural"] * len(keep)
    v6p = np.concatenate(v6p)
    nat = np.array(nat)
    assert len(v6p) == len(pred["presence"]) == len(refs)

    def corr(evs, probs):
        return [correct_events(e, p, TABLE) for e, p in zip(evs, probs)]

    R = [decode_proposals(p, a, b, 0.5) for p, a, b in zip(pred["presence"], pred["d_on"], pred["d_off"])]
    mixp = 0.5 * (pred["presence"] + v6p)
    C1 = [decode_proposals(p, a, b, 0.5) for p, a, b in zip(mixp, pred["d_on"], pred["d_off"])]
    C2 = [refine_segments(s, a, b) for s, a, b in zip(viterbi_events(v6p), pred["d_on"], pred["d_off"])]
    base = corr(viterbi_events(v6p), v6p)
    from src.edge_correction import edge_contrast
    gate = np.array([True if not (ev := viterbi_events(p[None])[0]) else np.mean([edge_contrast(p, x, y) for x, y in ev]) < 0.5
                     for p in v6p])
    Rc = corr(R, v6p)
    gR = [r if g else b_ for r, b_, g in zip(R, base, gate)]
    gRc = [r if g else b_ for r, b_, g in zip(Rc, base, gate)]
    print(f"gate: Natural clips routed to refiner {gate[nat].mean():.1%}, synthetic {gate[~nat].mean():.1%}", flush=True)
    cands = {"gated R": gR, "gated R +corr": gRc, "v6 recipe": base, "R": R, "R +corr": corr(R, v6p), "C1": C1, "C1 +corr": corr(C1, mixp),
             "C2": C2, "C2 +corr": corr(C2, v6p)}
    rows = {k: per_clip(refs, v) for k, v in cands.items()}
    comb = lambda r: f1_only(r) + r[:, 3].mean()
    rng = np.random.default_rng(0)
    ni, si = np.where(nat)[0], np.where(~nat)[0]
    boots = [np.concatenate([rng.choice(ni, len(ni)), rng.choice(si, len(si))]) for _ in range(2000)]
    b0 = np.array([comb(rows["v6 recipe"][b]) for b in boots])  # baseline
    for k, r in rows.items():
        d = np.array([comb(r[b]) for b in boots]) - b0
        print(f"HOLD30 [{k:10s}] combined={comb(r):.4f} vs v6 recipe {d.mean():+.4f} (5%={np.percentile(d, 5):+.4f}, "
              f"P>0={np.mean(d > 0):.3f}) | natF1={f1_only(r[nat]):.4f} natDice={r[nat, 3].mean():.4f} "
              f"synF1={f1_only(r[~nat]):.4f} synDice={r[~nat, 3].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
