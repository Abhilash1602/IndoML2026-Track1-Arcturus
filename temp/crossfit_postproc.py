"""E2 + E3: Natural-only onset/offset post-processing for the v6-style ensemble, estimated
honestly by 2-fold cross-fitting on hold30 (fit on half A, score on half B, then swap).

Ensemble = v4's 5 members (cached logits) + the 5 V1 (VT_*) EMA teachers, i.e. the v6
structure with train70-fine-tuned proxies for v6's VA members. Every choice is made
on the COMBINED score (Event F1 + Dice), so an F1 gain that costs Dice is rejected.
Only clips/events the sharpness gate calls Natural are touched; synthetic keeps v6's setup.

  base   -- v6 recipe: decode 0.9802/0.9489 + the v5 correction table
  E2     -- correction table refit on half A's predictions (median edge offsets per
            predicted-duration bin), max-duration cap chosen on A from {0.7, 1.0, 1.5}
  E3     -- decode (p_stay_silence, p_stay_event) for gated-Natural clips chosen on A
            from a 5x5 grid, with the v5 correction kept
  E2+E3  -- decode chosen on A, then the correction refit on A under that decode
Usage: python -m temp.crossfit_postproc
"""
import itertools
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from src import edge_correction as ec
from src.edge_correction import edge_contrast
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.eval_dual_gated import S, V4
from temp.eval_holdout import FT, hold_logits
from temp.official_metric import macro_segment_dice, match_events
from src.config import Config
from train import official_val_subset

H = 0.01
BASE_PS, BASE_PE = 0.9802, 0.9489
GRID = list(itertools.product([0.95, 0.98, 0.99, 0.995, 0.998], [0.85, 0.9, 0.95, 0.97, 0.98]))
G = {}


def decode(p, ps, pe):
    return [(e["onset"], e["offset"]) for e in extract_events_viterbi(viterbi_decode(p, p_stay_silence=ps, p_stay_event=pe), H)]


def clip_gate(p):
    ev = decode(p, BASE_PS, BASE_PE)
    return True if not ev else np.mean([edge_contrast(p, a, b) for a, b in ev]) < 0.5


def apply_corr(ev, p, table, max_dur):
    old = ec.MAX_DUR
    ec.MAX_DUR = max_dur
    try:
        return ec.correct_events(ev, p, table)
    finally:
        ec.MAX_DUR = old


def predict(i, nat_decode, table, max_dur):
    p = G["P"][i]
    ps, pe = nat_decode if G["gate"][i] else (BASE_PS, BASE_PE)
    ev = decode(p, ps, pe)
    return apply_corr(ev, p, table, max_dur) if table is not None else ev


def clip_row(args):
    i, nat_decode, table, max_dur = args
    ev = predict(i, nat_decode, table, max_dur)
    tp, fp, fn = match_events(G["refs"][i], ev)
    return tp, fp, fn, macro_segment_dice([G["refs"][i]], [ev], hop_sec=H)


def combined(pool, idx, nat_decode, table, max_dur):
    r = np.array(pool.map(clip_row, [(i, nat_decode, table, max_dur) for i in idx], chunksize=16))
    nat = np.array([G["is_nat"][i] for i in idx])
    f1 = lambda a: 2 * a[:, 0].sum() / max(2 * a[:, 0].sum() + a[:, 1].sum() + a[:, 2].sum(), 1)
    return f1(r) + r[:, 3].mean(), f1(r[nat]), r[nat, 3].mean(), f1(r[~nat]), r[~nat, 3].mean()


def fit_table_on(idx, nat_decode):
    refs = [G["refs"][i] for i in idx if G["gate"][i]]
    preds = [decode(G["P"][i], *nat_decode) for i in idx if G["gate"][i]]
    return ec.fit_table(refs, preds)


def main():
    config = Config()
    hold = json.load(open("./data/val_split_s0.json"))["hold30"]
    ds = {sub: official_val_subset(config, sub, "hold30") for sub in ("natural", "synthetic")}
    L, refs, is_nat = [], [], []
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        pos = {str(c): k for k, c in enumerate(d["ids"])}
        order = [ds[sub].dataset.rows[i]["segmentFileName"] for i in ds[sub].indices]
        assert set(order) == set(hold[sub])
        sel = [pos[c] for c in order]
        members = [d["logits"].astype(np.float32)[sel, k] for k in range(len(V4))]
        for m, (cls, stem, ov) in FT.items():
            members.append(hold_logits(cls, stem.format(P="VT") + "_teacher.pth", ov, {sub: ds[sub]})[sub])
        L.append(np.mean(members, axis=0))
        refs += [[(e["onset"], e["offset"]) for e in extract_events_from_mask(m.astype(np.int32), H)] for m in d["mask"][sel]]
        is_nat += [sub == "natural"] * len(sel)
    G["P"] = 1 / (1 + np.exp(-np.concatenate(L)))
    G["refs"], G["is_nat"] = refs, np.array(is_nat)
    G["gate"] = np.array([clip_gate(p) for p in G["P"]])
    torch.cuda.empty_cache()
    n = len(G["P"])
    print(f"hold30 clips {n}; gate routes {G['gate'][G['is_nat']].mean():.1%} of Natural and "
          f"{G['gate'][~G['is_nat']].mean():.1%} of synthetic clips to Natural processing", flush=True)

    rng = np.random.default_rng(1)
    folds = [[], []]
    for dom in (True, False):
        idx = np.where(G["is_nat"] == dom)[0]
        rng.shuffle(idx)
        folds[0] += list(idx[: len(idx) // 2])
        folds[1] += list(idx[len(idx) // 2:])
    v5_table = ec.load_table("./checkpoints/edge_correction_v5.json")
    base_dec = (BASE_PS, BASE_PE)
    out = {k: [] for k in ("base", "E2", "E3", "E2+E3")}
    with mp.get_context("fork").Pool(48) as pool:
        for f in (0, 1):
            A, B = folds[f], folds[1 - f]
            out["base"].append(combined(pool, B, base_dec, v5_table, 1.0))
            # E2: refit correction on A (base decode); cap chosen on A
            t2 = fit_table_on(A, base_dec)
            cap = max((0.7, 1.0, 1.5), key=lambda c: combined(pool, A, base_dec, t2, c)[0])
            out["E2"].append(combined(pool, B, base_dec, t2, cap))
            # E3: Natural decode chosen on A (v5 correction kept)
            dec = max(GRID, key=lambda g: combined(pool, A, g, v5_table, 1.0)[0])
            out["E3"].append(combined(pool, B, dec, v5_table, 1.0))
            # E2+E3: decode chosen on A, then correction refit on A under it
            t23 = fit_table_on(A, dec)
            cap23 = max((0.7, 1.0, 1.5), key=lambda c: combined(pool, A, dec, t23, c)[0])
            out["E2+E3"].append(combined(pool, B, dec, t23, cap23))
            print(f"fold {f}: E2 cap={cap} | E3 natural decode={dec} | E2+E3 cap={cap23}", flush=True)
    for k, v in out.items():
        v = np.array(v)
        m = v.mean(0)
        d = (v[:, 0] - np.array(out["base"])[:, 0])
        print(f"[{k:6s}] combined={m[0]:.4f} (vs base, per fold {d[0]:+.4f} / {d[1]:+.4f}, mean {d.mean():+.4f}) | "
              f"natF1={m[1]:.4f} natDice={m[2]:.4f} synF1={m[3]:.4f} synDice={m[4]:.4f}", flush=True)


if __name__ == "__main__":
    main()
