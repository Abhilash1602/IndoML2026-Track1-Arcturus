"""S1: snap Natural-looking predicted OFFSETS to the next VAD speech onset.

Measurement (temp/vad_boundary_analysis.py, train70): human breath offsets sit within 0.05 s
of the next speech onset 47.6% of the time (random-shift null 2.3%), all Natural events 38.0%
(null 4.4%). Onsets show no such alignment, so only offsets are touched.

Rule: for a predicted event whose edges look Natural (edge contrast < 0.5), take the first
Silero speech onset u in [offset - W, offset + W]; set offset = u - DELTA (if still > onset).
(W, DELTA, whether to keep the v5 edge correction, max event duration) are chosen on train70
Natural+synthetic clips using the v4 ensemble (never trained on validation); the chosen rule is
then scored ONCE on hold30 with the v6 recipe (v4 + V1 VT teachers + correction), with a
paired bootstrap against the unsnapped v6 recipe.

Usage: python -m temp.vad_snap_eval
"""
import itertools
import json
import multiprocessing as mp

import numpy as np

from src import edge_correction as ec
from src.config import Config
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.eval_dual_gated import S, V4
from temp.eval_holdout import FT, hold_logits
from temp.official_metric import macro_segment_dice, match_events
from temp.vad_boundary_analysis import speech_edges, vad_probs
from train import official_val_subset

H = 0.01
TABLE = ec.load_table("./checkpoints/edge_correction_v5.json")
GRID = [dict(W=w, D=d, corr=c, maxd=m) for w, d, c, m in
        itertools.product((0.1, 0.15, 0.2, 0.3), (0.0, 0.03, 0.05), (True, False), (2.0, 99.0))]
G = {}


def decode(p):
    return [(e["onset"], e["offset"]) for e in extract_events_viterbi(viterbi_decode(p, p_stay_silence=0.9802, p_stay_event=0.9489), H)]


def snap(ev, p, ups, rule):
    out = []
    for a, b in ev:
        if b - a < rule["maxd"] and ec.edge_contrast(p, a, b) < 0.5:
            cand = ups[(ups >= b - rule["W"]) & (ups <= b + rule["W"])]
            if len(cand):
                nb = float(cand[np.argmin(np.abs(cand - b))]) - rule["D"]
                if nb - a >= 0.02:
                    b = nb
        out.append((a, b))
    return out


def events_for(i, rule):
    p = G["P"][i]
    ev = decode(p)
    if rule is None:
        return ec.correct_events(ev, p, TABLE)
    if rule["corr"]:
        ev = ec.correct_events(ev, p, TABLE)
    return snap(ev, p, G["ups"][i], rule)


def row(args):
    i, rule = args
    ev = events_for(i, rule)
    tp, fp, fn = match_events(G["refs"][i], ev)
    return tp, fp, fn, macro_segment_dice([G["refs"][i]], [ev], hop_sec=H)


def table_rows(pool, idx, rule):
    return np.array(pool.map(row, [(i, rule) for i in idx], chunksize=8))


def f1(a):
    return 2 * a[:, 0].sum() / max(2 * a[:, 0].sum() + a[:, 1].sum() + a[:, 2].sum(), 1)


def comb(r):
    return f1(r) + r[:, 3].mean()


def load_part(part, with_vt):
    config = Config()
    vad = vad_probs()
    split = json.load(open("./data/val_split_s0.json"))[part]
    P, refs, ups, nat = [], [], [], []
    for sub in ("natural", "synthetic"):
        ds = official_val_subset(config, sub, part)
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        pos = {str(c): k for k, c in enumerate(d["ids"])}
        order = [ds.dataset.rows[i]["segmentFileName"] for i in ds.indices]
        assert set(order) == set(split[sub])
        sel = [pos[c] for c in order]
        members = [d["logits"].astype(np.float32)[sel, k] for k in range(len(V4))]
        if with_vt:
            for cls, stem, ov in FT.values():
                members.append(hold_logits(cls, stem.format(P="VT") + "_teacher.pth", ov, {sub: ds})[sub])
        P.append(1 / (1 + np.exp(-np.mean(members, axis=0))))
        refs += [[(e["onset"], e["offset"]) for e in extract_events_from_mask(m.astype(np.int32), H)] for m in d["mask"][sel]]
        ups += [speech_edges(vad[n])[0] for n in order]
        nat += [sub == "natural"] * len(order)
    return np.concatenate(P), refs, ups, np.array(nat)


def main():
    # ---- choose the rule on train70 with the v4 ensemble (never trained on validation) ----
    G["P"], G["refs"], G["ups"], nat = load_part("train70", with_vt=False)
    idx = np.arange(len(G["P"]))
    with mp.get_context("fork").Pool(48) as pool:
        base = table_rows(pool, idx, None)
        scores = [(comb(table_rows(pool, idx, r)), r) for r in GRID]
    best_score, rule = max(scores, key=lambda x: x[0])
    print(f"train70 (v4 ensemble): base combined {comb(base):.4f} natF1 {f1(base[nat]):.4f} | "
          f"best rule {rule} -> {best_score:.4f} ({best_score - comb(base):+.4f})", flush=True)
    for s, r in sorted(scores, key=lambda x: -x[0])[:5]:
        print(f"   {s:.4f} {r}", flush=True)

    # ---- score once on hold30 with the v6 recipe ----
    G["P"], G["refs"], G["ups"], nat = load_part("hold30", with_vt=True)
    idx = np.arange(len(G["P"]))
    with mp.get_context("fork").Pool(48) as pool:
        b = table_rows(pool, idx, None)
        s = table_rows(pool, idx, rule)
    rng = np.random.default_rng(0)
    ni, si = np.where(nat)[0], np.where(~nat)[0]
    d = []
    for _ in range(2000):
        bi = np.concatenate([rng.choice(ni, len(ni)), rng.choice(si, len(si))])
        d.append(comb(s[bi]) - comb(b[bi]))
    d = np.array(d)
    for name, r in (("v6 recipe", b), ("v6 recipe + VAD offset snap", s)):
        print(f"HOLD30 [{name:28s}] combined={comb(r):.4f} natF1={f1(r[nat]):.4f} natDice={r[nat, 3].mean():.4f} "
              f"synF1={f1(r[~nat]):.4f} synDice={r[~nat, 3].mean():.4f}", flush=True)
    print(f"HOLD30 delta {comb(s) - comb(b):+.4f} (5%={np.percentile(d, 5):+.4f}, P>0={np.mean(d > 0):.3f})", flush=True)
    json.dump({"rule": rule, "hold30_delta": float(comb(s) - comb(b)), "p": float(np.mean(d > 0))},
              open(f"{S}/vad_snap_decision.json", "w"), indent=1)


if __name__ == "__main__":
    main()
