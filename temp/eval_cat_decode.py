"""Phase C evaluation: category-aware event construction on hold30, 2-fold cross-fitted.

Any-noise probabilities = the v6 recipe (v4's 5 cached members + the 5 V1 VT teachers).
Category probabilities = mean of the Phase C models' 7 category channels (cat_*_last_teacher).
Only clips the sharpness gate calls Natural are changed; everything else keeps the v6 recipe.

  base -- decode 0.9802/0.9489 + v5 correction (the v6 recipe)
  (ii) -- per-clip dominant predicted category -> that category's decode params
          (chosen on fold A per category from a 4x4 grid, combined score), + v5 correction
  (i)  -- per-category Viterbi on each category's probability with its own params
          (chosen on A), events unioned and merged, + v5 correction
Also reports how often the predicted dominant category matches the true one.

Usage: python -m temp.eval_cat_decode cOwB cbS2
"""
import ast
import itertools
import json
import multiprocessing as mp
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import edge_correction as ec
from src.architecture import StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.dataset import CATEGORY_LIST
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.eval_dual_gated import S, V4
from temp.eval_holdout import FT, hold_logits
from temp.official_metric import macro_segment_dice, match_events
from train import official_val_subset

H, BASE = 0.01, (0.9802, 0.9489)
GRID = list(itertools.product([0.95, 0.98, 0.995, 0.998], [0.85, 0.95, 0.97, 0.98]))
TABLE = ec.load_table("./checkpoints/edge_correction_v5.json")
G = {}


def decode(p, ps, pe):
    return [(e["onset"], e["offset"]) for e in extract_events_viterbi(viterbi_decode(p, p_stay_silence=ps, p_stay_event=pe), H)]


def merge(ev):
    out = []
    for a, b in sorted(ev):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def events_for(i, mode, params):
    p = G["P"][i]
    if not G["gate"][i] or mode == "base":
        ev = decode(p, *BASE)
    elif mode == "ii":
        ev = decode(p, *params.get(G["dom"][i], BASE))
    else:  # "i": per-category decode, union
        ev = []
        for c in range(len(CATEGORY_LIST)):
            if G["C"][i][:, c].max() >= 0.5:
                ev += decode(G["C"][i][:, c], *params.get(c, BASE))
        ev = merge(ev) if ev else decode(p, *BASE)
    return ec.correct_events(ev, p, TABLE)


def row(args):
    i, mode, params = args
    ev = events_for(i, mode, params)
    tp, fp, fn = match_events(G["refs"][i], ev)
    return tp, fp, fn, macro_segment_dice([G["refs"][i]], [ev], hop_sec=H)


def score(pool, idx, mode, params):
    r = np.array(pool.map(row, [(i, mode, params) for i in idx], chunksize=8))
    nat = G["is_nat"][idx]
    f1 = lambda a: 2 * a[:, 0].sum() / max(2 * a[:, 0].sum() + a[:, 1].sum() + a[:, 2].sum(), 1)
    return f1(r) + r[:, 3].mean(), f1(r[nat]), r[nat, 3].mean(), f1(r[~nat])


@torch.no_grad()
def cat_probs(tag, ds):
    cls = StandaloneDashengSED if tag.startswith("cOw") else StandaloneBEATsSED
    config = Config()
    config.CAT_HEAD, config.OUTPUT_HEAD = True, "anycat"
    m = cls(config)
    m.load_state_dict(torch.load(f"./checkpoints/cat_{tag}_last_teacher.pth", map_location="cpu", weights_only=True))
    m = m.to(config.DEVICE).eval()
    out = []
    for sub in ("natural", "synthetic"):
        loader = DataLoader(ds[sub], batch_size=16, num_workers=config.NUM_WORKERS)
        out.append(np.concatenate([torch.sigmoid(m(b[0].to(config.DEVICE)))[:, :1000, 1:].float().cpu().numpy() for b in loader]))
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    tags = sys.argv[1:] or ["cOwB", "cbS2"]
    config = Config()
    hold = json.load(open("./data/val_split_s0.json"))["hold30"]
    ds = {sub: official_val_subset(config, sub, "hold30") for sub in ("natural", "synthetic")}
    meta = {r["segmentFileName"]: r for r in json.load(open("./data/official_validation_raw/validation/validationMetadata.json", encoding="utf-8"))}
    L, refs, is_nat, true_dom = [], [], [], []
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
        for name in order:
            v = meta[name].get("NoiseSubCategoryTimeStamp") or "[]"
            v = ast.literal_eval(v) if isinstance(v, str) else v
            dur = {}
            for e in v:
                dur[e["category"]] = dur.get(e["category"], 0) + float(e["end"]) - float(e["start"])
            true_dom.append(CATEGORY_LIST.index(max(dur, key=dur.get)) if dur and max(dur, key=dur.get) in CATEGORY_LIST else -1)
    G["P"] = 1 / (1 + np.exp(-np.concatenate(L)))
    G["refs"], G["is_nat"] = refs, np.array(is_nat)
    G["C"] = np.mean([cat_probs(t, ds) for t in tags], axis=0)
    G["gate"] = np.array([True if not (ev := decode(p, *BASE)) else np.mean([ec.edge_contrast(p, a, b) for a, b in ev]) < 0.5
                          for p in G["P"]])
    G["dom"] = [int((G["C"][i] * (G["P"][i][:, None] > 0.5)).sum(0).argmax()) for i in range(len(G["P"]))]
    td = np.array(true_dom)
    ok = G["is_nat"] & (td >= 0)
    print(f"dominant-category accuracy on hold30 Natural clips: {np.mean(np.array(G['dom'])[ok] == td[ok]):.3f} "
          f"(majority-class baseline {np.bincount(td[ok]).max() / ok.sum():.3f})", flush=True)

    rng = np.random.default_rng(1)
    folds = [[], []]
    for dom in (True, False):
        idx = np.where(G["is_nat"] == dom)[0]
        rng.shuffle(idx)
        folds[0] += list(idx[: len(idx) // 2]); folds[1] += list(idx[len(idx) // 2:])
    res = {k: [] for k in ("base", "ii", "i")}
    with mp.get_context("fork").Pool(48) as pool:
        for f in (0, 1):
            A, B = np.array(folds[f]), np.array(folds[1 - f])
            res["base"].append(score(pool, B, "base", {}))
            for mode in ("ii", "i"):
                params = {}
                for c in range(len(CATEGORY_LIST)):
                    sub = A[[G["gate"][i] and (G["dom"][i] == c if mode == "ii" else G["C"][i][:, c].max() >= 0.5) for i in A]]
                    if len(sub) >= 15:
                        params[c] = max(GRID, key=lambda g: score(pool, sub, mode, {**params, c: g})[0])
                res[mode].append(score(pool, B, mode, params))
                print(f"fold {f} ({mode}): params {{{', '.join(f'{CATEGORY_LIST[c]}: {v}' for c, v in params.items())}}}", flush=True)
    for k, v in res.items():
        v = np.array(v); d = v[:, 0] - np.array(res["base"])[:, 0]
        print(f"[{k:4s}] combined={v[:, 0].mean():.4f} (vs base per fold {d[0]:+.4f} / {d[1]:+.4f}, mean {d.mean():+.4f}) | "
              f"natF1={v[:, 1].mean():.4f} natDice={v[:, 2].mean():.4f} synF1={v[:, 3].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
