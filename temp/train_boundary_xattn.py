"""Lane A (Oct 11): train/evaluate the event-query cross-attention boundary head
(src/boundary_xattn.py) on clips the 5 v4 feature members never saw.

Labelled pool: Gold held-out (2,223, Natural) + official validation train70 (Natural + Synthetic).
Gate set: hold30 (scored once, with the configuration chosen by 5-fold CV inside the pool).
Proposals = the v4 ensemble's Viterbi + gated-correction events (exactly the base decode), plus
jittered copies during training. Targets per proposal, from the reference with the largest
overlap: onset/offset bins (+/-0.3 s, Gaussian-smoothed by the official tolerance
max(0.2*dur, 0.05)), match (both edges within tolerance), pure FP (no overlap).

Decision rule (Dice-safe): for base events shorter than SHORT s, an edge moves to the bin c
(|c| <= CAP bins) maximising the probability mass within the tolerance when that mass beats
the mass at the current position by more than delta. Optional P(FP) drop with the 0.70
confidence guard.

Usage: python -m temp.train_boundary_xattn cv <config>      # 5-fold OOF on the pool
       python -m temp.train_boundary_xattn final <config>   # train on the pool, score hold30 once
configs: full | mean_member | concat | no_global
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from src.boundary_xattn import HALF, BoundaryXAttn
from src.edge_correction import correct_events, load_table
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events

W = "./work"
OUT = f"{W}/xattn"
TABLE = load_table("./checkpoints/edge_correction_v5.json")
CONFIGS = {"full": {}, "mean_member": dict(member="mean"), "concat": dict(fusion="concat"),
           "no_global": dict(use_global=False)}
SHORT, CAP = 1.0, 15
DEV = os.environ.get("VAANI_DEVICE", "cuda:0")
EPOCHS = int(os.environ.get("XA_EPOCHS", "15"))


def load_sets():
    split = json.load(open("./data/val_split_s0.json"))
    sets = {}
    for name in ("gold_heldout", "official_natural", "official_synthetic"):
        d = np.load(f"{W}/refiner_{name}.npz")
        ids = [str(x) for x in d["ids"]]
        emb = np.load(f"{W}/verifier/{name}_emb.npy", mmap_mode="r")
        prob = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
        base = [correct_events(e, p, TABLE) for e, p in zip(viterbi_events(prob), prob)]
        part = {"gold_heldout": None, "official_natural": "natural", "official_synthetic": "synthetic"}[name]
        for i, cid in enumerate(ids):
            tag = "pool" if part is None or cid in set(split["train70"][part]) else "hold"
            sets.setdefault(tag, []).append(dict(set=name, i=i, dom="syn" if part == "synthetic" else "nat",
                                                 base=base[i], prob=prob[i]))
        sets.setdefault("_arr", {})[name] = (d["logits"], d["mel"], emb, refs_of(d["mask"]))
    return sets


def batch_inputs(clips, arr):
    lg = torch.tensor(np.stack([arr[c["set"]][0][c["i"]] for c in clips]).astype(np.float32))
    mel = torch.tensor(np.stack([arr[c["set"]][1][c["i"]] for c in clips]).astype(np.float32))
    emb = torch.tensor(np.stack([np.asarray(arr[c["set"]][2][c["i"]]) for c in clips]).astype(np.float32))
    return mel.to(DEV), lg.to(DEV), emb.to(DEV)


def targets(props, refs):
    """props list of (on, off) s; refs list of (on, off) s -> per-proposal target dict."""
    out = []
    for on, off in props:
        best, bo = None, 0.0
        for r in refs:
            ov = min(off, r[1]) - max(on, r[0])
            if ov > bo:
                best, bo = r, ov
        if best is None:
            out.append(dict(fp=1.0, match=0.0, on=None, off=None, tol=None))
            continue
        tol = max(0.2 * (best[1] - best[0]), 0.05)
        out.append(dict(fp=0.0, match=float(abs(on - best[0]) <= tol and abs(off - best[1]) <= tol),
                        on=(best[0] - on) * 100, off=(best[1] - off) * 100, tol=tol * 100))
    return out


def soft_bins(delta, tol):
    b = torch.arange(-HALF, HALF + 1).float()
    s = max(1.0, tol / 2)
    p = torch.exp(-0.5 * ((b - delta) / s) ** 2)
    return p / p.sum()


def make_batch(clips, arr, jitter, rng):
    props, tg = [], []
    for k, c in enumerate(clips):
        refs = arr[c["set"]][3][c["i"]]
        ev = list(c["base"])
        if jitter:
            for on, off in c["base"]:
                for _ in range(2):
                    a, b = on + rng.normal(0, 0.06), off + rng.normal(0, 0.06)
                    if b - a > 0.05:
                        ev.append((max(0.0, a), min(10.0, b)))
        for (on, off), t in zip(ev, targets(ev, refs)):
            props.append((k, on * 100, off * 100))
            tg.append(t)
    return torch.tensor(props, dtype=torch.float32, device=DEV), tg


def loss_fn(out, tg):
    fp = torch.tensor([t["fp"] for t in tg], device=DEV)
    match = torch.tensor([t["match"] for t in tg], device=DEV)
    loss = F.binary_cross_entropy_with_logits(out["fp"], fp) + F.binary_cross_entropy_with_logits(out["match"], match)
    for k in ("on", "off"):
        idx = [j for j, t in enumerate(tg) if t[k] is not None and abs(t[k]) <= HALF]
        if idx:
            tgt = torch.stack([soft_bins(tg[j][k], tg[j]["tol"]) for j in idx]).to(DEV)
            logit = out[k][idx]
            tgt = tgt * (logit > -1e3)  # bins outside the clip carry no target mass
            tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-8)
            loss = loss - (tgt * logit.log_softmax(-1)).sum(-1).mean()
    return loss


def train(clips, arr, cfg, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BoundaryXAttn(**CONFIGS[cfg]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    steps = EPOCHS * (len(clips) // 16)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(clips))
        losses = []
        for b in range(len(clips) // 16):
            cb = [clips[j] for j in order[b * 16:(b + 1) * 16]]
            props, tg = make_batch(cb, arr, True, rng)
            if not len(props):
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(*batch_inputs(cb, arr), props)
            loss = loss_fn({k: v.float() for k, v in out.items()}, tg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"  [{cfg}] epoch {ep + 1}/{EPOCHS} loss {np.mean(losses):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, clips, arr):
    """Per clip: list of (on_probs (61,), off_probs (61,), p_match, p_fp) for its base events."""
    model.eval()
    res = []
    for b in range(0, len(clips), 32):
        cb = clips[b:b + 32]
        props, _ = make_batch(cb, arr, False, None)
        per = [[] for _ in cb]
        if len(props):
            out = model(*batch_inputs(cb, arr), props)
            pon, poff = out["on"].float().softmax(-1).cpu().numpy(), out["off"].float().softmax(-1).cpu().numpy()
            pm, pf = torch.sigmoid(out["match"]).cpu().numpy(), torch.sigmoid(out["fp"]).cpu().numpy()
            for j, k in enumerate(props[:, 0].long().cpu().numpy()):
                per[k].append((pon[j], poff[j], float(pm[j]), float(pf[j])))
        res.extend(per)
    return res


def apply(clips, preds, delta, fp_tau=None, guard=0.70):
    out = []
    for c, pr in zip(clips, preds):
        ev = []
        for (on, off), (pon, poff, pm, pf) in zip(c["base"], pr):
            if fp_tau is not None and pf > fp_tau:
                a, b = int(on * 100), max(int(off * 100), int(on * 100) + 1)
                if c["prob"][a:b].mean() < guard:
                    continue
            if off - on < SHORT:
                tol = int(round(max(0.2 * (off - on), 0.05) * 100))
                new = []
                for p, x in ((pon, on), (poff, off)):
                    mass = np.convolve(p, np.ones(2 * tol + 1), mode="same")
                    cand = np.arange(HALF - CAP, HALF + CAP + 1)
                    c_best = cand[np.argmax(mass[cand])]
                    new.append(x + (c_best - HALF) / 100 if mass[c_best] - mass[HALF] > delta else x)
                if new[1] - new[0] > 0.03:
                    on, off = new
            ev.append((on, off))
        out.append(ev)
    return out


def pooled(clips, events, arr):
    rows = per_clip([arr[c["set"]][3][c["i"]] for c in clips], events)
    nat = np.array([c["dom"] == "nat" for c in clips])
    return dict(pooled=f1_only(rows) + rows[:, 3].mean(), natF1=f1_only(rows[nat]), natDice=rows[nat, 3].mean(),
                synF1=f1_only(rows[~nat]) if (~nat).any() else float("nan"))


def sweep(clips, preds, arr, tag):
    base = pooled(clips, [c["base"] for c in clips], arr)
    print(f"[{tag}] base: " + " ".join(f"{k} {v:.4f}" for k, v in base.items()), flush=True)
    best = (0.0, None)
    for fp_tau in (None, 0.7, 0.8, 0.9):
        for delta in (0.0, 0.05, 0.1, 0.2, 0.3):
            r = pooled(clips, apply(clips, preds, delta, fp_tau), arr)
            d = r["pooled"] - base["pooled"]
            print(f"[{tag}] delta {delta} fp_tau {fp_tau}: {d:+.4f} | natF1 {r['natF1']:.4f} natDice {r['natDice']:.4f} "
                  f"synF1 {r['synF1']:.4f}", flush=True)
            if d > best[0]:
                best = (d, (delta, fp_tau))
    return best


def main():
    mode, cfg = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    sets = load_sets()
    arr, pool, hold = sets["_arr"], sets["pool"], sets["hold"]
    print(f"pool {len(pool)} clips, hold30 {len(hold)} clips", flush=True)
    if mode == "cv":
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(pool))
        oof = [None] * len(pool)
        for f in range(5):
            tr = [c for c, k in zip(pool, fold) if k != f]
            te_idx = [j for j, k in enumerate(fold) if k == f]
            model = train(tr, arr, cfg, seed=f)
            for j, p in zip(te_idx, predict(model, [pool[j] for j in te_idx], arr)):
                oof[j] = p
        best = sweep(pool, oof, arr, f"cv-{cfg}")
        print(f"CV {cfg}: best {best[0]:+.4f} at (delta, fp_tau) = {best[1]}", flush=True)
        json.dump(dict(cfg=cfg, best=best[0], params=best[1]), open(f"{OUT}/cv_{cfg}.json", "w"))
    else:
        params = json.load(open(f"{OUT}/cv_{cfg}.json"))["params"]
        model = train(pool, arr, cfg, seed=0)
        torch.save(dict(state=model.state_dict(), cfg=cfg, params=params), f"./checkpoints/boundary_xattn_{cfg}.pth")
        preds = predict(model, hold, arr)
        base = pooled(hold, [c["base"] for c in hold], arr)
        r = pooled(hold, apply(hold, preds, *params), arr)
        rng = np.random.default_rng(0)
        refs = [arr[c["set"]][3][c["i"]] for c in hold]
        rb, rn = per_clip(refs, [c["base"] for c in hold]), per_clip(refs, apply(hold, preds, *params))
        comb = lambda r: f1_only(r) + r[:, 3].mean()
        diffs = []
        for _ in range(2000):
            s = rng.integers(0, len(hold), len(hold))
            diffs.append(comb(rn[s]) - comb(rb[s]))
        print(f"HOLD30 (params {params}): base " + " ".join(f"{k} {v:.4f}" for k, v in base.items()), flush=True)
        print(f"HOLD30 xattn: " + " ".join(f"{k} {v:.4f}" for k, v in r.items())
              + f" | delta {r['pooled'] - base['pooled']:+.4f} P>0 {np.mean(np.array(diffs) > 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
