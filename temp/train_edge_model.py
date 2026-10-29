"""Train the short-event edge model on Gold (plan Step 2).

Training data: Gold TRAIN split (scratchpad/edge_gold_train.npz, 8,888 clips, merged any-noise
mask = the scoring convention). Every short reference event (0.1-0.8 s, not touching the clip
borders) gives two edges; each epoch every edge gets K candidate positions drawn from the
empirical v4/v5-decode error model (scratchpad/edge_errors.npy, measured on Gold held-out; 80%)
or uniform +-0.3 s (20%). Target = true edge frame inside the 1.2 s crop centred on the
candidate; soft (Gaussian sigma 1.5 bins) cross-entropy on the matching head.

Early stopping: Gold HELD-OUT (refiner_gold_heldout.npz; no v4 member or edge model trained on
it) with the REAL v5-decode predictions of the v4 ensemble: base events vs edges of gated short
events moved, scored F1 + Dice. Official validation is not touched here.

Usage: EDGE_SEED=0 VAANI_DEVICE=cuda:0 python -m temp.train_edge_model
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.edge_correction import correct_events, edge_contrast, load_table
from src.edge_model import HALF, HOP, PAD_DB, WIN, EdgeModel, move_edges, normalise, span_rows
from temp.edge_data import MAX_DUR, MIN_DUR
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

DEVICE = os.environ.get("VAANI_DEVICE", "cuda:0")
SEED = int(os.environ.get("EDGE_SEED", "0"))
EPOCHS, BATCH, LR, K = int(os.environ.get("EDGE_EPOCHS", "30")), 256, 1e-3, 4
FRAMES = 1000
TABLE = load_table("./checkpoints/edge_correction_v5.json")
OUT = f"./checkpoints/edge_model_gold_s{SEED}.pth"


def edges_of(mask):
    """(clip, on_frame, off_frame) for short reference events away from the clip borders."""
    out = []
    for i, ev in enumerate(refs_of(mask)):
        for a, b in ev:
            if MIN_DUR <= b - a <= MAX_DUR and a > 0 and b < FRAMES * HOP - HOP:
                out.append((i, int(round(a / HOP)), int(round(b / HOP))))
    return np.array(out, np.int64)


def sample_items(E, errs, rng):
    """K perturbed candidates per edge -> arrays clip, centre, cand_on, cand_off, side, target."""
    n = len(E) * K
    ev = np.repeat(E, K, axis=0)
    emp = errs[rng.integers(0, len(errs), n)] + rng.normal(0, 0.01, (n, 2))
    uni = rng.uniform(-0.3, 0.3, (n, 2))
    e = np.where(rng.random(n)[:, None] < 0.8, emp, uni)
    c_on = ev[:, 1] + np.round(e[:, 0] / HOP).astype(np.int64)
    c_off = ev[:, 2] + np.round(e[:, 1] / HOP).astype(np.int64)
    side = rng.integers(0, 2, n)
    centre = np.where(side == 0, c_on, c_off)
    target = np.where(side == 0, ev[:, 1], ev[:, 2]) - centre + HALF
    ok = (c_off - c_on >= 2) & (target >= 0) & (target < WIN) & (centre >= 0) & (centre < FRAMES)
    return ev[ok, 0], centre[ok], c_on[ok], c_off[ok], side[ok], target[ok]


def crops_from(padded, clip, centre):
    idx = torch.as_tensor(centre, device=padded.device)[:, None] + torch.arange(WIN, device=padded.device)[None]
    x = padded[torch.as_tensor(clip, device=padded.device)[:, None], :, idx]  # (B, WIN, 80)
    return normalise(x.permute(0, 2, 1).float())


def heldout_eval(model, H):
    model.eval()
    moved = [move_edges(model, H["mel"][i], ev, DEVICE) if g else ev for i, (ev, g) in enumerate(zip(H["base"], H["gate"]))]
    model.train()
    r = per_clip(H["refs"], moved)
    return f1_only(r) + r[:, 3].mean(), f1_only(r), r[:, 3].mean()


def load_heldout():
    d = np.load(f"{S}/refiner_gold_heldout.npz")
    prob = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
    vit = viterbi_events(prob)
    base = [correct_events(e, p, TABLE) for e, p in zip(vit, prob)]
    gate = [True if not e else np.mean([edge_contrast(p, a, b) for a, b in e]) < 0.5 for e, p in zip(vit, prob)]
    return {"mel": d["mel"].astype(np.float32), "base": base, "gate": gate, "refs": refs_of(d["mask"].astype(np.float32))}


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    d = np.load(f"{S}/edge_gold_train.npz")
    E = edges_of(d["mask"].astype(np.float32))
    errs = np.load(f"{S}/edge_errors.npy")
    padded = F.pad(torch.from_numpy(d["mel"]).to(DEVICE), (HALF, HALF), value=PAD_DB)  # (N, 80, T + WIN), f16
    H = load_heldout()
    rb = per_clip(H["refs"], H["base"])
    base_score = f1_only(rb) + rb[:, 3].mean()
    print(f"seed {SEED}: {len(E)} short Gold-train events; heldout base F1 {f1_only(rb):.4f} Dice {rb[:, 3].mean():.4f} "
          f"combined {base_score:.4f}", flush=True)
    model = EdgeModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * (len(E) * K // BATCH + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    bins = torch.arange(WIN, device=DEVICE).float()
    best, since = -1.0, 0
    for ep in range(EPOCHS):
        clip, centre, c_on, c_off, side, target = sample_items(E, errs, rng)
        perm = rng.permutation(len(clip))
        tot, n = 0.0, 0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            if len(b) < 2:
                continue
            x = crops_from(padded, clip[b], centre[b])
            cen = torch.as_tensor(centre[b], device=DEVICE)
            span = span_rows(cen, torch.as_tensor(c_on[b], device=DEVICE), torch.as_tensor(c_off[b], device=DEVICE))
            logits = model(x, span)
            sd = torch.as_tensor(side[b], device=DEVICE)
            lg = logits[torch.arange(len(b), device=DEVICE), sd]  # (B, WIN)
            t = torch.as_tensor(target[b], device=DEVICE).float()
            soft = torch.softmax(-0.5 * ((bins[None] - t[:, None]) / 1.5) ** 2, -1)
            loss = -(soft * F.log_softmax(lg, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        comb, f1, dice = heldout_eval(model, H)
        print(f"  ep {ep:2d} loss {tot / max(n, 1):.3f} | heldout F1 {f1:.4f} Dice {dice:.4f} combined {comb:.4f} "
              f"({comb - base_score:+.4f})", flush=True)
        if comb > best:
            best, since = comb, 0
            torch.save({"state": model.state_dict(), "n_extra": 0, "epoch": ep, "heldout": float(comb)}, OUT)
        else:
            since += 1
            if since >= 6:
                break
    print(f"best heldout combined {best:.4f} ({best - base_score:+.4f} vs base); saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
