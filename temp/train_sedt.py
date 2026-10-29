"""Train the Dice-locked SEDT (plan Step 2) and report gate G1 on Gold held-out.

Train: Gold train (scratchpad/sedt/gold_train_*; 8,888 clips, pretrained Dasheng tokens +
10 ms log-mel, merged any-noise references). Augmentation: random time shift of +-1 s (whole
tokens, zero/floor padding, references shifted and clipped).

Every EVAL_EVERY epochs on Gold HELD-OUT (no v4 member, no SEDT trained on it):
  * standalone SEDT event F1 (confidence >= 0.5, non-overlapping);
  * the lock: v4-ensemble v5-decode events on Natural-gated clips re-cut at SEDT event edges
    (src/event_lock.lock_partition) for c_min in C_GRID -> F1 gain; Dice is identical by
    construction (asserted);
  * cut accounting: cuts that gain matches vs cuts that break them.
Checkpoint = best locked F1 gain (c_min stored). Official validation is NOT touched here.

Usage: SEDT_SEED=0 VAANI_DEVICE=cuda:0 python -m temp.train_sedt
"""
import json
import os

import numpy as np
import torch

from src.event_lock import frames_equal, lock_partition
from src.sedt import SEDT, predict_events, select_events, sedt_loss
from temp.train_edge_model import load_heldout
from temp.train_refiner import f1_only, per_clip, refs_of
from temp.train_refiner_val import S

DEVICE = os.environ.get("VAANI_DEVICE", "cuda:0")
SEED = int(os.environ.get("SEDT_SEED", "0"))
EPOCHS = int(os.environ.get("SEDT_EPOCHS", "150"))
BATCH, EVAL_EVERY, PATIENCE = 64, 5, 6
LR = float(os.environ.get("SEDT_LR", "1e-4"))
CLIP_NORM = float(os.environ.get("SEDT_CLIP", "0.1"))
KWARGS = {"anchors": os.environ.get("SEDT_ANCHORS") == "1"}
TAG = os.environ.get("SEDT_TAG", "")
C_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
D = f"{S}/sedt"
OUT = f"./checkpoints/sedt_gold{TAG}_s{SEED}.pth"


def load_set(name):
    tok = torch.from_numpy(np.load(f"{D}/{name}_tok.npy")).to(DEVICE)  # f16
    mel = torch.from_numpy(np.load(f"{D}/{name}_mel.npy")).to(DEVICE)
    refs = refs_of(np.load(f"{D}/{name}_mask.npy").astype(np.float32))
    return tok, mel, refs, json.load(open(f"{D}/{name}_ids.json"))


def shifted(tok, mel, refs, k):
    """Shift one clip by k tokens (40 ms each; k > 0 = later)."""
    if k == 0:
        return tok, mel, refs
    t, m = torch.zeros_like(tok), torch.full_like(mel, -100.0)
    if k > 0:
        t[k:], m[:, 4 * k:] = tok[:-k], mel[:, :-4 * k]
    else:
        t[:k], m[:, :4 * k] = tok[-k:], mel[:, -4 * k:]
    s = 0.04 * k
    r = [(max(a + s, 0.0), min(b + s, 10.0)) for a, b in refs]
    return t, m, [(a, b) for a, b in r if b - a >= 0.02]


@torch.no_grad()
def predict_all(model, tok, mel, bs=128):
    model.eval()
    out = []
    for i in range(0, len(tok), bs):
        out += predict_events(model, tok[i:i + bs].float(), mel[i:i + bs].float())
    model.train()
    return out


def match_count(pieces, rs):
    from temp.official_metric import match_events
    return match_events(rs, pieces)[0] if pieces else 0


def gate_g1(pred, H, refs):
    """Standalone F1 + locked gains on Gold held-out; returns (report dict, best c_min)."""
    alone = per_clip(refs, [select_events(p, 0.5) for p in pred])
    r0 = per_clip(refs, H["base"])
    rep = {"standalone_F1": f1_only(alone), "base_F1": f1_only(r0)}
    best_c, best_gain = None, -1.0
    for c in C_GRID:
        locked, gain_cuts, break_cuts = [], 0, 0
        for ev, g, p, rs in zip(H["base"], H["gate"], pred, refs):
            if not g:
                locked.append(ev)
                continue
            sel = [(a, b) for a, b, cc in p if cc >= c]
            new = lock_partition(ev, sel)
            assert frames_equal(new, ev)
            if new != ev:
                d = match_count(new, rs) - match_count(ev, rs)
                gain_cuts += d > 0
                break_cuts += d < 0
            locked.append(new)
        r = per_clip(refs, locked)
        assert abs(r[:, 3].mean() - r0[:, 3].mean()) < 1e-12
        gain = f1_only(r) - f1_only(r0)
        rep[f"c{c}"] = (round(gain, 4), gain_cuts, break_cuts)
        if gain > best_gain:
            best_gain, best_c = gain, c
    rep["best"] = (best_c, round(best_gain, 4))
    return rep, best_c, best_gain


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok, mel, refs, _ = load_set("gold_train")
    htok, hmel, hrefs, hids = load_set("gold_heldout")
    H = load_heldout()
    cached_ids = [str(x) for x in np.load(f"{S}/refiner_gold_heldout.npz")["ids"]]
    assert cached_ids == hids, "Gold held-out order differs from the v4-logit cache"
    assert refs_of(np.load(f"{S}/refiner_gold_heldout.npz")["mask"].astype(np.float32)) == hrefs
    print(f"seed {SEED}: train {len(tok)} clips ({sum(map(len, refs))} events); held-out {len(htok)}", flush=True)
    print(f"lr {LR} clip {CLIP_NORM} kwargs {KWARGS} -> {OUT}", flush=True)
    model = SEDT(**KWARGS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(EPOCHS * 0.7)], gamma=0.1)
    best, best_alone, since = 0.0, -1.0, 0
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tok))
        tot, n = 0.0, 0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            ts, ms, rs = [], [], []
            for i in b:
                t, m, r = shifted(tok[i], mel[i], refs[i], int(rng.integers(-25, 26)))
                ts.append(t); ms.append(m)
                rs.append(torch.tensor(r, dtype=torch.float32, device=DEVICE).reshape(-1, 2))
            loss, parts = sedt_loss(model(torch.stack(ts).float(), torch.stack(ms).float()), rs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            tot += loss.item(); n += 1
        sched.step()
        if (ep + 1) % EVAL_EVERY:
            continue
        rep, c, gain = gate_g1(predict_all(model, htok, hmel), H, hrefs)
        print(f"  ep {ep + 1:3d} loss {tot / n:.3f} {parts} | G1 {rep}", flush=True)
        ck = {"state": model.state_dict(), "kwargs": KWARGS, "c_min": float(c), "epoch": ep + 1, "gain": float(gain),
              "standalone_F1": float(rep["standalone_F1"])}
        # Patience follows standalone detection quality (is it still learning?); "no cuts" gives a
        # locked gain of exactly 0, so the lock gain alone would stop a model that is still improving.
        if rep["standalone_F1"] > best_alone:
            best_alone, since = rep["standalone_F1"], 0
            torch.save(ck, OUT.replace(".pth", "_alone.pth"))
        else:
            since += 1
        if gain > best and gain > 0:
            best = gain
            torch.save(ck, OUT)
        if since >= PATIENCE:
            break
    print(f"best Gold held-out locked F1 gain {best:+.4f} (saved {OUT} only if > 0); "
          f"best standalone F1 {best_alone:.4f}", flush=True)


if __name__ == "__main__":
    main()
