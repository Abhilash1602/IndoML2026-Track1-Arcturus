"""S2: the learned boundary refiner (src/refiner.py), retrained IN-DOMAIN on official-validation
train70 (the earlier refiner failed only because its synthetic training data came from our own
synthesiser, which does not match the organisers'), with and without the Silero VAD speech
probability as an extra input channel (human offsets sit at speech onsets far above chance).

Inputs per frame: the 5 v4 members' logits (never trained on validation), 80-band log-mel,
[VAD speech probability upsampled to 10 ms]. Training: train70 minus a 15% tuning slice
(early stopping + decode choice). One-shot evaluation on hold30 against the v5 pipeline
(v4 logit mean, Viterbi 0.9802/0.9489, gated edge correction) with a paired bootstrap.

Usage: python -m temp.train_refiner_val
"""
import json
import os

import numpy as np
import torch

from src.edge_correction import correct_events, load_table
from src.refiner import BoundaryRefiner, boundary_targets, decode_proposals, refine_segments, refiner_loss
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.vad_boundary_analysis import STEP, vad_probs

S = "./work"
DEVICE = os.environ.get("VAANI_DEVICE", "cuda:0")
EPOCHS, BATCH, LR = 60, 32, 1e-3
TABLE = load_table("./checkpoints/edge_correction_v5.json")


def load_all():
    split = json.load(open("./data/val_split_s0.json"))
    vad = vad_probs()
    out = {}
    for sub in ("natural", "synthetic"):
        d = np.load(f"{S}/refiner_official_{sub}.npz")
        ids = [str(x) for x in d["ids"]]
        V = np.zeros((len(ids), 1, 1000), np.float32)
        for k, name in enumerate(ids):
            p = vad[name]
            t = np.minimum((np.arange(1000) * 0.01 / STEP).astype(int), len(p) - 1) if len(p) else None
            if t is not None:
                V[k, 0] = p[t]
        for part in ("train70", "hold30"):
            idx = [k for k, n in enumerate(ids) if n in set(split[part][sub])]
            out[(part, sub)] = (d["logits"][idx].astype(np.float32), d["mel"][idx].astype(np.float32),
                                d["mask"][idx].astype(np.float32), V[idx])
    return out


@torch.no_grad()
def predict(model, L, M, V):
    model.eval()
    res = {"presence": [], "d_on": [], "d_off": []}
    for i in range(0, len(L), 64):
        o = model(torch.from_numpy(L[i:i + 64]).to(DEVICE), torch.from_numpy(M[i:i + 64]).to(DEVICE),
                  torch.from_numpy(V[i:i + 64]).to(DEVICE) if model.n_extra else None)
        res["presence"].append(torch.sigmoid(o["presence"]).cpu().numpy())
        res["d_on"].append(o["d_on"].cpu().numpy())
        res["d_off"].append(o["d_off"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in res.items()}


def events(kind, pred, base_prob, thr=0.5):
    if kind == "v5":
        return [correct_events(ev, p, TABLE) for ev, p in zip(viterbi_events(base_prob), base_prob)]
    if kind == "hybrid_base":
        return [refine_segments(s, a, b) for s, a, b in zip(viterbi_events(base_prob), pred["d_on"], pred["d_off"])]
    if kind == "hybrid":
        return [refine_segments(s, a, b) for s, a, b in zip(viterbi_events(pred["presence"]), pred["d_on"], pred["d_off"])]
    return [decode_proposals(p, a, b, thr) for p, a, b in zip(pred["presence"], pred["d_on"], pred["d_off"])]


def combined(rows):
    return f1_only(rows) + rows[:, 3].mean()


def refiner_kwargs():
    """Architecture variants (defaults = every v7-v11 refiner): REFINER_WIDTH, REFINER_GRU, REFINER_ARCH (gru|tf)."""
    return dict(width=int(os.environ.get("REFINER_WIDTH", "128")), gru=int(os.environ.get("REFINER_GRU", "128")),
                arch=os.environ.get("REFINER_ARCH", "gru"))


def train_one(n_extra, L, M, Y, V, tune, train, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tgt = boundary_targets(torch.from_numpy(Y))
    model = BoundaryRefiner(n_members=L.shape[1], n_extra=n_extra, **refiner_kwargs()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    t_refs = refs_of(Y[tune])
    t_base = 1 / (1 + np.exp(-L[tune].mean(1)))
    best, since, state, kind = -1, 0, None, None
    for ep in range(EPOCHS):
        model.train()
        order = train.copy()
        rng.shuffle(order)
        for i in range(0, len(order), BATCH):
            b = np.sort(order[i:i + BATCH])
            bt = torch.from_numpy(b)
            out = model(torch.from_numpy(L[b]).to(DEVICE), torch.from_numpy(M[b]).to(DEVICE),
                        torch.from_numpy(V[b]).to(DEVICE) if n_extra else None)
            loss, _ = refiner_loss(out, torch.from_numpy(Y[b]).to(DEVICE), tuple(t[bt].to(DEVICE) for t in tgt))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        pred = predict(model, L[tune], M[tune], V[tune])
        sc = {k: combined(per_clip(t_refs, events(k, pred, t_base))) for k in ("hybrid_base", "hybrid", "proposals")}
        top = max(sc, key=sc.get)
        if sc[top] > best:
            best, since, kind = sc[top], 0, top
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= 10:
                break
    model.load_state_dict(state)
    v5_tune = combined(per_clip(t_refs, events("v5", None, t_base)))
    print(f"  [n_extra={n_extra}] tune slice: best decode '{kind}' combined {best:.4f} vs v5 {v5_tune:.4f}", flush=True)
    return model, kind


def train_all():
    """v7: the same no-VAD refiner trained on 100% of the validation set (85% train, 15% tuning
    slice for early stopping, the same procedure as the train70 run). No held-out score exists."""
    seed = int(os.environ.get("REFINER_SEED", "0"))  # extra seeds vote on test merges/drops (v8)
    tag = os.environ.get("REFINER_TAG", "")  # architecture variant name; "" = the original refiner
    data = load_all()
    cat = lambda j: np.concatenate([data[(p, s)][j] for p in ("train70", "hold30") for s in ("natural", "synthetic")])
    L, M, Y, V = (cat(j) for j in range(4))
    n_extra = 0
    if os.environ.get("REFINER_EXTRA_SRC") == "emb":
        from temp.refiner_emb import extra_all_data
        V, n_extra = extra_all_data("train_all"), 32
    perm = np.random.default_rng(seed).permutation(len(L))
    tune, train = perm[: int(0.15 * len(L))], perm[int(0.15 * len(L)):]
    print(f"all validation, seed {seed}, variant '{tag}': {len(L)} clips (train {len(train)}, tune {len(tune)})", flush=True)
    model, kind = train_one(n_extra, L, M, Y, V, tune, train, seed=seed)
    if tag:
        path = f"./checkpoints/refiner_val_all_{tag}_s{seed}.pth"
    else:
        path = "./checkpoints/refiner_val_all_x0.pth" if seed == 0 else f"./checkpoints/refiner_val_all_x0_s{seed}.pth"
    torch.save({"state": model.state_dict(), "n_extra": n_extra, "kind": kind, "kwargs": refiner_kwargs(),
                "extra_src": os.environ.get("REFINER_EXTRA_SRC", "")}, path)
    print(f"saved {path} (decode '{kind}')", flush=True)


def main():
    if os.environ.get("REFINER_ALL") == "1":
        return train_all()
    data = load_all()
    cat = lambda part, j: np.concatenate([data[(part, "natural")][j], data[(part, "synthetic")][j]])
    L, M, Y, V = (cat("train70", j) for j in range(4))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(L))
    tune, train = perm[: int(0.15 * len(L))], perm[int(0.15 * len(L)):]
    print(f"train70: {len(L)} clips (train {len(train)}, tune {len(tune)})", flush=True)
    HL, HM, HY, HV = (cat("hold30", j) for j in range(4))
    n_nat = len(data[("hold30", "natural")][0])
    h_refs = refs_of(HY)
    h_base = 1 / (1 + np.exp(-HL.mean(1)))
    rows = {"v5 pipeline": per_clip(h_refs, events("v5", None, h_base))}
    for n_extra in [int(x) for x in os.environ.get("REFINER_EXTRA", "0,1").split(",")]:
        model, kind = train_one(n_extra, L, M, Y, V, tune, train)
        torch.save({"state": model.state_dict(), "n_extra": n_extra, "kind": kind},
                   f"./checkpoints/refiner_val_train70_x{n_extra}.pth")
        pred = predict(model, HL, HM, HV)
        np.savez(f"{S}/refiner_val_hold30_x{n_extra}.npz", **pred)
        name = f"refiner{' +VAD' if n_extra else ''} [{kind}]"
        rows[name] = per_clip(h_refs, events(kind, pred, h_base))
        corr = [correct_events(ev, p, TABLE) for ev, p in zip(events(kind, pred, h_base), h_base)]
        rows[name + " +corr"] = per_clip(h_refs, corr)
    nat = np.arange(len(h_refs)) < n_nat
    boots = [np.concatenate([rng.choice(np.where(nat)[0], nat.sum()), rng.choice(np.where(~nat)[0], (~nat).sum())])
             for _ in range(2000)]
    b0 = np.array([combined(rows["v5 pipeline"][b]) for b in boots])
    for k, r in rows.items():
        d = np.array([combined(r[b]) for b in boots]) - b0
        print(f"HOLD30 [{k:34s}] combined={combined(r):.4f} (vs v5 {d.mean():+.4f}, P>0={np.mean(d > 0):.3f}) "
              f"natF1={f1_only(r[nat]):.4f} natDice={r[nat, 3].mean():.4f} synF1={f1_only(r[~nat]):.4f} "
              f"synDice={r[~nat, 3].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
