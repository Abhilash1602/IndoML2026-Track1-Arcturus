"""Plan Step 4: train the boundary refiner (src/refiner.py) on the frozen v4 ensemble's
outputs, using only data no v4 member trained on (gold_heldout + synth_val, 85/15
split for early stopping and decode choice), then evaluate ONCE on the official
validation set against the v4 Viterbi baseline -- Codabench-style pooled score,
eq_avg, Natural F1, and a paired clip bootstrap.

Decodes compared (chosen on the 15% tuning split, never on validation):
  viterbi   -- baseline: v4 mean logit, Viterbi 0.9802/0.9489 (submission_v4's decode)
  hybrid    -- Viterbi segments on the refined presence, edges moved to the median
               regressed onset/offset of the segment's frames
  proposals -- paper's Algorithm 1 on refined presence + regressed distances

Usage: python -m temp.train_refiner
"""
import os

import numpy as np
import torch

from src.refiner import BoundaryRefiner, boundary_targets, decode_proposals, refine_segments, refiner_loss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, match_events, micro_event_f1

S = "./work"
OUT = "./checkpoints/boundary_refiner_v4" + ("_gold" if os.environ.get("REFINER_DATA") == "gold" else "") + ".pth"
PS, PE, HOP = 0.9802, 0.9489, 0.01
DEVICE = os.environ.get("VAANI_DEVICE", "cuda:0")
EPOCHS, BATCH, LR = int(os.environ.get("REFINER_EPOCHS", "60")), 32, 1e-3


def load(name):
    d = np.load(f"{S}/refiner_{name}.npz")
    return d["logits"].astype(np.float32), d["mel"].astype(np.float32), d["mask"].astype(np.float32)


def refs_of(mask):
    return [[(e["onset"], e["offset"]) for e in extract_events_from_mask(m.astype(np.int32), HOP)] for m in mask]


def viterbi_events(prob, ps=PS, pe=PE):
    return [[(e["onset"], e["offset"]) for e in extract_events_viterbi(
        viterbi_decode(p, p_stay_silence=ps, p_stay_event=pe), HOP)] for p in prob]


@torch.no_grad()
def predict(model, logits, mel):
    model.eval()
    outs = {k: [] for k in ("presence", "d_on", "d_off")}
    for i in range(0, len(logits), 64):
        o = model(torch.from_numpy(logits[i:i + 64]).to(DEVICE), torch.from_numpy(mel[i:i + 64]).to(DEVICE))
        outs["presence"].append(torch.sigmoid(o["presence"]).cpu().numpy())
        outs["d_on"].append(o["d_on"].cpu().numpy())
        outs["d_off"].append(o["d_off"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def decode_all(kind, pred, base_prob, thr=0.5):
    if kind == "viterbi":
        return viterbi_events(base_prob)
    if kind == "hybrid":
        segs = viterbi_events(pred["presence"])
        return [refine_segments(s, a, b) for s, a, b in zip(segs, pred["d_on"], pred["d_off"])]
    if kind == "hybrid_base":  # baseline segmentation, refined edges
        segs = viterbi_events(base_prob)
        return [refine_segments(s, a, b) for s, a, b in zip(segs, pred["d_on"], pred["d_off"])]
    return [decode_proposals(p, a, b, thr) for p, a, b in zip(pred["presence"], pred["d_on"], pred["d_off"])]


def per_clip(refs, preds):
    rows = []
    for g, p in zip(refs, preds):
        tp, fp, fn = match_events(g, p)
        d = macro_segment_dice([g], [p], hop_sec=HOP)
        rows.append((tp, fp, fn, d))
    return np.array(rows, float)


def pooled(arr_list, idx_list):
    a = np.concatenate([x[i] for x, i in zip(arr_list, idx_list)])
    tp, fp, fn = a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum()
    return 2 * tp / max(2 * tp + fp + fn, 1) + a[:, 3].mean()


def f1_only(arr):
    tp, fp, fn = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    parts = [load("gold_heldout")] + ([] if os.environ.get("REFINER_DATA") == "gold" else [load("synth_val")])
    L = np.concatenate([p[0] for p in parts]); M = np.concatenate([p[1] for p in parts])
    Y = np.concatenate([p[2] for p in parts])
    n = len(L)
    perm = rng.permutation(n)
    tune, train = perm[:int(0.15 * n)], perm[int(0.15 * n):]
    print(f"refiner data: {n} clips ({[len(p_[0]) for p_ in parts]}); "
          f"train {len(train)}, tune {len(tune)}", flush=True)
    tgt = boundary_targets(torch.from_numpy(Y))

    model = BoundaryRefiner(n_members=L.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    tune_refs = refs_of(Y[tune])
    tune_base = 1 / (1 + np.exp(-L[tune].mean(1)))
    base_tune = per_clip(tune_refs, decode_all("viterbi", None, tune_base))
    best, since, best_state = -1, 0, None
    for ep in range(EPOCHS):
        model.train()
        rng.shuffle(train)
        losses = []
        for i in range(0, len(train), BATCH):
            b = np.sort(train[i:i + BATCH])
            bt = torch.from_numpy(b)
            out = model(torch.from_numpy(L[b]).to(DEVICE), torch.from_numpy(M[b]).to(DEVICE))
            y = torch.from_numpy(Y[b]).to(DEVICE)
            loss, parts_ = refiner_loss(out, y, tuple(t[bt].to(DEVICE) for t in tgt))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        pred = predict(model, L[tune], M[tune])
        score = {k: f1_only(per_clip(tune_refs, decode_all(k, pred, tune_base))) for k in ("hybrid", "hybrid_base", "proposals")}
        top = max(score.values())
        print(f"E{ep + 1}: loss {np.mean(losses):.4f} | tune F1 viterbi {f1_only(base_tune):.4f} "
              + " ".join(f"{k} {v:.4f}" for k, v in score.items()), flush=True)
        if top > best:
            best, since, best_state = top, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_kind = max(score, key=score.get)
        else:
            since += 1
            if since >= 10:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, OUT)
    pred = predict(model, L[tune], M[tune])
    thr_scores = {thr: f1_only(per_clip(tune_refs, decode_all("proposals", pred, tune_base, thr)))
                  for thr in (0.3, 0.4, 0.5, 0.6, 0.7)}
    best_thr = max(thr_scores, key=thr_scores.get)
    print(f"\nchosen on tune split: best decode '{best_kind}' (tune F1 {best:.4f}); proposal thr {best_thr} "
          f"({thr_scores})", flush=True)

    # ---- one-shot evaluation on the official validation set ----
    ev = {s: load(f"official_{s}") for s in ("natural", "synthetic")}
    results = {}
    for kind in ("viterbi", "hybrid", "hybrid_base", "proposals"):
        arrs = []
        for s in ("natural", "synthetic"):
            l, m, y = ev[s]
            p = predict(model, l, m)
            base = 1 / (1 + np.exp(-l.mean(1)))
            arrs.append(per_clip(refs_of(y), decode_all(kind, p, base, best_thr)))
        results[kind] = arrs
        nat, syn = arrs
        print(f"OFFICIAL VAL [{kind:11s}] pooled(Codabench-style)={pooled(arrs, [np.arange(len(nat)), np.arange(len(syn))]):.4f} "
              f"natF1={f1_only(nat):.4f} natDice={nat[:, 3].mean():.4f} synF1={f1_only(syn):.4f} "
              f"eq_avg={0.5 * (f1_only(nat) + nat[:, 3].mean() + f1_only(syn) + syn[:, 3].mean()):.4f}", flush=True)
    nn_, ns = len(results["viterbi"][0]), len(results["viterbi"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(1000)]
    for kind in ("hybrid", "hybrid_base", "proposals"):
        d = np.array([pooled(results[kind], bt) - pooled(results["viterbi"], bt) for bt in boots])
        dn = np.array([f1_only(results[kind][0][bt[0]]) - f1_only(results["viterbi"][0][bt[0]]) for bt in boots])
        print(f"bootstrap {kind:11s} vs v4 viterbi: pooled delta mean {d.mean():+.4f} 5%={np.percentile(d, 5):+.4f} "
              f"P(>0)={np.mean(d > 0):.3f} | natF1 delta mean {dn.mean():+.4f} P(>0)={np.mean(dn > 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
