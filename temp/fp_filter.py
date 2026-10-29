"""P3: false-positive event filter (plan Phase 1).

Dropping a predicted event that overlaps no reference raises Event F1 (one fewer prediction)
AND Segment Dice (fewer wrong frames). A gradient-boosted classifier on 15 event features from
the 5 v4 members (never trained on Gold held-out / official validation) predicts P(pure FP).
Out-of-sample (trained on Gold held-out only, scored once on validation): AUC 0.845, pooled
+0.0020. For test it is trained on Gold held-out + all validation; the threshold is re-checked
by 5-fold CV on that pool. Applied to v6's own events on Natural-looking clips (v6-probability
gate, as v7 used); features always come from the v4 members, the distribution it was trained on.

Usage: python -m temp.fp_filter [--thr 0.7]   -> submission_probe_P3.zip (+ report)
"""
import argparse
import json
import zipfile

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.edge_correction import correct_events, edge_contrast, load_table
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

TABLE = load_table("./checkpoints/edge_correction_v5.json")
V4_IN_V6 = slice(5, 10)  # v6 MEMBERS order: 5 VA members, then the 5 v4 members


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def gate_of(P, events_vit):
    return [True if not e else np.mean([edge_contrast(p, a, b) for a, b in e]) < 0.5 for e, p in zip(events_vit, P)]


def features(L, P, E, ev, k):
    a, b = ev[k]
    ia, ib = int(a * 100), max(int(b * 100), int(a * 100) + 1)
    ib = min(ib, P.shape[-1])
    ia = min(ia, ib - 1)
    p, lm = P[ia:ib], L[:, ia:ib].mean(1)
    lo, hi = max(ia - 50, 0), min(ib + 50, len(E))
    ctx = np.r_[E[lo:ia], E[ib:hi]]
    return [b - a, p.mean(), p.max(), p.min(), p.std(), (lm > 0).mean(), lm.std(), lm.min(), edge_contrast(P, a, b),
            len(ev), a - (ev[k - 1][1] if k else 0), (ev[k + 1][0] if k + 1 < len(ev) else len(P) * 0.01) - b, a,
            E[ia:ib].mean() - (ctx.mean() if len(ctx) else E[ia:ib].mean()), P.mean()]


def labelled_set(name):
    d = np.load(f"{S}/refiner_{name}.npz")
    L = d["logits"].astype(np.float32)
    P = 1 / (1 + np.exp(-L.mean(1)))
    vit = viterbi_events(P)
    base = [correct_events(e, p, TABLE) for e, p in zip(vit, P)]
    gate = gate_of(P, vit)
    E = d["mel"].astype(np.float32).mean(1)
    refs = refs_of(d["mask"].astype(np.float32))
    X, y, key = [], [], []
    for i, (ev, g) in enumerate(zip(base, gate)):
        if not g:
            continue
        for k in range(len(ev)):
            X.append(features(L[i], P[i], E[i], ev, k))
            y.append(0 if any(overlap(ev[k], r) for r in refs[i]) else 1)
            key.append((name, i, k))
    return np.array(X, np.float32), np.array(y), key, base, refs


def make_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=30,
                                          l2_regularization=1.0, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=None, help="drop threshold (default: chosen by CV)")
    args = ap.parse_args()
    sets = {n: labelled_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")}
    X = np.concatenate([s[0] for s in sets.values()])
    y = np.concatenate([s[1] for s in sets.values()])
    key = sum((s[2] for s in sets.values()), [])
    groups = np.array([f"{k[0]}:{k[1]}" for k in key])
    ug = np.unique(groups)
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(ug):
        m = np.isin(groups, ug[tr])
        oof[~m] = make_model().fit(X[m], y[m]).predict_proba(X[~m])[:, 1]
    print(f"pool: {len(y)} gated events, pure-FP rate {y.mean():.1%}, 5-fold CV AUC {roc_auc_score(y, oof):.3f}", flush=True)
    best = (None, -9.0)
    for thr in (0.5, 0.6, 0.7, 0.8):
        tot = 0.0
        for name, (_, _, kk, base, refs) in sets.items():
            idx = [j for j, k in enumerate(key) if k[0] == name]
            drop = {(key[j][1], key[j][2]) for j in idx if oof[j] > thr}
            out = [[e for k, e in enumerate(ev) if (i, k) not in drop] for i, ev in enumerate(base)]
            r0, r1 = per_clip(refs, base), per_clip(refs, out)
            tot += (f1_only(r1) - f1_only(r0)) + (r1[:, 3].mean() - r0[:, 3].mean())
        print(f"  CV thr {thr}: summed (F1 + Dice) change over the three sets {tot:+.4f}", flush=True)
        if tot > best[1]:
            best = (thr, tot)
    thr = args.thr if args.thr is not None else best[0]
    model = make_model().fit(X, y)

    # ---- apply to v6's events on test ----
    ids = json.load(open(f"{S}/verifier/test_ids.json"))
    logits = np.load(f"{S}/verifier/test_logits.npy").astype(np.float32)
    mel = np.load(f"{S}/verifier/test_mel.npy").astype(np.float32)
    rows = [json.loads(l) for l in zipfile.ZipFile("submission_v6.zip").read("predictions.jsonl").decode().splitlines()]
    v6 = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}
    out, dropped, n_clips = {}, 0, 0
    for i, cid in enumerate(ids):
        ev = v6[cid]
        P6 = 1 / (1 + np.exp(-logits[i].mean(0)))
        if not ev or not gate_of([P6], viterbi_events(P6[None]))[0]:
            continue
        L4 = logits[i, V4_IN_V6]
        P4 = 1 / (1 + np.exp(-L4.mean(0)))
        E = mel[i].mean(0)
        s = model.predict_proba(np.array([features(L4, P4, E, ev, k) for k in range(len(ev))], np.float32))[:, 1]
        keep = [e for e, sc in zip(ev, s) if sc <= thr]
        if len(keep) != len(ev):
            out[cid] = keep
            dropped += len(ev) - len(keep)
            n_clips += 1
    final = {r["clip_id"]: out.get(r["clip_id"], v6[r["clip_id"]]) for r in rows}
    assert len(final) == 5517 and all(0 <= a < b for ev in final.values() for a, b in ev)
    print(f"test: threshold {thr}; dropped {dropped} v6 events on {n_clips} clips "
          f"({sum(map(len, v6.values()))} -> {sum(map(len, final.values()))})", flush=True)
    with zipfile.ZipFile("submission_probe_P3.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "".join(
            json.dumps({"clip_id": r["clip_id"], "events": [{"onset": a, "offset": b} for a, b in final[r["clip_id"]]]}) + "\n"
            for r in rows))
    print("wrote submission_probe_P3.zip", flush=True)


if __name__ == "__main__":
    main()
