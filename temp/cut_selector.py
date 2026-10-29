"""Dice-neutral event cutting: feasibility of a learned cut selector.

Frames stay exactly v5/v6's (Segment Dice identical by construction); only the partition of each
predicted event into touching sub-events changes, which only Event F1 sees. Cut positions come
from the ensemble probability's high-confidence core (first/last frame >= t). Oracle selection of
which events/sides to cut is worth +0.10..+0.12 Natural F1 on validation; blanket cutting is
-0.02..-0.23, so the selector is everything.

Selector: per (predicted event, cut option) features -> gradient-boosted regression of the
option's gain in matched references minus LAM per extra piece. Trained on Gold HELD-OUT
(refiner_gold_heldout.npz; v4 never trained on it), margin chosen by 5-fold CV there, then scored
ONCE on the official validation set (gated Natural-looking clips only; synthetic reported too).

Usage: python -m temp.cut_selector
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src.edge_correction import correct_events, edge_contrast, load_table
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S

TABLE = load_table("./checkpoints/edge_correction_v5.json")
THRS = (0.6, 0.7, 0.8, 0.9)
LAM = 0.3
MIN_PIECE = 0.03


def load(name):
    d = np.load(f"{S}/refiner_{name}.npz")
    P = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
    vit = viterbi_events(P)
    base = [[(round(a, 2), round(b, 2)) for a, b in correct_events(e, p, TABLE)] for e, p in zip(vit, P)]
    gate = [True if not e else np.mean([edge_contrast(p, a, b) for a, b in e]) < 0.5 for e, p in zip(vit, P)]
    energy = d["mel"].astype(np.float32).mean(1)  # (N, T) mean log-mel per frame
    return P, energy, refs_of(d["mask"].astype(np.float32)), base, gate


def match(p, r):
    t = max(0.2 * (r[1] - r[0]), 0.05)
    return abs(p[0] - r[0]) <= t and abs(p[1] - r[1]) <= t


def n_matched(pieces, rs):
    used, n = set(), 0
    for q in pieces:
        for k, r in enumerate(rs):
            if k not in used and match(q, r):
                used.add(k); n += 1
                break
    return n


def options(a, b, p):
    """[(pieces, (t, cut_left, cut_right, na, nb))] -- cut options at the probability core."""
    ia, ib = int(round(a * 100)), int(round(b * 100))
    seg = p[ia:ib]
    out = []
    for t in THRS:
        idx = np.where(seg >= t)[0]
        if not len(idx):
            continue
        na, nb = round((ia + idx[0]) / 100, 2), round((ia + idx[-1] + 1) / 100, 2)
        L, R = na - a >= MIN_PIECE, b - nb >= MIN_PIECE and nb - na >= MIN_PIECE
        if L:
            out.append(([(a, na), (na, b)], (t, 1, 0, na, nb)))
        if R:
            out.append(([(a, nb), (nb, b)], (t, 0, 1, na, nb)))
        if L and R:
            out.append(([(a, na), (na, nb), (nb, b)], (t, 1, 1, na, nb)))
    return out


def feats(a, b, info, p, e, clip_events, k):
    t, cl, cr, na, nb = info
    ia, ib, ja, jb = (int(round(x * 100)) for x in (a, b, na, nb))
    T = len(p)
    step = lambda j: float(e[min(j + 5, T - 1)] - e[max(j - 5, 0)]) if 0 <= j < T else 0.0
    prev_gap = a - clip_events[k - 1][1] if k > 0 else 10.0
    next_gap = clip_events[k + 1][0] - b if k + 1 < len(clip_events) else 10.0
    seg = p[ia:ib] if ib > ia else p[ia:ia + 1]
    return [b - a, t, cl, cr, na - a, b - nb, nb - na, (nb - na) / max(b - a, 1e-3),
            float(seg.mean()), float(seg.max()), float(p[min(ia, T - 1)]), float(p[min(max(ib - 1, 0), T - 1)]),
            float(p[ia:ja].mean()) if ja > ia else 1.0, float(p[jb:ib].mean()) if ib > jb else 1.0,
            step(ja), step(jb), step(ia), step(ib),
            float(e[ia:ja].mean() - e[ja:jb].mean()) if ja > ia and jb > ja else 0.0,
            float(e[jb:ib].mean() - e[ja:jb].mean()) if ib > jb and jb > ja else 0.0,
            edge_contrast(p, a, b), len(clip_events), prev_gap, next_gap]


def build(data, with_labels=True):
    P, E, refs, base, gate = data
    X, y, key = [], [], []
    for i, (ev, g) in enumerate(zip(base, gate)):
        if not g:
            continue
        for k, (a, b) in enumerate(ev):
            rs = [r for r in refs[i] if min(r[1], b) > max(r[0], a)]
            v0 = n_matched([(a, b)], rs) - LAM
            for pieces, info in options(a, b, P[i]):
                X.append(feats(a, b, info, P[i], E[i], ev, k))
                y.append(n_matched(pieces, rs) - LAM * len(pieces) - v0)
                key.append((i, k, pieces))
    return np.array(X, np.float32), np.array(y, np.float32), key


def apply(data, key, pred, margin):
    base = data[3]
    best = {}
    for (i, k, pieces), s in zip(key, pred):
        if s > margin and s > best.get((i, k), (-1e9, None))[0]:
            best[(i, k)] = (s, pieces)
    out = []
    for i, ev in enumerate(base):
        res = []
        for k, e in enumerate(ev):
            res += best[(i, k)][1] if (i, k) in best else [e]
        out.append(res)
    return out


def main():
    gold = load("gold_heldout")
    Xg, yg, kg = build(gold)
    print(f"Gold held-out: {len(Xg)} (event, option) rows; positive-gain rows {np.mean(yg > 0):.1%}", flush=True)
    mk = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    # margin by 5-fold CV on Gold held-out (folds by clip)
    clips = np.array([k[0] for k in kg])
    oof = np.zeros(len(yg), np.float32)
    uc = np.unique(clips)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(uc):
        m = np.isin(clips, uc[tr])
        oof[~m] = mk().fit(Xg[m], yg[m]).predict(Xg[~m])
    r0 = per_clip(gold[2], gold[3])
    best_m, best_f = None, -1
    for margin in (0.0, 0.1, 0.2, 0.3, 0.4):
        r = per_clip(gold[2], apply(gold, kg, oof, margin))
        print(f"  Gold held-out CV margin {margin:.1f}: F1 {f1_only(r0):.4f} -> {f1_only(r):.4f} ({f1_only(r) - f1_only(r0):+.4f}) "
              f"Dice {r[:, 3].mean() - r0[:, 3].mean():+.5f}", flush=True)
        if f1_only(r) > best_f:
            best_f, best_m = f1_only(r), margin
    model = mk().fit(Xg, yg)
    print(f"chosen margin {best_m}; scoring once on official validation", flush=True)
    for sub in ("official_natural", "official_synthetic"):
        data = load(sub)
        Xv, _, kv = build(data)
        out = apply(data, kv, model.predict(Xv) if len(Xv) else [], best_m)
        rb, rc = per_clip(data[2], data[3]), per_clip(data[2], out)
        print(f"VALIDATION {sub}: F1 {f1_only(rb):.4f} -> {f1_only(rc):.4f} ({f1_only(rc) - f1_only(rb):+.4f}) | "
              f"Dice {rb[:, 3].mean():.4f} -> {rc[:, 3].mean():.4f} | events {sum(map(len, data[3]))} -> {sum(map(len, out))}",
              flush=True)


if __name__ == "__main__":
    main()
