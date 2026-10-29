"""Tolerance-hit-maximising edge decisions (kNN posterior), 5-fold out-of-fold on validation recipe events."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.edge_correction import correct_events, edge_contrast
from temp.crossfit_refiner import all_data, folds
from temp.eval_crossfit_refiner import clip, durations
from temp.eval_recipe_refiners import T, presence, seeds, recipe
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.verifier_recipe import W
L, _, Y, _, nat = all_data(); N, sec = len(L), durations()
refs, fold = refs_of(Y), folds(nat)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
P = 1 / (1 + np.exp(-(W * L.mean(1) + (1 - W) * lg(presence("crossfit", N)))))
base = [clip(correct_events(e, p, T), s) for e, p, s in zip(viterbi_events(P), P, sec)]
Rs = seeds("crossfit", N, P, sec)
rec = [recipe(x, Rs, i) for i, x in enumerate(base)]

def feats(p, a, b):
    ia, ib = int(a * 100), max(int(b * 100), int(a * 100) + 1); n = len(p)
    ins = p[ia:ib]; d = b - a
    pre, post = p[max(0, ia - 10):ia], p[ib:min(n, ib + 10)]
    return [np.log(d), ins.mean(), ins.max(), edge_contrast(p, a, b),
            ins[:5].mean() - (pre.mean() if len(pre) else 0), ins[-5:].mean() - (post.mean() if len(post) else 0),
            (np.argmax(ins) + .5) / len(ins), np.mean(ins > 0.7)]

GO = np.linspace(-0.4, 0.4, 17)      # relative shifts (fraction of predicted duration)
rows, ev_idx = [], []
for i, ev in enumerate(rec):
    for k, (a, b) in enumerate(ev):
        rows.append(feats(P[i], a, b)); ev_idx.append((i, k))
X = np.array(rows, np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
# for each event: the unique overlapping ref (if exactly one), else None
pair = []
for i, k in ev_idx:
    a, b = rec[i][k]; ov = [r for r in refs[i] if r[0] < b and r[1] > a]
    pair.append(ov[0] if len(ov) == 1 else None)

def hits(e, r):
    """(17,17) matrix: does shifted event match ref r within tolerance."""
    a, b = e; d = b - a; tol = max(0.2 * (r[1] - r[0]), 0.05)
    on = a + GO * d; off = b + GO * d
    return (np.abs(on - r[0]) <= tol)[:, None] & (np.abs(off - r[1]) <= tol)[None, :] & ((off[None, :] - on[:, None]) > 0.02)

H = [hits(rec[i][k], pair[j]) if pair[j] is not None else None for j, (i, k) in enumerate(ev_idx)]
efold = np.array([fold[i] for i, _ in ev_idx])
comb = lambda r: f1_only(r) + r[:, 3].mean()
r0 = per_clip(refs, rec)
z = len(GO) // 2
for K, margin in ((100, 0.03), (200, 0.03), (200, 0.06), (400, 0.03)):
    new = [list(e) for e in rec]; moved = 0
    for f in range(5):
        tr = np.where(efold != f)[0]; te = np.where(efold == f)[0]
        nn = NearestNeighbors(n_neighbors=K).fit(X[tr])
        _, idx = nn.kneighbors(X[te])
        for j, nb in zip(te, idx):
            M = np.zeros((len(GO), len(GO)))
            for q in tr[nb]:
                if H[q] is not None: M += H[q]
            M /= K
            bo, bf = np.unravel_index(np.argmax(M), M.shape)
            if M[bo, bf] - M[z, z] > margin:
                i, k = ev_idx[j]; a, b = rec[i][k]; d = b - a
                new[i][k] = (max(0.0, a + GO[bo] * d), min(sec[i], b + GO[bf] * d)); moved += 1
    r = per_clip(refs, new)
    print(f"K {K} margin {margin}: moved {moved}/{len(ev_idx)} | pooled {comb(r):.4f} ({comb(r) - comb(r0):+.4f}) | Nat F1 {f1_only(r[nat]):.4f} "
          f"({f1_only(r[nat]) - f1_only(r0[nat]):+.4f}) Dice {r[nat, 3].mean():.4f} ({r[nat, 3].mean() - r0[nat, 3].mean():+.4f}) | "
          f"Syn F1 {f1_only(r[~nat]):.4f} ({f1_only(r[~nat]) - f1_only(r0[~nat]):+.4f}) Dice {r[~nat, 3].mean() - r0[~nat, 3].mean():+.4f}", flush=True)
