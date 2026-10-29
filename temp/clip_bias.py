"""Per-clip sensitivity: predict, for each clip, the logit offset b applied before the (unchanged)
Viterbi + edge-correction decode.

Why: on validation Natural, choosing the best b in [-3, 3] per clip with the labels (oracle)
lifts Natural F1 0.396 -> 0.59-0.68 and Dice 0.718 -> 0.84-0.86, and the best b is spread over the
whole range -- one global decode setting is badly wrong for many clips. This measures how much
of that is predictable from label-free clip features.

Model: one gradient-boosted regressor over (clip features, b) -> the clip's score at b, where
score = Dice + 0.5 * (2TP - FP - FN) / (TP + FP + FN) (rewards Dice and matched events); the
chosen b per clip is the argmax over the grid. Features are from models that never trained on
these clips: v4 member logits, the anchor SEDT, the refiners (cross-fit on validation,
all-validation on Gold held-out), 80-mel energy, pooled v4 embeddings (PCA).
Training pool: Gold held-out + validation Natural, 5-fold CV grouped by clip; evaluation is the
out-of-fold choice on validation Natural (pooled F1 and macro Dice), vs b = 0.

Usage: python -m temp.clip_bias
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src.edge_correction import correct_events
from temp.fp_filter import TABLE
from temp.merge_model import presence_all_val, presence_val
from temp.train_refiner import f1_only, per_clip, viterbi_events
from temp.train_refiner_val import S
from temp.verifier import load_set

BIASES = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
N_PCA = 16


def bias_rows(D):
    """(nB, N, 4) per-clip (tp, fp, fn, dice) of the decode at each bias."""
    Lm = D["L"].mean(1)
    out = []
    for b in BIASES:
        P = 1 / (1 + np.exp(-(Lm + b)))
        ev = [correct_events(e, p, TABLE) for e, p in zip(viterbi_events(P), P)]
        out.append(per_clip(D["refs"], ev))
    return np.stack(out)


def clip_feats(D, pres, i):
    L, P, E = D["L"][i], D["P"][i], D["E"][i]
    valid = E > -99
    n = max(int(valid.sum()), 1)
    Pv, Ev, Lv = P[valid] if valid.any() else P, E[valid] if valid.any() else E, L[:, valid] if valid.any() else L
    rp = pres[i].astype(np.float32)[:, :len(P)]
    sedt = [c for a, b, c in D["sedt"][i]]
    return [n / 100, Pv.mean(), *np.percentile(Pv, [50, 75, 90, 95, 99]), (Pv > 0.3).mean(), (Pv > 0.5).mean(), (Pv > 0.7).mean(),
            *Lv.mean(1), Lv.std(0).mean(), (Lv > 0).mean(), Ev.mean(), Ev.std(), *np.percentile(Ev, [10, 50, 90]),
            rp.mean(), (rp > 0.5).mean(), np.abs(rp.mean(0) - P).mean(),
            sum(c > 0.5 for c in sedt), sum(c for c in sedt), max(sedt, default=0.0)]


def clip_emb(D, i):
    return np.asarray(D["emb"][i], np.float32).mean(1).reshape(-1)


def score(R):
    tp, fp, fn, dice = R[..., 0], R[..., 1], R[..., 2], R[..., 3]
    return dice + 0.5 * (2 * tp - fp - fn) / np.maximum(tp + fp + fn, 1)


def main():
    gold = load_set("gold_heldout")
    nat = load_set("official_natural")
    gd = np.load(f"{S}/refiner_gold_heldout.npz")
    pres = {"gold_heldout": presence_all_val(gd["logits"].astype(np.float32), gd["mel"].astype(np.float32)),
            "official_natural": presence_val()["official_natural"]}
    X, Em, R, grp = [], [], [], []
    for D in (gold, nat):
        rows = bias_rows(D)  # (nB, N, 4)
        for i in range(len(D["L"])):
            X.append(clip_feats(D, pres[D["name"]], i)); Em.append(clip_emb(D, i)); R.append(rows[:, i]); grp.append((D["name"], i))
    X, Em, R = np.array(X, np.float32), np.array(Em, np.float32), np.array(R)  # R (Nc, nB, 4)
    XF = np.c_[X, PCA(N_PCA, random_state=0).fit_transform(Em)]
    nB = len(BIASES)
    Xb = np.repeat(XF, nB, 0)
    Xb = np.c_[Xb, np.tile(BIASES, len(XF))]
    yb = score(R).reshape(-1)
    clip_id = np.repeat(np.arange(len(XF)), nB)
    pred = np.zeros(len(yb))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(len(XF))):
        m = np.isin(clip_id, tr)
        model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, min_samples_leaf=40, random_state=0)
        pred[~m] = model.fit(Xb[m], yb[m]).predict(Xb[~m])
    pick = pred.reshape(len(XF), nB).argmax(1)
    is_nat = np.array([g[0] == "official_natural" for g in grp])
    b0 = BIASES.index(0)
    for name, mask in (("validation Natural", is_nat), ("Gold held-out", ~is_nat)):
        r0 = R[mask, b0]
        rs = R[mask][np.arange(mask.sum()), pick[mask]]
        ro = R[mask][np.arange(mask.sum()), score(R[mask]).argmax(1)]
        print(f"{name}: b=0 F1 {f1_only(r0):.4f} Dice {r0[:, 3].mean():.4f} | LEARNED per-clip b F1 {f1_only(rs):.4f} "
              f"({f1_only(rs) - f1_only(r0):+.4f}) Dice {rs[:, 3].mean():.4f} ({rs[:, 3].mean() - r0[:, 3].mean():+.4f}) | "
              f"oracle F1 {f1_only(ro):.4f} Dice {ro[:, 3].mean():.4f}", flush=True)
        print(f"   chosen b distribution {np.bincount(pick[mask], minlength=nB)} (grid {BIASES})", flush=True)


if __name__ == "__main__":
    main()
