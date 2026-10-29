"""Oct 14, residual Step 0 tests (CPU). Speech-enhancement residual r = x - s_hat (public pretrained
MossFormerGAN_SE_16K, temp/cache_residual.py) as new evidence for Natural event edges.

(a) edge evidence: spread (IQR) of the relative energy level at reference edges, mixture vs residual,
    and the edge contrast (mean dB 100 ms inside minus 100 ms outside the edge);
(b) residual edge snap: unguarded short events (< 1 s, mean prob < 0.70) get each edge moved to the
    residual-energy crossing (level lam between floor and event peak) within +/- cap; lam and cap fitted
    on Gold held-out, scored once on validation Natural + Synthetic (paired bootstrap);
(c) verifier v4: verifier v1 features + residual features per event, grouped 5-fold OOF vs v1.
Pre-registered: (b) >= +0.002 pooled with synthetic within +/-0.0005, or (c) AUC >= v1 + 0.01 and pooled
>= +0.001 (drops at 0.35 under the 0.70 guard).

Usage: python -m temp.residual_tests
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import temp.verifier as vf
from src.edge_correction import correct_events, load_table
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events

W = "./work"
T5 = load_table("./checkpoints/edge_correction_v5.json")
SETS = ("gold_heldout", "official_natural", "official_synthetic")


def energy(mel):
    return mel.astype(np.float32).mean(0)  # mean dB over 80 bands, (T,)


def floor_of(e):
    v = e[e > e.min() + 1e-3]
    return np.percentile(v, 10) if len(v) > 10 else e.min()


def load(n):
    d = np.load(f"{W}/refiner_{n}.npz")
    P = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
    base = [correct_events(e, p, T5) for e, p in zip(viterbi_events(P), P)]
    return dict(P=P, refs=refs_of(d["mask"]), base=base, mix=d["mel"], res=np.load(f"{W}/residual/{n}_res_mel.npy"),
                spe=np.load(f"{W}/residual/{n}_spe_mel.npy"))


def edge_evidence(D):
    out = {}
    for kind in ("mix", "res"):
        lv, ct = [], []
        for i, refs in enumerate(D["refs"]):
            e = energy(D[kind][i]); fl = floor_of(e)
            for on, off in refs:
                a, b = int(round(on * 100)), int(round(off * 100))
                if not (10 < a < b < 990) or off - on >= 1.0:
                    continue
                pk = e[a:b].max()
                for idx, inside, outside in ((a, e[a:a + 10], e[a - 10:a]), (b - 1, e[b - 10:b], e[b:b + 10])):
                    lv.append((e[idx] - fl) / max(pk - fl, 1e-3))
                    ct.append(inside.mean() - outside.mean())
        lv, ct = np.array(lv), np.array(ct)
        out[kind] = (np.percentile(lv, 25), np.percentile(lv, 75), np.median(ct))
    return out


def snap(D, lam, cap):
    new = []
    for i, ev in enumerate(D["base"]):
        P, e = D["P"][i], energy(D["res"][i])
        fl, out = floor_of(e), []
        for a, b in ev:
            ia, ib = int(round(a * 100)), int(round(b * 100))
            conf = P[ia:max(ib, ia + 1)].mean()
            if b - a >= 1.0 or conf >= 0.70 or ib - ia < 3:
                out.append((a, b)); continue
            pk = e[ia:ib].max()
            thr = fl + lam * (pk - fl)
            lo, hi = max(0, ia - cap), min(len(e), ib + cap)
            above = np.where(e[lo:hi] >= thr)[0] + lo
            if len(above) == 0:
                out.append((a, b)); continue
            na = min(max(above[0], ia - cap), ia + cap)
            nb = max(min(above[-1] + 1, ib + cap), ib - cap)
            out.append((na / 100, nb / 100) if nb - na >= 3 else (a, b))
        new.append(out)
    return new


def res_features(D, i, a, b):
    e, s, P = energy(D["res"][i]), energy(D["spe"][i]), D["P"][i]
    T = len(e)
    ia, ib = min(int(a * 100), T - 1), min(max(int(b * 100), int(a * 100) + 1), T)
    ma, mb = max(0, ia - 20), min(T, ib + 20)
    inside, margin = e[ia:ib], np.r_[e[ma:ia], e[ib:mb]]
    fl = floor_of(e)
    return [inside.mean() - fl, inside.max() - fl, (margin.mean() if len(margin) else fl) - fl,
            inside.mean() - (margin.mean() if len(margin) else fl), (e[ia:ib] - s[ia:ib]).mean(),
            e[min(ia + 5, T - 1)] - e[max(ia - 5, 0)], e[max(ib - 5, 0)] - e[min(ib + 5, T - 1)]]


def main():
    D = {n: load(n) for n in SETS}
    print("(a) edge evidence on short refs (relative level IQR; median edge contrast dB):", flush=True)
    for n in ("gold_heldout", "official_natural"):
        ev = edge_evidence(D[n])
        print(f"   {n}: mixture IQR [{ev['mix'][0]:.2f}, {ev['mix'][1]:.2f}] contrast {ev['mix'][2]:.2f} | "
              f"residual IQR [{ev['res'][0]:.2f}, {ev['res'][1]:.2f}] contrast {ev['res'][2]:.2f}", flush=True)

    comb = lambda r: f1_only(r) + r[:, 3].mean()
    g0 = comb(per_clip(D["gold_heldout"]["refs"], D["gold_heldout"]["base"]))
    best = max(((comb(per_clip(D["gold_heldout"]["refs"], snap(D["gold_heldout"], lam, cap))), lam, cap)
                for lam in (0.2, 0.3, 0.4, 0.5, 0.6) for cap in (5, 10, 15)), key=lambda x: x[0])
    print(f"(b) snap fit on Gold held-out: base {g0:.4f} -> {best[0]:.4f} at lam {best[1]} cap {best[2] * 10} ms", flush=True)
    rn0 = per_clip(D["official_natural"]["refs"], D["official_natural"]["base"])
    rs0 = per_clip(D["official_synthetic"]["refs"], D["official_synthetic"]["base"])
    rn1 = per_clip(D["official_natural"]["refs"], snap(D["official_natural"], best[1], best[2]))
    rs1 = per_clip(D["official_synthetic"]["refs"], snap(D["official_synthetic"], best[1], best[2]))
    b0, b1 = comb(np.r_[rn0, rs0]), comb(np.r_[rn1, rs1])
    rng = np.random.default_rng(0)
    boot = [comb(np.r_[rn1[i], rs1[j]]) - comb(np.r_[rn0[i], rs0[j]]) for i, j in
            ((rng.integers(0, len(rn0), len(rn0)), rng.integers(0, len(rs0), len(rs0))) for _ in range(1000))]
    print(f"(b) VALIDATION snap: pooled {b0:.4f} -> {b1:.4f} ({b1 - b0:+.4f}, P>0 {np.mean(np.array(boot) > 0):.3f}) | "
          f"Nat F1 {f1_only(rn0):.4f}->{f1_only(rn1):.4f} Dice {rn0[:, 3].mean():.4f}->{rn1[:, 3].mean():.4f} | "
          f"Syn F1 {f1_only(rs0):.4f}->{f1_only(rs1):.4f} Dice {rs0[:, 3].mean():.4f}->{rs1[:, 3].mean():.4f}", flush=True)

    sets = [vf.load_set(n) for n in SETS]
    ev_X, ev_E, ev_y, ev_key = vf.build_training(sets)[:4]
    by = {s["name"]: s for s in sets}
    ev_R = np.array([res_features(D[n], i, *by[n]["base"][i][k]) for n, i, k in ev_key], np.float32)
    pca = PCA(vf.N_PCA, random_state=0).fit(ev_E)
    X1 = np.c_[ev_X, pca.transform(ev_E)]
    X4 = np.c_[X1, ev_R]
    o1, o4 = vf.oof_scores(X1, ev_y, ev_key), vf.oof_scores(X4, ev_y, ev_key)
    a1, a4 = roc_auc_score(ev_y, o1), roc_auc_score(ev_y, o4)
    score = {}
    for tag, o in (("v1", o1), ("v4", o4)):
        drop = {key for key, s in zip(ev_key, o) if s > 0.35}
        rows = {}
        for s in sets[1:]:
            evs = []
            for i, ev in enumerate(s["base"]):
                P = s["P"][i]
                conf = lambda e: float(P[int(e[0] * 100):max(int(e[1] * 100), int(e[0] * 100) + 1)].mean())
                evs.append([e for k, e in enumerate(ev) if (s["name"], i, k) not in drop or conf(e) >= 0.70])
            rows[s["name"]] = per_clip(s["refs"], evs)
        score[tag] = comb(np.r_[rows["official_natural"], rows["official_synthetic"]])
    print(f"(c) verifier: OOF AUC v1 {a1:.4f} v4(+residual) {a4:.4f} | pooled v1 {score['v1']:.4f} v4 {score['v4']:.4f} "
          f"({score['v4'] - score['v1']:+.4f})", flush=True)
    syn_ok = abs(f1_only(rs1) - f1_only(rs0)) <= 0.0005 and abs(rs1[:, 3].mean() - rs0[:, 3].mean()) <= 0.0005
    gate_b = b1 - b0 >= 0.002 and syn_ok
    gate_c = a4 - a1 >= 0.01 and score["v4"] - score["v1"] >= 0.001
    print(f"GATE: (b) {'PASS' if gate_b else 'FAIL'} | (c) {'PASS' if gate_c else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
