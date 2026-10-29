"""Merge model v3: is the gap between two adjacent predicted events real noise?

Why: real probes showed merging fragments raises Event F1 AND Dice (P2 +0.0054; majority-vote
merges of 5 refiners in v8_combo better still), and on validation perfect merges on top of the
current recipe are worth another +0.025 pooled. The v1 verifier's merge head was weak (AUC 0.70,
1,866 training gaps). v3 changes:
  * data -- adjacent pairs from five decodes of the same clips (the production v5 decode, two
    less sticky Viterbi settings, p >= 0.5 and p >= 0.7 segments), several times more gaps;
  * evidence -- the refiners' raw per-frame presence inside the gap (mean / min / max, averaged
    over refiners, and the fraction of refiners that keep the whole gap above 0.5), plus the
    ensemble probability, per-member logits, embeddings and energy in the gap.
Label: >= 50% of the gap lies inside a reference event.

Refiner presence is out-of-sample everywhere: validation uses the 5-fold cross-fit seeds 0-2,
Gold held-out and test use the all-validation refiners.

Evaluation (validation, out-of-fold by clip): replace the majority-vote merges of the local
current-recipe analog (majority merges + drop if no seed supports; +0.0126 pooled over the v5
decode) with learned merges at several thresholds.

Test: `--write` builds v6 + learned merges + the v9a_d35 drops (real-proven P1 + P3 + verifier
v1 > 0.35), i.e. v9a_d35_m60 with its merge component replaced.

Usage: python -m temp.merge_model [--write --thr 0.5 0.6]
"""
import argparse
import glob
import json
import zipfile

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from release.v6_inference import MEMBERS
from release.v7_inference import REFINER_ORDER
from src.edge_correction import correct_events
from src.refiner import BoundaryRefiner
from temp.crossfit_refiner import OUT as CF_OUT, all_data
from temp.fp_filter import TABLE, overlap
from temp.train_refiner import f1_only, per_clip, viterbi_events
from temp.train_refiner_val import S
from temp.verifier import gap_in_refs, load_set, pooled
from temp.verifier2 import ALL_VAL, voters_val

MAX_GAP = 3.0
N_PCA = 16


def presence_all_val(L, M, device="cuda:0"):
    """(N, K, T) presence of the all-validation refiners; L in refiner channel order."""
    out = np.zeros((len(L), len(ALL_VAL), L.shape[-1]), np.float16)
    for r, path in enumerate(ALL_VAL):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        m = BoundaryRefiner(n_members=5).to(device)
        m.load_state_dict(ck["state"])
        m.eval()
        with torch.no_grad():
            for i in range(0, len(L), 128):
                o = m(torch.from_numpy(np.ascontiguousarray(L[i:i + 128])).to(device),
                      torch.from_numpy(np.ascontiguousarray(M[i:i + 128])).to(device))
                out[i:i + 128, r] = torch.sigmoid(o["presence"]).cpu().numpy()
    return out


def presence_val():
    """Cross-fit presence (N, 3, T) re-indexed to scratchpad/refiner_official_* order."""
    L, _, _, _, _ = all_data()
    N = len(L)
    split = json.load(open("./data/val_split_s0.json"))
    names = []
    for sub in ("natural", "synthetic"):
        ids = [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]]
        for part in ("train70", "hold30"):
            keep = set(split[part][sub])
            names += [(sub, n) for n in ids if n in keep]
    pres = np.zeros((N, 3, 1000), np.float16)
    for s in range(3):
        for f in glob.glob(f"{CF_OUT}/f*_s{s}.npz"):
            d = np.load(f)
            pres[d["idx"], s] = d["presence"]
    out = {}
    for sub in ("natural", "synthetic"):
        ids = [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]]
        pos = {n: i for i, (sb, n) in enumerate(names) if sb == sub}
        out[f"official_{sub}"] = pres[[pos[n] for n in ids]]
    return out


def segments(p, thr):
    m = np.r_[False, p >= thr, False]
    d = np.diff(m.astype(int))
    return [(a / 100, b / 100) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= 2]


def decodes(P):
    """Five decodes per clip; index 0 is the production v5 decode (with the edge correction)."""
    v0 = viterbi_events(P)
    out = [[correct_events(e, p, TABLE) for e, p in zip(v0, P)],
           viterbi_events(P, 0.98, 0.90), viterbi_events(P, 0.95, 0.80),
           [segments(p, 0.5) for p in P], [segments(p, 0.7) for p in P]]
    return out


def pair_feats(D, pres, i, e1, e2):
    b1, a2 = e1[1], e2[0]
    ga, gb = int(round(b1 * 100)), max(int(round(a2 * 100)), int(round(b1 * 100)) + 1)
    gb = min(gb, 1000)
    ga = min(ga, gb - 1)
    pg, lg = D["P"][i][ga:gb], D["L"][i][:, ga:gb]
    rp = pres[i][:, ga:gb].astype(np.float32)  # (K, gap)
    E = D["E"][i]
    ev_e = np.r_[E[int(e1[0] * 100):int(b1 * 100)], E[int(a2 * 100):int(e2[1] * 100)]]
    sp = [cf for a, b, cf in D["sedt"][i] if a <= b1 + 0.05 and b >= a2 - 0.05]
    f = [a2 - b1, e1[1] - e1[0], e2[1] - e2[0], pg.mean(), pg.max(), pg.min(), np.median(pg), (pg > 0.3).mean(),
         *lg.mean(1), (lg.mean(1) > 0).mean(), lg.std(0).mean(),
         rp.mean(), rp.min(1).mean(), rp.max(1).mean(), (rp.min(1) > 0.5).mean(), (rp.mean(1) > 0.5).mean(),
         E[ga:gb].mean() - (ev_e.mean() if len(ev_e) else E[ga:gb].mean()), max(sp, default=0.0), e1[0]]
    return f, pooled(D["emb"][i], b1, max(a2, b1 + 0.04))


def build(D, pres, with_labels=True):
    X, Em, y, key = [], [], [], []
    decs = decodes(D["P"]) if D["base"] is not None else None
    for i, g in enumerate(D["gate"]):
        if not g:
            continue
        seen = set()
        for di, dec in enumerate(decs):
            ev = sorted(dec[i])
            for k in range(len(ev) - 1):
                gap = ev[k + 1][0] - ev[k][1]
                sig = (round(ev[k][1], 2), round(ev[k + 1][0], 2))
                if gap <= 0 or gap > MAX_GAP or sig in seen:
                    continue
                seen.add(sig)
                f, e = pair_feats(D, pres, i, ev[k], ev[k + 1])
                X.append(f); Em.append(e)
                y.append(int(gap_in_refs(D["refs"][i], ev[k][1], ev[k + 1][0]) >= 0.5))
                key.append((D["name"], i, di, k))
    return np.array(X, np.float32), np.array(Em, np.float32), np.array(y), key


def gbm():
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)


def recipe(base_ev, R3, merge_pairs):
    """Local current-recipe analog with a given merge set: merge pairs, then drop events no seed supports."""
    sup = [sum(any(overlap(e, f) for f in R3[s]) for s in range(len(R3))) for e in base_ev]
    m, sp = [], []
    for k, e in enumerate(base_ev):
        if m and (k - 1) in merge_pairs:
            m[-1] = (m[-1][0], max(m[-1][1], e[1])); sp[-1] = 1
        else:
            m.append(e); sp.append(sup[k])
    return [e for e, s in zip(m, sp) if s > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    args = ap.parse_args()
    sets = [load_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")]
    gd = np.load(f"{S}/refiner_gold_heldout.npz")
    pres = {"gold_heldout": presence_all_val(gd["logits"].astype(np.float32), gd["mel"].astype(np.float32)), **presence_val()}
    parts = [build(D, pres[D["name"]]) for D in sets]
    X = np.concatenate([p[0] for p in parts]); Em = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts]); key = sum((p[3] for p in parts), [])
    pca = PCA(N_PCA, random_state=0).fit(Em)
    XF = np.c_[X, pca.transform(Em)]
    groups = np.array([f"{k[0]}:{k[1]}" for k in key])
    ug = np.unique(groups)
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(ug):
        m = np.isin(groups, ug[tr])
        oof[~m] = gbm().fit(XF[m], y[m]).predict_proba(XF[~m])[:, 1]
    prod = np.array([k[2] == 0 for k in key])
    print(f"pairs: {len(y)} from 5 decodes ({prod.sum()} from the production decode); gap-is-noise rate {y.mean():.1%}", flush=True)
    print(f"OOF AUC all pairs {roc_auc_score(y, oof):.3f} | production-decode pairs {roc_auc_score(y[prod], oof[prod]):.3f} "
          f"(verifier v1 merge head: 0.70)", flush=True)

    # local merge effect on validation, replacing the majority-vote merges of the current-recipe analog
    V = voters_val()
    for D in sets[1:]:
        name = D["name"]
        R3 = V[name]
        base = D["base"]
        pk = {(k[1], k[3]): s for k, s in zip(key, oof) if k[0] == name and k[2] == 0}
        refs = D["refs"]
        r0 = per_clip(refs, base)
        comb = lambda r: f1_only(r) + r[:, 3].mean()

        def run(pairs_fn):
            out = [recipe(b, R3[i], pairs_fn(i, b)) if D["gate"][i] else b for i, b in enumerate(base)]
            return per_clip(refs, out)

        maj = run(lambda i, b: {k for k in range(len(b) - 1)
                                if sum(any(overlap(b[k], f) and overlap(b[k + 1], f) for f in R3[i][s]) for s in range(3)) >= 2})
        print(f"{name}: majority-vote merges (current) F1 {f1_only(maj) - f1_only(r0):+.4f} Dice {maj[:, 3].mean() - r0[:, 3].mean():+.4f} "
              f"comb {comb(maj) - comb(r0):+.4f}", flush=True)
        for t in args.thr:
            r = run(lambda i, b: {k for k in range(len(b) - 1) if pk.get((i, k), 0) > t})
            print(f"   learned merges > {t}: F1 {f1_only(r) - f1_only(r0):+.4f} Dice {r[:, 3].mean() - r0[:, 3].mean():+.4f} "
                  f"comb {comb(r) - comb(r0):+.4f}  (vs majority {comb(r) - comb(maj):+.4f})", flush=True)

    if not args.write:
        return
    model = gbm().fit(XF, y)
    T = load_set("test")
    names = [n for n, _, _, _ in MEMBERS]
    ridx = [names.index(n) for n in REFINER_ORDER]
    Lr = np.load(f"{S}/verifier/test_logits.npy").astype(np.float32)[:, ridx]
    tpres = presence_all_val(Lr, np.load(f"{S}/verifier/test_mel.npy").astype(np.float32))

    def load_zip(p):
        rows = [json.loads(l) for l in zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines()]
        return {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}, [r["clip_id"] for r in rows]

    v6, order = load_zip("submission_v6.zip")
    best, _ = load_zip("submission_v9a_d35_m60.zip")
    v8c, _ = load_zip("submission_v8_combo.zip")
    idx = {c: i for i, c in enumerate(T["ids"])}
    for t in args.thr:
        out, merges = {}, 0
        for c in order:
            ev = v6[c]
            # the drop set of the current best: v6 events absent from v9a_d35 and not inside a v8_combo merge
            kept_or_merged = set(best[c])
            dropset = {e for e in ev if e not in best[c] and not any(overlap(e, f) for f in best[c])}
            i = idx.get(c)
            if i is None or not T["gate"][i] or len(ev) < 2:
                out[c] = [e for e in ev if e not in dropset] if dropset else best[c]
                continue
            pairs = set()
            for k in range(len(ev) - 1):
                gap = ev[k + 1][0] - ev[k][1]
                if 0 < gap <= MAX_GAP:
                    f, e = pair_feats(T, tpres, i, ev[k], ev[k + 1])
                    if model.predict_proba(np.c_[np.array([f], np.float32), pca.transform(np.array([e]))])[0, 1] > t:
                        pairs.add(k)
            m, src = [], []
            for k, e in enumerate(ev):
                if m and (k - 1) in pairs:
                    m[-1] = (m[-1][0], max(m[-1][1], e[1])); src[-1] = None; merges += 1
                else:
                    m.append(e); src.append(e)
            out[c] = [e for e, s in zip(m, src) if not (s is not None and s in dropset)]
        assert len(out) == 5517 and all(0 <= a < b for ev in out.values() for a, b in ev)
        name = f"submission_v10_merge{int(t * 100)}.zip"
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.jsonl", "".join(json.dumps({"clip_id": c, "events": [
                {"onset": a, "offset": b} for a, b in out[c]]}) + "\n" for c in order))
        diff = sum(out[c] != best[c] for c in order)
        print(f"test: learned merges > {t}: {merges} merges; events {sum(map(len, out.values()))} "
              f"(v9a_d35: {sum(map(len, best.values()))}); clips differing from v9a_d35 {diff} -> {name}", flush=True)


if __name__ == "__main__":
    main()
