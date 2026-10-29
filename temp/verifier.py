"""Phase 2 event verifier, v1: learned DROP and MERGE decisions on top of the v6 events.

Real-test probes showed both actions pay: dropping refiner-unsupported events (P1 +0.0043),
merging fragments one refiner event spans (P2 +0.0054, F1 AND Dice up) and a 15-feature FP
classifier (P3 +0.0030, although local CV predicted ~0). This verifier learns both decisions
from richer evidence, each a gradient-boosted classifier:

  DROP  (per predicted event)    label = overlaps no reference (pure false positive)
  MERGE (per adjacent event pair) label = most of the gap lies inside a reference
                                          (the gap is real noise -> merging raises Dice and lets
                                          one event match the reference)

Features (all from models that never trained on Gold held-out / validation, so they mean the
same on test): the 15 temp/fp_filter event features; PCA of the 5 v4 members' frame embeddings
pooled inside the span (and in the gap for pairs); member-logit statistics; and the opinion of
an independent detector -- the anchor SEDT (Gold-train only), via its best-overlapping confident
event (confidence, IoU) and, for pairs, whether one SEDT event spans both.

Training pool: Gold held-out + all validation (gated Natural-looking clips), 5-fold CV grouped
by clip for out-of-fold scores and thresholds. Test: applied to v6's own events.

Usage: python -m temp.verifier [--drop-thr X] [--merge-thr Y] [--base submission_probe_P6.zip]
"""
import argparse
import json
import zipfile

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.edge_correction import correct_events
from src.sedt import SEDT
from temp.fp_filter import TABLE, features as fp_features, gate_of, overlap
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S
from temp.train_sedt import predict_all

SEDT_CKPT = "./checkpoints/sedt_gold_anc_s0_alone.pth"
N_PCA = 24
MAX_GAP = 3.0


def sedt_predictions(tok, mel, device="cuda:0"):
    ck = torch.load(SEDT_CKPT, map_location="cpu", weights_only=False)
    model = SEDT(**ck.get("kwargs", {})).to(device)
    model.load_state_dict(ck["state"])
    out = []
    for i in range(0, len(tok), 512):
        t = torch.from_numpy(np.asarray(tok[i:i + 512])).to(device)
        m = torch.from_numpy(np.asarray(mel[i:i + 512])).to(device)
        out += predict_all(model, t, m)
    return out


def load_set(name):
    """-> dict with v4 logits L (N,5,T), prob P, energy E, embeddings emb (N,5,250,144),
    SEDT events, refs (None for test), base events, gate."""
    if name == "test":
        L = np.load(f"{S}/verifier/test_logits.npy").astype(np.float32)[:, 5:10]
        mel = np.load(f"{S}/verifier/test_mel.npy")
        tok = np.load(f"{S}/verifier/test_tok.npy", mmap_mode="r")
        refs = None
        L6 = np.load(f"{S}/verifier/test_logits.npy").astype(np.float32)
        P6 = 1 / (1 + np.exp(-L6.mean(1)))
        gate = gate_of(P6, viterbi_events(P6))
        ids = json.load(open(f"{S}/verifier/test_ids.json"))
        base = None
    else:
        d = np.load(f"{S}/refiner_{name}.npz")
        L = d["logits"].astype(np.float32)
        mel = np.load(f"{S}/sedt/{name}_mel.npy")
        tok = np.load(f"{S}/sedt/{name}_tok.npy", mmap_mode="r")
        refs = refs_of(d["mask"].astype(np.float32))
        P4 = 1 / (1 + np.exp(-L.mean(1)))
        vit = viterbi_events(P4)
        base = [correct_events(e, p, TABLE) for e, p in zip(vit, P4)]
        gate = gate_of(P4, vit)
        ids = json.load(open(f"{S}/sedt/{name}_ids.json"))
    P = 1 / (1 + np.exp(-L.mean(1)))
    emb = np.load(f"{S}/verifier/{name}_emb.npy", mmap_mode="r")
    return dict(name=name, L=L, P=P, E=mel.astype(np.float32).mean(1), emb=emb, sedt=sedt_predictions(tok, mel),
                refs=refs, base=base, gate=gate, ids=ids)


def pooled(emb_i, a, b):
    """Mean of the 5 members' 144-d embeddings over [a, b) seconds (25 Hz) -> (720,)."""
    n = emb_i.shape[1]  # 250 for 10 s clips; longer for full-length long clips
    ia, ib = int(a * 25), max(int(np.ceil(b * 25)), int(a * 25) + 1)
    ia, ib = min(ia, n - 1), min(max(ib, ia + 1), n)
    return np.asarray(emb_i[:, ia:ib], np.float32).mean(1).reshape(-1)


def iou(x, y):
    inter = max(0.0, min(x[1], y[1]) - max(x[0], y[0]))
    return inter / max(max(x[1], y[1]) - min(x[0], y[0]), 1e-6)


def sedt_support(span, sedt_ev, c=0.5):
    hits = [(cf, iou(span, (a, b))) for a, b, cf in sedt_ev if overlap(span, (a, b))]
    conf_hits = [h for h in hits if h[0] >= c]
    return [max([h[0] for h in hits], default=0.0), max([h[1] for h in conf_hits], default=0.0), len(conf_hits)]


def event_rows(D, i, ev):
    rows = []
    for k, (a, b) in enumerate(ev):
        rows.append((fp_features(D["L"][i], D["P"][i], D["E"][i], ev, k) + sedt_support((a, b), D["sedt"][i]),
                     pooled(D["emb"][i], a, b)))
    return rows


def pair_rows(D, i, ev):
    rows = []
    P, L = D["P"][i], D["L"][i]
    for k in range(len(ev) - 1):
        (a1, b1), (a2, b2) = ev[k], ev[k + 1]
        gap = a2 - b1
        if gap > MAX_GAP or gap <= 0:
            continue
        ga, gb = int(b1 * 100), max(int(a2 * 100), int(b1 * 100) + 1)
        pg, lg = P[ga:gb], L[:, ga:gb].mean(1)
        spanning = [cf for a, b, cf in D["sedt"][i] if a <= b1 + 0.05 and b >= a2 - 0.05 and overlap((a, b), ev[k]) and overlap((a, b), ev[k + 1])]
        e_gap, e_ev = D["E"][i][ga:gb].mean(), np.r_[D["E"][i][int(a1 * 100):int(b1 * 100)], D["E"][i][int(a2 * 100):int(b2 * 100)]]
        rows.append(([gap, b1 - a1, b2 - a2, pg.mean(), pg.max(), pg.min(), (lg > 0).mean(), lg.mean(),
                      max(spanning, default=0.0), len(spanning), e_gap - (e_ev.mean() if len(e_ev) else e_gap), a1, len(ev)],
                     pooled(D["emb"][i], b1, a2), k))
    return rows


def gap_in_refs(refs, b1, a2):
    n = max(int(round((a2 - b1) * 100)), 1)
    t = b1 + (np.arange(n) + 0.5) / 100
    return float(np.mean([any(r[0] <= x < r[1] for r in refs) for x in t]))


def build_training(sets):
    ev_X, ev_E, ev_y, ev_key, pr_X, pr_E, pr_y, pr_key = [], [], [], [], [], [], [], []
    for D in sets:
        for i, (ev, g) in enumerate(zip(D["base"], D["gate"])):
            if not g:
                continue
            for k, (f, e) in enumerate(event_rows(D, i, ev)):
                ev_X.append(f); ev_E.append(e)
                ev_y.append(0 if any(overlap(ev[k], r) for r in D["refs"][i]) else 1)
                ev_key.append((D["name"], i, k))
            for f, e, k in pair_rows(D, i, ev):
                pr_X.append(f); pr_E.append(e)
                pr_y.append(int(gap_in_refs(D["refs"][i], ev[k][1], ev[k + 1][0]) >= 0.5))
                pr_key.append((D["name"], i, k))
    return (np.array(ev_X, np.float32), np.array(ev_E, np.float32), np.array(ev_y), ev_key,
            np.array(pr_X, np.float32), np.array(pr_E, np.float32), np.array(pr_y), pr_key)


def gbm():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, min_samples_leaf=30,
                                          l2_regularization=1.0, random_state=0)


def oof_scores(X, y, key):
    groups = np.array([f"{k[0]}:{k[1]}" for k in key])
    ug = np.unique(groups)
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(ug):
        m = np.isin(groups, ug[tr])
        oof[~m] = gbm().fit(X[m], y[m]).predict_proba(X[~m])[:, 1]
    return oof


def apply_decisions(base, drop, merge):
    """drop: set of (i, k); merge: set of (i, k) meaning merge event k with k+1."""
    out = []
    for i, ev in enumerate(base):
        new, k = [], 0
        while k < len(ev):
            a, b = ev[k]
            while (i, k) in merge and k + 1 < len(ev):
                k += 1
                b = max(b, ev[k][1])
            if not ((i, k) in drop and ev[k] == (a, b)):  # never drop a merged event
                new.append((a, b))
            k += 1
        out.append(new)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-thr", type=float, nargs="*", default=[0.4, 0.5])
    ap.add_argument("--merge-thr", type=float, nargs="*", default=[0.5, 0.6])
    ap.add_argument("--base", default="submission_probe_P6.zip", help="test base the verifier edits (v6 or a probe)")
    ap.add_argument("--tag", default="v8", help="output name prefix (never overwrite a submitted zip)")
    args = ap.parse_args()
    sets = [load_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")]
    ev_X, ev_E, ev_y, ev_key, pr_X, pr_E, pr_y, pr_key = build_training(sets)
    pca_e = PCA(N_PCA, random_state=0).fit(ev_E)
    pca_p = PCA(N_PCA, random_state=0).fit(pr_E)
    XE = np.c_[ev_X, pca_e.transform(ev_E)]
    XP = np.c_[pr_X, pca_p.transform(pr_E)]
    oe, op = oof_scores(XE, ev_y, ev_key), oof_scores(XP, pr_y, pr_key)
    base_only = oof_scores(ev_X[:, :15], ev_y, ev_key)
    print(f"DROP : {len(ev_y)} events, pure-FP rate {ev_y.mean():.1%}, OOF AUC {roc_auc_score(ev_y, oe):.3f} "
          f"(15 fp_filter features only: {roc_auc_score(ev_y, base_only):.3f})", flush=True)
    print(f"MERGE: {len(pr_y)} adjacent pairs (gap <= {MAX_GAP}s), gap-is-noise rate {pr_y.mean():.1%}, "
          f"OOF AUC {roc_auc_score(pr_y, op):.3f}", flush=True)

    # local out-of-fold effect of each decision rule (validation + Gold held-out)
    for dt in [None] + args.drop_thr:
        for mt in [None] + args.merge_thr:
            tot = []
            for D in sets:
                drop = {(k[1], k[2]) for k, s in zip(ev_key, oe) if k[0] == D["name"] and dt is not None and s > dt}
                merge = {(k[1], k[2]) for k, s in zip(pr_key, op) if k[0] == D["name"] and mt is not None and s > mt}
                out = apply_decisions(D["base"], drop, merge)
                r0, r1 = per_clip(D["refs"], D["base"]), per_clip(D["refs"], out)
                tot.append((D["name"], f1_only(r1) - f1_only(r0), r1[:, 3].mean() - r0[:, 3].mean()))
            print(f"  OOF drop>{dt} merge>{mt}: " + " | ".join(f"{n} F1 {f:+.4f} Dice {d:+.4f}" for n, f, d in tot), flush=True)

    # ---- test ----
    me, mp = gbm().fit(XE, ev_y), gbm().fit(XP, pr_y)
    T = load_set("test")
    rows = [json.loads(l) for l in zipfile.ZipFile(args.base).read("predictions.jsonl").decode().splitlines()]
    base = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}
    idx = {c: i for i, c in enumerate(T["ids"])}
    scores_e, scores_p = {}, {}
    for cid, i in idx.items():
        ev = base[cid]
        if not ev or not T["gate"][i]:
            continue
        er = event_rows(T, i, ev)
        se = me.predict_proba(np.c_[np.array([f for f, _ in er], np.float32), pca_e.transform(np.array([e for _, e in er]))])[:, 1]
        scores_e[cid] = se
        pr = pair_rows(T, i, ev)
        if pr:
            sp = mp.predict_proba(np.c_[np.array([f for f, _, _ in pr], np.float32), pca_p.transform(np.array([e for _, e, _ in pr]))])[:, 1]
            scores_p[cid] = [(k, s) for (_, _, k), s in zip(pr, sp)]
    for dt in args.drop_thr:
        for mt in args.merge_thr:
            out = {}
            for cid in base:
                if cid not in scores_e:
                    out[cid] = base[cid]
                    continue
                drop = {(0, k) for k, s in enumerate(scores_e[cid]) if s > dt}
                merge = {(0, k) for k, s in scores_p.get(cid, []) if s > mt}
                out[cid] = apply_decisions([base[cid]], drop, merge)[0]
            assert all(0 <= a < b for ev in out.values() for a, b in ev)
            n0, n1 = sum(map(len, base.values())), sum(map(len, out.values()))
            name = f"submission_{args.tag}_d{int(dt * 100)}_m{int(mt * 100)}.zip"
            assert args.tag != "v8" or args.base == "submission_probe_P6.zip", "pick a new --tag for a new base"
            with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("predictions.jsonl", "".join(json.dumps({"clip_id": r["clip_id"], "events": [
                    {"onset": round(a, 3), "offset": round(b, 3)} for a, b in out[r["clip_id"]]]}) + "\n" for r in rows))
            print(f"test ({args.base} + verifier) drop>{dt} merge>{mt}: events {n0} -> {n1} "
                  f"(clips changed {sum(out[c] != base[c] for c in base)}) -> {name}", flush=True)


if __name__ == "__main__":
    main()
