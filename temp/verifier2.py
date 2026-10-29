"""Verifier v2: verifier v1 + the refiner-ensemble votes as features.

Real results: refiner support (P1 drops, P2 merges, majority-vote merges) and the learned
verifier's drops (v8_d40_m60, +0.0014 over P6) each add. v2 lets one model weigh all of it:
per event the fraction of refiners that support it and their IoU with it; per adjacent pair the
fraction of refiners with one event spanning both. Votes are fractions, so the 3 out-of-fold
seeds used on validation and the 9 refiners used on test mean the same thing.

Out-of-sample refiner opinions everywhere:
  gold_heldout       -- all-validation refiners (trained on official validation only)
  official_*         -- the 5-fold cross-fit refiner seeds 0-2 (scratchpad/crossfit)
  test               -- all refiners (temp.refiner_votes.REFINERS)

Usage: python -m temp.verifier2 --base submission_v8_combo.zip --tag v9b --drop-thr 0.35 0.4 0.5
"""
import argparse
import glob
import json
import zipfile

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from src.edge_correction import correct_events
from src.refiner import BoundaryRefiner, decode_proposals
from temp.crossfit_refiner import OUT as CF_OUT, all_data
from temp.eval_crossfit_refiner import durations
from temp.fp_filter import overlap
from temp.refiner_votes import REFINERS, refiner_test_events
from temp.train_refiner_val import S
import temp.verifier as v1

ALL_VAL = [p for p in REFINERS if "train70" not in p]


def iou(x, y):
    inter = max(0.0, min(x[1], y[1]) - max(x[0], y[0]))
    return inter / max(max(x[1], y[1]) - min(x[0], y[0]), 1e-6)


def decode(pres, d_on, d_off, prob, sec=10.0):
    ev = correct_events(decode_proposals(pres, d_on, d_off, 0.5), prob, v1.TABLE)
    ev = [(a, min(b, sec)) for a, b in ev]
    return [(round(a, 3), round(b, 3)) for a, b in ev if b - a >= 0.02]


def voters_gold():
    d = np.load(f"{S}/refiner_gold_heldout.npz")
    L, M = d["logits"].astype(np.float32), d["mel"].astype(np.float32)
    P = 1 / (1 + np.exp(-L.mean(1)))
    out = [[] for _ in range(len(L))]
    for path in ALL_VAL:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        m = BoundaryRefiner(n_members=5).to("cuda:0")
        m.load_state_dict(ck["state"])
        m.eval()
        with torch.no_grad():
            for i in range(0, len(L), 128):
                o = m(torch.from_numpy(L[i:i + 128]).cuda(), torch.from_numpy(M[i:i + 128]).cuda())
                pr, a, b = torch.sigmoid(o["presence"]).cpu().numpy(), o["d_on"].cpu().numpy(), o["d_off"].cpu().numpy()
                for j in range(len(pr)):
                    out[i + j].append(decode(pr[j], a[j], b[j], P[i + j]))
    return out


def voters_val():
    """Cross-fit (out-of-fold) refiner seeds 0-2, re-indexed to scratchpad/refiner_official_* order."""
    L, _, _, _, _ = all_data()
    N = len(L)
    sec = durations()
    P = 1 / (1 + np.exp(-L.mean(1)))
    split = json.load(open("./data/val_split_s0.json"))
    names = []
    for sub in ("natural", "synthetic"):
        ids = [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]]
        for part in ("train70", "hold30"):
            keep = set(split[part][sub])
            names += [(sub, n) for n in ids if n in keep]
    per = [[] for _ in range(N)]
    for s in range(3):
        R = {k: np.zeros((N, 1000), np.float32) for k in ("presence", "d_on", "d_off")}
        for f in glob.glob(f"{CF_OUT}/f*_s{s}.npz"):
            d = np.load(f)
            for k in R:
                R[k][d["idx"]] = d[k]
        for i in range(N):
            per[i].append(decode(R["presence"][i], R["d_on"][i], R["d_off"][i], P[i], sec[i]))
    out = {}
    for sub in ("natural", "synthetic"):
        ids = [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]]
        pos = {n: i for i, (sb, n) in enumerate(names) if sb == sub}
        out[f"official_{sub}"] = [per[pos[n]] for n in ids]
    return out


def vote_event(span, voters):
    K = max(len(voters), 1)
    best = [max([iou(span, f) for f in v if overlap(span, f)], default=0.0) for v in voters]
    return [sum(b > 0 for b in best) / K, float(np.mean(best)) if best else 0.0, max(best, default=0.0)]


def vote_pair(e1, e2, voters):
    K = max(len(voters), 1)
    span = sum(any(overlap(e1, f) and overlap(e2, f) for f in v) for v in voters) / K
    both = sum(any(overlap(e1, f) for f in v) and any(overlap(e2, f) for f in v) for v in voters) / K
    return [span, both]


def patch_rows(voters_by_set):
    """Wrap verifier v1's row builders so they append the vote features."""
    ev_rows, pr_rows = v1.event_rows, v1.pair_rows

    def event_rows(D, i, ev):
        V = voters_by_set[D["name"]][i] or []
        return [(f + vote_event(ev[k], V), e) for k, (f, e) in enumerate(ev_rows(D, i, ev))]

    def pair_rows(D, i, ev):
        V = voters_by_set[D["name"]][i] or []
        return [(f + vote_pair(ev[k], ev[k + 1], V), e, k) for f, e, k in pr_rows(D, i, ev)]

    v1.event_rows, v1.pair_rows = event_rows, pair_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submission_v8_combo.zip")
    ap.add_argument("--tag", default="v9b")
    ap.add_argument("--drop-thr", type=float, nargs="+", default=[0.35, 0.4, 0.5])
    ap.add_argument("--merge-thr", type=float, nargs="+", default=[0.6])
    args = ap.parse_args()
    assert args.tag not in ("v8", "v9a"), "never overwrite submitted zips"
    ids, R, gate = refiner_test_events()
    print(f"test voters: {len(R)} refiners", flush=True)
    voters = {"gold_heldout": voters_gold(), **voters_val(),
              "test": [[R[p][i] or [] for p in REFINERS] if gate[i] else None for i in range(len(ids))]}
    patch_rows(voters)
    import sys
    sys.argv = ["verifier", "--base", args.base, "--tag", args.tag,
                "--drop-thr", *map(str, args.drop_thr), "--merge-thr", *map(str, args.merge_thr)]
    v1.main()


if __name__ == "__main__":
    main()
