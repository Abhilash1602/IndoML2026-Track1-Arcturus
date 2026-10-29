"""v8: refiner-ensemble votes on v6's events (merges by majority, drops by consensus).

Real probes: dropping v6 events one refiner does not support (P1) +0.0043; merging v6 events one
refiner event spans (P2) +0.0054 with F1 AND Dice up. Out-of-fold on validation the local analogs
track the real results (merges well, drops conservatively), and with 3 refiner seeds the best rule
is: merge an adjacent pair when a MAJORITY of refiners have one event spanning both; drop an event
only when NO refiner supports it (natural F1 +0.012, Dice -0.0002, pooled +0.0126 locally).

Here K refiners (all trained on official validation only: v7's all-validation refiner, v7b's
train70 refiner, and extra all-validation seeds) are run on the cached test features (the 5 v4
members' logits in the refiner's channel order + 80-mel; scratchpad/verifier/test_*), decoded
exactly like release/v7_inference.refiner_events, and vote on v6's own events on Natural-looking
short clips (v6-probability gate). Everything else is byte-identical to v6.

Usage: python -m temp.refiner_votes --merge-need 3 --drop-need 5 [--merge-need ... ]
"""
import argparse
import glob
import json
import os
import zipfile

import numpy as np
import soundfile as sf
import torch

from release.v7_inference import REFINER_ORDER, natural_gate, refiner_events
from release.v6_inference import MEMBERS
from src.edge_correction import load_table
from src.refiner import BoundaryRefiner
from temp.train_refiner_val import S

REFINERS = ["./checkpoints/refiner_val_all_x0.pth", "./checkpoints/refiner_val_train70_x0.pth"] + \
    sorted(glob.glob("./checkpoints/refiner_val_all_x0_s*.pth"))
DEVICE = "cuda:0"


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def refiner_test_events():
    """-> ids, {refiner path: [events per short test clip]} on gated clips (None elsewhere), gate."""
    ids = json.load(open(f"{S}/verifier/test_ids.json"))
    logits = np.load(f"{S}/verifier/test_logits.npy").astype(np.float32)
    mel = np.load(f"{S}/verifier/test_mel.npy").astype(np.float32)
    names = [n for n, _, _, _ in MEMBERS]
    ridx = [names.index(n) for n in REFINER_ORDER]
    table = load_table("./checkpoints/edge_correction_v5.json")
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("./data/test_audio/**/*.wav", recursive=True)}
    secs = []
    for c in ids:
        try:
            d = sf.info(paths[c]).frames / sf.info(paths[c]).samplerate
        except Exception:
            d = 0.0
        secs.append(min(d, 10.0) if d > 0 else None)
    P6 = 1 / (1 + np.exp(-logits.mean(1)))
    gate = [natural_gate(p, 0.01) for p in P6]
    out = {}
    for path in REFINERS:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = BoundaryRefiner(n_members=len(REFINER_ORDER), n_extra=ck["n_extra"]).to(DEVICE)
        model.load_state_dict(ck["state"])
        model.eval()
        evs = [None] * len(ids)
        with torch.no_grad():
            for i in range(0, len(ids), 128):
                o = model(torch.from_numpy(logits[i:i + 128][:, ridx]).to(DEVICE), torch.from_numpy(mel[i:i + 128]).to(DEVICE))
                pres = torch.sigmoid(o["presence"]).cpu().numpy()
                d_on, d_off = o["d_on"].cpu().numpy(), o["d_off"].cpu().numpy()
                for j in range(len(pres)):
                    if gate[i + j]:
                        evs[i + j] = [(e["onset"], e["offset"]) for e in
                                      refiner_events(pres[j], d_on[j], d_off[j], P6[i + j][:1000], table, secs[i + j])]
        out[path] = evs
        print(f"refiner {os.path.basename(path)} (decode chosen in training: {ck.get('kind')}) done", flush=True)
    return ids, out, gate


def vote(ev, refs_list, merge_need, drop_need):
    K = len(refs_list)
    pair_votes = [sum(any(overlap(ev[k], f) and overlap(ev[k + 1], f) for f in r) for r in refs_list) for k in range(len(ev) - 1)]
    support = [sum(any(overlap(e, f) for f in r) for r in refs_list) for e in ev]
    merged, sup = [], []
    for k, e in enumerate(ev):
        if merged and pair_votes[k - 1] >= merge_need:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e[1]))
            sup[-1] = K  # a merged event is supported by construction
        else:
            merged.append(e)
            sup.append(support[k])
    return [e for e, s in zip(merged, sup) if K - s < drop_need]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-need", type=int, nargs="+", default=[3])
    ap.add_argument("--drop-need", type=int, nargs="+", default=[5])
    args = ap.parse_args()
    ids, R, gate = refiner_test_events()
    K = len(R)
    rows = [json.loads(l) for l in zipfile.ZipFile("submission_v6.zip").read("predictions.jsonl").decode().splitlines()]
    v6 = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}
    # sanity: v7's refiner recomputed from the cache vs the real v7 zip (gated clips)
    v7 = {json.loads(l)["clip_id"]: [(e["onset"], e["offset"]) for e in json.loads(l)["events"]]
          for l in zipfile.ZipFile("submission_v7.zip").read("predictions.jsonl").decode().splitlines()}
    g = [i for i in range(len(ids)) if gate[i]]
    same = np.mean([R[REFINERS[0]][i] == v7[ids[i]] for i in g])
    close = np.mean([len(R[REFINERS[0]][i]) == len(v7[ids[i]]) for i in g])
    print(f"K={K} refiners; gated short clips {len(g)} (v7 used 2,792); v7 refiner recomputed: identical {same:.1%}, "
          f"same event count {close:.1%}", flush=True)
    for m in args.merge_need:
        for d in args.drop_need:
            out = dict(v6)
            for i in g:
                c = ids[i]
                out[c] = vote(v6[c], [R[p][i] for p in REFINERS], m, d)
            assert len(out) == 5517 and all(0 <= a < b for ev in out.values() for a, b in ev)
            name = f"submission_v8_vote_m{m}_d{d}.zip"
            with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("predictions.jsonl", "".join(json.dumps({"clip_id": r["clip_id"], "events": [
                    {"onset": a, "offset": b} for a, b in out[r["clip_id"]]]}) + "\n" for r in rows))
            n0, n1 = sum(map(len, v6.values())), sum(map(len, out.values()))
            print(f"merge >= {m}/{K}, drop if unsupported by >= {d}/{K}: events {n0} -> {n1} "
                  f"(clips changed {sum(out[c] != v6[c] for c in v6)}) -> {name}", flush=True)


if __name__ == "__main__":
    main()
