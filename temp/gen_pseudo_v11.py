"""Iteration-4 pseudo-labels: label the 79,526-clip Silver+Bronze pool with our full current
system (v11g_w70, real 1.5865) -- the 10-member ensemble, the domain-gated refiner blend (w 0.70),
refiner-vote merges, refiner / false-positive-filter drops and the verifier -- so that the
students learn the event-level decisions the post-processing currently has to repair (false
positives, fragments). DCASE 2024 Task 4 winner recipe (multi-iteration distillation from the
best ensemble); our i2 round gave a real +0.0033 with a much weaker teacher.

Clips are the BronzeInferenceDataset items (`row_<idx>`, padded/trimmed to the 10 s training
window), so labels align exactly with what the students see. Output (resumable, per shard):
work/pseudo_v11/shard{K}of{N}.npz with `ids` and `hard` (uint8 (n, 1000) frame masks of the final
events) and `soft` (float16 base probability after the blend).

Usage: SHARD=0 NSHARDS=4 VAANI_DEVICE=cuda:0 python -m temp.gen_pseudo_v11
"""
import os

import numpy as np
from datasets import Audio

import release.v6_inference as v6
from src.dataset import BronzeInferenceDataset, load_full_dataset
from src.edge_correction import load_table
from temp.pipeline_v11 import base_prob, extract_items, final_events, postproc_models, verifier_view

W = 0.70
OUT = os.environ.get("PSEUDO_OUT", "./work/pseudo_v11")
MEMBER_PREFIX = os.environ.get("MEMBER_PREFIX", "VA")  # VS = label with the v14 students


def main():
    k, n = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/shard{k}of{n}.npz"
    if os.path.exists(path):
        print(f"{path} exists", flush=True)
        return
    config = v6.make_config()
    full_ds = load_full_dataset().cast_column("audio", Audio(decode=False))
    ds = BronzeInferenceDataset(full_ds, config)
    idx = list(range(k, len(ds), n))
    print(f"shard {k}/{n}: {len(idx)} of {len(ds)} Silver+Bronze clips", flush=True)
    post, table = postproc_models(), load_table("./checkpoints/edge_correction_v5.json")
    ids, hard, soft = [], [], []
    stats = dict(gated=0, merges=0, drops=0, vdrops=0, vmerges=0)

    def items():
        for j in idx:
            wav, clip_id, _ = ds[j]
            yield clip_id, wav

    def sink(d):
        D = verifier_view([d])
        ev = final_events(d, D, 0, W, post, table, gated_blend=True, no_gate=False, st=stats)
        m = np.zeros(1000, np.uint8)
        for a, b in ev:
            m[int(round(a * 100)):int(round(b * 100))] = 1
        ids.append(d["id"])
        hard.append(m)
        soft.append(base_prob(d, W, True)[:1000].astype(np.float16))

    extract_items(items(), member_prefix=MEMBER_PREFIX, total=len(idx), sink=sink)
    np.savez(path, ids=np.array(ids), hard=np.stack(hard), soft=np.stack(soft))
    print(f"saved {path}: {len(ids)} clips, positive frame rate {np.stack(hard).mean():.3f}; {stats}", flush=True)


if __name__ == "__main__":
    main()
