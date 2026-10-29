"""Oct 13: synthetic-safety check for the guarded decision sweep. For each variant zip vs the base
zip, report removed / added events split by the ensemble's mean probability over the event
(>= 0.70 = guard-protected, i.e. synthetic-like), from the cached pipeline features.
Usage: python -m temp.sweep_safety BASE.zip VARIANT.zip [...]"""
import json
import pickle
import sys
import zipfile

import numpy as np

from temp.pipeline_v11 import base_prob


def load(f):
    return {json.loads(l)["clip_id"]: [(e["onset"], e["offset"]) for e in json.loads(l)["events"]]
            for l in zipfile.ZipFile(f).read("predictions.jsonl").decode().splitlines() if l.strip()}


def main():
    data = {d["id"]: d for d in pickle.load(open("./work/verifier/pipeline_v11_features_base_VS.pkl", "rb"))}
    base = load(sys.argv[1])
    for f in sys.argv[2:]:
        v = load(f)
        stat = dict(removed_lo=0, removed_hi=0, added_lo=0, added_hi=0, clips=0)
        for c in base:
            a, b = set(base[c]), set(v[c])
            if a == b:
                continue
            stat["clips"] += 1
            P = base_prob(data[c], 0.7, False)
            conf = lambda e: float(P[int(e[0] * 100):max(int(e[1] * 100), int(e[0] * 100) + 1)].mean())
            for e in a - b:
                stat["removed_hi" if conf(e) >= 0.7 else "removed_lo"] += 1
            for e in b - a:
                stat["added_hi" if conf(e) >= 0.7 else "added_lo"] += 1
        print(f"{f}: {stat}", flush=True)


if __name__ == "__main__":
    main()
