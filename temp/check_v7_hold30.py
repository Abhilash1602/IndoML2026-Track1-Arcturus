"""End-to-end check of release/v7_inference.py on hold30 audio (never trained on): V1 (VT)
members + the train70 refiner, vs the same code with the refiner off (= the v6 recipe).
Full-length merged references from validationMetadata.json. Expect about +0.011 (eval_refiner_v6).

Usage: python -m temp.check_v7_hold30
"""
import json
import os

import numpy as np

from release.v7_inference import run
from temp.long_clip_eval import merged_refs
from temp.official_metric import macro_segment_dice, match_events

ROOT = "./data/official_validation_raw/validation"
S = "./work"


def main():
    split = json.load(open("./data/val_split_s0.json"))["hold30"]
    meta = {r["segmentFileName"]: r for r in json.load(open(f"{ROOT}/validationMetadata.json", encoding="utf-8"))}
    files, nat = [], {}
    for sub, folder in (("natural", "naturalNoisyAudio"), ("synthetic", "syntheticNoiseAudio")):
        for n in split[sub]:
            files.append(os.path.join(ROOT, folder, n))
            nat[os.path.splitext(n)[0]] = sub == "natural"
    files = sorted(files)
    res = {}
    for name, ref_path in (("v6 recipe (VT)", None), ("v7 (VT + train70 refiner)", "./checkpoints/refiner_val_train70_x0.pth")):
        entries = run(None, f"{S}/v7check_{'on' if ref_path else 'off'}", "./checkpoints", ref_path, member_prefix="VT", files=files)
        rows, is_nat = [], []
        for e in entries:
            row = meta[e["clip_id"] + ".wav"]
            g = merged_refs(row, float(row["duration"]))
            p = [(x["onset"], x["offset"]) for x in e["events"]]
            tp, fp, fn = match_events(g, p)
            rows.append((tp, fp, fn, macro_segment_dice([g], [p], hop_sec=0.01)))
            is_nat.append(nat[e["clip_id"]])
        res[name] = (np.array(rows, float), np.array(is_nat))
    f1 = lambda a: 2 * a[:, 0].sum() / max(2 * a[:, 0].sum() + a[:, 1].sum() + a[:, 2].sum(), 1)
    for name, (r, n) in res.items():
        print(f"[{name:28s}] combined={f1(r) + r[:, 3].mean():.4f} natF1={f1(r[n]):.4f} natDice={r[n, 3].mean():.4f} "
              f"synF1={f1(r[~n]):.4f} synDice={r[~n, 3].mean():.4f}", flush=True)
    (a, n), (b, _) = res.values()
    print(f"delta v7 - v6 recipe: {f1(b) + b[:, 3].mean() - f1(a) - a[:, 3].mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
