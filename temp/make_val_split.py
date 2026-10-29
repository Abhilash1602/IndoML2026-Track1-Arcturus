"""Plan Phase V0: fixed, seeded 70/30 split of the official validation set for measuring
validation-as-training. train70 may be trained on; hold30 is only ever scored.
Stratified by domain (natural/synthetic) and clip duration (<=5 s, 5-10 s, >10 s).

Output: data/val_split_s0.json = {"seed": 0, "train70": {"natural": [...], "synthetic": [...]},
                                   "hold30": {...}} (segmentFileName lists)
Usage: python -m temp.make_val_split
"""
import json

import numpy as np

META = "./data/official_validation_raw/validation/validationMetadata.json"
OUT = "./data/val_split_s0.json"
SEED, HOLD = 0, 0.30


def main():
    rows = json.load(open(META, encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    split = {"seed": SEED, "train70": {"natural": [], "synthetic": []}, "hold30": {"natural": [], "synthetic": []}}
    for dom in ("natural", "synthetic"):
        rs = [r for r in rows if bool(r.get("syntheticData")) == (dom == "synthetic")]
        for lo, hi in ((0, 5), (5, 10), (10, 1e9)):
            names = sorted(r["segmentFileName"] for r in rs if lo < float(r["duration"]) <= hi or (lo == 0 and float(r["duration"]) <= hi))
            names = sorted(set(names))
            rng.shuffle(names)
            k = int(round(HOLD * len(names)))
            split["hold30"][dom] += names[:k]
            split["train70"][dom] += names[k:]
    all_names = {r["segmentFileName"] for r in rows}
    tr = set(split["train70"]["natural"] + split["train70"]["synthetic"])
    ho = set(split["hold30"]["natural"] + split["hold30"]["synthetic"])
    assert not (tr & ho), "train70 and hold30 overlap"
    assert tr | ho == all_names, "split does not cover the validation set"
    json.dump(split, open(OUT, "w"), indent=1)
    for part in ("train70", "hold30"):
        print(part, {d: len(v) for d, v in split[part].items()})


if __name__ == "__main__":
    main()
