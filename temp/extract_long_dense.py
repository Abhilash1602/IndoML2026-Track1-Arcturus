"""Oct 14: denser windows for the 717 long test clips (> 10 s; 16.6% of predicted events). The
pipeline normally uses v6's 10 s windows with 2 s overlap (stride 8 s); here OVERLAP s (e.g. 5 or 8,
stride 5 or 2 s), so each long-clip frame is averaged over more windows. Short clips are untouched:
their records are copied from the base cache, and the long-clip records are re-extracted with the
same members/refiners (temp.pipeline_v11.extract_items) and spliced in.

Output: work/verifier/pipeline_v11_features_base_VS_ov<OVERLAP>.pkl (use with pipeline --cache).
Usage: CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 python -m temp.extract_long_dense 8
"""
import glob
import os
import pickle
import sys

import release.v6_inference as v6

OV = float(sys.argv[1])
v6.OVERLAP_SEC = OV  # read at call time by temp.long_clips.windows

from temp.long_clips import FIVE  # noqa: E402
from temp.pipeline_v11 import extract_items, path_items  # noqa: E402

BASE = "./work/verifier/pipeline_v11_features_base_VS.pkl"
OUT = f"./work/verifier/pipeline_v11_features_base_VS_ov{int(OV)}.pkl"


def main():
    data = pickle.load(open(BASE, "rb"))
    long_ids = {d["id"] for d in data if not d["short"]}
    paths = [p for p in sorted(glob.glob("./data/test_audio/**/*.wav", recursive=True))
             if os.path.splitext(os.path.basename(p))[0] in long_ids]
    print(f"{len(paths)} long clips re-extracted with {OV:g} s overlap", flush=True)
    new = {d["id"]: d for d in extract_items(path_items(paths, v6.make_config()), FIVE, "VS", total=len(paths))}
    assert set(new) == long_ids
    out = [new.get(d["id"], d) for d in data]
    pickle.dump(out, open(OUT, "wb"))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
