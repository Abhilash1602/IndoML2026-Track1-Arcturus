"""Refit the gated edge-correction table (checkpoints/edge_correction_v5.json).

Per predicted-duration bin: the median (reference - predicted) onset and offset offsets, over 1:1
overlapping pairs of the v4 ensemble's Viterbi events (p_stay 0.9802 / 0.9489) and the references
on the Gold held-out split (2,223 clips no v4 member trained on). Inputs come from
temp/cache_refiner_data.py (work/refiner_gold_heldout.npz: the 5 v4 members' logits + mask).

Usage: python tools/fit_edge_table.py [--out checkpoints/edge_correction_v5.json]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.edge_correction import fit_table, save_table  # noqa: E402
from temp.train_refiner import refs_of, viterbi_events  # noqa: E402
from temp.train_refiner_val import S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./work/edge_correction_refit.json")
    args = ap.parse_args()
    d = np.load(f"{S}/refiner_gold_heldout.npz")
    P = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
    table = fit_table(refs_of(d["mask"].astype(np.float32)), viterbi_events(P))
    save_table(table, args.out)
    print(f"fitted on {len(P)} Gold held-out clips -> {args.out}")


if __name__ == "__main__":
    main()
