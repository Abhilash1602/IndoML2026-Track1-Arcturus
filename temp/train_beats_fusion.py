"""Session 7d: train Phase 0 + Phase 2 with the new frozen-BEATs-fusion
architecture, saving to separate checkpoint paths so today's proven
Session 7c results (checkpoints/dualbranch_baseline.pth /
dualbranch_distilled.pth, and their submission zips already generated)
are never at risk regardless of how this experiment goes.

Usage: python -m temp.train_beats_fusion
"""
import os

import torch
from torch.utils.data import DataLoader

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import BronzeInferenceDataset
import train as T

BEATS_BASELINE_PATH = "./checkpoints/dualbranch_baseline_beats.pth"
BEATS_DISTILLED_PATH = "./checkpoints/dualbranch_distilled_beats.pth"


def main():
    config = Config()
    full_ds, train_loader, val_loader = T.make_gold_loaders(config)

    original_baseline_path, original_distilled_path = T.BASELINE_PATH, T.DISTILLED_PATH
    T.BASELINE_PATH, T.DISTILLED_PATH = BEATS_BASELINE_PATH, BEATS_DISTILLED_PATH
    try:
        baseline = DualBranchSED(config).to(config.DEVICE)
        T.fit_baseline(baseline, train_loader, val_loader, config)
        baseline.load_state_dict(torch.load(BEATS_BASELINE_PATH, map_location=config.DEVICE))

        silver_bronze = BronzeInferenceDataset(full_ds, config)
        silver_bronze_loader = DataLoader(
            silver_bronze, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
        )
        print(f"Silver+Bronze (self-distillation pool) samples: {len(silver_bronze)}")
        pseudo_labels = T.generate_pseudo_labels(baseline, silver_bronze_loader, config)

        self_distill_loader = T.build_self_distillation_loader(
            full_ds, silver_bronze, pseudo_labels, train_loader.dataset, config
        )
        distilled = DualBranchSED(config).to(config.DEVICE)
        T.fit_self_distilled(distilled, self_distill_loader, val_loader, config)
    finally:
        T.BASELINE_PATH, T.DISTILLED_PATH = original_baseline_path, original_distilled_path

    print(f"Done. BEATs-fusion checkpoints saved to {BEATS_BASELINE_PATH} and {BEATS_DISTILLED_PATH}")


if __name__ == "__main__":
    main()
