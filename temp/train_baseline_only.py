"""Session 6 FDY-Conv ablation: retrain Phase 0 baseline only (plain Conv2d,
just reverted in src/architecture.py) and nothing else, so the result can be
compared directly against the FDY-Conv baseline's real Codabench result
(Combined 1.0597, Event F1 0.3821) and Session 5's Run A (1.0899, 0.4265)
without Mean Teacher/Phase 2 in the picture at all.

Usage: python -m temp.train_baseline_only
"""
import os

from src.architecture import DualBranchSED
from src.config import Config
from train import BASELINE_PATH, CHECKPOINT_DIR, fit_baseline, make_gold_loaders


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = Config()
    _, train_loader, val_loader = make_gold_loaders(config)
    baseline = DualBranchSED(config).to(config.DEVICE)
    fit_baseline(baseline, train_loader, val_loader, config)
    print(f"Done. Best checkpoint saved to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
