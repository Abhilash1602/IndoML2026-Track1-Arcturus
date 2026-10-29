"""Re-tune Viterbi decode params specifically for the BEATs-fusion checkpoints
(checkpoints/dualbranch_baseline_beats.pth / dualbranch_distilled_beats.pth) --
optimize_postprocessing.py's load_models() hardcodes the live BASELINE_PATH/
DISTILLED_PATH (today's proven non-BEATs checkpoints), so this reuses its other
functions directly with explicit BEATs paths instead, rather than touching those
module-level constants at all.

Usage: python -m temp.optimize_postprocessing_beats
"""
import torch

from src.architecture import DualBranchSED
from src.config import Config
from optimize_postprocessing import (
    cache_validation_predictions,
    load_validation_loader,
    tune_viterbi_with_optuna,
)

BEATS_BASELINE_PATH = "./checkpoints/dualbranch_baseline_beats.pth"
BEATS_DISTILLED_PATH = "./checkpoints/dualbranch_distilled_beats.pth"


def load_beats_models(config):
    models = []
    for path in (BEATS_BASELINE_PATH, BEATS_DISTILLED_PATH):
        model = DualBranchSED(config).to(config.DEVICE)
        model.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True))
        model.eval()
        models.append(model)
    return models


def main():
    config = Config()
    validation_loader = load_validation_loader(config)
    models = load_beats_models(config)
    probabilities, ground_truth = cache_validation_predictions(models, validation_loader, config)
    print(f"Cached probabilities: {probabilities.shape}")
    print(f"Cached ground truth: {ground_truth.shape}")

    print("\n=== Tuning Viterbi decoder via Optuna (BEATs-fusion checkpoints) ===")
    result = tune_viterbi_with_optuna(probabilities, ground_truth, config)
    print(f"P(stay silence):    {result['p_stay_silence']:.4f}")
    print(f"P(stay event):      {result['p_stay_event']:.4f}")
    print(f"Segment Dice:       {result['dice']:.6f}")
    print(f"Event F1:           {result['event_f1']:.6f}")
    print(f"Combined Score:     {result['combined_score']:.6f}")


if __name__ == "__main__":
    main()
