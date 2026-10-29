"""Evaluate an ensemble of Phase F + Phase G2 against the real organizer validation
set -- the free first check in STRATEGY.md's "Next steps" plan, testing whether
their complementary strengths (F better Natural, G2 better Synthetic) combine to
beat either alone before committing to any bigger, more expensive lever.

Usage: python -m temp.eval_ensemble_fg2
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import micro_event_f1, macro_segment_dice

VAL_ROOT = "./data/official_validation_raw/validation"
CKPTS = ["./checkpoints/dualbranch_phasef.pth", "./checkpoints/dualbranch_phaseg2.pth"]


def load_models(config):
    models = []
    for path in CKPTS:
        m = DualBranchSED(config).to(config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True))
        m.eval()
        models.append(m)
    return models


def cache_ensemble(models, loader, config):
    probs, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in tqdm(loader, desc="caching"):
            audio = audio.to(config.DEVICE, non_blocking=True)
            ensembled = torch.sigmoid(sum(m(audio) for m in models) / len(models))
            frames = min(ensembled.shape[1], masks.shape[1])
            probs.append(ensembled[:, :frames].cpu().numpy())
            targets.append(masks[:, :frames].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score(probabilities, ground_truth, p_stay_silence, p_stay_event, config):
    hop = config.HOP_LENGTH / config.SR
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, p, r = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, p, r


def main():
    config = Config()
    models = load_models(config)

    cached = {}
    for subset in ("natural", "synthetic"):
        ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        probs, targets = cache_ensemble(models, loader, config)
        cached[subset] = (probs, targets)

    for name, ps, pe in [("phaseF params (0.9989/0.8355)", 0.9989, 0.8355),
                         ("phaseG2 params (0.9950/0.8639)", 0.9950, 0.8639)]:
        results = {}
        for subset, (probs, targets) in cached.items():
            dice, f1, p, r = score(probs, targets, ps, pe, config)
            results[subset] = (f1, dice)
            print(f"[{name}] [{subset}] F1={f1:.4f} (P={p:.3f} R={r:.3f}) Dice={dice:.4f} "
                  f"combined={f1+dice:.4f}", flush=True)
        eq_avg = 0.5 * (sum(results["natural"]) + sum(results["synthetic"]))
        print(f"[{name}] eq-weighted overall combined = {eq_avg:.4f}\n", flush=True)


if __name__ == "__main__":
    main()
