"""Evaluate our checkpoints against the REAL organizer-provided validation set
(Codabench Files tab -> validation.zip -> data/official_validation_raw/validation/),
using the corrected official micro-F1 + global-matching metric (temp/official_metric.py).

This is the first local proxy built from real held-out data in the true eval
distribution (not our own gold_val slice or from-scratch synth_val). Prints each
checkpoint's Natural/Synthetic/overall numbers next to the REAL Codabench numbers we
already have, to see how well this new proxy actually predicts the real leaderboard --
if it correlates well, it replaces gold_val/synth_val as our primary local signal
going forward.

Usage: python -m temp.eval_official_validation
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

CKPTS = {
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6),
                   decode=(0.9986, 0.6944)),
    "phasee": dict(path="./checkpoints/dualbranch_phasee.pth", overrides={},
                   decode=(0.9963, 0.5257)),
    "phasef": dict(path="./checkpoints/dualbranch_phasef.pth", overrides={},
                   decode=(0.9989, 0.8355)),
}

REAL_RESULTS = {
    # (Natural F1, Natural Dice, Synthetic F1, Synthetic Dice) from actual Codabench submissions this session
    "phased": (0.3824, 0.6396, 0.9289, 0.9410),
    "phasee": (0.4028, 0.6661, 0.8984, 0.9296),
}


def cache(model, loader, config):
    probs, targets, names = [], [], []
    with torch.no_grad():
        for audio, masks, clip_ids in tqdm(loader, desc="caching"):
            audio = audio.to(config.DEVICE, non_blocking=True)
            logits = model(audio)
            frames = min(logits.shape[1], masks.shape[1])
            probs.append(torch.sigmoid(logits[:, :frames]).cpu().numpy())
            targets.append(masks[:, :frames].numpy())
            names.extend(clip_ids)
    return np.concatenate(probs), np.concatenate(targets), names


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
    for name, spec in CKPTS.items():
        config = Config()
        for k, v in spec["overrides"].items():
            setattr(config, k, v)
        model = DualBranchSED(config).to(config.DEVICE)
        model.load_state_dict(torch.load(spec["path"], map_location=config.DEVICE, weights_only=True))
        model.eval()
        ps, pe = spec["decode"]

        print(f"\n=== {name} (decode {ps}/{pe}) ===", flush=True)
        results = {}
        for subset in ("natural", "synthetic"):
            ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
            loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
            probs, targets, _ = cache(model, loader, config)
            dice, f1, p, r = score(probs, targets, ps, pe, config)
            results[subset] = (f1, dice)
            print(f"  [{subset}] n={len(ds)} F1={f1:.4f} (P={p:.3f} R={r:.3f}) Dice={dice:.4f} "
                  f"combined={f1+dice:.4f}", flush=True)

        eq_avg = 0.5 * (sum(results["natural"]) + sum(results["synthetic"]))
        print(f"  eq-weighted overall combined = {eq_avg:.4f}", flush=True)

        real = REAL_RESULTS.get(name)
        if real:
            rn_f1, rn_dice, rs_f1, rs_dice = real
            real_eq = 0.5 * ((rn_f1 + rn_dice) + (rs_f1 + rs_dice))
            print(f"  REAL Codabench: natural={rn_f1+rn_dice:.4f} synthetic={rs_f1+rs_dice:.4f} "
                  f"overall={real_eq:.4f}", flush=True)


if __name__ == "__main__":
    main()
