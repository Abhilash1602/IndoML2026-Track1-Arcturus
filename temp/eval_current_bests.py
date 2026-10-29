"""Evaluate Phase H's current best (E45) and the WavLM stage-1 smoke-test checkpoint
against the real validation set, for an accurate, up-to-date comparison.

Usage: python -m temp.eval_current_bests
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
    "phaseh (E45, capacity+selfdistill)": dict(
        path="./checkpoints/dualbranch_phaseh.pth",
        overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6),
        decode=(0.9986, 0.6944)),
    "wavlm_stage1 (E15 smoke test)": dict(
        path="./checkpoints/dualbranch_wavlm_stage1.pth",
        overrides=dict(USE_WAVLM=True),
        decode=(0.9986, 0.6944)),
}


def cache(model, loader, config):
    probs, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in tqdm(loader, desc="caching"):
            audio = audio.to(config.DEVICE, non_blocking=True)
            logits = model(audio)
            frames = min(logits.shape[1], masks.shape[1])
            probs.append(torch.sigmoid(logits[:, :frames]).cpu().numpy())
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
    for name, spec in CKPTS.items():
        config = Config()
        for k, v in spec["overrides"].items():
            setattr(config, k, v)
        model = DualBranchSED(config).to(config.DEVICE)
        model.load_state_dict(torch.load(spec["path"], map_location=config.DEVICE, weights_only=True))
        model.eval()
        ps, pe = spec["decode"]
        print(f"\n=== {name} ===", flush=True)
        results = {}
        for subset in ("natural", "synthetic"):
            ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
            probs, targets = cache(model, loader, config)
            dice, f1, p, r = score(probs, targets, ps, pe, config)
            results[subset] = (f1, dice)
            print(f"  [{subset}] F1={f1:.4f} (P={p:.3f} R={r:.3f}) Dice={dice:.4f} combined={f1+dice:.4f}", flush=True)
        eq_avg = 0.5 * (sum(results["natural"]) + sum(results["synthetic"]))
        print(f"  eq-weighted overall combined = {eq_avg:.4f}", flush=True)
    print("\n(reference: Phase F=1.4502, Phase G2=1.4602 on this same proxy; "
          "Phase G2 real Codabench=1.4742)", flush=True)


if __name__ == "__main__":
    main()
