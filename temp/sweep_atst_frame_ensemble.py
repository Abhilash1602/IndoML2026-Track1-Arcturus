"""Add the new ATST-Frame checkpoint (temp/train_atst_frame_stage1_full.py's
best, E6, standalone eq_avg 1.4759 at fixed decode) into the ensemble search,
following the exact same logic as temp/sweep_broader_ensemble.py that found the
current-best 4-model combo (Phase H + WavLM full1 + Phase D + Phase G2, real
Codabench 1.5019). WavLM full1 alone wasn't independently competitive either
(fixed-decode eq_avg ~1.4694) yet contributed a real +0.0167 via ensembling --
this checks whether ATST-Frame, a structurally different frame-level-SED-
pretrained transformer, does the same.

Usage: python -m temp.sweep_atst_frame_ensemble
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

CANDIDATES = {
    "phaseh": dict(path="./checkpoints/dualbranch_phaseh.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "wavlm_full1": dict(path="./checkpoints/dualbranch_wavlm_stage1_full.pth", overrides=dict(USE_WAVLM=True)),
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "phaseg2": dict(path="./checkpoints/dualbranch_phaseg2.pth", overrides={}),
    "atst_frame": dict(path="./checkpoints/dualbranch_atst_frame_stage1_full.pth",
                        overrides=dict(USE_ATST_FRAME=True)),
}

# Current 4-model winner's own tuned decode -- a robust, representative choice.
P_STAY_SILENCE, P_STAY_EVENT = 0.9937, 0.9488


def cache_model(name, spec):
    config = Config()
    for k, v in spec["overrides"].items():
        setattr(config, k, v)
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(spec["path"], map_location=config.DEVICE, weights_only=True))
    model.eval()
    result = {}
    for subset in ("natural", "synthetic"):
        ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
        probs, targets = [], []
        with torch.no_grad():
            for audio, masks, _ in tqdm(loader, desc=f"{name}/{subset}"):
                audio = audio.to(config.DEVICE, non_blocking=True)
                logits = model(audio)
                frames = min(logits.shape[1], masks.shape[1])
                probs.append(torch.sigmoid(logits[:, :frames]).cpu().numpy())
                targets.append(masks[:, :frames].numpy())
        result[subset] = (np.concatenate(probs), np.concatenate(targets))
    del model
    torch.cuda.empty_cache()
    return result


def score(probabilities, ground_truth, hop):
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, _, _ = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice + f1


def eq_avg_for(names, cache, hop):
    results = {}
    for subset in ("natural", "synthetic"):
        probs_list = [cache[n][subset][0] for n in names]
        targets = cache[names[0]][subset][1]
        avg_probs = np.mean(probs_list, axis=0)
        results[subset] = score(avg_probs, targets, hop)
    return 0.5 * (results["natural"] + results["synthetic"])


def main():
    config = Config()
    hop = config.HOP_LENGTH / config.SR
    print("Caching all candidates (once each)...", flush=True)
    cache = {name: cache_model(name, spec) for name, spec in CANDIDATES.items()}

    print("\n=== Individual models (fixed decode 0.9937/0.9488) ===", flush=True)
    for name in CANDIDATES:
        eq = eq_avg_for([name], cache, hop)
        print(f"{name}: eq_avg={eq:.4f}", flush=True)

    current_best = ["phaseh", "wavlm_full1", "phased", "phaseg2"]
    current_eq = eq_avg_for(current_best, cache, hop)
    print(f"\ncurrent best 4-model ({'+'.join(current_best)}): eq_avg={current_eq:.4f} "
          f"(reference: real Codabench 1.5019)", flush=True)

    print("\n=== Current best + ATST-Frame (5-model) ===", flush=True)
    five = current_best + ["atst_frame"]
    eq5 = eq_avg_for(five, cache, hop)
    print(f"{'+'.join(five)}: eq_avg={eq5:.4f}  (delta vs 4-model: {eq5-current_eq:+.4f})", flush=True)

    print("\n=== Smaller combos including ATST-Frame ===", flush=True)
    for combo in (
        ["phaseh", "atst_frame"],
        ["phaseh", "wavlm_full1", "atst_frame"],
        ["phaseh", "wavlm_full1", "phased", "atst_frame"],
        ["phaseh", "wavlm_full1", "phaseg2", "atst_frame"],
        ["phased", "phaseg2", "atst_frame"],
    ):
        eq = eq_avg_for(combo, cache, hop)
        print(f"{'+'.join(combo)}: eq_avg={eq:.4f}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"Best known before this sweep: 1.4855 (4-model, fixed decode) / 1.5019 (real)", flush=True)
    print(f"5-model (adding ATST-Frame): {eq5:.4f}", flush=True)
    if eq5 > current_eq:
        print("ATST-Frame HELPS the ensemble -- worth decode-tuning and generating a submission.", flush=True)
    else:
        print("ATST-Frame does not improve the current best ensemble at this fixed decode -- "
              "check the smaller-combo results above before concluding it's not useful at all.", flush=True)


if __name__ == "__main__":
    main()
