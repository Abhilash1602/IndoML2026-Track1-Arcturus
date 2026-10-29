"""Session 9 plan, Priority A step 3 ("the test that actually matters"):
swap the new ATST-Frame Stage 2 (E4, LLRD+MT+ICT, eq_avg 1.4782 standalone --
genuinely beats the frozen Stage 1 gate before the run later collapsed) into
the current best ensemble in place of the frozen Stage 1 checkpoint, and
compare against the current ensemble unchanged. Standalone gate-beating isn't
required for an ensemble member to help (Phase D/Phase G2 precedent) -- this
is the real test.

Usage: python -m temp.swap_test_atst_stage2
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

VAL_ROOT = "./data/official_validation_raw/validation"

CANDIDATES = {
    "phaseh": dict(path="./checkpoints/dualbranch_phaseh.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "wavlm_full1": dict(path="./checkpoints/dualbranch_wavlm_stage1_full.pth", overrides=dict(USE_WAVLM=True)),
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "atst_stage1": dict(path="./checkpoints/dualbranch_atst_frame_stage1_full.pth",
                         overrides=dict(USE_ATST_FRAME=True)),
    "atst_stage2": dict(path="./checkpoints/dualbranch_atst_frame_stage2.pth",
                         overrides=dict(USE_ATST_FRAME=True)),
}

# Current best ensemble's own tuned decode (Phase H + WavLM full1 + Phase D + ATST-Frame Stage1).
P_STAY_SILENCE, P_STAY_EVENT = 0.9883, 0.7954


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
    return dice, f1, dice + f1


def eq_avg_for(names, cache, hop):
    results = {}
    for subset in ("natural", "synthetic"):
        probs_list = [cache[n][subset][0] for n in names]
        targets = cache[names[0]][subset][1]
        avg_probs = np.mean(probs_list, axis=0)
        _, _, results[subset] = score(avg_probs, targets, hop)
    return results["natural"], results["synthetic"], 0.5 * (results["natural"] + results["synthetic"])


def main():
    config = Config()
    hop = config.HOP_LENGTH / config.SR
    print("Caching all candidates (once each)...", flush=True)
    cache = {name: cache_model(name, spec) for name, spec in CANDIDATES.items()}

    base = ["phaseh", "wavlm_full1", "phased"]
    print("\n=== Individual ATST-Frame checkpoints ===", flush=True)
    for name in ("atst_stage1", "atst_stage2"):
        nc, sc, eq = eq_avg_for([name], cache, hop)
        print(f"{name} alone: natural={nc:.4f} synthetic={sc:.4f} eq_avg={eq:.4f}", flush=True)

    print("\n=== Current best ensemble (with frozen Stage 1) ===", flush=True)
    nc1, sc1, eq1 = eq_avg_for(base + ["atst_stage1"], cache, hop)
    print(f"phaseh+wavlm_full1+phased+atst_stage1: natural={nc1:.4f} synthetic={sc1:.4f} "
          f"eq_avg={eq1:.4f}  (reference: real Codabench 1.5230)", flush=True)

    print("\n=== Swap test: Stage 2 (E4) in place of Stage 1 ===", flush=True)
    nc2, sc2, eq2 = eq_avg_for(base + ["atst_stage2"], cache, hop)
    print(f"phaseh+wavlm_full1+phased+atst_stage2: natural={nc2:.4f} synthetic={sc2:.4f} "
          f"eq_avg={eq2:.4f}  (delta vs Stage 1 ensemble: {eq2-eq1:+.4f})", flush=True)

    print("\n=== Also try: both Stage 1 and Stage 2 together (5-model) ===", flush=True)
    nc3, sc3, eq3 = eq_avg_for(base + ["atst_stage1", "atst_stage2"], cache, hop)
    print(f"phaseh+wavlm_full1+phased+atst_stage1+atst_stage2: natural={nc3:.4f} synthetic={sc3:.4f} "
          f"eq_avg={eq3:.4f}  (delta vs Stage 1 ensemble: {eq3-eq1:+.4f})", flush=True)

    print("\n=== Summary ===", flush=True)
    best_eq = max(eq1, eq2, eq3)
    if best_eq > eq1 + 0.001:
        winner = "Stage 2 swap" if best_eq == eq2 else "5-model (both stages)"
        print(f"{winner} HELPS -- worth decode-tuning and generating a submission.", flush=True)
    else:
        print("Stage 2 checkpoint does not improve the ensemble -- keep the current best as-is.", flush=True)


if __name__ == "__main__":
    main()
