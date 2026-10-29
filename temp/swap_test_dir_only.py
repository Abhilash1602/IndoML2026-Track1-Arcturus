"""Session 9 engineering-hack plan, Priority B: test the new DIR-only
ATST-Frame checkpoint (temp/train_atst_frame_dir_only_full.py, standalone
eq_avg 1.4822, +0.0063 over the frozen Stage 1 baseline) in the current best
ensemble -- same "the test that actually matters" step used for every prior
checkpoint this session.

Current best ensemble: Phase H + WavLM full1 + Phase D + ATST-Frame Stage 1
+ ATST-Frame Stage 2 (real Codabench 1.5326). Tests: add DIR-only as a 6th
member, and swap it in place of each ATST-Frame variant individually.

Usage: python -m temp.swap_test_dir_only
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
    "atst_dir": dict(path="./checkpoints/dualbranch_atst_frame_dir_only_full.pth",
                      overrides=dict(USE_ATST_FRAME=True)),
}

P_STAY_SILENCE, P_STAY_EVENT = 0.9005, 0.8901


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

    base5 = ["phaseh", "wavlm_full1", "phased", "atst_stage1", "atst_stage2"]
    print("\n=== Individual: DIR-only checkpoint alone ===", flush=True)
    nc, sc, eq = eq_avg_for(["atst_dir"], cache, hop)
    print(f"atst_dir alone: natural={nc:.4f} synthetic={sc:.4f} eq_avg={eq:.4f}", flush=True)

    print("\n=== Current best 5-model ensemble ===", flush=True)
    nc0, sc0, eq0 = eq_avg_for(base5, cache, hop)
    print(f"{'+'.join(base5)}: natural={nc0:.4f} synthetic={sc0:.4f} eq_avg={eq0:.4f} "
          f"(reference: real Codabench 1.5326)", flush=True)

    print("\n=== Add DIR-only as a 6th member ===", flush=True)
    nc1, sc1, eq1 = eq_avg_for(base5 + ["atst_dir"], cache, hop)
    print(f"6-model (+atst_dir): natural={nc1:.4f} synthetic={sc1:.4f} eq_avg={eq1:.4f} "
          f"(delta vs 5-model: {eq1-eq0:+.4f})", flush=True)

    print("\n=== Swap DIR-only in place of Stage 1 ===", flush=True)
    swap1 = ["phaseh", "wavlm_full1", "phased", "atst_dir", "atst_stage2"]
    nc2, sc2, eq2 = eq_avg_for(swap1, cache, hop)
    print(f"{'+'.join(swap1)}: natural={nc2:.4f} synthetic={sc2:.4f} eq_avg={eq2:.4f} "
          f"(delta vs 5-model: {eq2-eq0:+.4f})", flush=True)

    print("\n=== Swap DIR-only in place of Stage 2 ===", flush=True)
    swap2 = ["phaseh", "wavlm_full1", "phased", "atst_stage1", "atst_dir"]
    nc3, sc3, eq3 = eq_avg_for(swap2, cache, hop)
    print(f"{'+'.join(swap2)}: natural={nc3:.4f} synthetic={sc3:.4f} eq_avg={eq3:.4f} "
          f"(delta vs 5-model: {eq3-eq0:+.4f})", flush=True)

    print("\n=== Summary ===", flush=True)
    options = {"5-model (current)": eq0, "6-model (+DIR)": eq1,
               "swap DIR<-Stage1": eq2, "swap DIR<-Stage2": eq3}
    best_name = max(options, key=options.get)
    best_val = options[best_name]
    print({k: round(v, 4) for k, v in options.items()}, flush=True)
    if best_val > eq0 + 0.001:
        print(f"BEST: {best_name} ({best_val:.4f}) HELPS -- worth decode-tuning and generating "
              f"a submission.", flush=True)
    else:
        print("DIR-only checkpoint does not meaningfully improve the ensemble -- keep current best.",
              flush=True)


if __name__ == "__main__":
    main()
