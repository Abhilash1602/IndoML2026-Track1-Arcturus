"""Session 9 plan, Part 2 Stage 2A full-mix run, Part 5: test the new
boundary-aux checkpoint (temp/train_boundary_aux_full.py, standalone
eq_avg 1.4772, +0.0043 over Phase H's own proxy under an identical
from-scratch recipe -- teacher=Phase F, same data mix, same capacity, same
LR schedule, only the boundary-aux loss added) in the current best ensemble
-- same "the test that actually matters" step used for every prior
checkpoint this session.

Current best ensemble: Phase H + WavLM full1 + Phase D + ATST-Frame Stage 1
+ ATST-Frame Stage 2 (real Codabench 1.5326). Tests: add boundary_aux as a
6th member, and swap it in place of each existing member individually --
unlike DIR-only (a third ATST-Frame variant that failed to add diversity),
boundary_aux shares Phase H/Phase D's base architecture but a structurally
different training signal, so it's tested against every member, not just
the ATST-Frame ones.

Uses the inference-clean checkpoint (boundary_head weights stripped,
verified strict-load-compatible with a boundary-off Config).

Usage: python -m temp.swap_test_boundary_aux
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
    "boundary_aux": dict(path="./checkpoints/dualbranch_boundary_aux_full_inference.pth",
                          overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
}

# Current best ensemble's own tuned decode (Phase H + WavLM full1 + Phase D + ATST-Frame Stage1/2).
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
    print("\n=== Individual: boundary_aux checkpoint alone ===", flush=True)
    nc, sc, eq = eq_avg_for(["boundary_aux"], cache, hop)
    print(f"boundary_aux alone: natural={nc:.4f} synthetic={sc:.4f} eq_avg={eq:.4f}", flush=True)

    print("\n=== Current best 5-model ensemble ===", flush=True)
    nc0, sc0, eq0 = eq_avg_for(base5, cache, hop)
    print(f"{'+'.join(base5)}: natural={nc0:.4f} synthetic={sc0:.4f} eq_avg={eq0:.4f} "
          f"(reference: real Codabench 1.5326)", flush=True)

    print("\n=== Add boundary_aux as a 6th member ===", flush=True)
    nc1, sc1, eq1 = eq_avg_for(base5 + ["boundary_aux"], cache, hop)
    print(f"6-model (+boundary_aux): natural={nc1:.4f} synthetic={sc1:.4f} eq_avg={eq1:.4f} "
          f"(delta vs 5-model: {eq1-eq0:+.4f})", flush=True)

    options = {"5-model (current)": eq0, "6-model (+boundary_aux)": eq1}
    print("\n=== Swap boundary_aux in place of each member ===", flush=True)
    for drop in base5:
        swap = ["boundary_aux"] + [n for n in base5 if n != drop]
        ncx, scx, eqx = eq_avg_for(swap, cache, hop)
        label = f"swap boundary_aux<-{drop}"
        print(f"{'+'.join(swap)}: natural={ncx:.4f} synthetic={scx:.4f} eq_avg={eqx:.4f} "
              f"(delta vs 5-model: {eqx-eq0:+.4f})", flush=True)
        options[label] = eqx

    print("\n=== Summary ===", flush=True)
    best_name = max(options, key=options.get)
    best_val = options[best_name]
    print({k: round(v, 4) for k, v in options.items()}, flush=True)
    if best_val > eq0 + 0.001:
        print(f"BEST: {best_name} ({best_val:.4f}) HELPS -- worth decode-tuning and generating "
              f"a submission.", flush=True)
    else:
        print("boundary_aux checkpoint does not meaningfully improve the ensemble -- keep current best.",
              flush=True)


if __name__ == "__main__":
    main()
