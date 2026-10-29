"""Plan Part 3, item 1: broader ensembling. Cache sigmoid probabilities for every
strong candidate checkpoint ONCE on the real validation set, then cheaply try
extending the known-good Phase H + WavLM-full1 pair with each other candidate
(and promising combinations of two), since exhaustively trying all C(8,k)
subsets is unnecessary -- we already know that pair works, so the natural next
search is "does adding a third/fourth member help further".

Usage: python -m temp.sweep_broader_ensemble
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
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "phasee": dict(path="./checkpoints/dualbranch_phasee.pth", overrides={}),
    "phasef": dict(path="./checkpoints/dualbranch_phasef.pth", overrides={}),
    "phaseg2": dict(path="./checkpoints/dualbranch_phaseg2.pth", overrides={}),
    "wavlm_smoke": dict(path="./checkpoints/dualbranch_wavlm_stage1.pth", overrides=dict(USE_WAVLM=True)),
    "wavlm_full1": dict(path="./checkpoints/dualbranch_wavlm_stage1_full.pth", overrides=dict(USE_WAVLM=True)),
    "wavlm_full2": dict(path="./checkpoints/dualbranch_wavlm_stage1_full2.pth", overrides=dict(USE_WAVLM=True)),
}

# Phase H's tuned decode params -- a robust, representative choice (this session
# repeatedly found decode params matter little once models are strong/ensembled).
P_STAY_SILENCE, P_STAY_EVENT = 0.9955, 0.8222


def cache_model(name, spec, config_template):
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
    f1, p, r = micro_event_f1(gt_events, pred_events)
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
    cache = {name: cache_model(name, spec, config) for name, spec in CANDIDATES.items()}

    print("\n=== Individual models ===", flush=True)
    for name in CANDIDATES:
        eq = eq_avg_for([name], cache, hop)
        print(f"{name}: eq_avg={eq:.4f}", flush=True)

    print("\n=== Known-good pair ===", flush=True)
    base = ["phaseh", "wavlm_full1"]
    base_eq = eq_avg_for(base, cache, hop)
    print(f"{'+'.join(base)}: eq_avg={base_eq:.4f}  (reference: real Codabench 1.4982)", flush=True)

    print("\n=== Pair + one more (broader ensembling search) ===", flush=True)
    others = [n for n in CANDIDATES if n not in base]
    triple_results = []
    for extra in others:
        combo = base + [extra]
        eq = eq_avg_for(combo, cache, hop)
        triple_results.append((eq, extra))
        print(f"{'+'.join(combo)}: eq_avg={eq:.4f}  (delta vs pair: {eq-base_eq:+.4f})", flush=True)

    triple_results.sort(reverse=True)
    best_eq, best_extra = triple_results[0]
    if best_eq > base_eq:
        print(f"\n=== Best triple + one more (4-model search) ===", flush=True)
        best_triple = base + [best_extra]
        for extra in [n for n in others if n != best_extra]:
            combo = best_triple + [extra]
            eq = eq_avg_for(combo, cache, hop)
            print(f"{'+'.join(combo)}: eq_avg={eq:.4f}  (delta vs best triple: {eq-best_eq:+.4f})", flush=True)
    else:
        print("\nNo 3rd member improved on the pair -- skipping 4-model search.", flush=True)


if __name__ == "__main__":
    main()
