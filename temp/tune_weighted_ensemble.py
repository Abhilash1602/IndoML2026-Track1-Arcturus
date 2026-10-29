"""Engineering-hack plan, item 1: replace the uniform 1/N sigmoid average with a
learned per-model weight, jointly Optuna-tuned alongside the existing
(p_stay_silence, p_stay_event) decode params against OfficialValidationDataset.

Every ensemble script in this project (temp/tune_gen_atst_ensemble.py,
temp/sweep_atst_frame_ensemble.py, generate_submission.py, ~10 others) combines
sigmoid outputs via sum(...)/len(...) or np.mean(...) -- a uniform average that
gives every model equal weight regardless of individual strength. This is
exactly the mechanism behind the diagnosed "ensemble dilution" problem:
ATST-Frame was individually the strongest checkpoint on the local sweep
(eq_avg 1.4748 alone, beating every other candidate) yet only delivered
+0.0054 real Combined once diluted to a forced 1/4 weight. Weighted averaging
is an established DCASE ensemble technique that simply hasn't been tried here.

Runs two searches:
  1. The current 4-model set (Phase H, WavLM full1, Phase D, ATST-Frame) with
     learned weights instead of the swap-based Phase-G2-vs-ATST-Frame choice.
  2. All 5 candidates (adding Phase G2 back) with learned weights, in case a
     non-zero weight for Phase G2 alongside ATST-Frame beats the hard
     swap-out that was chosen based on uniform averaging.

Usage: python -m temp.tune_weighted_ensemble
"""
import numpy as np
import optuna
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
    "atst_frame": dict(path="./checkpoints/dualbranch_atst_frame_stage1_full.pth",
                        overrides=dict(USE_ATST_FRAME=True)),
    "phaseg2": dict(path="./checkpoints/dualbranch_phaseg2.pth", overrides={}),
}

# Known uniform-average references for the sanity check (Section "Verification"
# of the plan): forcing all weights equal must reproduce these.
KNOWN_UNIFORM_TUNED_4MODEL = 1.4937  # phaseh+wavlm_full1+phased+atst_frame, Optuna-tuned decode


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


def score(probabilities, ground_truth, p_stay_silence, p_stay_event, hop):
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, _, _ = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, dice + f1


def weighted_eq_avg(names, weights, cache, p_stay_silence, p_stay_event, hop):
    results = {}
    for subset in ("natural", "synthetic"):
        probs_list = [cache[n][subset][0] for n in names]
        targets = cache[names[0]][subset][1]
        avg_probs = np.average(np.stack(probs_list, axis=0), axis=0, weights=weights)
        _, _, results[subset] = score(avg_probs, targets, p_stay_silence, p_stay_event, hop)
    return 0.5 * (results["natural"] + results["synthetic"])


def tune(names, cache, hop, n_trials=120):
    def objective(trial):
        weights = [trial.suggest_float(f"w_{n}", 0.0, 2.0) for n in names]
        if sum(weights) < 1e-6:
            return 0.0  # degenerate all-zero weights
        p_stay_silence = trial.suggest_float("p_stay_silence", 0.90, 0.999)
        p_stay_event = trial.suggest_float("p_stay_event", 0.4, 0.95)
        return weighted_eq_avg(names, weights, cache, p_stay_silence, p_stay_event, hop)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial


def main():
    config = Config()
    hop = config.HOP_LENGTH / config.SR
    print("Caching all candidates (once each)...", flush=True)
    cache = {name: cache_model(name, spec) for name, spec in CANDIDATES.items()}

    four = ["phaseh", "wavlm_full1", "phased", "atst_frame"]
    print("\n=== Sanity check: equal weights must reproduce the known uniform-average result ===", flush=True)
    eq_uniform = weighted_eq_avg(four, [1.0] * 4, cache, 0.9883, 0.7954, hop)
    print(f"weighted_eq_avg(equal weights, known-best decode) = {eq_uniform:.4f} "
          f"(reference: {KNOWN_UNIFORM_TUNED_4MODEL:.4f})", flush=True)
    assert abs(eq_uniform - KNOWN_UNIFORM_TUNED_4MODEL) < 0.01, \
        "Sanity check failed -- weighted combination code path disagrees with the known uniform-average result"

    print("\n=== Tuning weights: 4-model set (Phase H + WavLM full1 + Phase D + ATST-Frame) ===", flush=True)
    best4 = tune(four, cache, hop)
    weights4 = {n: best4.params[f"w_{n}"] for n in four}
    print(f"Best eq_avg={best4.value:.4f}  weights={weights4}  "
          f"p_stay_silence={best4.params['p_stay_silence']:.4f} "
          f"p_stay_event={best4.params['p_stay_event']:.4f}  "
          f"(delta vs uniform: {best4.value - KNOWN_UNIFORM_TUNED_4MODEL:+.4f})", flush=True)

    five = ["phaseh", "wavlm_full1", "phased", "atst_frame", "phaseg2"]
    print("\n=== Tuning weights: 5-model set (adding Phase G2 back) ===", flush=True)
    best5 = tune(five, cache, hop)
    weights5 = {n: best5.params[f"w_{n}"] for n in five}
    print(f"Best eq_avg={best5.value:.4f}  weights={weights5}  "
          f"p_stay_silence={best5.params['p_stay_silence']:.4f} "
          f"p_stay_event={best5.params['p_stay_event']:.4f}  "
          f"(delta vs 4-model weighted: {best5.value - best4.value:+.4f})", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"Uniform 4-model (known): {KNOWN_UNIFORM_TUNED_4MODEL:.4f}", flush=True)
    print(f"Weighted 4-model:        {best4.value:.4f}", flush=True)
    print(f"Weighted 5-model:        {best5.value:.4f}", flush=True)
    overall_best = max(best4.value, best5.value)
    if overall_best > KNOWN_UNIFORM_TUNED_4MODEL + 0.001:
        print("Weighted ensemble HELPS -- worth generating a submission with the winning weights.", flush=True)
    else:
        print("Weighted ensemble does not meaningfully beat uniform averaging here.", flush=True)


if __name__ == "__main__":
    main()
