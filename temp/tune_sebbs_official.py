"""SEBBs (Sound Event Bounding Boxes) post-processing, recalibrated against
OfficialValidationDataset -- the real organizer validation set -- instead of
the homegrown Natural-only gold_val split that caused both prior SEBB attempts
to score ~0.38 (see STRATEGY.md). Same 4-model ensemble as the current best
local candidate (Phase H + WavLM full1 + Phase D + Phase G2, real Codabench
1.5019), compared apples-to-apples against that ensemble's own Optuna-tuned
Viterbi decode (local eq_avg 1.4855) using the corrected official metric
(temp/official_metric.py's micro F1 + macro Dice) for both.

Usage: python -m temp.tune_sebbs_official
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.sebbs_postprocessing import sebbs_predicted_events, tune_sebbs
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

VAL_ROOT = "./data/official_validation_raw/validation"
VITERBI_P_STAY_SILENCE, VITERBI_P_STAY_EVENT = 0.9937, 0.9488  # the 4-model ensemble's own tuned decode

# Exact same MEMBERS spec as temp/tune_gen_4model_ensemble.py (the script that
# produced the real Codabench 1.5019 submission) -- Phase D uses Phase H's own
# capacity overrides (NOT a separate bigger config), and Phase G2 uses the
# plain default Config with no overrides at all.
MEMBERS = {
    "phaseh": dict(path="./checkpoints/dualbranch_phaseh.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "wavlm_full1": dict(path="./checkpoints/dualbranch_wavlm_stage1_full.pth", overrides=dict(USE_WAVLM=True)),
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "phaseg2": dict(path="./checkpoints/dualbranch_phaseg2.pth", overrides={}),
}


def build_models():
    models = []
    for name, spec in MEMBERS.items():
        config = Config()
        for k, v in spec["overrides"].items():
            setattr(config, k, v)
        model = DualBranchSED(config).to(config.DEVICE)
        model.load_state_dict(torch.load(spec["path"], map_location=config.DEVICE, weights_only=True))
        model.eval()
        models.append(model)
    return models, Config()


@torch.no_grad()
def cache_ensemble(models, loader, device):
    probs, targets = [], []
    for audio, masks, _ in tqdm(loader, desc="caching"):
        audio = audio.to(device, non_blocking=True)
        ensembled = torch.sigmoid(sum(m(audio) for m in models) / len(models))
        frames = min(ensembled.shape[1], masks.shape[1])
        probs.append(ensembled[:, :frames].cpu().numpy())
        targets.append(masks[:, :frames].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score_viterbi(probabilities, ground_truth, hop):
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=VITERBI_P_STAY_SILENCE, p_stay_event=VITERBI_P_STAY_EVENT)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, _, _ = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, dice + f1


def score_sebbs(predictor, sed_scores, clip_ids, ground_truth, hop):
    pred_events, gt_events = [], []
    for cid, target in zip(clip_ids, ground_truth):
        events = sebbs_predicted_events(predictor, sed_scores, cid)
        pred_events.append([(e["onset"], e["offset"]) for e in events])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, _, _ = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, dice + f1


def tune_and_score_subset(name, probabilities, ground_truth, hop, clip_duration_sec):
    print(f"\n=== {name} ===", flush=True)
    vd, vf, vc = score_viterbi(probabilities, ground_truth, hop)
    print(f"Viterbi (ensemble's own tuned decode): dice={vd:.4f} f1={vf:.4f} combined={vc:.4f}", flush=True)

    predictor, best_values, sed_scores, sebbs_gt, audio_durations, clip_ids = tune_sebbs(
        probabilities, ground_truth, hop, clip_duration_sec,
        step_filter_lengths=(0.16, 0.24, 0.32, 0.48, 0.64, 0.80),
        merge_thresholds_abs=(0.10, 0.15, 0.20, 0.30),
        merge_thresholds_rel=(1.5, 2.0, 3.0, 4.0),
        either_abs_or_rel_threshold=False,
    )
    print(f"SEBBs tuned params: {best_values}", flush=True)
    sd, sf, sc = score_sebbs(predictor, sed_scores, clip_ids, ground_truth, hop)
    print(f"SEBBs (re-scored w/ official metric): dice={sd:.4f} f1={sf:.4f} combined={sc:.4f}", flush=True)
    return vc, sc


def main():
    models, config = build_models()
    device = config.DEVICE
    hop = config.HOP_LENGTH / config.SR
    clip_duration_sec = config.MAX_DURATION

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)

    print("Caching natural predictions (4-model ensemble)...", flush=True)
    np_, nt = cache_ensemble(models, nat_loader, device)
    print("Caching synthetic predictions (4-model ensemble)...", flush=True)
    sp, st = cache_ensemble(models, syn_loader, device)

    nat_viterbi, nat_sebbs = tune_and_score_subset("NATURAL", np_, nt, hop, clip_duration_sec)
    syn_viterbi, syn_sebbs = tune_and_score_subset("SYNTHETIC", sp, st, hop, clip_duration_sec)

    eq_viterbi = 0.5 * nat_viterbi + 0.5 * syn_viterbi
    eq_sebbs = 0.5 * nat_sebbs + 0.5 * syn_sebbs
    print(f"\n=== SUMMARY (4-model ensemble, real validation set) ===", flush=True)
    print(f"Viterbi eq_avg: {eq_viterbi:.4f}", flush=True)
    print(f"SEBBs   eq_avg: {eq_sebbs:.4f}", flush=True)
    if eq_sebbs > eq_viterbi:
        print(f"SEBBs WINS by {eq_sebbs - eq_viterbi:+.4f} -- worth generating a SEBBs-based submission.", flush=True)
    else:
        print(f"Viterbi still wins by {eq_viterbi - eq_sebbs:+.4f} -- keep the Viterbi-decoded submission.", flush=True)


if __name__ == "__main__":
    main()
