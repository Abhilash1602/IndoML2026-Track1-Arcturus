"""Decode-tune and generate the submission for the new best 4-model ensemble
found by temp/sweep_atst_frame_ensemble.py: Phase H + WavLM full1 + Phase D +
ATST-Frame (swapping ATST-Frame in for Phase G2 -- eq_avg 1.4896 vs. the
previous 4-model's 1.4842 at the same fixed decode, and vs. 1.4839 for the
Phase-G2 variant with ATST-Frame added on top instead of swapped in).

Usage: python -m temp.tune_gen_atst_ensemble
"""
import json
import zipfile

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import micro_event_f1, macro_segment_dice
from generate_submission import LocalTestDataset

VAL_ROOT = "./data/official_validation_raw/validation"

MEMBERS = {
    "phaseh": dict(path="./checkpoints/dualbranch_phaseh.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "wavlm_full1": dict(path="./checkpoints/dualbranch_wavlm_stage1_full.pth", overrides=dict(USE_WAVLM=True)),
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "atst_frame": dict(path="./checkpoints/dualbranch_atst_frame_stage1_full.pth",
                        overrides=dict(USE_ATST_FRAME=True)),
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
    return models


def cache_ensemble(models, loader, device):
    probs, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in tqdm(loader, desc="caching"):
            audio = audio.to(device, non_blocking=True)
            ensembled = torch.sigmoid(sum(m(audio) for m in models) / len(models))
            frames = min(ensembled.shape[1], masks.shape[1])
            probs.append(ensembled[:, :frames].cpu().numpy())
            targets.append(masks[:, :frames].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score(probabilities, ground_truth, p_stay_silence, p_stay_event, hop):
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, p, r = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, dice + f1


def main():
    config = Config()
    device = config.DEVICE
    hop = config.HOP_LENGTH / config.SR
    models = build_models()

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)

    print("Caching natural predictions...", flush=True)
    np_, nt = cache_ensemble(models, nat_loader, device)
    print("Caching synthetic predictions...", flush=True)
    sp, st = cache_ensemble(models, syn_loader, device)

    def objective(trial):
        p_stay_silence = trial.suggest_float("p_stay_silence", 0.90, 0.999)
        p_stay_event = trial.suggest_float("p_stay_event", 0.4, 0.95)
        _, _, nc = score(np_, nt, p_stay_silence, p_stay_event, hop)
        _, _, sc = score(sp, st, p_stay_silence, p_stay_event, hop)
        eq = 0.5 * nc + 0.5 * sc
        trial.set_user_attr("natural_combined", nc)
        trial.set_user_attr("synthetic_combined", sc)
        return eq

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=80, show_progress_bar=True)
    best = study.best_trial
    ps, pe = best.params["p_stay_silence"], best.params["p_stay_event"]
    print(f"\n=== BEST (Phase H + WavLM full1 + Phase D + ATST-Frame, REAL validation set) ===", flush=True)
    print(f"p_stay_silence={ps:.4f} p_stay_event={pe:.4f} "
          f"natural_combined={best.user_attrs['natural_combined']:.4f} "
          f"synthetic_combined={best.user_attrs['synthetic_combined']:.4f} "
          f"eq_avg={best.value:.4f}", flush=True)

    print("\nGenerating submission with tuned params...", flush=True)
    dataset = LocalTestDataset("./data/test_audio", config)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    entries = []
    with torch.no_grad():
        for audio, clip_ids in tqdm(loader, desc="Inference"):
            audio = audio.to(device)
            probabilities = torch.sigmoid(sum(m(audio) for m in models) / len(models)).cpu().numpy()
            for probability, clip_id in zip(probabilities, clip_ids):
                states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
                events = extract_events_viterbi(states, hop)
                entries.append({"clip_id": clip_id, "events": events})

    with open("predictions_atst_ensemble.jsonl", "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    with zipfile.ZipFile("submission_atst_ensemble.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write("predictions_atst_ensemble.jsonl", arcname="predictions.jsonl")
    print("Done -> submission_atst_ensemble.zip", flush=True)


if __name__ == "__main__":
    main()
