"""Decode-tune and generate the submission for the new best 5-model ensemble
found by temp/swap_test_atst_stage2.py: Phase H + WavLM full1 + Phase D +
ATST-Frame Stage 1 (frozen) + ATST-Frame Stage 2 (E4, LLRD+MT+ICT) --
eq_avg 1.4948 vs. the current best 4-model's 1.4900 at the same fixed decode
(+0.0048, both Natural and Synthetic improving together). Includes both
ATST-Frame checkpoints since they apparently provide complementary diversity
despite sharing the same base architecture (swapping Stage 2 in alone gave
only +0.0041; adding both together gave +0.0048, the best of the three
combos tried).

Also uses chunked/overlap-add inference (src/chunked_inference.py) for test
clips exceeding Config.MAX_DURATION, preserving the real +0.0157 gain that
delivered the current best real submission (1.5230).

Usage: python -m temp.tune_gen_5model_ensemble
"""
import glob
import json
import os
import zipfile

import numpy as np
import optuna
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.chunked_inference import predict_long_clip
from src.config import Config
from src.dataset import OfficialValidationDataset, pad_or_trim
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

VAL_ROOT = "./data/official_validation_raw/validation"
TEST_DIR = "./data/test_audio"
OUT_JSONL = "predictions_5model.jsonl"
OUT_ZIP = "submission_5model.zip"

MEMBERS = {
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


def load_wav(path, config):
    wav, sample_rate = torchaudio.load(path)
    wav = wav.mean(dim=0) if wav.size(0) > 1 else wav.squeeze(0)
    if sample_rate != config.SR:
        wav = torchaudio.functional.resample(wav, sample_rate, config.SR)
    return wav


class ShortClipDataset(Dataset):
    def __init__(self, file_paths, config):
        self.file_paths = file_paths
        self.config = config

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            wav = load_wav(path, self.config)
            wav = pad_or_trim(wav, self.config.target_samples)
        except Exception:
            wav = torch.zeros(self.config.target_samples)
        return wav, os.path.splitext(os.path.basename(path))[0]


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
    print(f"\n=== BEST (5-model ensemble, REAL validation set) ===", flush=True)
    print(f"p_stay_silence={ps:.4f} p_stay_event={pe:.4f} "
          f"natural_combined={best.user_attrs['natural_combined']:.4f} "
          f"synthetic_combined={best.user_attrs['synthetic_combined']:.4f} "
          f"eq_avg={best.value:.4f}", flush=True)

    print("\nGenerating submission with tuned params + chunked inference for long clips...", flush=True)
    all_files = sorted(glob.glob(os.path.join(TEST_DIR, "**", "*.wav"), recursive=True))
    short_files, long_files = [], []
    for path in tqdm(all_files, desc="checking durations"):
        try:
            info = sf.info(path)
            duration = info.frames / info.samplerate
        except Exception:
            duration = 0.0
        (short_files if duration <= config.MAX_DURATION else long_files).append(path)
    print(f"{len(short_files)} short clips (batched), {len(long_files)} long clips (chunked)", flush=True)

    entries = []
    dataset = ShortClipDataset(short_files, config)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    with torch.no_grad():
        for audio, clip_ids in tqdm(loader, desc="short clips (batched)"):
            audio = audio.to(device)
            probabilities = torch.sigmoid(sum(m(audio) for m in models) / len(models)).cpu().numpy()
            for probability, clip_id in zip(probabilities, clip_ids):
                states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
                events = extract_events_viterbi(states, hop)
                entries.append({"clip_id": clip_id, "events": events})

    for path in tqdm(long_files, desc="long clips (chunked)"):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = load_wav(path, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        probability = predict_long_clip(models, wav, config)
        states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
        events = extract_events_viterbi(states, hop)
        entries.append({"clip_id": clip_id, "events": events})

    assert len(entries) == len(all_files), f"entry count mismatch: {len(entries)} vs {len(all_files)}"

    with open(OUT_JSONL, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUT_JSONL, arcname="predictions.jsonl")
    print(f"Done -> {OUT_ZIP} ({len(entries)} entries, {len(long_files)} used chunked inference)", flush=True)


if __name__ == "__main__":
    main()
