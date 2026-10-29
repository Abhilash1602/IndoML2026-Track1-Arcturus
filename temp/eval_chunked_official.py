"""Engineering-hack plan, item 2, verification: measure chunked/overlap-add
inference's real impact on OfficialValidationDataset HONESTLY, since that
dataset also truncates every clip to Config.MAX_DURATION via the same
pad_or_trim call LocalTestDataset uses (src/dataset.py:96-145's
OfficialValidationDataset.__getitem__ builds both audio and the ground-truth
mask from the truncated length). Only ~3.5-3.8% of validation clips exceed 10s
(vs. 15% of the real test set) -- this script re-scores the SAME 4-model
ensemble on the FULL, untruncated audio + full-length ground truth for every
validation clip, using src/chunked_inference.py's predict_long_clip (which
reduces to today's exact single-pass behavior for the ~96% of clips that are
already <= 10s), so the fix's real local delta can be measured before
trusting it enough to submit.

Usage: python -m temp.eval_chunked_official
"""
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.chunked_inference import predict_long_clip
from src.config import Config
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

VAL_ROOT = "./data/official_validation_raw/validation"
P_STAY_SILENCE, P_STAY_EVENT = 0.9883, 0.7954  # current best ensemble's tuned decode

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


def load_full_audio(path, config):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = torch.as_tensor(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=torch.float32)
    if sr != config.SR:
        audio = torchaudio.functional.resample(audio, sr, config.SR)
    return audio


def full_length_mask(row, n_frames, hop_sec):
    mask = np.zeros(n_frames, dtype=np.int32)
    for event in row.get("NoiseSubCategoryTimeStamp") or []:
        if "start" not in event or "end" not in event:
            continue
        start = min(max(int(float(event["start"]) / hop_sec), 0), n_frames - 1)
        end = min(max(int(float(event["end"]) / hop_sec), 0), n_frames - 1)
        if start < end:
            mask[start:end] = 1
    return mask


def score_subset(rows, models, config, hop_sec, tag):
    pred_events, gt_events = [], []
    n_long = 0
    for row in tqdm(rows, desc=tag):
        folder = "syntheticNoiseAudio" if row.get("syntheticData") else "naturalNoisyAudio"
        path = os.path.join(VAL_ROOT, folder, row["segmentFileName"])
        try:
            audio = load_full_audio(path, config)
        except Exception:
            continue
        if audio.numel() > config.target_samples:
            n_long += 1
        probability = predict_long_clip(models, audio, config)
        n_frames = len(probability)
        gt_mask = full_length_mask(row, n_frames, hop_sec)
        states = viterbi_decode(probability, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop_sec)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(gt_mask, hop_sec)])
    f1, _, _ = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop_sec)
    print(f"{tag}: {n_long}/{len(rows)} clips exceeded {config.MAX_DURATION}s "
          f"({100*n_long/len(rows):.1f}%)", flush=True)
    return dice, f1, dice + f1


def main():
    config = Config()
    hop_sec = config.HOP_LENGTH / config.SR
    models = build_models()

    with open(os.path.join(VAL_ROOT, "validationMetadata.json"), encoding="utf-8") as fh:
        rows = json.load(fh)
    nat_rows = [r for r in rows if not r.get("syntheticData")]
    syn_rows = [r for r in rows if r.get("syntheticData")]

    nd, nf, nc = score_subset(nat_rows, models, config, hop_sec, "natural (chunked, full-length)")
    sd, sf_, sc = score_subset(syn_rows, models, config, hop_sec, "synthetic (chunked, full-length)")
    eq = 0.5 * nc + 0.5 * sc

    print(f"\n=== Chunked/full-length eq_avg ===", flush=True)
    print(f"natural: dice={nd:.4f} f1={nf:.4f} combined={nc:.4f}", flush=True)
    print(f"synthetic: dice={sd:.4f} f1={sf_:.4f} combined={sc:.4f}", flush=True)
    print(f"eq_avg={eq:.4f}  (reference, truncated/known: 1.4937)", flush=True)
    if eq > 1.4937:
        print(f"Chunked inference HELPS locally by {eq-1.4937:+.4f} -- worth submitting "
              f"submission_chunked.zip.", flush=True)
    else:
        print(f"Chunked inference does not show a positive local delta ({eq-1.4937:+.4f}) -- "
              f"given the local proxy is known to underrepresent this fix's real-world impact "
              f"(only ~3.5-4% of validation clips affected vs. 15% of the real test set), "
              f"still consider submitting given the fix is a genuine correctness improvement, "
              f"not just a hopeful tweak.", flush=True)


if __name__ == "__main__":
    main()
