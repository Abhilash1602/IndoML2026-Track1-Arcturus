"""Engineering-hack plan, item 2: generate a submission using chunked/overlap-add
inference (src/chunked_inference.py) for clips exceeding Config.MAX_DURATION,
instead of generate_submission.py's silent pad_or_trim truncation. Measured:
15% of a 200-file real test-set sample exceeds 10s (up to 19.6s) and was being
silently cut off in every submission generated so far this session.

Short clips (<= MAX_DURATION, ~85% of the test set) still go through the fast
batched DataLoader path exactly as before -- only the long clips are routed
through the new per-clip chunked path, keeping this close to today's runtime.

Usage: python -m temp.gen_submission_chunked
"""
import glob
import json
import os
import zipfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.chunked_inference import predict_long_clip
from src.config import Config
from src.dataset import pad_or_trim
from src.utils import extract_events_viterbi, viterbi_decode

TEST_DIR = "./data/test_audio"
OUT_JSONL = "predictions_chunked.jsonl"
OUT_ZIP = "submission_chunked.zip"

# Current best ensemble (STRATEGY.md: real Codabench 1.5073) and its tuned decode.
MEMBERS = {
    "phaseh": dict(path="./checkpoints/dualbranch_phaseh.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "wavlm_full1": dict(path="./checkpoints/dualbranch_wavlm_stage1_full.pth", overrides=dict(USE_WAVLM=True)),
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)),
    "atst_frame": dict(path="./checkpoints/dualbranch_atst_frame_stage1_full.pth",
                        overrides=dict(USE_ATST_FRAME=True)),
}
P_STAY_SILENCE, P_STAY_EVENT = 0.9883, 0.7954


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


def load_wav(path, config):
    wav, sample_rate = torchaudio.load(path)
    wav = wav.mean(dim=0) if wav.size(0) > 1 else wav.squeeze(0)
    if sample_rate != config.SR:
        wav = torchaudio.functional.resample(wav, sample_rate, config.SR)
    return wav


class ShortClipDataset(Dataset):
    """Same fixed-length batched path as generate_submission.py's LocalTestDataset,
    restricted to clips already known to be <= MAX_DURATION."""

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
    models = build_models()
    hop_sec = config.HOP_LENGTH / config.SR

    all_files = sorted(glob.glob(os.path.join(TEST_DIR, "**", "*.wav"), recursive=True))
    print(f"{len(all_files)} test files found, checking durations...", flush=True)

    short_files, long_files = [], []
    for path in tqdm(all_files, desc="checking durations"):
        try:
            info = sf.info(path)
            duration = info.frames / info.samplerate
        except Exception:
            duration = 0.0
        (short_files if duration <= config.MAX_DURATION else long_files).append(path)

    print(f"{len(short_files)} short clips (<= {config.MAX_DURATION}s, batched path), "
          f"{len(long_files)} long clips (> {config.MAX_DURATION}s, chunked path, "
          f"{100 * len(long_files) / len(all_files):.1f}% of total)", flush=True)

    entries = []

    # Fast batched path for short clips -- identical to today's behavior.
    dataset = ShortClipDataset(short_files, config)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    with torch.no_grad():
        for audio, clip_ids in tqdm(loader, desc="short clips (batched)"):
            audio = audio.to(config.DEVICE)
            probabilities = torch.sigmoid(sum(m(audio) for m in models) / len(models)).cpu().numpy()
            for probability, clip_id in zip(probabilities, clip_ids):
                states = viterbi_decode(probability, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT)
                events = extract_events_viterbi(states, hop_sec)
                entries.append({"clip_id": clip_id, "events": events})

    # Chunked overlap-add path for long clips.
    for path in tqdm(long_files, desc="long clips (chunked)"):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = load_wav(path, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        probability = predict_long_clip(models, wav, config)
        states = viterbi_decode(probability, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT)
        events = extract_events_viterbi(states, hop_sec)
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
