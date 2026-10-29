"""Engineering-hack plan, item 2: chunked/overlap-add inference for clips
exceeding Config.MAX_DURATION.

Every dataset path in this project (LocalTestDataset, OfficialValidationDataset,
VaaniDataset) calls pad_or_trim (src/dataset.py), which hard-truncates any clip
longer than target_samples via `audio[:target_len]` -- silently dropping all
audio and events past that point. Measured directly: 15% of a 200-file real
test-set sample exceeds the 10s cap (up to 19.6s), while only ~3.5-3.8% of the
local validation set does -- the trusted local proxy underestimates this bug's
real-world impact by roughly 4x.

`predict_long_clip` fixes this at inference time only (no retraining, no model
changes): for clips longer than one window, it splits the waveform into
overlapping fixed-length windows (the same length every model was trained on),
runs the existing ensemble on each window exactly as today, and merges the
per-window probability sequences by averaging in the overlapping regions --
standard overlap-add for long-form audio (the same idea used for MAESTRO-style
long recordings in recent DCASE Task 4 systems). For clips at or under one
window, this reduces to exactly today's single forward pass.

`viterbi_decode`/`extract_events_viterbi` (src/utils.py) already operate on an
arbitrary-length 1D probability array -- nothing there needs to change.
"""
import numpy as np
import torch

from src.dataset import pad_or_trim


@torch.no_grad()
def predict_long_clip(models, wav, config, window_sec=10.0, overlap_sec=2.0):
    """wav: 1D torch tensor, raw waveform at config.SR, ARBITRARY length (not yet
    padded/trimmed). Returns a 1D numpy array of ensemble-averaged sigmoid
    probabilities at config.HOP_LENGTH/config.SR resolution, covering the clip's
    own real duration (wav.numel() // config.HOP_LENGTH frames) -- never the
    padded length.
    """
    window_samples = int(window_sec * config.SR)
    hop = config.HOP_LENGTH
    total_frames = wav.numel() // hop

    if wav.numel() <= window_samples:
        # Reduces to exactly today's single padded forward pass.
        x = pad_or_trim(wav, window_samples).unsqueeze(0).to(config.DEVICE)
        probs = torch.sigmoid(sum(m(x) for m in models) / len(models))[0].cpu().numpy()
        return probs[:total_frames]

    overlap_samples = int(overlap_sec * config.SR)
    stride_samples = window_samples - overlap_samples
    assert stride_samples > 0, "overlap_sec must be smaller than window_sec"

    starts = list(range(0, wav.numel() - window_samples + 1, stride_samples))
    if not starts or starts[-1] + window_samples < wav.numel():
        starts.append(wav.numel() - window_samples)  # tail-aligned final window, never dropped

    prob_sum = np.zeros(total_frames, dtype=np.float64)
    weight_sum = np.zeros(total_frames, dtype=np.float64)
    for start in starts:
        chunk = wav[start:start + window_samples].unsqueeze(0).to(config.DEVICE)
        probs = torch.sigmoid(sum(m(chunk) for m in models) / len(models))[0].cpu().numpy()
        frame_start = start // hop
        frame_end = min(frame_start + len(probs), total_frames)
        n_eff = frame_end - frame_start
        prob_sum[frame_start:frame_end] += probs[:n_eff]
        weight_sum[frame_start:frame_end] += 1.0

    # Every frame is covered by at least one window by construction (starts
    # span [0, wav.numel()-window_samples] with the tail explicitly appended).
    assert np.all(weight_sum > 0), "chunked_inference: found an uncovered frame -- window coverage bug"
    return prob_sum / weight_sum
