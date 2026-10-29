"""Scaper-style soundscape synthesis for Vaani noise-event SED.

Builds strong-labelled 10 s clips by mixing isolated noise events onto (relatively)
clean speech backgrounds at a controlled SNR -- reproducing the organizers' Synthetic
eval subset ("clean speech + synthetically added noise", per dataset_info.txt) so we
can (a) validate locally against a signal that predicts the Synthetic 40% of the
leaderboard, and (b) generate large strong-labelled training data for the loud/diverse
event categories the current model badly underfits (16-23% recall, per the Phase A
diagnostic).

No `scaper` dependency -- a ~120-line numpy/soundfile mixer is more robust than fighting
scaper's sox requirement, and this task only needs single-background + few-foreground
mixing.
"""
import numpy as np
import soundfile as sf

SR = 16000
CLIP_SEC = 10.0
CLIP_LEN = int(SR * CLIP_SEC)


def load_mono(path, sr=SR):
    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def fit_length(wav, target_len, rng, crop=True):
    """Returns (fitted_wav, crop_offset_sec). crop_offset is what to SUBTRACT from
    any pre-existing background-event timestamps so they stay valid after fitting."""
    if len(wav) == target_len:
        return wav, 0.0
    if len(wav) > target_len:
        start = int(rng.integers(0, len(wav) - target_len + 1)) if crop else 0
        return wav[start:start + target_len], start / SR
    return np.pad(wav, (0, target_len - len(wav))), 0.0


def trim_silence(ev, thresh_rel=0.02, pad_sec=0.01):
    """Trim leading/trailing near-silence from an extracted event snippet so its
    labelled onset/offset are tight, keeping a small pad."""
    if len(ev) == 0:
        return ev
    env = np.abs(ev)
    peak = env.max() + 1e-9
    active = np.where(env > thresh_rel * peak)[0]
    if len(active) == 0:
        return ev
    pad = int(pad_sec * SR)
    lo = max(0, active[0] - pad)
    hi = min(len(ev), active[-1] + 1 + pad)
    return ev[lo:hi]


def synthesize_clip(background_path, event_paths, *, n_events=(1, 3),
                    snr_db=(0.0, 15.0), min_gap_sec=0.2, event_len_sec=(0.15, 6.0),
                    background_events=None, crop=True, rng=None):
    """Mix `n_events` events (sampled from `event_paths`) onto the background at a
    sampled per-event SNR relative to the background RMS.

    background_events: optional list of {"onset","offset"} for noise already present
    in the background clip -- merged into the returned label list (shifted for any
    crop) so the labels are complete. Pass crop=False for validation sets so those
    timestamps stay exactly valid.

    Returns (mixed_wav float32 [CLIP_LEN], [(onset_sec, offset_sec), ...] sorted).
    """
    rng = rng or np.random.default_rng()
    bg, crop_off = fit_length(load_mono(background_path), CLIP_LEN, rng, crop=crop)
    peak = np.max(np.abs(bg)) + 1e-9
    bg = bg / peak * 0.7
    bg_rms = rms(bg)
    if bg_rms < 1e-4:
        bg_rms = 1e-3  # near-silent background: give events something to scale against

    k = int(rng.integers(n_events[0], n_events[1] + 1))
    idxs = rng.permutation(len(event_paths))[:max(k * 3, k)]  # oversample; some get skipped
    mix = bg.copy()
    occupied = []  # (start_frame, end_frame)
    labels = []
    for idx in idxs:
        if len(labels) >= k:
            break
        try:
            ev = trim_silence(load_mono(event_paths[int(idx)]))
        except Exception:
            continue
        dur = len(ev) / SR
        if not (event_len_sec[0] <= dur <= event_len_sec[1]) or len(ev) >= CLIP_LEN:
            continue
        gap = int(min_gap_sec * SR)
        placed_start = None
        for _ in range(25):
            start = int(rng.integers(0, CLIP_LEN - len(ev)))
            end = start + len(ev)
            if all(end + gap <= s or start - gap >= e for (s, e) in occupied):
                placed_start = start
                break
        if placed_start is None:
            continue
        end = placed_start + len(ev)
        snr = float(rng.uniform(*snr_db))
        target_rms = bg_rms * (10.0 ** (snr / 20.0))
        ev = ev / (rms(ev) + 1e-9) * target_rms
        # light fade to avoid click artifacts at placement edges
        fade = min(int(0.005 * SR), len(ev) // 4)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            ev[:fade] *= ramp
            ev[-fade:] *= ramp[::-1]
        mix[placed_start:end] += ev
        occupied.append((placed_start, end))
        labels.append((placed_start / SR, end / SR))

    for ev in (background_events or []):
        a, b = ev["onset"] - crop_off, ev["offset"] - crop_off
        a, b = max(0.0, a), min(CLIP_SEC, b)
        if b - a >= 0.05:
            labels.append((a, b))

    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 1.0:
        mix = mix / peak
    labels.sort()
    return mix.astype(np.float32), labels


def write_dataset(out_dir, backgrounds, event_paths, n_clips, *, prefix="syn",
                  seed=0, **synth_kwargs):
    """Generate `n_clips` synthetic clips into `out_dir/audio/` and a
    `out_dir/labels.jsonl` of {"clip_id", "events":[{"onset","offset"}]}."""
    import json
    import os

    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []
    for i in range(n_clips):
        bg = backgrounds[int(rng.integers(0, len(backgrounds)))]
        wav, labels = synthesize_clip(bg, event_paths, rng=rng, **synth_kwargs)
        clip_id = f"{prefix}_{i:06d}"
        sf.write(os.path.join(out_dir, "audio", f"{clip_id}.wav"), wav, SR, subtype="PCM_16")
        manifest.append({"clip_id": clip_id,
                         "events": [{"onset": round(a, 3), "offset": round(b, 3)} for a, b in labels]})
    with open(os.path.join(out_dir, "labels.jsonl"), "w", encoding="utf-8") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")
    return manifest
