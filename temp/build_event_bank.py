"""Phase A: extract isolated noise-event snippets from Gold clips by their precise
timestamps, to serve as the foreground-event bank for src/synthesis.py.

Skips [breathing]/[inhaling] (can't isolate cleanly from speech, and the Synthetic
eval subset almost certainly lacks speaker-breath events -- it's "clean speech +
ADDED noise"). Keeps the loud/diverse categories the current model underfits.

Output: data/event_bank/<category>/<clip>__<idx>.wav  +  data/event_bank/manifest.jsonl
"""
import io
import json
import os
from collections import Counter

import numpy as np
import soundfile as sf
from datasets import Audio

from src.dataset import load_full_dataset

SR = 16000
OUT = "data/event_bank"
SKIP_TAGS = {"[breathing]", "[inhaling]", "[exhaling]"}
PAD_SEC = 0.03          # context pad around the labelled span
MIN_DUR, MAX_DUR = 0.10, 8.0


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_full_dataset()  # audio decoded
    q = ds["annotationQuality"]
    gold_idx = [i for i, x in enumerate(q) if x == "verified_timestamps"]
    print(f"Gold clips: {len(gold_idx)}", flush=True)

    manifest = []
    cat_counts = Counter()
    for n, i in enumerate(gold_idx):
        row = ds[i]
        events = row.get("NoiseSubCategoryTimeStamp") or []
        if not events:
            continue
        try:
            wav = np.asarray(row["audio"]["array"], dtype=np.float32)
            file_sr = row["audio"]["sampling_rate"]
        except Exception:
            continue
        if file_sr != SR and len(wav):
            import librosa
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=SR)
        clip_stub = os.path.splitext(os.path.basename(row.get("imageFileName", f"clip_{i}")))[0]
        for j, ev in enumerate(events):
            tag = (ev.get("tag") or "").strip()
            cat = (ev.get("category") or "unknown").strip()
            if tag in SKIP_TAGS or "start" not in ev or "end" not in ev:
                continue
            s, e = float(ev["start"]), float(ev["end"])
            if not (MIN_DUR <= e - s <= MAX_DUR):
                continue
            a = max(0, int((s - PAD_SEC) * SR))
            b = min(len(wav), int((e + PAD_SEC) * SR))
            snip = wav[a:b]
            if len(snip) < int(MIN_DUR * SR):
                continue
            cat_dir = os.path.join(OUT, cat)
            os.makedirs(cat_dir, exist_ok=True)
            fn = f"{clip_stub}__{j}.wav"
            sf.write(os.path.join(cat_dir, fn), snip, SR, subtype="PCM_16")
            manifest.append({"path": os.path.join(cat, fn), "category": cat, "tag": tag,
                             "duration": round(len(snip) / SR, 3)})
            cat_counts[cat] += 1
        if (n + 1) % 1000 == 0:
            print(f"  {n+1}/{len(gold_idx)} clips, {sum(cat_counts.values())} events extracted", flush=True)

    with open(os.path.join(OUT, "manifest.jsonl"), "w", encoding="utf-8") as fh:
        for r in manifest:
            fh.write(json.dumps(r) + "\n")
    print(f"\nDONE. {len(manifest)} event snippets:", dict(cat_counts.most_common()), flush=True)


if __name__ == "__main__":
    main()
