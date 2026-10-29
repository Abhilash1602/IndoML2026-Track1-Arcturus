"""Extract Silver clips (1-2 approximate events) to disk as synthesis backgrounds,
carrying their approximate event labels. src/synthesis.py adds Gold-extracted loud
events on top at exact times; the final label list = pre-existing Silver events +
added events.

Output: data/bg_silver/audio/<id>.wav  +  data/bg_silver/bg_labels.jsonl
"""
import json
import os

import numpy as np
import soundfile as sf

from src.dataset import load_full_dataset

SR = 16000
OUT = "data/bg_silver"
N = 6000


def main():
    os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
    ds = load_full_dataset()
    tbl = ds.data.table if hasattr(ds.data, "table") else ds.data
    q = tbl.column("annotationQuality").to_pylist()
    ts = tbl.column("NoiseSubCategoryTimeStamp").to_pylist()

    cand = [i for i, (qi, t) in enumerate(zip(q, ts))
            if qi == "unverified_timestamps" and 1 <= len(t or []) <= 2]
    rng = np.random.default_rng(0)
    rng.shuffle(cand)
    cand = cand[:N]
    print(f"{len(cand)} Silver background candidates", flush=True)

    manifest = []
    for n, i in enumerate(cand):
        row = ds[i]
        try:
            wav = np.asarray(row["audio"]["array"], dtype=np.float32)
            sr = row["audio"]["sampling_rate"]
        except Exception:
            continue
        if sr != SR and len(wav):
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        if len(wav) < SR:  # < 1s, skip
            continue
        cid = f"bg_{n:06d}"
        sf.write(os.path.join(OUT, "audio", f"{cid}.wav"), wav, SR, subtype="PCM_16")
        evs = []
        for ev in (row["NoiseSubCategoryTimeStamp"] or []):
            if "start" in ev and "end" in ev and float(ev["end"]) > float(ev["start"]):
                evs.append({"onset": round(float(ev["start"]), 3), "offset": round(float(ev["end"]), 3)})
        manifest.append({"clip_id": cid, "path": f"audio/{cid}.wav", "events": evs})
        if (n + 1) % 1000 == 0:
            print(f"  {n+1}/{len(cand)}", flush=True)

    with open(os.path.join(OUT, "bg_labels.jsonl"), "w", encoding="utf-8") as fh:
        for r in manifest:
            fh.write(json.dumps(r) + "\n")
    print(f"DONE. {len(manifest)} backgrounds written.", flush=True)


if __name__ == "__main__":
    main()
