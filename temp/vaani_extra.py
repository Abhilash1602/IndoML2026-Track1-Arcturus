"""Extra in-domain unlabelled audio for self-training (plan Phase N1b): the public Vaani corpus
(ARTPARK-IISc/Vaani on Hugging Face, CC-BY-4.0), streamed per language config.

Overlap guard: the Track 1 test set is "split by state/district/speaker" from the same corpus, so
every (state, district) present in the 5,517 test clip IDs (work/test_districts.txt, 156 districts)
is excluded, and so is any speaker id seen in excluded districts. Clips must have a transcript
with at least one noise tag (<...> or [...]), the Bronze-style weak "noise present" signal. Per
district / per speaker caps keep it diverse. Audio -> 16 kHz mono, first 10 s, int16.

Output: work/vaani_extra/<language>.npz (audio int16 (n, 160000), ids, district, speaker) and
work/vaani_extra/summary.json (counts, districts used, exclusions).

Usage: VE_LANGS=Hindi,Bengali VE_TARGET=4000 python -m temp.vaani_extra
"""
import io
import itertools
import json
import os
import re

import numpy as np

OUT = "./work/vaani_extra"
TAG = re.compile(r"<[^<>]{2,40}>|\[[A-Za-z_ ]{2,30}\]")
PER_DISTRICT, PER_SPEAKER = 800, 12


def test_districts():
    return {tuple(l.rstrip("\n").split("\t")) for l in open("./work/test_districts.txt") if l.strip()}


def norm(x):
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def main():
    import soundfile as sf
    import torch
    import torchaudio
    from datasets import Audio, load_dataset
    os.makedirs(OUT, exist_ok=True)
    langs = os.environ.get("VE_LANGS", "Hindi").split(",")
    target = int(os.environ.get("VE_TARGET", "4000"))
    excl = {(norm(s), norm(d)) for s, d in test_districts()}
    excl_d = {d for _, d in excl}
    summary = {}
    for lang in langs:
        path = f"{OUT}/{lang}.npz"
        if os.path.exists(path):
            continue
        ds = load_dataset("ARTPARK-IISc/Vaani", lang, streaming=True, token=os.environ.get("HF_TOKEN"))["train"]
        ds = ds.cast_column("audio", Audio(decode=False))
        audio, ids, dist, spk = [], [], [], []
        per_d, per_s, seen, skipped = {}, {}, 0, {"test district": 0, "no tag": 0, "cap": 0, "decode": 0}
        for r in ds:
            seen += 1
            if seen > 400000 or len(ids) >= target:
                break
            sd = (norm(r.get("state")), norm(r.get("district")))
            if sd in excl or sd[1] in excl_d:
                skipped["test district"] += 1
                continue
            if r.get("isTranscriptionAvailable") != "Yes" or not TAG.search(r.get("transcript") or ""):
                skipped["no tag"] += 1
                continue
            if per_d.get(sd, 0) >= PER_DISTRICT or per_s.get(r.get("speakerID"), 0) >= PER_SPEAKER:
                skipped["cap"] += 1
                continue
            try:
                w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                w = torch.as_tensor(w if w.ndim == 1 else w.mean(1))
                if sr != 16000:
                    w = torchaudio.functional.resample(w, sr, 16000)
            except Exception:
                skipped["decode"] += 1
                continue
            w = torch.nn.functional.pad(w, (0, max(0, 160000 - w.numel())))[:160000]
            audio.append((w.clamp(-1, 1).numpy() * 32767).astype(np.int16))
            ids.append(f"vx_{lang}_{len(ids)}")
            dist.append("/".join(sd))
            spk.append(r.get("speakerID") or "")
            per_d[sd] = per_d.get(sd, 0) + 1
            per_s[r.get("speakerID")] = per_s.get(r.get("speakerID"), 0) + 1
            if len(ids) % 500 == 0:
                print(f"{lang}: {len(ids)} kept / {seen} seen; skipped {skipped}", flush=True)
        if ids:
            np.savez(path, audio=np.stack(audio), ids=np.array(ids), district=np.array(dist), speaker=np.array(spk))
        summary[lang] = dict(kept=len(ids), seen=seen, skipped=skipped, districts=sorted(set(dist)))
        assert not any(tuple(d.split("/")) in excl for d in dist), "test district leaked"
        print(f"{lang}: saved {len(ids)} clips from {len(set(dist))} non-test districts; skipped {skipped}", flush=True)
        json.dump(summary, open(f"{OUT}/summary_{lang}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
    os._exit(0)  # datasets streaming threads can crash interpreter shutdown
