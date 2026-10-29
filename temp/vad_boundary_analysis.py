"""S1 measurement: do human noise-event boundaries line up with speech boundaries?

Silero VAD frame probabilities (512-sample = 32 ms windows at 16 kHz) for every official
validation clip are cached to scratchpad/vad_val.npz. On train70 Natural clips only (hold30
stays untouched), for each reference noise event:
  offset gap = (first speech onset at/after the event's offset - 0.3 s) - event offset
  onset gap  = event onset - (last speech offset at/before the event's onset + 0.3 s)
compared with a null where each event is shifted by a random +-0.25..0.75 s.
Reported overall and for breaths (human_non_speech) separately, plus how our v4-ensemble
predicted offsets relate to the same speech onsets.

Usage: python -m temp.vad_boundary_analysis
"""
import ast
import json
import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = "./data/official_validation_raw/validation"
S = "./work"
CACHE = os.path.join(S, "vad_val.npz")
WIN = 512  # samples per Silero window at 16 kHz = 32 ms
STEP = WIN / 16000.0


def vad_probs():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return dict(zip(d["names"], d["probs"]))
    torch.set_num_threads(8)
    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    rows = json.load(open(f"{ROOT}/validationMetadata.json", encoding="utf-8"))
    out = {}
    for r in tqdm(rows, desc="VAD"):
        folder = "syntheticNoiseAudio" if r.get("syntheticData") else "naturalNoisyAudio"
        w, sr = sf.read(os.path.join(ROOT, folder, r["segmentFileName"]), dtype="float32")
        w = w if w.ndim == 1 else w.mean(1)
        assert sr == 16000
        model.reset_states()
        x = torch.from_numpy(w)
        n = len(w) // WIN
        with torch.no_grad():
            p = [float(model(x[i * WIN:(i + 1) * WIN], sr)) for i in range(n)]
        out[r["segmentFileName"]] = np.array(p, np.float32)
    np.savez(CACHE, names=np.array(list(out)), probs=np.array(list(out.values()), dtype=object))
    return out


def speech_edges(p, thr=0.5):
    on = p >= thr
    ups = [(i) * STEP for i in range(1, len(on)) if on[i] and not on[i - 1]]
    downs = [(i) * STEP for i in range(1, len(on)) if not on[i] and on[i - 1]]
    return np.array(ups), np.array(downs)


def gaps(events, ups, downs):
    off_g, on_g = [], []
    for s, e in events:
        nxt = ups[ups >= e - 0.3]
        if len(nxt):
            off_g.append(nxt[0] - e)
        prv = downs[downs <= s + 0.3]
        if len(prv):
            on_g.append(s - prv[-1])
    return off_g, on_g


def summarise(name, g):
    g = np.array(g)
    if not len(g):
        return f"{name}: n=0"
    return (f"{name}: n={len(g)} median={np.median(g):+.3f}s | |gap|<=0.05s {np.mean(np.abs(g) <= 0.05):.1%} "
            f"<=0.1s {np.mean(np.abs(g) <= 0.1):.1%} <=0.2s {np.mean(np.abs(g) <= 0.2):.1%}")


def main():
    probs = vad_probs()
    rows = {r["segmentFileName"]: r for r in json.load(open(f"{ROOT}/validationMetadata.json", encoding="utf-8"))}
    train = json.load(open("./data/val_split_s0.json"))["train70"]["natural"]
    rng = np.random.default_rng(0)
    real = {"all": ([], []), "breath": ([], [])}
    null = {"all": ([], []), "breath": ([], [])}
    for name in train:
        v = rows[name].get("NoiseSubCategoryTimeStamp") or "[]"
        v = ast.literal_eval(v) if isinstance(v, str) else v
        ups, downs = speech_edges(probs[name])
        for e in v:
            s, t = float(e["start"]), float(e["end"])
            if t <= s:
                continue
            keys = ["all"] + (["breath"] if e["category"] == "human_non_speech" else [])
            sh = rng.choice([-1, 1]) * rng.uniform(0.25, 0.75)
            for k in keys:
                o, n = gaps([(s, t)], ups, downs)
                real[k][0].extend(o); real[k][1].extend(n)
                o, n = gaps([(s + sh, t + sh)], ups, downs)
                null[k][0].extend(o); null[k][1].extend(n)
    speech_frac = np.mean([np.mean(probs[n] >= 0.5) for n in train])
    print(f"train70 Natural clips: {len(train)}; mean speech fraction {speech_frac:.2f}", flush=True)
    for k in ("all", "breath"):
        print(f"\n== {k} events ==", flush=True)
        print("  " + summarise("offset -> next speech onset  REAL", real[k][0]), flush=True)
        print("  " + summarise("offset -> next speech onset  NULL", null[k][0]), flush=True)
        print("  " + summarise("prev speech offset -> onset   REAL", real[k][1]), flush=True)
        print("  " + summarise("prev speech offset -> onset   NULL", null[k][1]), flush=True)


if __name__ == "__main__":
    main()
