"""Plan lever C: long-clip inference. 13% of test clips are over 10 s (717/5517) vs 3.7%
of validation, and local evaluation (OfficialValidationDataset pads/trims to 10 s) has
never scored anything past 10 s. This scores the validation clips over 10 s at FULL
length with the v4 ensemble (logit mean, decode 0.9802/0.9489) under:
  trunc10     -- predictions only for the first 10 s (what local eval implicitly did)
  chunk_o2    -- current submission path: 10 s windows, 2 s overlap, flat average
  chunk_o5    -- 5 s overlap, flat average
  chunk_o5tri -- 5 s overlap, triangular cross-fade weights
References are the full-length merged annotations from validationMetadata.json.

Usage: python -m temp.long_clip_eval
"""
import ast
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

import temp.search_ensemble_v2 as v2
import temp.search_ensemble_v3 as v3
from src.config import Config
from src.dataset import pad_or_trim
from src.utils import extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

MEMBERS = ("atst_stage2", "dasheng_owB", "beats_s2", "dasheng_owB_t", "beats_owA")
ROOT = "./data/official_validation_raw/validation"
PS, PE, HOP = 0.9802, 0.9489, 0.01


def merged_refs(row, dur):
    v = row.get("NoiseSubCategoryTimeStamp") or "[]"
    v = ast.literal_eval(v) if isinstance(v, str) else v
    n = int(round(dur * 100))
    m = np.zeros(n + 1, bool)
    for e in v:
        s, t = float(e["start"]), float(e["end"])
        if t > s:
            m[max(0, int(round(s * 100))):min(n, int(round(t * 100)))] = True
    ev, start = [], None
    for i, x in enumerate(m):
        if x and start is None:
            start = i
        elif not x and start is not None:
            ev.append((start * HOP, i * HOP))
            start = None
    return ev


@torch.no_grad()
def member_logits(model, wav, config, overlap_sec, tri):
    win = int(10.0 * config.SR)
    total = wav.numel() // config.HOP_LENGTH
    if wav.numel() <= win:
        x = pad_or_trim(wav, win).unsqueeze(0).to(config.DEVICE)
        return model(x).float()[0].cpu().numpy()[:total]
    stride = win - int(overlap_sec * config.SR)
    starts = list(range(0, wav.numel() - win + 1, stride))
    if starts[-1] + win < wav.numel():
        starts.append(wav.numel() - win)
    acc, wsum = np.zeros(total), np.zeros(total)
    for st in starts:
        out = model(wav[st:st + win].unsqueeze(0).to(config.DEVICE)).float()[0].cpu().numpy()
        f0 = st // config.HOP_LENGTH
        f1 = min(f0 + len(out), total)
        n = f1 - f0
        w = np.ones(n)
        if tri:
            ramp = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(float)
            w = np.minimum(ramp / (overlap_sec * 100 / 2), 1.0) + 1e-3
        acc[f0:f1] += out[:n] * w
        wsum[f0:f1] += w
    return acc / wsum


def main():
    for name, (cls, path, overrides) in v3.NEW_MEMBERS.items():
        if os.path.exists(path):
            v2.MEMBERS[name] = dict(cls=cls, path=path, overrides=overrides)
    config = Config()
    rows = [r for r in json.load(open(os.path.join(ROOT, "validationMetadata.json"), encoding="utf-8"))
            if float(r["duration"]) > 10.0]
    wavs, refs, dom = [], [], []
    for r in rows:
        folder = "syntheticNoiseAudio" if r.get("syntheticData") else "naturalNoisyAudio"
        w, sr = sf.read(os.path.join(ROOT, folder, r["segmentFileName"]), dtype="float32", always_2d=False)
        w = torch.as_tensor(w if w.ndim == 1 else w.mean(axis=1))
        if sr != config.SR:
            w = torchaudio.functional.resample(w, sr, config.SR)
        wavs.append(w)
        refs.append(merged_refs(r, w.numel() / config.SR))
        dom.append("synthetic" if r.get("syntheticData") else "natural")
    print(f"{len(rows)} validation clips over 10 s ({dom.count('natural')} natural, {dom.count('synthetic')} synthetic); "
          f"{sum(map(len, refs))} reference events, {sum(1 for e in refs for _, t in e if t > 10.0)} ending after 10 s", flush=True)

    variants = {"chunk_o2": (2.0, False), "chunk_o5": (5.0, False), "chunk_o5tri": (5.0, True)}
    logits = {v: [np.zeros(w.numel() // config.HOP_LENGTH) for w in wavs] for v in variants}
    for member in MEMBERS:
        model = v2.load_member(member)
        for v, (ov, tri) in variants.items():
            for i, w in enumerate(wavs):
                logits[v][i] += member_logits(model, w, config, ov, tri) / len(MEMBERS)
        del model
        torch.cuda.empty_cache()
    trunc = [np.where(np.arange(len(x)) < 1000, x, -20.0) for x in logits["chunk_o2"]]
    # trunc10 uses the chunk_o2 first-window logits for frames < 1000 and predicts nothing after 10 s
    logits["trunc10"] = trunc
    for v in ("trunc10", "chunk_o2", "chunk_o5", "chunk_o5tri"):
        preds = [[(e["onset"], e["offset"]) for e in extract_events_viterbi(
            viterbi_decode(1 / (1 + np.exp(-x)), p_stay_silence=PS, p_stay_event=PE), HOP)] for x in logits[v]]
        line = []
        for d in ("natural", "synthetic", "all"):
            idx = [i for i in range(len(rows)) if d == "all" or dom[i] == d]
            f, p, r = micro_event_f1([refs[i] for i in idx], [preds[i] for i in idx])
            dice = macro_segment_dice([refs[i] for i in idx], [preds[i] for i in idx], hop_sec=HOP)
            line.append(f"{d}: F1={f:.4f} Dice={dice:.4f}")
        print(f"[{v:11s}] " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
