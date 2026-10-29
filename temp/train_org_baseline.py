"""Oct 12: the organisers' own Track 1 baseline (v-3 WavLM SED, github.com/nagshubhadip/
noise-event-detection-and-removal, cloned in third_party/org_baseline) trained on our local data as
a new lineage. Its modules (config, data, model, losses, train) are imported unchanged; only the
data source and paths are adapted:
  * rows from the local HF cache (not streaming) with the repo's "full" budget caps (all Gold,
    Silver 20,000 with their human timestamps at weight 0.4, Bronze 8,000 clip tags), plus the
    official validation train70 clips as Gold-tier rows (exact metadata spans + categories);
  * their 10% Gold split selects the checkpoint; hold30 is never trained on or selected on;
  * after training, hold30 posteriors (any-noise channel, their whole-clip inference) are written to
    work/hold30_logits/ORG.npz on our 10 ms grid for the ensemble gate.

Usage: CUDA_VISIBLE_DEVICES=1 python -m temp.train_org_baseline
"""
import ast
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ORG = os.path.abspath("./third_party/org_baseline/track1_detection/versions/v-3_indoml2026_track1/modular")
sys.path.insert(0, ORG)
import config as oc  # noqa: E402  (the repo's own modules)

oc.WORK = Path("./work/org_baseline").absolute()
oc.WORK.mkdir(parents=True, exist_ok=True)
oc.CKPT = oc.WORK / "t1_wavlm_VT.pt"
oc.CFG["num_workers"] = 8
import data as od  # noqa: E402
import model as om  # noqa: E402
import train as otr  # noqa: E402
import inference as oi  # noqa: E402

CAPS = {"verified_timestamps": None, "unverified_timestamps": 20000, "no_timestamps": 8000}
VAL_ROOT = "./data/official_validation_raw/validation"


def hf_rows():
    from datasets import Audio
    from src.dataset import load_full_dataset
    ds = load_full_dataset().cast_column("audio", Audio(decode=False))
    q = ds["annotationQuality"]
    idx = {k: [i for i, x in enumerate(q) if x == k] for k in CAPS}
    rng = random.Random(42)
    keep = []
    for k, v in idx.items():
        rng.shuffle(v)
        keep += v if CAPS[k] is None else v[:CAPS[k]]
    rows = []
    for n, i in enumerate(sorted(keep)):
        ex = ds[i]
        try:
            w, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32", always_2d=False)
        except Exception:
            continue
        if w.ndim > 1:
            w = w.mean(1)
        if sr != oc.CFG["sr"]:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=oc.CFG["sr"])
        if len(w) < oc.CFG["sr"] * 0.2:
            continue
        qq = ex["annotationQuality"]
        strong = qq in ("verified_timestamps", "unverified_timestamps")
        rows.append(dict(wav=(np.clip(w, -1, 1) * 32767).astype(np.int16),
                         lab=od.build_labels(od.spans_from(ex), 1 + len(w) // oc.CFG["hop"], strong),
                         tags=od.clip_tags(ex), quality=qq, has_strong=strong))
        if n % 5000 == 0:
            print(f"decoded {n}/{len(keep)}", flush=True)
    return rows


def val_rows(part):
    split = json.load(open("./data/val_split_s0.json"))[part]
    names = set(split["natural"]) | set(split["synthetic"])
    meta = json.load(open(f"{VAL_ROOT}/validationMetadata.json", encoding="utf-8"))
    out = []
    for r in meta:
        if r["segmentFileName"] not in names:
            continue
        syn = str(r.get("syntheticData")) == "True"
        path = os.path.join(VAL_ROOT, "syntheticNoiseAudio" if syn else "naturalNoisyAudio", r["segmentFileName"])
        if not os.path.exists(path):
            path += ".wav"
        w, sr = sf.read(path, dtype="float32", always_2d=False)
        w = w if w.ndim == 1 else w.mean(1)
        spans = ast.literal_eval(r["NoiseSubCategoryTimeStamp"]) if isinstance(r["NoiseSubCategoryTimeStamp"], str) \
            else (r["NoiseSubCategoryTimeStamp"] or [])
        cats = ast.literal_eval(r["NoiseCategory"]) if isinstance(r["NoiseCategory"], str) else (r["NoiseCategory"] or [])
        ex = dict(NoiseSubCategoryTimeStamp=spans, NoiseCategory=cats)
        out.append(dict(name=r["segmentFileName"], syn=syn, wav_f=w, row=dict(
            wav=(np.clip(w, -1, 1) * 32767).astype(np.int16),
            lab=od.build_labels(od.spans_from(ex), 1 + len(w) // oc.CFG["hop"], True),
            tags=od.clip_tags(ex), quality="verified_timestamps", has_strong=True)))
    return out


def main():
    dev = "cuda"
    od.seed_torch()
    rows = hf_rows()
    gold, silver, bronze = od.bucket_by_tier(rows)
    v70 = [v["row"] for v in val_rows("train70")]
    random.Random(oc.CFG["seed"]).shuffle(gold)
    n_val = max(1, int(len(gold) * oc.CFG["val_frac"]))
    val, train_rows = gold[:n_val], gold[n_val:] + v70 + silver + bronze
    random.Random(oc.CFG["seed"]).shuffle(train_rows)
    print(f"train {len(train_rows)} (gold {len(gold) - n_val}, val70 {len(v70)}, silver {len(silver)}, bronze {len(bronze)}) "
          f"| select-val {len(val)} Gold", flush=True)
    from torch.utils.data import DataLoader
    tdl = DataLoader(od.SEDData(train_rows, True), batch_size=16, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    vdl = DataLoader(od.SEDData(val, False), batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    otr.train(tdl, vdl, dev, int(os.environ.get("ORG_S1", "6")), int(os.environ.get("ORG_S2", "16")))

    # hold30 posteriors on our 10 ms grid (first 10 s, as the cached ensemble logits)
    ck = torch.load(oc.CKPT, map_location=dev, weights_only=False)
    model = om.WavLMSED().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    import importlib.util
    from src.config import Config
    spec = importlib.util.spec_from_file_location("vaani_train", os.path.abspath("./train.py"))
    vt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vt)  # our train.py (the repo's own `train` module shadows the name)
    official_val_subset = vt.official_val_subset
    hold = {v["name"]: v for v in val_rows("hold30")}
    out = {}
    for sub in ("natural", "synthetic"):
        ds = official_val_subset(Config(), sub, "hold30")
        order = [ds.dataset.rows[i]["segmentFileName"] for i in ds.indices]
        L = np.full((len(order), 1000), -9.0, np.float32)
        for k, name in enumerate(order):
            post, _ = oi.t1_posteriors(hold[name]["wav_f"], model, dev)
            p10 = np.repeat(post[0], 2)[:1000]
            p10 = np.clip(p10, 1e-4, 1 - 1e-4)
            L[k, :len(p10)] = np.log(p10 / (1 - p10))
        out[sub] = L
    os.makedirs("./work/hold30_logits", exist_ok=True)
    np.savez("./work/hold30_logits/ORG.npz", **out)
    print(f"saved hold30 logits; checkpoint {oc.CKPT} ({ck['which']}, select-val {ck['score']:.4f})", flush=True)


if __name__ == "__main__":
    main()
