"""submission_v5 = submission_v4 (atst_stage2 + dasheng_owB + dasheng_owB_t + beats_s2 +
beats_owA, logit mean, Viterbi 0.9802/0.9489, same short/chunked inference) + the gated edge
correction (src/edge_correction.py, table fitted on the Gold held-out split, not validation:
checkpoints/edge_correction_v5.json). Validation: pooled 1.4553 -> 1.4598 (+0.0045, bootstrap
P(>0)=0.988), Natural F1 0.3839 -> 0.3955, synthetic essentially unchanged.

Usage: python -m temp.gen_submission_v5
"""
import glob
import json
import os
import zipfile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import temp.gen_submission_v4 as v4gen  # registers the v4 members in v2.MEMBERS
import temp.search_ensemble_v2 as v2
from src.config import Config
from src.edge_correction import correct_events, load_table
from src.utils import extract_events_viterbi, viterbi_decode

TABLE = "./checkpoints/edge_correction_v5.json"
OUT_JSONL, OUT_ZIP = "predictions_v5.jsonl", "submission_v5.zip"


def to_events(prob, table, ps, pe, hop, clip_sec):
    ev = [(e["onset"], e["offset"]) for e in extract_events_viterbi(viterbi_decode(prob, p_stay_silence=ps, p_stay_event=pe), hop)]
    return [{"onset": round(float(a), 3), "offset": round(float(b), 3)} for a, b in correct_events(ev, prob, table, clip_sec)]


def main():
    config = Config()
    table = load_table(TABLE)
    models = [v2.load_member(n) for n in v4gen.NAMES]
    ps, pe, mode, hop = v4gen.PS, v4gen.PE, v4gen.MODE, config.HOP_LENGTH / config.SR
    files = sorted(glob.glob(os.path.join(v2.TEST_DIR, "**", "*.wav"), recursive=True))
    short, long_, dur = [], [], {}
    for path in files:
        try:
            info = sf.info(path)
            d = info.frames / info.samplerate
        except Exception:
            d = 0.0
        dur[path] = d
        (short if d <= config.MAX_DURATION else long_).append(path)
    entries = []
    loader = DataLoader(v2.ShortClipDataset(short, config), batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    k = 0
    with torch.no_grad():
        for audio, ids in tqdm(loader, desc="short clips"):
            probs = v2.combine([m(audio.to(config.DEVICE)) for m in models], mode).cpu().numpy()
            for prob, cid in zip(probs, ids):
                clip_sec = min(dur[short[k]], config.MAX_DURATION) if dur[short[k]] > 0 else None
                entries.append({"clip_id": cid, "events": to_events(prob, table, ps, pe, hop, clip_sec)})
                k += 1
    for path in tqdm(long_, desc="long clips (chunked)"):
        cid = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = v2.load_wav(path, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        prob = v2.predict_long_clip_mode(models, wav, config, mode)
        entries.append({"clip_id": cid, "events": to_events(prob, table, ps, pe, hop, dur[path] or None)})
    assert len(entries) == len(files) and len({e["clip_id"] for e in entries}) == len(entries)
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_JSONL, arcname="predictions.jsonl")
    print(f"Done -> {OUT_ZIP} ({len(entries)} clips, {len(long_)} chunked)", flush=True)


if __name__ == "__main__":
    main()
