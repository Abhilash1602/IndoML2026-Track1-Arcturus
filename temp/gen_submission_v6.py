"""submission_v6 (plan Phase V2): the 5 v4 members fine-tuned with 100% of the official
validation set (VAL_TRAIN=all, tags VA_*, EMA teachers; organisers confirmed validation may be
used for fine-tuning), assembled exactly as the hold30 decision (scratchpad/v1_decision.json)
chose: optionally together with v4's original 5 members ("UNION"), optionally with v5's gated
edge correction. Same inference as v4/v5 (short clips batched, long clips chunked, logit
mean, Viterbi 0.9802/0.9489). Writes submission_v6.zip and validates it against v5.

Usage: python -m temp.gen_submission_v6
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

import temp.gen_submission_v4 as v4gen  # registers v4's members in v2.MEMBERS
import temp.search_ensemble_v2 as v2
from src.architecture import DualBranchSED, StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.edge_correction import correct_events, load_table
from src.utils import extract_events_viterbi, viterbi_decode

DECISION = os.path.join(v2.SCRATCH, "v1_decision.json")
VA = {  # fine-tuned-with-all-validation members (EMA teachers)
    "va_atst": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2_owbceVA_atst_teacher.pth", dict(USE_ATST_FRAME=True)),
    "va_owB": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceVA_owB_teacher.pth", {}),
    "va_bS2": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceVA_bS2_teacher.pth", {}),
    "va_owBt": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceVA_owBt_teacher.pth", {}),
    "va_bA": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceVA_bA_teacher.pth", {}),
}
OUT_JSONL, OUT_ZIP = "predictions_v6.jsonl", "submission_v6.zip"


def main():
    decision = json.load(open(DECISION))
    assert decision["commit"], "hold30 gate not met -- v6 must not be built"
    for name, (cls, path, ov) in VA.items():
        assert os.path.exists(path), f"missing {path}"
        v2.MEMBERS[name] = dict(cls=cls, path=path, overrides=ov)
    names = list(VA) + (list(v4gen.NAMES) if decision["union"] else [])
    correct = decision["correct"]
    print(f"v6 members: {names} | correction: {correct} | hold30 choice: {decision['choice']}", flush=True)

    config = Config()
    table = load_table("./checkpoints/edge_correction_v5.json")
    models = [v2.load_member(n) for n in names]
    ps, pe, hop = v4gen.PS, v4gen.PE, config.HOP_LENGTH / config.SR

    def to_events(prob, clip_sec):
        ev = [(e["onset"], e["offset"]) for e in extract_events_viterbi(
            viterbi_decode(prob, p_stay_silence=ps, p_stay_event=pe), hop)]
        if correct:
            ev = correct_events(ev, prob, table, clip_sec)
        return [{"onset": round(float(a), 3), "offset": round(float(b), 3)} for a, b in ev]

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
    entries, k = [], 0
    loader = DataLoader(v2.ShortClipDataset(short, config), batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    with torch.no_grad():
        for audio, ids in tqdm(loader, desc="short clips"):
            probs = v2.combine([m(audio.to(config.DEVICE)) for m in models], "logit").cpu().numpy()
            for prob, cid in zip(probs, ids):
                d = dur[short[k]]
                entries.append({"clip_id": cid, "events": to_events(prob, min(d, config.MAX_DURATION) if d > 0 else None)})
                k += 1
    for path in tqdm(long_, desc="long clips (chunked)"):
        cid = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = v2.load_wav(path, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        prob = v2.predict_long_clip_mode(models, wav, config, "logit")
        entries.append({"clip_id": cid, "events": to_events(prob, dur[path] or None)})
    assert len(entries) == len(files) and len({e["clip_id"] for e in entries}) == len(entries)
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_JSONL, arcname="predictions.jsonl")

    # ---- validation against v5 ----
    def load(zp):
        with zipfile.ZipFile(zp) as f:
            return f.namelist(), {json.loads(l)["clip_id"]: json.loads(l)["events"] for l in f.read("predictions.jsonl").decode().splitlines()}
    n6, a6 = load(OUT_ZIP)
    _, a5 = load("submission_v5.zip")
    ev6 = [(x["onset"], x["offset"]) for v in a6.values() for x in v]
    ev5 = [(x["onset"], x["offset"]) for v in a5.values() for x in v]
    bad = sum(not (0 <= s < e) for s, e in ev6)
    ok = n6 == ["predictions.jsonl"] and set(a6) == set(a5) and len(a6) == 5517 and bad == 0
    print(f"ZIP CHECK {'PASS' if ok else 'FAIL'}: files={n6} clips={len(a6)} same_ids_as_v5={set(a6) == set(a5)} "
          f"events v6={len(ev6)} v5={len(ev5)} malformed={bad} empty v6={sum(not v for v in a6.values())} "
          f"v5={sum(not v for v in a5.values())} median dur v6={np.median([e - s for s, e in ev6]):.3f}s "
          f"v5={np.median([e - s for s, e in ev5]):.3f}s", flush=True)
    assert ok, "zip check failed"


if __name__ == "__main__":
    main()
