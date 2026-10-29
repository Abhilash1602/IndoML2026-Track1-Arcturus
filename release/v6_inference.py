"""IndoML 2026 Track 1 (noise event detection) -- inference for our final submission (v6).

Ensemble of 10 frame-level sound-event detectors (5 fine-tuned on Gold + official validation,
5 earlier Stage-2 members), logit-averaged at 10 ms, decoded with a 2-state Viterbi, then the
gated edge correction for Natural-looking short events. Clips longer than 10 s are processed in
10 s windows with 2 s overlap and averaged. Reproduces submission_v6.zip exactly.
"""
import glob
import json
import os
import zipfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.architecture import DualBranchSED, StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.dataset import pad_or_trim
from src.edge_correction import correct_events, load_table
from src.utils import extract_events_viterbi, viterbi_decode

P_STAY_SILENCE, P_STAY_EVENT = 0.9802, 0.9489
WINDOW_SEC, OVERLAP_SEC = 10.0, 2.0
ATST = dict(USE_ATST_FRAME=True)
# Order matters for bit-exact reproduction (floating-point summation order of the logit mean).
MEMBERS = [
    # fine-tuned with Gold + 100% of the official validation set (EMA teachers)
    ("va_atst", DualBranchSED, "dualbranch_atst_frame_stage2_i2_owbceVA_atst_teacher.pth", ATST),
    ("va_owB", StandaloneDashengSED, "standalone_dasheng_stage2_i2_owbceVA_owB_teacher.pth", {}),
    ("va_bS2", StandaloneBEATsSED, "standalone_beats_stage2_owbceVA_bS2_teacher.pth", {}),
    ("va_owBt", StandaloneDashengSED, "standalone_dasheng_stage2_i2_owbceVA_owBt_teacher.pth", {}),
    ("va_bA", StandaloneBEATsSED, "standalone_beats_stage2_owbceVA_bA_teacher.pth", {}),
    # earlier Stage-2 members (submission_v4)
    ("atst_stage2", DualBranchSED, "dualbranch_atst_frame_stage2.pth", ATST),
    ("dasheng_owB", StandaloneDashengSED, "standalone_dasheng_stage2_i2_owbceB.pth", {}),
    ("dasheng_owB_t", StandaloneDashengSED, "standalone_dasheng_stage2_i2_owbceB_teacher.pth", {}),
    ("beats_s2", StandaloneBEATsSED, "standalone_beats_stage2.pth", {}),
    ("beats_owA", StandaloneBEATsSED, "standalone_beats_stage2_owbceA.pth", {}),
]
CORRECTION_TABLE = "edge_correction_v5.json"
# Every environment-controlled switch in src/config.py pinned to the value used for v6.
PINNED = dict(BEATS_FREQ_RESHAPE=False, DUAL_HEAD=False, CAT_HEAD=False, OUTPUT_HEAD="nat",
              USE_CATEGORY_HEAD=False, USE_BOUNDARY_HEAD=False, OWBCE_HEIGHT=0.0)


def make_config(overrides=None):
    config = Config()
    for k, v in {**PINNED, **(overrides or {})}.items():
        setattr(config, k, v)
    return config


def load_models(ckpt_dir):
    models = []
    for name, cls, fname, overrides in MEMBERS:
        config = make_config(overrides)
        model = cls(config).to(config.DEVICE)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, fname), map_location=config.DEVICE, weights_only=True))
        models.append(model.eval())
    return models


def ensemble_prob(models, x):
    return torch.sigmoid(torch.stack([m(x) for m in models]).mean(0))


def load_wav(path, config):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0) if wav.size(0) > 1 else wav.squeeze(0)
    return torchaudio.functional.resample(wav, sr, config.SR) if sr != config.SR else wav


class ShortClips(Dataset):
    def __init__(self, paths, config):
        self.paths, self.config = paths, config

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            wav = pad_or_trim(load_wav(self.paths[i], self.config), self.config.target_samples)
        except Exception:
            wav = torch.zeros(self.config.target_samples)
        return wav, os.path.splitext(os.path.basename(self.paths[i]))[0]


@torch.no_grad()
def long_clip_prob(models, wav, config):
    win, hop = int(WINDOW_SEC * config.SR), config.HOP_LENGTH
    total = wav.numel() // hop
    stride = win - int(OVERLAP_SEC * config.SR)
    starts = list(range(0, wav.numel() - win + 1, stride))
    if not starts or starts[-1] + win < wav.numel():
        starts.append(wav.numel() - win)
    acc, cnt = np.zeros(total), np.zeros(total)
    for s in starts:
        p = ensemble_prob(models, wav[s:s + win].unsqueeze(0).to(config.DEVICE))[0].cpu().numpy()
        a = s // hop
        b = min(a + len(p), total)
        acc[a:b] += p[:b - a]
        cnt[a:b] += 1.0
    return acc / cnt


def to_events(prob, table, hop, clip_sec):
    states = viterbi_decode(prob, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT)
    ev = [(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)]
    return [{"onset": round(float(a), 3), "offset": round(float(b), 3)} for a, b in correct_events(ev, prob, table, clip_sec)]


def run(test_dir, out_dir, ckpt_dir):
    config = make_config()
    table = load_table(os.path.join(ckpt_dir, CORRECTION_TABLE))
    models = load_models(ckpt_dir)
    hop = config.HOP_LENGTH / config.SR
    files = sorted(glob.glob(os.path.join(test_dir, "**", "*.wav"), recursive=True))
    dur, short, long_ = {}, [], []
    for f in files:
        try:
            info = sf.info(f)
            dur[f] = info.frames / info.samplerate
        except Exception:
            dur[f] = 0.0
        (short if dur[f] <= config.MAX_DURATION else long_).append(f)
    entries, k = [], 0
    with torch.no_grad():
        for audio, ids in tqdm(DataLoader(ShortClips(short, config), batch_size=8, shuffle=False,
                                          num_workers=config.NUM_WORKERS), desc="short clips"):
            probs = ensemble_prob(models, audio.to(config.DEVICE)).cpu().numpy()
            for prob, cid in zip(probs, ids):
                d = dur[short[k]]
                entries.append({"clip_id": cid, "events": to_events(prob, table, hop, min(d, config.MAX_DURATION) if d > 0 else None)})
                k += 1
    for f in tqdm(long_, desc="long clips (chunked)"):
        try:
            wav = load_wav(f, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        entries.append({"clip_id": os.path.splitext(os.path.basename(f))[0],
                        "events": to_events(long_clip_prob(models, wav, config), table, hop, dur[f] or None)})
    assert len(entries) == len(files) and len({e["clip_id"] for e in entries}) == len(entries)
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "predictions.jsonl")
    with open(jsonl, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl, arcname="predictions.jsonl")
    return jsonl
