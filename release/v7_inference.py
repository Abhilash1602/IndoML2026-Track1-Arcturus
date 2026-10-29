"""Submission v7 = v6 + the in-domain boundary refiner on Natural-looking clips.

For every clip <= 10 s: the v6 ensemble (10 members, logit mean) gives a probability track and
v6's events (Viterbi + gated edge correction). If the clip looks Natural (mean edge contrast of
its v6 events < 0.5, or no events), its events are replaced by the refiner's: presence and
regressed onset/offset distances from the 5 v4 members' logits + an 80-band log-mel, decoded
with greedy non-overlapping proposals, then the same edge correction. Synthetic-looking clips
and clips > 10 s keep v6's events. Hold-out evidence (refiner trained on validation train70,
scored on hold30): 1.4745 vs 1.4632 for the v6 recipe; Natural F1 0.4124 vs 0.3953.
"""
import glob
import json
import os
import zipfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from release.v6_inference import (MEMBERS, ShortClips, load_models, load_wav, long_clip_prob, make_config,
                                  to_events, P_STAY_EVENT, P_STAY_SILENCE)
from src.edge_correction import correct_events, edge_contrast, load_table
from src.refiner import BoundaryRefiner, decode_proposals
from src.utils import extract_events_viterbi, viterbi_decode

# The refiner's member channels, in the order it was trained with (temp/cache_refiner_data.py).
REFINER_ORDER = ["atst_stage2", "dasheng_owB", "beats_s2", "dasheng_owB_t", "beats_owA"]
FRAMES = 1000


def load_refiner(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["n_extra"] == 0 and ck["kind"] == "proposals", ck.get("kind")
    model = BoundaryRefiner(n_members=len(REFINER_ORDER))
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


def natural_gate(prob, hop):
    ev = [(e["onset"], e["offset"]) for e in extract_events_viterbi(
        viterbi_decode(prob, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT), hop)]
    return True if not ev else float(np.mean([edge_contrast(prob, a, b) for a, b in ev])) < 0.5


def refiner_events(pres, d_on, d_off, prob, table, clip_sec):
    ev = decode_proposals(pres, d_on, d_off, 0.5)
    ev = correct_events(ev, prob, table, clip_sec)
    end = clip_sec if clip_sec is not None else FRAMES * 0.01
    ev = [(a, min(b, end)) for a, b in ev if min(b, end) - a >= 0.02]
    return [{"onset": round(float(a), 3), "offset": round(float(b), 3)} for a, b in ev]


def cover(ev):
    return sum(e["offset"] - e["onset"] for e in ev)


def run(test_dir, out_dir, ckpt_dir, refiner_path, member_prefix="VA", files=None, fallback_ratio=None):
    """fallback_ratio (v7c): on a gated clip keep v6's events when v6 has events and the refiner's
    are empty or cover less than fallback_ratio x v6's total duration (cross-fit: 0.8 recovers
    ~40% of the refiner's Natural Dice cost)."""
    config = make_config()
    table = load_table(os.path.join(ckpt_dir, "edge_correction_v5.json"))
    members = [(n, c, f.replace("owbceVA_", f"owbce{member_prefix}_"), o) for n, c, f, o in MEMBERS]
    import release.v6_inference as v6
    saved, v6.MEMBERS = v6.MEMBERS, members
    try:
        models = load_models(ckpt_dir)
    finally:
        v6.MEMBERS = saved
    names = [n for n, _, _, _ in members]
    ridx = [names.index(n) for n in REFINER_ORDER]
    refiner = load_refiner(refiner_path, config.DEVICE) if refiner_path else None  # None -> plain v6 events
    mel_tf = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
        torchaudio.transforms.AmplitudeToDB()).to(config.DEVICE)
    hop = config.HOP_LENGTH / config.SR
    files = files if files is not None else sorted(glob.glob(os.path.join(test_dir, "**", "*.wav"), recursive=True))
    dur, short, long_ = {}, [], []
    for f in files:
        try:
            info = sf.info(f)
            dur[f] = info.frames / info.samplerate
        except Exception:
            dur[f] = 0.0
        (short if dur[f] <= config.MAX_DURATION else long_).append(f)
    entries, k, n_ref, n_fb = [], 0, 0, 0
    with torch.no_grad():
        for audio, ids in tqdm(DataLoader(ShortClips(short, config), batch_size=8, shuffle=False,
                                          num_workers=config.NUM_WORKERS), desc="short clips"):
            x = audio.to(config.DEVICE)
            logits = torch.stack([m(x) for m in models])  # (M, B, T)
            probs = torch.sigmoid(logits.mean(0)).cpu().numpy()
            if refiner is not None:
                rin = logits[ridx].permute(1, 0, 2)[:, :, :FRAMES].float()
                mel = mel_tf(x)[..., :FRAMES]
                out = refiner(rin, mel)
                pres = torch.sigmoid(out["presence"]).cpu().numpy()
                d_on, d_off = out["d_on"].cpu().numpy(), out["d_off"].cpu().numpy()
            for j, (prob, cid) in enumerate(zip(probs, ids)):
                d = dur[short[k]]
                clip_sec = min(d, config.MAX_DURATION) if d > 0 else None
                if refiner is not None and natural_gate(prob, hop):
                    ev = refiner_events(pres[j], d_on[j], d_off[j], prob[:FRAMES], table, clip_sec)
                    n_ref += 1
                    if fallback_ratio is not None:
                        base = to_events(prob, table, hop, clip_sec)
                        if base and (not ev or cover(ev) < fallback_ratio * cover(base)):
                            ev = base
                            n_fb += 1
                else:
                    ev = to_events(prob, table, hop, clip_sec)
                entries.append({"clip_id": cid, "events": ev})
                k += 1
    for f in tqdm(long_, desc="long clips (v6 path)"):
        try:
            wav = load_wav(f, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        import release.v6_inference as v6m
        entries.append({"clip_id": os.path.splitext(os.path.basename(f))[0],
                        "events": to_events(v6m.long_clip_prob(models, wav, config), table, hop, dur[f] or None)})
    assert len(entries) == len(files) and len({e["clip_id"] for e in entries}) == len(entries)
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "predictions.jsonl")
    with open(jsonl, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl, arcname="predictions.jsonl")
    print(f"{len(entries)} clips; refiner used on {n_ref} short clips ({n_ref / max(len(short), 1):.1%}); "
          f"fell back to v6 on {n_fb}", flush=True)
    return entries
