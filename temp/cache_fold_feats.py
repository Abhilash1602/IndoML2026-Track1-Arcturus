"""Oct 14, verifier v5 inputs: out-of-fold frame logits of the cross-fitted Dasheng lanes (FA, FB;
work/chain_folds.sh) on every Gold clip (11,111, verified-timestamps order of make_gold_loaders) and
every official validation clip (refiner-cache order), plus targets and 80-band mel energy; and the
test logits as the mean of each lane's 5 fold models (pipeline windowing).

Outputs (work/folds/):
  gold_{logits,mask,E}.npy, gold_heldout_idx.npy (rows of Gold in the old 20% held-out split),
  official_{natural,synthetic}_{logits,mask,E}.npy   logits (N, 2, 1000) f16 = [FA, FB]
  test_logits.pkl = {clip_id: (2, T) f16}
Usage: CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 python -m temp.cache_fold_feats [part]   part: oof | test | all
"""
import glob
import os
import pickle
import sys

import numpy as np
import torch
import torchaudio
from datasets import Audio
from torch.utils.data import DataLoader, Subset

import release.v6_inference as v6
from src.architecture import StandaloneDashengSED
from src.config import Config
from src.dataset import OfficialValidationDataset, VaaniDataset, load_full_dataset, pad_or_trim
from temp.long_clips import windows
from temp.pipeline_v11 import path_items
from train import gold_fold_of, val_fold_of

OUT = "./work/folds"
LANES, NF = ("FA", "FB"), 5
VAL_ROOT = "./data/official_validation_raw/validation"


def load(lane, k):
    for v in ("CONTEXT", "DASHENG_NAME"):
        os.environ.pop(v, None)
    config = Config()
    m = StandaloneDashengSED(config)
    m.load_state_dict(torch.load(f"./checkpoints/standalone_dasheng_stage2_i2_owbce{lane}{k}_last_teacher.pth",
                                 map_location="cpu", weights_only=True))
    return m.to(config.DEVICE).eval(), config


@torch.no_grad()
def run(model, ds, dev, mel_tf=None):
    L, M, E = [], [], []
    for audio, mask, _ in DataLoader(ds, batch_size=16, shuffle=False, num_workers=8):
        x = audio.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            L.append(model(x).float()[:, :1000].cpu().numpy())
        M.append(mask[:, :1000].numpy())
        if mel_tf is not None:
            E.append(mel_tf(x)[..., :1000].mean(1).cpu().numpy())
    return np.concatenate(L), np.concatenate(M), (np.concatenate(E) if E else None)


def oof():
    os.makedirs(OUT, exist_ok=True)
    config = Config()
    dev = config.DEVICE
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold = full.filter(lambda r: r["annotationQuality"] == "verified_timestamps")
    n = len(gold)
    gf = gold_fold_of(n, NF)
    test_rows = set(gold.add_column("_gi", list(range(n))).train_test_split(test_size=0.2, seed=42)["test"]["_gi"])
    np.save(f"{OUT}/gold_heldout_idx.npy", np.array(sorted(test_rows)))
    sets = {"gold": (n, lambda k: [i for i in range(n) if gf[i] == k])}
    vds = {s: OfficialValidationDataset(VAL_ROOT, config, subset=s) for s in ("natural", "synthetic")}
    for s, d in vds.items():
        sets[f"official_{s}"] = (len(d), (lambda d: lambda k: [i for i, r in enumerate(d.rows)
                                                                if val_fold_of(r["segmentFileName"], NF) == k])(d))
    mel_tf = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
                                 torchaudio.transforms.AmplitudeToDB()).to(dev)
    arr = {s: dict(L=np.zeros((N, len(LANES), 1000), np.float16), M=np.zeros((N, 1000), np.uint8),
                   E=np.zeros((N, 1000), np.float16)) for s, (N, _) in sets.items()}
    for li, lane in enumerate(LANES):
        for k in range(NF):
            model, _ = load(lane, k)
            for s, (N, idx_of) in sets.items():
                idx = idx_of(k)
                if s == "gold":
                    sel = gold.select(idx)
                    ds = VaaniDataset(sel.cast_column("audio", Audio(decode=True)), sel, config)
                else:
                    ds = Subset(vds[s.split("_")[1]], idx)
                L, M, E = run(model, ds, dev, mel_tf if li == 0 else None)
                arr[s]["L"][idx, li] = L.astype(np.float16)
                if li == 0:
                    arr[s]["M"][idx] = (M > 0.5).astype(np.uint8)
                    arr[s]["E"][idx] = E.astype(np.float16)
            print(f"{lane}{k}: out-of-fold features done", flush=True)
            del model
            torch.cuda.empty_cache()
    for s, a in arr.items():
        for key, name in (("L", "logits"), ("M", "mask"), ("E", "E")):
            np.save(f"{OUT}/{s}_{name}.npy", a[key])
    print("saved OOF features", flush=True)


@torch.no_grad()
def test():
    paths = sorted(glob.glob("./data/test_audio/**/*.wav", recursive=True))
    config = v6.make_config()
    hop, win = config.HOP_LENGTH, config.target_samples
    out = {}
    for li, lane in enumerate(LANES):
        for k in range(NF):
            model, cfg = load(lane, k)
            for cid, wav in path_items(paths, config):
                n = wav.numel()
                short = n / config.SR <= config.MAX_DURATION
                if short:
                    wav = pad_or_trim(wav, win)
                    n = wav.numel()
                starts = [0] if short else windows(n, config.SR)
                T = n // hop
                acc, cnt = np.zeros(T, np.float32), np.zeros(T, np.float32)
                for s in starts:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = model(wav[s:s + win].unsqueeze(0).to(cfg.DEVICE)).float()[0].cpu().numpy()
                    a = s // hop
                    b = min(a + len(lg), T)
                    acc[a:b] += lg[:b - a]
                    cnt[a:b] += 1
                o = out.setdefault(cid, np.zeros((len(LANES), T), np.float32))
                o[li] += acc / np.maximum(cnt, 1) / NF
            print(f"test {lane}{k} done", flush=True)
            del model
            torch.cuda.empty_cache()
    pickle.dump({k: v.astype(np.float16) for k, v in out.items()}, open(f"{OUT}/test_logits.pkl", "wb"))
    print(f"saved test logits for {len(out)} clips", flush=True)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("oof", "all"):
        oof()
    if part in ("test", "all"):
        test()
