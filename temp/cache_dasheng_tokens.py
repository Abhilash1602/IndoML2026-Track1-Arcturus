"""Oct 14, idea #2 (clip-tag evidence) step 1: frozen PRETRAINED Dasheng-base tokens (25 Hz, 768-d; never
fine-tuned, so no label leakage) for every HF clip (first 10 s), the official validation set (refiner order)
and the test clips (first 10 s; long clips keep their first window only -- tagger features for events
beyond 10 s fall back to the clip-level probability).

Outputs (work/tags/): hf_tok.npy (90637, 250, 768) f16 memmap + hf_tags.npy (90637, 8) + hf_quality.npy,
official_{natural,synthetic}_tok.npy, test_tok.npy + test_ids.json.
Usage: CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 python -m temp.cache_dasheng_tokens
"""
import glob
import json
import os

import numpy as np
import torch
from datasets import Audio
from torch.utils.data import DataLoader, Dataset

from src.architecture import StandaloneDashengSED
from src.config import Config
from src.dataset import BronzeInferenceDataset, OfficialValidationDataset, load_full_dataset, pad_or_trim
from temp.pipeline_v11 import path_items

OUT = "./work/tags"
CATS = ["animal", "vehicle_traffic", "baby_child", "singing_music", "phone_signal_alarm", "appliance_machine",
        "human_non_speech"]
VAL_ROOT = "./data/official_validation_raw/validation"


class AllRows(Dataset):
    def __init__(self, full, config):
        self.inner = BronzeInferenceDataset(full, config)
        self.inner.indices = list(range(len(full)))

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        return self.inner[i][0]


class First(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0]


@torch.no_grad()
def dump(model, loader, path, n, dev):
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, 250, 768))
    k = 0
    for wav in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.dasheng(model.mel_forward(wav.to(dev))).hidden_states.float()
        h = h[:, :250]
        mm[k:k + len(h), :h.shape[1]] = h.cpu().numpy().astype(np.float16)
        k += len(h)
    mm.flush()
    assert k == n, (path, k, n)


def main():
    os.makedirs(OUT, exist_ok=True)
    for v in ("CONTEXT", "DASHENG_NAME"):
        os.environ.pop(v, None)
    config = Config()
    dev = config.DEVICE
    model = StandaloneDashengSED(config).to(dev).eval()  # pretrained Dasheng only (no checkpoint loaded)
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    meta = full.select_columns(["annotationQuality", "NoiseCategory"])
    tags = np.zeros((len(full), 8), np.float32)
    for i, cats in enumerate(meta["NoiseCategory"]):
        cats = cats or []
        tags[i, 0] = float(len(cats) > 0)
        for c in cats:
            if c in CATS:
                tags[i, 1 + CATS.index(c)] = 1.0
    np.save(f"{OUT}/hf_tags.npy", tags)
    np.save(f"{OUT}/hf_quality.npy", np.array(meta["annotationQuality"]))
    dump(model, DataLoader(AllRows(full, config), batch_size=32, num_workers=24), f"{OUT}/hf_tok.npy", len(full), dev)
    print("hf done", flush=True)
    for s in ("natural", "synthetic"):
        ds = OfficialValidationDataset(VAL_ROOT, config, subset=s)
        dump(model, DataLoader(First(ds), batch_size=32, num_workers=8), f"{OUT}/official_{s}_tok.npy", len(ds), dev)
    print("validation done", flush=True)
    paths = sorted(glob.glob("./data/test_audio/**/*.wav", recursive=True))
    ids, wavs = [], []
    for cid, wav in path_items(paths, config):
        ids.append(cid)
        wavs.append(pad_or_trim(wav, config.target_samples))
    json.dump(ids, open(f"{OUT}/test_ids.json", "w"))
    dump(model, DataLoader(wavs, batch_size=32), f"{OUT}/test_tok.npy", len(ids), dev)
    print(f"test done ({len(ids)} clips)", flush=True)


if __name__ == "__main__":
    main()
