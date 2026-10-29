"""SEDT inputs (plan Step 1): for every clip, the PRETRAINED mispeech/dasheng-base hidden states
(never fine-tuned on our labels, so there is no in-sample leak on Gold), an 80-band 10 ms
log-mel, and the merged any-noise target mask (the scoring convention).

Sets: gold_train / gold_heldout (make_gold_loaders' seed-42 80/20 split; audio read directly,
no training-noise augmentation), official_natural / official_synthetic (same order as
scratchpad/refiner_official_*.npz). Clips are padded/trimmed to 10 s like everywhere else.

Output: scratchpad/sedt/{set}_{tok,mel,mask}.npy (tok (N, Tt, 768) f16, mel (N, 80, 1000) f16,
mask (N, 1000) u8) + {set}_ids.json.

Usage: VAANI_DEVICE=cuda:0 python -m temp.cache_sedt_features [set ...]
"""
import json
import os
import sys

import numpy as np
import torch
import torchaudio
from datasets import Audio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Config
from src.dataset import OfficialValidationDataset, _build_targets, load_full_dataset, pad_or_trim
from temp.train_refiner_val import S

OUT = os.path.join(S, "sedt")
VAL_ROOT = "./data/official_validation_raw/validation"
FRAMES = 1000


class CleanGold(Dataset):
    def __init__(self, hf, config):
        self.hf, self.config = hf, config

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        row = self.hf[i]
        # datasets 5 returns a torchcodec AudioDecoder: it supports ["array"] / ["sampling_rate"]
        # (as VaaniDataset uses) but not dict .get(); a decode failure must not silently become
        # silence, so only genuinely broken files fall back to zeros and they are counted.
        try:
            dec = row["audio"]
            audio = torch.as_tensor(np.asarray(dec["array"]), dtype=torch.float32)
            if audio.ndim > 1:
                audio = audio.mean(0) if audio.shape[0] < audio.shape[-1] else audio.mean(-1)
            sr = int(dec["sampling_rate"])
            if sr != self.config.SR:
                audio = torchaudio.functional.resample(audio, sr, self.config.SR)
        except Exception as e:
            print(f"decode failure on row {i}: {e!r}", flush=True)
            audio = torch.zeros(self.config.target_samples)
        audio = pad_or_trim(audio, self.config.target_samples)
        mask = _build_targets(row.get("NoiseSubCategoryTimeStamp"), FRAMES, self.config)[0]
        return audio, mask, str(row.get("imageFileName", i))


def gold_sets(config):
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold = full.filter(lambda r: r["annotationQuality"] == "verified_timestamps")
    split = gold.train_test_split(test_size=0.2, seed=42)  # identical to train.make_gold_loaders
    return {"gold_train": CleanGold(split["train"].cast_column("audio", Audio(decode=True)), config),
            "gold_heldout": CleanGold(split["test"].cast_column("audio", Audio(decode=True)), config)}


def main():
    config = Config()
    device = os.environ.get("VAANI_DEVICE", "cuda:0")
    os.makedirs(OUT, exist_ok=True)
    wanted = sys.argv[1:] or ["official_natural", "official_synthetic", "gold_heldout", "gold_train"]
    sets = {"official_natural": OfficialValidationDataset(VAL_ROOT, config, subset="natural"),
            "official_synthetic": OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")}
    if any(w.startswith("gold") for w in wanted):
        sets.update(gold_sets(config))
    from transformers import AutoModel
    dasheng = AutoModel.from_pretrained("mispeech/dasheng-base", trust_remote_code=True).to(device).eval()
    d_mel = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=160,
                                             f_min=0, f_max=8000, n_mels=64, center=True),
        torchaudio.transforms.AmplitudeToDB(top_db=120)).to(device)  # = StandaloneDashengSED.mel_forward
    mel80 = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
        torchaudio.transforms.AmplitudeToDB()).to(device)  # = release/v7_inference, refiner caches
    for name in wanted:
        ds = sets[name]
        if os.path.exists(f"{OUT}/{name}_tok.npy"):
            continue
        tok = mel = mask = None
        ids, pos = [], 0
        with torch.no_grad():
            for audio, m, cid in tqdm(DataLoader(ds, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS), desc=name):
                x = audio.to(device)
                h = dasheng(d_mel(x)).hidden_states  # (B, Tt, 768)
                if tok is None:
                    tok = np.lib.format.open_memmap(f"{OUT}/{name}_tok.npy.part", "w+", np.float16, (len(ds),) + tuple(h.shape[1:]))
                    mel = np.zeros((len(ds), 80, FRAMES), np.float16)
                    mask = np.zeros((len(ds), FRAMES), np.uint8)
                b = x.shape[0]
                tok[pos:pos + b] = h.cpu().numpy().astype(np.float16)
                mm = mel80(x)[..., :FRAMES]
                mel[pos:pos + b, :, :mm.shape[-1]] = mm.cpu().numpy()
                mask[pos:pos + b] = (m[:, :FRAMES].numpy() > 0.5)
                ids += list(cid)
                pos += b
        assert pos == len(ds)
        tok.flush()
        del tok
        os.replace(f"{OUT}/{name}_tok.npy.part", f"{OUT}/{name}_tok.npy")
        np.save(f"{OUT}/{name}_mel.npy", mel)
        np.save(f"{OUT}/{name}_mask.npy", mask)
        json.dump(ids, open(f"{OUT}/{name}_ids.json", "w"))
        print(f"{name}: {len(ds)} clips, tokens {np.load(f'{OUT}/{name}_tok.npy', mmap_mode='r').shape}, "
              f"positive frame rate {mask.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
