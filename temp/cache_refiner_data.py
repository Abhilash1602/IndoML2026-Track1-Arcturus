"""Plan Step 4 data: inputs for the boundary refiner.

For every clip: the 5 submission_v4 members' frame logits (atst_stage2, dasheng_owB,
beats_s2, dasheng_owB_t, beats_owA), an 80-band 10 ms log-mel (DualBranchSED's front
end), and the any-noise target mask. Datasets:
  * gold_heldout -- the 20% Gold split (make_gold_loaders, seed 42) no v4 member trained on
  * synth_val    -- data/synth_val (never trained on)
  * official_natural / official_synthetic -- the official validation set (evaluation only)
Refiner training uses only the first two, so there is no validation leakage.

Output: scratchpad/refiner_<name>.npz with logits (N, 5, 1000) f16, mel (N, 80, 1000) f16,
mask (N, 1000) u8, ids.

Usage: python -m temp.cache_refiner_data
"""
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

import temp.search_ensemble_v2 as v2
import temp.search_ensemble_v3 as v3
from src.config import Config
from src.dataset import OfficialValidationDataset, SyntheticSEDDataset
from train import SYNTH_VAL_DIR, make_gold_loaders

MEMBERS = ("atst_stage2", "dasheng_owB", "beats_s2", "dasheng_owB_t", "beats_owA")
SCRATCH = v2.SCRATCH
VAL_ROOT = "./data/official_validation_raw/validation"
FRAMES = 1000


def main():
    for name, (cls, path, overrides) in v3.NEW_MEMBERS.items():
        if os.path.exists(path):
            v2.MEMBERS[name] = dict(cls=cls, path=path, overrides=overrides)
    config = Config()
    device = config.DEVICE
    _, _, gold_val_loader = make_gold_loaders(config)
    datasets = {
        "gold_heldout": gold_val_loader.dataset,
        "synth_val": SyntheticSEDDataset(SYNTH_VAL_DIR, config, add_noise=False),
        "official_natural": OfficialValidationDataset(VAL_ROOT, config, subset="natural"),
        "official_synthetic": OfficialValidationDataset(VAL_ROOT, config, subset="synthetic"),
    }
    loaders = {k: DataLoader(d, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS)
               for k, d in datasets.items()}
    mel_tf = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
        torchaudio.transforms.AmplitudeToDB()).to(device)

    out = {k: {"logits": np.zeros((len(d), len(MEMBERS), FRAMES), np.float16),
               "mel": np.zeros((len(d), 80, FRAMES), np.float16),
               "mask": np.zeros((len(d), FRAMES), np.uint8), "ids": []} for k, d in datasets.items()}
    for mi, member in enumerate(MEMBERS):
        model = v2.load_member(member)
        for k, loader in loaders.items():
            pos = 0
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"{member}/{k}"):
                    audio, mask, ids = batch[0], batch[1], batch[2]
                    audio = audio.to(device)
                    b = audio.shape[0]
                    logit = model(audio).float()[:, :FRAMES]
                    out[k]["logits"][pos:pos + b, mi, :logit.shape[1]] = logit.cpu().numpy()
                    if mi == 0:
                        mel = mel_tf(audio)[..., :FRAMES]
                        out[k]["mel"][pos:pos + b, :, :mel.shape[-1]] = mel.cpu().numpy()
                        m = mask[:, :FRAMES].numpy() > 0.5
                        out[k]["mask"][pos:pos + b, :m.shape[1]] = m
                        out[k]["ids"].extend(list(ids))
                    pos += b
            assert pos == len(datasets[k])
        del model
        torch.cuda.empty_cache()
    for k, d in out.items():
        np.savez(os.path.join(SCRATCH, f"refiner_{k}.npz"), logits=d["logits"], mel=d["mel"], mask=d["mask"],
                 ids=np.array(d["ids"]))
        print(f"{k}: {len(d['ids'])} clips, positive frame rate {d['mask'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
