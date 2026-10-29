"""Pre-train src/context_mae.ContextNet by masked reconstruction of frozen Dasheng features on all
in-domain audio we may use: Gold (11,111) + Silver + Bronze (all HF rows) + validation train70.
hold30 (the gate set) and the test audio are excluded. The backbone is the VT dasheng_owB teacher's
Dasheng (the member the new context network will replace), frozen, bf16.

Output: ./checkpoints/context_mae_pretrained.pth (ContextNet state + feature mean/std).
Usage: CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 python -m temp.pretrain_context_mae
"""
import os
import time

import numpy as np
import torch
from datasets import Audio
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.architecture import StandaloneDashengSED
from src.config import Config
from src.context_mae import ContextNet, span_mask
from src.dataset import BronzeInferenceDataset, load_full_dataset
from train import official_val_subset

BACKBONE = os.environ.get("CTX_BACKBONE", "./checkpoints/standalone_dasheng_stage2_i2_owbceVT_owB_teacher.pth")
OUT = os.environ.get("CTX_OUT", "./checkpoints/context_mae_pretrained.pth")
EPOCHS = float(os.environ.get("CTX_EPOCHS", "2"))
BATCH = 32


class AllRows(Dataset):
    """Every HF row (Gold + Silver + Bronze), audio only, via BronzeInferenceDataset's decoder."""

    def __init__(self, full_ds, config):
        self.inner = BronzeInferenceDataset(full_ds, config)
        self.inner.indices = list(range(len(full_ds)))

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        return self.inner[i][0]


class AudioOnly(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0]


def main():
    config = Config()
    dev = config.DEVICE
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    data = ConcatDataset([AllRows(full, config)] +
                         [AudioOnly(official_val_subset(config, s, "train70")) for s in ("natural", "synthetic")])
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, num_workers=24, drop_last=True, persistent_workers=True)
    sed = StandaloneDashengSED(config)
    sed.load_state_dict(torch.load(BACKBONE, map_location="cpu", weights_only=True))
    sed = sed.to(dev).eval()
    ctx = ContextNet(d_in=768).to(dev)
    steps = int(EPOCHS * len(loader))
    opt = torch.optim.AdamW(ctx.parameters(), lr=5e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.05)
    print(f"{len(data)} clips, {len(loader)} batches/epoch, {steps} steps -> {OUT}", flush=True)

    @torch.no_grad()
    def feats(wav):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return sed.dasheng(sed.mel_forward(wav)).hidden_states.float()  # (B, 250, 768)

    # feature standardisation from the first 40 batches
    it = iter(loader)
    acc = torch.cat([feats(next(it).to(dev)).reshape(-1, 768) for _ in range(40)])
    mu, sd = acc.mean(0), acc.std(0) + 1e-5
    del acc
    step, t0, losses = 0, time.time(), []
    g = torch.Generator().manual_seed(0)
    while step < steps:
        for wav in loader:
            f = (feats(wav.to(dev)) - mu) / sd
            m = span_mask(f.size(0), f.size(1), 0.3, 4, 10, dev, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, rec = ctx(f, m)
            loss = ((rec.float() - f) ** 2).mean(-1)[m].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctx.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} loss {np.mean(losses[-200:]):.4f} ({(time.time() - t0) / step:.2f} s/step)", flush=True)
            if step % 1000 == 0 or step == steps:
                torch.save(dict(state=ctx.state_dict(), mu=mu.cpu(), sd=sd.cpu(), step=step), OUT)
            if step >= steps:
                break
    print(f"done; final loss {np.mean(losses[-200:]):.4f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
