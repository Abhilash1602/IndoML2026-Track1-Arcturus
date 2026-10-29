"""Win-strategy Lane A, Stage 1 (Oct 10): head-only training on a frozen larger Dasheng
(DASHENG_NAME, e.g. mispeech/dasheng-0.6B), on the student data mix, so Stage 2
(temp/train_dasheng_stage2_i2_owbce.py with WARM_START=<this checkpoint>) starts from a
trained head instead of a random one.

Data: make_train_loaders (VAL_TRAIN / GOLD_ALL / PSEUDO_LABELS / PSEUDO_SHARE env, as the
student runs). With VAL_TRAIN=train70 the per-epoch scoring set is hold30 only.

Usage: DASHENG_NAME=mispeech/dasheng-0.6B VAL_TRAIN=train70 GOLD_ALL=1 \
       PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 STAGE1_TAG=d06 \
       python -m temp.train_dasheng_big_stage1
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import FusionSED, StandaloneDashengSED
from src.config import Config
from src.loss import SEDLoss
from temp.train_dasheng_stage2_i2_owbce import cache_official, score_official
from train import apply_mixup, eval_val_dataset, make_train_loaders

TAG = os.environ.get("STAGE1_TAG", "d06")
MODEL = os.environ.get("STAGE1_MODEL", "dasheng")  # dasheng | fusion (FusionSED, FUSION_BRANCH env, e.g. whisper)
OUT = (f"./checkpoints/standalone_dasheng_big_stage1_{TAG}.pth" if MODEL == "dasheng"
       else f"./checkpoints/fusion_{TAG}_stage1.pth")
EPOCHS = int(os.environ.get("STAGE1_EPOCHS", "4"))


def main():
    config = Config()
    config.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
    config.OWBCE_HEIGHT = float(os.environ.get("OWBCE_HEIGHT", "4"))
    config.OWBCE_HALF_WIDTH = int(os.environ.get("OWBCE_HALF_WIDTH", "10"))
    _, train_loader, _, _ = make_train_loaders(config, synth_frac=float(os.environ.get("SYNTH_FRAC", "0.3")))

    if MODEL == "fusion":
        config.FUSION_BRANCH = os.environ.get("FUSION_BRANCH", "whisper")
        model = FusionSED(config).to(config.DEVICE)
    else:
        model = StandaloneDashengSED(config).to(config.DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{MODEL}: {sum(p.numel() for p in params)/1e6:.2f}M trainable, "
          f"{sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6:.1f}M frozen; "
          f"{len(train_loader)} batches/epoch -> {OUT}", flush=True)

    nat_loader = DataLoader(eval_val_dataset(config, "natural"), batch_size=8, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(eval_val_dataset(config, "synthetic"), batch_size=8, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR
    criterion = SEDLoss(config).to(config.DEVICE)
    # LoRA adapters (w2v-BERT branch) get their own, lower LR (LORA_LR); everything else config.LR.
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    lora_lr = float(os.environ.get("LORA_LR", "2e-4"))
    groups = [{"params": rest, "lr": config.LR}] + ([{"params": lora, "lr": lora_lr}] if lora else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=EPOCHS, pct_start=0.2)
    best = -1.0
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for audio, masks, _ in tqdm(train_loader, desc=f"DashengBig-S1-{TAG} {epoch + 1}/{EPOCHS}", mininterval=60):
            audio, masks = apply_mixup(audio.to(config.DEVICE), masks.to(config.DEVICE))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(audio)
            fr = min(logits.shape[1], masks.shape[1])
            loss = criterion(logits[:, :fr].float(), masks[:, :fr])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        model.eval()
        nd, nf, nc = score_official(*cache_official(model, nat_loader, config.DEVICE), hop)
        sd, sf, sc = score_official(*cache_official(model, syn_loader, config.DEVICE), hop)
        eq = 0.5 * nc + 0.5 * sc
        print(f"E{epoch + 1}: loss={np.mean(losses):.4f} | natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
              f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f}", flush=True)
        if eq > best:
            best = eq
            torch.save(model.state_dict(), OUT)
            print(f"  -> saved {OUT}", flush=True)
    torch.save(model.state_dict(), OUT.replace(".pth", "_last.pth"))  # final epoch, not selected on validation
    print(f"Done. best eq_avg {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
