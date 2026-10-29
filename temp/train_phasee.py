"""Phase E: Gold + self-distilled Silver/Bronze (teacher = Phase B) + synthetic
loud events, three-way mix, targeting Natural via 5x more Natural-domain training
volume than Gold alone. Checkpoint-selected 0.6*gold-val + 0.4*synth-val (flipped
from Phase B's 0.6*synth since Synthetic is already solved).

Usage: python -m temp.train_phasee
"""
import numpy as np
import torch
import wandb
from datasets import Audio
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import SyntheticSEDDataset, load_full_dataset
from src.loss import SEDLoss
from train import (
    apply_mixup,
    build_wandb_config_dict,
    evaluate_model,
    make_full_mix_loaders,
    make_gold_loaders,
)

TEACHER_PATH = "./checkpoints/dualbranch_baseline.pth"  # Phase B, real F1 0.89 Synthetic
OUT = "./checkpoints/dualbranch_phasee.pth"


def main():
    config = Config()
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)

    teacher = DualBranchSED(config).to(config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, synth_val_loader = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset)
    print(f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    model = DualBranchSED(config).to(config.DEVICE)
    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="phasee-full-mix",
               config=build_wandb_config_dict(Config), job_type="phasee")
    best, since = -1.0, 0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"PhaseE {epoch + 1}/{config.EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                audio, masks = apply_mixup(audio, masks)
                optimizer.zero_grad()
                logits = model(audio)
                fr = min(logits.shape[1], masks.shape[1])
                loss = criterion(logits[:, :fr], masks[:, :fr])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

            _, gd, gf, gs = evaluate_model(model, gold_val_loader, criterion, config)
            _, sd, sf, ss = evaluate_model(model, synth_val_loader, criterion, config)
            tr = float(np.mean(losses))
            sel = 0.6 * gs + 0.4 * ss  # Natural is the game now; Synthetic already solved
            print(f"E{epoch + 1}: loss={tr:.4f} | gold d={gd:.4f} f1={gf:.4f} ({gs:.4f}) | "
                  f"synth d={sd:.4f} f1={sf:.4f} ({ss:.4f}) | sel={sel:.4f}", flush=True)
            wandb.log({"epoch": epoch + 1, "train/loss": tr,
                       "gold/dice": gd, "gold/f1": gf, "gold/combined": gs,
                       "synth/dice": sd, "synth/f1": sf, "synth/combined": ss, "sel": sel})
            if sel > best:
                best, since = sel, 0
                torch.save(model.state_dict(), OUT)
            else:
                since += 1
                if since >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping (no improvement in {config.EARLY_STOP_PATIENCE}).", flush=True)
                    break
    finally:
        wandb.finish()
    print(f"Done. Best sel={best:.4f}, checkpoint -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
