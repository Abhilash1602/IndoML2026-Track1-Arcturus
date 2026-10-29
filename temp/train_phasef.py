"""Phase F: fix Phase E's real-result regression. Phase E's three-way mix used
synth_frac=0.25 (half of Phase B/D's 0.5), which halved synthetic-loud-event
gradient updates per epoch -- the direct, mechanical explanation for its real
Synthetic F1 drop (0.9289->0.8984) despite a genuine Natural F1 gain
(0.3824->0.4028). Local synth-val didn't catch this because its backgrounds
share lineage with the Silver+Bronze self-distillation pool (both drawn from
data/bg_silver/), so it isn't sensitive to the real eval's true clean-speech
backgrounds.

Fix: restore synth_frac to Phase B/D's proven 0.5, take the cut from
gold_frac/silver_bronze_frac instead -- keep Synthetic solved while still
injecting Natural-domain self-distillation volume.

Usage: python -m temp.train_phasef
"""
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from train import apply_mixup, build_wandb_config_dict, evaluate_model, make_full_mix_loaders, make_gold_loaders

TEACHER_PATH = "./checkpoints/dualbranch_baseline.pth"  # Phase B, real F1 0.93 Synthetic
OUT = "./checkpoints/dualbranch_phasef.pth"


def main():
    config = Config()
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)

    teacher = DualBranchSED(config).to(config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, synth_val_loader = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)
    print(f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    model = DualBranchSED(config).to(config.DEVICE)
    from src.loss import SEDLoss
    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="phasef-synth-restored",
               config=build_wandb_config_dict(Config), job_type="phasef")
    best, since = -1.0, 0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"PhaseF {epoch + 1}/{config.EPOCHS}"):
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
            # Equal-weighted, matching the real scoring formula (Phase E's 0.6/0.4
            # Natural-leaning selection was part of what let the Synthetic drop
            # through unnoticed locally).
            sel = 0.5 * gs + 0.5 * ss
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
