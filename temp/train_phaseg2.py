"""Phase G2: corrected retry of Phase G. Same recipe (teacher=Phase F,
gold_frac=0.20/silver_bronze_frac=0.30/synth_frac=0.50, union-mixup kept), but
POS_WEIGHT turned down from Phase G's 3.0 to 1.5 -- 3.0 was root-caused (via
temp/diagnose_frame_balance.py on the saved Phase G checkpoint) to push the
predicted-positive frame rate to 33% against a true 17.5%, crushing precision
(0.655->0.429) for a smaller recall gain (0.688->0.816), netting a worse F1/Dice.
Union-mixup and the teacher swap are very unlikely to be the cause (their
mechanisms don't explain a 2x shift in predicted-positive rate) so they're kept.

Usage: python -m temp.train_phaseg2
"""
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.loss import SEDLoss
from train import apply_mixup, build_wandb_config_dict, evaluate_model, make_full_mix_loaders, make_gold_loaders

TEACHER_PATH = "./checkpoints/dualbranch_phasef.pth"  # Phase F, real Combined 1.4727
OUT = "./checkpoints/dualbranch_phaseg2.pth"


def main():
    config = Config()
    config.POS_WEIGHT = 1.5
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)

    teacher = DualBranchSED(config).to(config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, synth_val_loader = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)
    print(f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    model = DualBranchSED(config).to(config.DEVICE)
    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="phaseg2-posweight1.5",
               config=build_wandb_config_dict(Config), job_type="phaseg2")
    best, since = -1.0, 0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"PhaseG2 {epoch + 1}/{config.EPOCHS}"):
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
