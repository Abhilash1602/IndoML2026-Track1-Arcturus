"""Phase C: Natural-focused. Synthetic is solved (real F1 0.89); Natural (real F1
0.38) is the entire remaining game. Retrain with a Gold-heavier mix and a
Gold-weighted selection metric.

Usage: python -m temp.train_phasec <synth_frac> <out_suffix>
  e.g. python -m temp.train_phasec 0.30 g30
"""
import sys
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.loss import SEDLoss
from train import apply_mixup, build_wandb_config_dict, evaluate_model, make_gold_synth_loaders

SYNTH_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "g30"
OUT = f"./checkpoints/dualbranch_{SUFFIX}.pth"


def main():
    config = Config()
    _, train_loader, gold_val_loader, synth_val_loader = make_gold_synth_loaders(
        config, synth_frac=SYNTH_FRAC)
    print(f"synth_frac={SYNTH_FRAC}  batches/epoch={len(train_loader)}  -> {OUT}", flush=True)

    model = DualBranchSED(config).to(config.DEVICE)
    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name=f"phasec-{SUFFIX}",
               config=build_wandb_config_dict(Config), job_type="phasec")
    best, since = -1.0, 0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"PhaseC {epoch + 1}/{config.EPOCHS}"):
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
            # Natural is the game now: weight gold-val 0.65, synth-val 0.35
            sel = 0.65 * gs + 0.35 * ss
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
