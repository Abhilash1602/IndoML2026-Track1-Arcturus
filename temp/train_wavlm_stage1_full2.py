"""Tier 3 plan, Stage 1 FULL run, take 2: fixes a real bug found in
train_wavlm_stage1_full.py's first attempt.

That script warm-started from the smoke test's already-converged weights but
then used the SAME fresh-start peak LR (1e-3) and a full OneCycleLR schedule as
if training from scratch. OneCycle ramps LR UP toward the peak in its early-to-
middle phase -- fine for random-init weights, actively disruptive to weights
that are already well-trained. This exactly matches what was observed: E1-E4
still benefited from the smoke test's momentum, then E5-E12 degraded as LR
climbed into disruptive territory, triggering an early stop that likely masked
real remaining headroom (the E4 checkpoint still scored 1.4716 on the real
validation set, nearly tying Phase H's 1.4729-1.4734, despite being cut short).

Fix: use a much lower peak LR (1e-4, 10x lower) and a plain CosineAnnealingLR
decay (no ramp-up) -- exactly the pattern train.py's own finetune_on_real
already uses for continued fine-tuning from a good checkpoint (3e-4 there,
vs. the initial training's 1e-3).

Usage: python -m temp.train_wavlm_stage1_full2
"""
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.loss import SEDLoss
from train import apply_mixup, build_wandb_config_dict, evaluate_model, make_full_mix_loaders, make_gold_loaders

TEACHER_PATH = "./checkpoints/dualbranch_phasef.pth"
WARM_START = "./checkpoints/dualbranch_wavlm_stage1.pth"  # the smoke-test's E15 best, NOT the disrupted full-run checkpoint
OUT = "./checkpoints/dualbranch_wavlm_stage1_full2.pth"
PEAK_LR = 1e-4  # 10x lower than fresh-start training (config.LR=1e-3) -- gentle continuation, not disruption


def main():
    config = Config()
    config.USE_WAVLM = True
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)

    teacher_config = Config()
    teacher = DualBranchSED(teacher_config).to(teacher_config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=teacher_config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, synth_val_loader = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)

    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(WARM_START, map_location=config.DEVICE, weights_only=True))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"warm-started from {WARM_START}, trainable: {n_trainable/1e6:.2f}M  peak_lr={PEAK_LR}  "
          f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=PEAK_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS * len(train_loader))

    wandb.init(project="vaani-track1", name="wavlm-stage1-full2-lowlr",
               config=build_wandb_config_dict(Config), job_type="wavlm_stage1_full2")
    best, since = -1.0, 0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"WavLM-S1-full2 {epoch + 1}/{config.EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                audio, masks = apply_mixup(audio, masks)
                optimizer.zero_grad()
                logits = model(audio)
                fr = min(logits.shape[1], masks.shape[1])
                loss = criterion(logits[:, :fr], masks[:, :fr])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
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
