"""Tier 3 plan, Stage 1: cheap smoke test of a frozen-WavLM fusion branch before
committing real GPU-days. Default (small) architecture + Config.USE_WAVLM=True,
WavLM fully frozen (matching the org baseline's own documented reasoning: naive
end-to-end fine-tuning of a large pretrained encoder from step 0 overfits and
underperforms a from-scratch model -- hence their two-stage recipe). This tests
only whether frozen WavLM features help AT ALL via FiLM-free concatenation
fusion, on a short run, before any stage-2 unfreezing.

Same self-distillation data mix as Phase F/G2 (teacher=Phase F, proven mix) for
a fair comparison, but capped at far fewer epochs (15) purely as a smoke test --
early stopping patience also shortened so a clearly-non-competitive run doesn't
burn the full budget.

Usage: python -m temp.train_wavlm_stage1
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
OUT = "./checkpoints/dualbranch_wavlm_stage1.pth"
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 5


def main():
    config = Config()
    config.USE_WAVLM = True
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)

    teacher_config = Config()  # Phase F is the DEFAULT small architecture, no WavLM
    teacher = DualBranchSED(teacher_config).to(teacher_config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=teacher_config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, synth_val_loader = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)

    model = DualBranchSED(config).to(config.DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable: {n_trainable/1e6:.2f}M  frozen(WavLM): {n_frozen/1e6:.2f}M  "
          f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=SMOKE_EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="wavlm-stage1-smoke",
               config=build_wandb_config_dict(Config), job_type="wavlm_stage1")
    best, since = -1.0, 0
    try:
        for epoch in range(SMOKE_EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"WavLM-S1 {epoch + 1}/{SMOKE_EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                audio, masks = apply_mixup(audio, masks)
                optimizer.zero_grad()
                logits = model(audio)
                fr = min(logits.shape[1], masks.shape[1])
                loss = criterion(logits[:, :fr], masks[:, :fr])
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
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
                if since >= SMOKE_PATIENCE:
                    print(f"Early stopping smoke test (no improvement in {SMOKE_PATIENCE}).", flush=True)
                    break
    finally:
        wandb.finish()
    print(f"Done. Best sel={best:.4f}, checkpoint -> {OUT}", flush=True)
    print("Compare vs Phase F/G2's sel at the SAME early epoch count (Phase F E5=0.9823, "
          "E10=1.0049; Phase G2 E5=0.9794, E10=1.0386) to judge whether WavLM fusion is "
          "even competitive before ever considering stage 2.", flush=True)


if __name__ == "__main__":
    main()
