"""Phase H: capacity rescaling combined with the self-distillation data volume --
untested combination. Phase D (11M params, 4.5x Phase B) only ever trained against
Gold-only Natural signal (11,111 clips) and tied Phase B; it never got a chance to
use the ~5x larger Natural-domain training volume that self-distillation (Phase
F/G2) provides. Two independent single-knob recipe tweaks on the small architecture
(Phase E's synth_frac cut, Phase G2's pos_weight) have both landed at flat trades,
suggesting a real plateau for THIS architecture at THIS capacity -- this tests
whether more capacity can use the extra data where the small model can't.

Uses Phase F's exact proven data mix (gold_frac=0.20, silver_bronze_frac=0.30,
synth_frac=0.50, teacher=Phase F) and its default POS_WEIGHT=1.0 (Phase F's own
recipe, which produced the session's only clean win with no domain trade-off) --
deliberately NOT combining with Phase G2's pos_weight=1.5, to isolate capacity as
the only new variable. LR peak lowered to 5e-4 (from Phase B/D/F's 1e-3) since
Phase D's own history flagged its LR was never re-tuned for the bigger model as
a real caveat on that "tied, not negative" result.

Usage: python -m temp.train_phaseh
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
OUT = "./checkpoints/dualbranch_phaseh.pth"
PEAK_LR = 5e-4


def main():
    config = Config()
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)

    # Phase F's checkpoint is the DEFAULT small architecture (2.44M params) -- must
    # build the teacher with an unmodified config, before overriding CNN_CH/CONF_DIM
    # for the (bigger) student below, or the state_dict load fails on shape mismatch.
    teacher_config = Config()
    teacher = DualBranchSED(teacher_config).to(teacher_config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=teacher_config.DEVICE, weights_only=True))
    teacher.eval()

    config.CNN_CH = (48, 64, 128, 256)
    config.CONF_DIM = 256
    config.CONF_HEADS = 8
    config.CONF_LAYERS = 6
    # Both GPUs on this shared machine are under heavy external contention right
    # now (another user's ~83GB VLLM process left only ~12GB free, not enough
    # headroom for this bigger model's batch-16 forward/backward -- crashed twice).
    # Halving batch size trades some throughput for fitting safely in what's
    # actually available; OneCycleLR's steps_per_epoch is derived from
    # len(train_loader) below so this needs no other change.
    config.BATCH_SIZE = 8

    train_loader, synth_val_loader = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)

    model = DualBranchSED(config).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M  batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=PEAK_LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="phaseh-capacity+selfdistill",
               config=build_wandb_config_dict(Config), job_type="phaseh")
    best, since = -1.0, 0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"PhaseH {epoch + 1}/{config.EPOCHS}"):
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
