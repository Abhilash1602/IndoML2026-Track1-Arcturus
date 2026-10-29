"""Session 9 plan, Stage 1 FULL run: the capped 15-epoch smoke test
(temp/train_atst_frame_stage1.py) justified this -- natural_combined reached
1.0163 at E11 and stayed at/above ~1.00-1.02 for most of the second half of the
run, matching/exceeding WavLM's own smoke-test reference (1.0102 by E15) that
originally justified WavLM's full run. Removes the epoch cap and uses the
standard EARLY_STOP_PATIENCE=8, otherwise identical recipe (frozen ATST-Frame,
Phase F's proven self-distillation mix). Resumes from the smoke-test checkpoint
rather than retraining from scratch.

Checkpoint selection stays against OfficialValidationDataset throughout (not
gold_val/synth_val) -- the whole point of fixing this from the smoke test
onward was to never repeat the WavLM full2 mistake of trusting a proxy that
can point in the wrong direction.

Usage: python -m temp.train_atst_frame_stage1_full
"""
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import apply_mixup, build_wandb_config_dict, make_full_mix_loaders, make_gold_loaders

TEACHER_PATH = "./checkpoints/dualbranch_phasef.pth"
WARM_START = "./checkpoints/dualbranch_atst_frame_stage1.pth"  # the smoke-test's E9 best
OUT = "./checkpoints/dualbranch_atst_frame_stage1_full.pth"
VAL_ROOT = "./data/official_validation_raw/validation"


@torch.no_grad()
def cache_official(model, loader, device):
    probs, targets = [], []
    for audio, masks, _ in loader:
        audio = audio.to(device, non_blocking=True)
        p = torch.sigmoid(model(audio))
        frames = min(p.shape[1], masks.shape[1])
        probs.append(p[:, :frames].cpu().numpy())
        targets.append(masks[:, :frames].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score_official(probabilities, ground_truth, hop, p_stay_silence=0.95, p_stay_event=0.7):
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, _, _ = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, dice + f1


def main():
    config = Config()
    config.USE_ATST_FRAME = True
    full_ds, gold_train_loader, _ = make_gold_loaders(config)

    teacher_config = Config()
    teacher = DualBranchSED(teacher_config).to(teacher_config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=teacher_config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, _ = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)

    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(WARM_START, map_location=config.DEVICE, weights_only=True))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"warm-started from {WARM_START}, trainable: {n_trainable/1e6:.2f}M  "
          f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="atst-frame-stage1-full",
               config=build_wandb_config_dict(Config), job_type="atst_frame_stage1_full")
    best, since = 1.4574, 0  # gate: must beat the smoke test's own best eq_avg to be worth keeping
    best_natural_combined, best_natural_f1 = 0.0, 0.0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"ATST-S1-full {epoch + 1}/{config.EPOCHS}"):
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

            model.eval()
            np_, nt = cache_official(model, nat_loader, config.DEVICE)
            sp, st = cache_official(model, syn_loader, config.DEVICE)
            nd, nf, nc = score_official(np_, nt, hop)
            sd, sf, sc = score_official(sp, st, hop)
            eq = 0.5 * nc + 0.5 * sc
            tr = float(np.mean(losses))
            print(f"E{epoch + 1}: loss={tr:.4f} | natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} (gate: {best:.4f})",
                  flush=True)
            wandb.log({"epoch": epoch + 1, "train/loss": tr,
                       "natural/dice": nd, "natural/f1": nf, "natural/combined": nc,
                       "synthetic/dice": sd, "synthetic/f1": sf, "synthetic/combined": sc,
                       "eq_avg": eq})
            if eq > best:
                best, since = eq, 0
                best_natural_combined, best_natural_f1 = nc, nf
                torch.save(model.state_dict(), OUT)
            else:
                since += 1
                if since >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping (no improvement in {config.EARLY_STOP_PATIENCE}).", flush=True)
                    break
    finally:
        wandb.finish()
    print(f"Done. Best eq_avg={best:.4f} (natural_combined={best_natural_combined:.4f}, "
          f"natural_f1={best_natural_f1:.4f}), checkpoint -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
