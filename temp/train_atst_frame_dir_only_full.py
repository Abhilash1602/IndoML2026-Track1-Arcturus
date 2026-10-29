"""Session 9 engineering-hack plan, Priority B follow-up: the DIR-only smoke
test (temp/train_atst_frame_dir_only_smoke.py) justified this -- it climbed
cleanly from eq_avg 1.4491 (E1) to 1.4818 (E15, still rising at the 15-epoch
cap), genuinely beating the frozen Stage 1 baseline (1.4759) with no sign of
the collapse that ended the Stage 2 (LLRD+MT+ICT) attempt. This removes the
epoch cap and uses the standard EARLY_STOP_PATIENCE, otherwise identical
recipe (frozen ATST-Frame, DIR augmentation only, Phase F's self-distillation
mix). Resumes from the smoke test's own best checkpoint rather than
retraining from scratch.

Usage: python -m temp.train_atst_frame_dir_only_full
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
WARM_START = "./checkpoints/dualbranch_atst_frame_dir_only.pth"  # smoke test's E15 best, eq_avg 1.4818
OUT = "./checkpoints/dualbranch_atst_frame_dir_only_full.pth"
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
    config.USE_FREQ_WARP = False      # keep DIR isolated -- this is what was validated
    config.USE_FREQ_MIXSTYLE = False
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
    print(f"warm-started from {WARM_START} (ATST-Frame frozen, DIR only), trainable: {n_trainable/1e6:.2f}M  "
          f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="atst-frame-dir-only-full",
               config=build_wandb_config_dict(Config), job_type="atst_frame_dir_only_full")
    best, since = 1.4818, 0  # gate: must beat the smoke test's own best to be worth keeping
    print(f"gate: must beat smoke-test eq_avg {best:.4f} on OfficialValidationDataset "
          f"(fixed decode 0.95/0.7)", flush=True)
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"ATST-DIR-full {epoch + 1}/{config.EPOCHS}"):
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
                torch.save(model.state_dict(), OUT)
                print(f"  -> new best, saved to {OUT}", flush=True)
            else:
                since += 1
                if since >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping (no improvement in {config.EARLY_STOP_PATIENCE}).", flush=True)
                    break
    finally:
        wandb.finish()
    print(f"Done. Best eq_avg={best:.4f}, checkpoint -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
