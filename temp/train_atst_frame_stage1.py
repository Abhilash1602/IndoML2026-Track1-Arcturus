"""Session 9 plan, Stage 1: cheap smoke test of a frozen-ATST-Frame fusion branch
before committing real GPU-days, modeled on temp/train_wavlm_stage1.py's capped
15-epoch structure -- BUT selecting checkpoints against OfficialValidationDataset
via temp/train_wavlm_stage2.py's real-metric pattern from epoch 1, not the
homegrown gold_val/synth_val proxies stage1 originally used (this project already
learned via the "WavLM full2" episode, documented in STRATEGY.md, that those
proxies can point in the wrong direction -- they overfit to gold_val's narrow,
~45%-breathing-dominated composition rather than measuring real generalization).

ATST-Frame (fschmid56/PretrainedSED, MIT) is pretrained specifically for
frame-level SED on AudioSet-Strong -- a supervised task much closer to ours than
BEATs'/WavLM's generic pretraining. Frozen only: WavLM's analogous "unfreeze at
low LR" Stage 2 was tried and decisively failed (never approached the real-
ensemble baseline across 4 epochs, no improving trend), so no unfreeze stage is
planned here regardless of this smoke test's outcome.

Usage: python -m temp.train_atst_frame_stage1
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

TEACHER_PATH = "./checkpoints/dualbranch_phasef.pth"  # Phase F, real Combined 1.4727
OUT = "./checkpoints/dualbranch_atst_frame_stage1.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 5

# Reference points for the decision rule at the end (see STRATEGY.md):
# WavLM's own smoke test showed Natural combined 1.0102 by E15, already beating
# Phase F/G2/H (~1.0072-1.0282) despite Synthetic lagging from undertraining --
# that early Natural-side edge was what justified WavLM's full run. The current
# real best's Natural F1 is 0.3978 (4-model ensemble, Combined 1.5019).
WAVLM_SMOKE_NATURAL_COMBINED = 1.0102
CURRENT_BEST_NATURAL_F1 = 0.3978


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
    """Fixed-decode-param scoring (matches evaluate_model's own fixed 0.95/0.7 so
    epoch-to-epoch comparisons aren't confounded by simultaneously re-tuning
    decode) against the REAL validation set with the corrected official
    micro-F1 + macro-Dice metric, not the homegrown macro-per-clip one."""
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

    teacher_config = Config()  # Phase F is the DEFAULT small architecture, no fusion branch
    teacher = DualBranchSED(teacher_config).to(teacher_config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=teacher_config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, _ = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)

    model = DualBranchSED(config).to(config.DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable: {n_trainable/1e6:.2f}M  frozen(ATST-Frame): {n_frozen/1e6:.2f}M  "
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
        epochs=SMOKE_EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="atst-frame-stage1-smoke",
               config=build_wandb_config_dict(Config), job_type="atst_frame_stage1")
    best, since = -1.0, 0
    best_natural_combined, best_natural_f1 = 0.0, 0.0
    try:
        for epoch in range(SMOKE_EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"ATST-S1 {epoch + 1}/{SMOKE_EPOCHS}"):
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
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f}", flush=True)
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
                if since >= SMOKE_PATIENCE:
                    print(f"Early stopping smoke test (no improvement in {SMOKE_PATIENCE}).", flush=True)
                    break
    finally:
        wandb.finish()
    print(f"Done. Best eq_avg={best:.4f} (natural_combined={best_natural_combined:.4f}, "
          f"natural_f1={best_natural_f1:.4f}), checkpoint -> {OUT}", flush=True)
    print(f"Decision rule: WavLM's own smoke test showed natural_combined="
          f"{WAVLM_SMOKE_NATURAL_COMBINED:.4f} by E15, which justified its full run. "
          f"Current real-best Natural F1 is {CURRENT_BEST_NATURAL_F1:.4f}. Only proceed "
          f"to a full, uncapped run (EARLY_STOP_PATIENCE=8, same frozen config, no "
          f"unfreeze stage) if this run shows a comparably clear early Natural-side "
          f"edge -- otherwise stop, per this project's established discipline.", flush=True)


if __name__ == "__main__":
    main()
