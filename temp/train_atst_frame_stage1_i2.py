"""Iteration 2 for the load-bearing ATST member: the frozen ATST-Frame fusion
Stage 1 (same recipe as temp/train_atst_frame_stage1.py, capped 15 epochs)
trained on the 1.5427 ensemble's pseudo-labels instead of Phase F's. Its
checkpoint warm-starts temp/train_atst_frame_stage2_i2.py -- the same path that
lifted Dasheng Stage 2 from 1.4630 to 1.4827 standalone.

Usage: python -m temp.train_atst_frame_stage1_i2
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
from temp.gen_ensemble_pseudo_labels import load_pseudo_labels
from train import apply_mixup, build_wandb_config_dict, make_full_mix_loaders, make_gold_loaders

OUT = "./checkpoints/dualbranch_atst_frame_stage1_i2.pth"
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

    train_loader, _ = make_full_mix_loaders(
        config, None, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50,
        pseudo_labels=load_pseudo_labels())

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

    wandb.init(project="vaani-track1", name="atst-frame-stage1-i2",
               config=build_wandb_config_dict(Config), job_type="atst_frame_stage1_i2")
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
