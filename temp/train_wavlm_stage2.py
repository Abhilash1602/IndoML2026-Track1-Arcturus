"""Tier 3 plan, Stage 2: unfreeze WavLM at a low LR, continuing from full1's
E4 checkpoint (the accidentally-undertrained run that's in our current best
1.4982 real ensemble -- see STRATEGY.md's root-cause section on why full1
generalizes better than the "fixed" full2).

Per "Plan: what it will take for the WavLM branch to actually beat Phase H"
(STRATEGY.md): two param groups (WavLM at ~5e-5, the rest at ~3e-4), and
critically, checkpoint selection against OfficialValidationDataset directly
using the corrected official metric (temp/official_metric.py's micro F1 +
macro Dice) -- NOT the old gold_val/synth_val proxies, which full2 proved can
point in the wrong direction (it over-calibrated to gold_val's own narrow,
~45%-breathing-dominated composition and scored worse in the real world
despite a better local number).

Usage: python -m temp.train_wavlm_stage2
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
WARM_START = "./checkpoints/dualbranch_wavlm_stage1_full.pth"  # full1's E4 best (the one in the 1.4982 ensemble)
OUT = "./checkpoints/dualbranch_wavlm_stage2.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
WAVLM_LR = 5e-5
REST_LR = 3e-4
PATIENCE = 4  # short: Sept 20 is close and OfficialValidationDataset (2,501 clips) is small enough
              # that only a decisive margin over Phase H's 1.4729-1.4734 / the ensemble's real 1.4982 counts


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
    decode) but against the REAL validation set with the corrected official
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
    config.USE_WAVLM = True
    # Backprop through the unfrozen 94M-param WavLM transformer is far more
    # memory-hungry than Stage 1's frozen forward-only pass -- the default
    # BATCH_SIZE=16 OOM'd on a GPU with ~11-13GB free (shared with another
    # user's large persistent process). Cut batch size, not architecture.
    config.BATCH_SIZE = 6
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
    model.unfreeze_wavlm()

    wavlm_params = list(model.wavlm.parameters())
    wavlm_ids = {id(p) for p in wavlm_params}
    rest_params = [p for p in model.parameters() if p.requires_grad and id(p) not in wavlm_ids]
    n_wavlm = sum(p.numel() for p in wavlm_params)
    n_rest = sum(p.numel() for p in rest_params)
    print(f"warm-started from {WARM_START}, wavlm unfrozen: {n_wavlm/1e6:.2f}M @ lr={WAVLM_LR}, "
          f"rest: {n_rest/1e6:.2f}M @ lr={REST_LR}  batches/epoch: {len(train_loader)}  -> {OUT}",
          flush=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": wavlm_params, "lr": WAVLM_LR},
        {"params": rest_params, "lr": REST_LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS * len(train_loader))

    wandb.init(project="vaani-track1", name="wavlm-stage2-unfreeze",
               config=build_wandb_config_dict(Config), job_type="wavlm_stage2")
    best, since = 1.4982, 0  # gate: must beat the CURRENT REAL BEST, not zero
    print(f"gate: must beat real-ensemble eq-avg {best:.4f} on OfficialValidationDataset "
          f"(fixed decode 0.95/0.7, same as evaluate_model's own convention)", flush=True)
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"WavLM-S2 {epoch + 1}/{config.EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                audio, masks = apply_mixup(audio, masks)
                optimizer.zero_grad()
                logits = model(audio)
                fr = min(logits.shape[1], masks.shape[1])
                loss = criterion(logits[:, :fr], masks[:, :fr])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wavlm_params + rest_params, 5.0)
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
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(gate: {best:.4f})", flush=True)
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
                if since >= PATIENCE:
                    print(f"Early stopping (no improvement in {PATIENCE}).", flush=True)
                    break
    finally:
        wandb.finish()
    if best > 1.4982:
        print(f"Done. Beat the real ensemble: eq_avg={best:.4f} -> {OUT}", flush=True)
    else:
        print(f"Done. Did NOT beat the real ensemble (1.4982) on the real validation-set proxy "
              f"(best eq_avg={best:.4f}) -- per the plan's stop condition, do not submit this branch.",
              flush=True)


if __name__ == "__main__":
    main()
