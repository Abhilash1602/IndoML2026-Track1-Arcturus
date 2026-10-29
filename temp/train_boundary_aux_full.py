"""Session 9 plan, Part 2 Stage 2A: full self-distillation-mix training run
for the boundary-aux head (OOL, arXiv:2601.04178). Deliberately mirrors
temp/train_phaseh.py's exact proven recipe line-for-line (teacher=Phase F,
gold_frac=0.20/silver_bronze_frac=0.30/synth_frac=0.50, same capacity, same
PEAK_LR=5e-4 OneCycleLR, fresh from-scratch model -- NOT warm-started) so
the boundary-aux loss is the ONLY new variable versus Phase H's real
Codabench result (1.4815) -- the cleanest possible isolation, per this
project's own "don't stack unverified changes" discipline (Phase G's
postmortem) and the user's explicit choice over a faster but confounded
warm-start alternative.

Two deltas from train_phaseh.py's checkpoint-selection convention (both
modernizations matching every later full-mix script's own established
practice, not deviations): (1) evaluates against OfficialValidationDataset
(cache_official/score_official, the primary decision signal since Phase D)
instead of Phase H's older gold_val/synth_val proxies; (2) discards
gold_val_loader/synth_val_loader entirely, since VaaniDataset/
SyntheticSEDDataset now return 5-tuples under USE_BOUNDARY_HEAD=True and
evaluate_model's fixed 3-tuple unpack would crash on them.

Usage: python -m temp.train_boundary_aux_full
"""
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.loss import OnsetOffsetLoss, SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import apply_mixup, build_wandb_config_dict, make_full_mix_loaders, make_gold_loaders

TEACHER_PATH = "./checkpoints/dualbranch_phasef.pth"  # Phase F, real Combined 1.4727 -- NOT Phase H
OUT = "./checkpoints/dualbranch_boundary_aux_full.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
PEAK_LR = 5e-4


@torch.no_grad()
def cache_official(model, loader, device):
    probs, targets = [], []
    for batch in loader:
        # OfficialValidationDataset returns a 4-tuple under USE_BOUNDARY_HEAD=True
        # (audio, mask, clip_id, boundary_mask) -- index-based access, not a
        # fixed unpack, so this works regardless of the flag.
        audio, masks = batch[0], batch[1]
        audio = audio.to(device, non_blocking=True)
        out = model(audio)
        main_logits = out[0] if isinstance(out, tuple) else out
        p = torch.sigmoid(main_logits)
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
    full_ds, gold_train_loader, _ = make_gold_loaders(config)  # gold_val_loader discarded, see module docstring

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
    config.BATCH_SIZE = 8  # shared-GPU contention precedent (Phase H), safety margin
    config.USE_BOUNDARY_HEAD = True
    config.BOUNDARY_LOSS_WEIGHT = 0.2

    train_loader, _ = make_full_mix_loaders(  # synth_val_loader discarded, see module docstring
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)

    model = DualBranchSED(config).to(config.DEVICE)  # fresh init -- NOT warm-started, see module docstring
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M  batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    boundary_criterion = OnsetOffsetLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=PEAK_LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="boundary-aux-full",
               config=build_wandb_config_dict(Config), job_type="boundary_aux_full")
    # Phase H's own real anchor, for reporting only -- best starts genuinely
    # unconstrained (-inf-equivalent), per this project's own twice-fixed
    # checkpoint-selection bug (never gate `best`'s initial value externally).
    PHASE_H_REAL, PHASE_H_PROXY = 1.4815, 1.4729
    best, since = -1.0, 0
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            main_losses, bnd_losses = [], []
            for audio, masks, _, boundary_masks, boundary_valid in tqdm(
                    train_loader, desc=f"BndAuxFull {epoch + 1}/{config.EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                boundary_masks, boundary_valid = boundary_masks.to(config.DEVICE), boundary_valid.to(config.DEVICE)

                # Pack boundary_valid as a third, time-broadcast channel of the
                # mixup extra tensor so apply_mixup's existing max/union rule
                # (train.py, unmodified) correctly propagates validity: mixing
                # valid+invalid keeps valid=1 (max(1,0)=1) with the valid
                # sample's real target passing through unchanged (max(real,0)
                # =real); mixing two invalid samples stays all-zero/invalid=0.
                valid_channel = boundary_valid.view(-1, 1, 1).expand(-1, boundary_masks.size(1), 1)
                combined_extra = torch.cat([boundary_masks, valid_channel], dim=-1)
                audio, masks, combined_extra = apply_mixup(audio, masks, extra=combined_extra)
                boundary_masks, boundary_valid = combined_extra[..., :2], combined_extra[..., 2].amax(dim=1)

                optimizer.zero_grad()
                main_logits, boundary_logits = model(audio)
                fr = min(main_logits.shape[1], masks.shape[1])
                main_loss = criterion(main_logits[:, :fr], masks[:, :fr])
                valid_idx = boundary_valid > 0.5
                if valid_idx.any():
                    bnd_loss = boundary_criterion(boundary_logits[valid_idx][:, :fr], boundary_masks[valid_idx][:, :fr])
                    bnd_losses.append(bnd_loss.item())
                else:
                    bnd_loss = torch.zeros((), device=config.DEVICE)  # rare (~0.007%/batch); well-defined no-op
                loss = main_loss + config.BOUNDARY_LOSS_WEIGHT * bnd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                main_losses.append(main_loss.item())

            model.eval()
            np_, nt = cache_official(model, nat_loader, config.DEVICE)
            sp, st = cache_official(model, syn_loader, config.DEVICE)
            nd, nf, nc = score_official(np_, nt, hop)
            sd, sf, sc = score_official(sp, st, hop)
            eq = 0.5 * nc + 0.5 * sc
            tr_main = float(np.mean(main_losses))
            tr_bnd = float(np.mean(bnd_losses)) if bnd_losses else 0.0
            print(f"E{epoch + 1}: main_loss={tr_main:.4f} bnd_loss={tr_bnd:.4f} | "
                  f"natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(local best: {best:.4f}, Phase H proxy: {PHASE_H_PROXY:.4f})", flush=True)
            wandb.log({"epoch": epoch + 1, "train/main_loss": tr_main, "train/bnd_loss": tr_bnd,
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
    print(f"Done. Best eq_avg={best:.4f} vs Phase H's real {PHASE_H_REAL:.4f} "
          f"(proxy {PHASE_H_PROXY:.4f}). Checkpoint -> {OUT}.", flush=True)


if __name__ == "__main__":
    main()
