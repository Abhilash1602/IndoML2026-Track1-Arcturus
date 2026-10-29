"""Session 9 plan, Part 2 Stage 2A: smoke test for the onset/offset auxiliary
head (OOL, arXiv:2601.04178). Deliberately IDENTICAL setup to Part 1's
category-aux smoke test AND its matched no-aux control (same warm-start,
Gold-only loader, batch size 8, LR schedule, epoch cap, patience) so this
run's per-epoch eq_avg can be compared directly against the already-measured
control trajectory below -- no need to launch a second control run, since
Gold-only fine-tuning's own instability (confirmed in Part 1: both its
category-aux and no-aux runs crashed identically hard at E7) is now a known,
matched baseline rather than something to re-derive.

Matched control trajectory (temp/train_category_aux_control_smoke.py,
USE_CATEGORY_HEAD=False, identical everything else, see STRATEGY.md):
  E1=1.0902 E2=1.2161 E3=1.2515(peak) E4=1.2370 E5=1.1994 E6=1.0320
  E7=0.8333(worst) E8=0.8857 E9=0.9513(stopped)

Usage: python -m temp.train_boundary_head_smoke
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
from train import apply_mixup, build_wandb_config_dict, make_gold_loaders

WARM_START = "./checkpoints/dualbranch_phaseh.pth"
OUT = "./checkpoints/dualbranch_boundary_aux_smoke.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 6

# Part 1's matched no-aux control, for direct epoch-by-epoch comparison
# without re-running a second control (see module docstring).
CONTROL_TRAJECTORY = {
    1: 1.0902, 2: 1.2161, 3: 1.2515, 4: 1.2370, 5: 1.1994,
    6: 1.0320, 7: 0.8333, 8: 0.8857, 9: 0.9513,
}


@torch.no_grad()
def cache_official(model, loader, device):
    probs, targets = [], []
    for batch in loader:
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
    config.CNN_CH = (48, 64, 128, 256)
    config.CONF_DIM = 256
    config.CONF_HEADS = 8
    config.CONF_LAYERS = 6
    config.BATCH_SIZE = 8  # match Part 1's smoke tests exactly, for a fair comparison
    config.USE_BOUNDARY_HEAD = True
    full_ds, train_loader, _ = make_gold_loaders(config)

    model = DualBranchSED(config).to(config.DEVICE)
    state_dict = torch.load(WARM_START, map_location=config.DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert all(k.startswith("boundary_head.") for k in missing), f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    print(f"warm-started from {WARM_START}, boundary_head randomly initialized "
          f"(missing keys, as expected): {missing}", flush=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    model.eval()
    np0, nt0 = cache_official(model, nat_loader, config.DEVICE)
    sp0, st0 = cache_official(model, syn_loader, config.DEVICE)
    _, _, nc0 = score_official(np0, nt0, hop)
    _, _, sc0 = score_official(sp0, st0, hop)
    gate = 0.5 * nc0 + 0.5 * sc0
    print(f"E0 (gate, = Phase H's own standalone score): natural_combined={nc0:.4f} "
          f"synthetic_combined={sc0:.4f} eq_avg={gate:.4f}", flush=True)

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    boundary_criterion = OnsetOffsetLoss(gamma=2.0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=SMOKE_EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="boundary-aux-head-smoke",
               config=build_wandb_config_dict(Config), job_type="boundary_aux_head_smoke")
    best, since = -1.0, 0
    print(f"tracking local best (starts unconstrained); reporting against Phase H's "
          f"own gate {gate:.4f} on OfficialValidationDataset (fixed decode 0.95/0.7), "
          f"and against Part 1's matched no-aux control trajectory", flush=True)
    try:
        for epoch in range(SMOKE_EPOCHS):
            model.train()
            main_losses, bnd_losses = [], []
            for audio, masks, _, boundary_masks in tqdm(train_loader, desc=f"BndAux {epoch + 1}/{SMOKE_EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                boundary_masks = boundary_masks.to(config.DEVICE)
                audio, masks, boundary_masks = apply_mixup(audio, masks, extra=boundary_masks)
                optimizer.zero_grad()
                main_logits, boundary_logits = model(audio)
                fr = min(main_logits.shape[1], masks.shape[1])
                main_loss = criterion(main_logits[:, :fr], masks[:, :fr])
                bnd_loss = boundary_criterion(boundary_logits[:, :fr], boundary_masks[:, :fr])
                loss = main_loss + config.BOUNDARY_LOSS_WEIGHT * bnd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                optimizer.step()
                scheduler.step()
                main_losses.append(main_loss.item())
                bnd_losses.append(bnd_loss.item())

            model.eval()
            np_, nt = cache_official(model, nat_loader, config.DEVICE)
            sp, st = cache_official(model, syn_loader, config.DEVICE)
            nd, nf, nc = score_official(np_, nt, hop)
            sd, sf, sc = score_official(sp, st, hop)
            eq = 0.5 * nc + 0.5 * sc
            tr_main, tr_bnd = float(np.mean(main_losses)), float(np.mean(bnd_losses))
            control_eq = CONTROL_TRAJECTORY.get(epoch + 1)
            delta_str = f" (control: {control_eq:.4f}, delta: {eq - control_eq:+.4f})" if control_eq else ""
            print(f"E{epoch + 1}: main_loss={tr_main:.4f} bnd_loss={tr_bnd:.4f} | "
                  f"natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(local best: {best:.4f}, gate: {gate:.4f}){delta_str}", flush=True)
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
                if since >= SMOKE_PATIENCE:
                    print(f"Early stopping (no improvement in {SMOKE_PATIENCE}).", flush=True)
                    break
    finally:
        wandb.finish()
    control_best = max(CONTROL_TRAJECTORY.values())
    print(f"Done. best eq_avg={best:.4f} vs gate={gate:.4f}, vs Part 1's matched "
          f"no-aux control best={control_best:.4f} (delta: {best - control_best:+.4f}). "
          f"Checkpoint saved to {OUT} regardless.", flush=True)


if __name__ == "__main__":
    main()
