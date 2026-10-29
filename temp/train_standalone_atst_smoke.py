"""3.5-week re-plan, Must-do #2: cheap smoke test of the standalone ATST-Frame
model (src/architecture.py's StandaloneATSTSED) before committing real
GPU-time to a full run. Modeled directly on temp/train_atst_frame_stage1.py's
structure (same teacher=Phase F, same full self-distillation mix,
gold_frac=0.20/silver_bronze_frac=0.30/synth_frac=0.50, capped 15 epochs,
OfficialValidationDataset scoring from epoch 1) -- the only difference is
the model itself: StandaloneATSTSED has no shared CNN+Conformer trunk with
any existing ensemble member, unlike the fused ATST-Frame branch that
script trained.

There's no exact matched baseline for a brand-new architecture (per the
plan: "compare against a sane floor rather than a matched control").
References used for the decision rule: the fused ATST-Frame Stage 1's own
smoke-test trajectory (justified its full run by beating Phase F/G2/H's
Natural combined ~1.007-1.026 early) and its eventual full-run standalone
ceiling (eq_avg 1.4759) -- the standalone model lacks a CNN branch, so
scoring meaningfully but not necessarily matching that ceiling this early
is still a reasonable signal to proceed.

Usage: python -m temp.train_standalone_atst_smoke
"""
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED, StandaloneATSTSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import apply_mixup, build_wandb_config_dict, make_full_mix_loaders, make_gold_loaders

TEACHER_PATH = "./checkpoints/dualbranch_phasef.pth"  # Phase F, real Combined 1.4727
OUT = "./checkpoints/standalone_atst_smoke.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 5

# Reference points for the decision rule (see module docstring).
FUSED_ATST_S1_SMOKE_NATURAL_COMBINED = None  # not directly logged in that script; use eq_avg instead
FUSED_ATST_S1_FULL_EQ_AVG = 1.4759
CURRENT_BEST_NATURAL_F1 = 0.4054


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
    full_ds, gold_train_loader, _ = make_gold_loaders(config)

    teacher_config = Config()  # Phase F is the DEFAULT small architecture, no fusion branch
    teacher = DualBranchSED(teacher_config).to(teacher_config.DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=teacher_config.DEVICE, weights_only=True))
    teacher.eval()

    train_loader, _ = make_full_mix_loaders(
        config, teacher, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50)

    model = StandaloneATSTSED(config).to(config.DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable(new head): {n_trainable/1e6:.2f}M  frozen(ATST-Frame): {n_frozen/1e6:.2f}M  "
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

    wandb.init(project="vaani-track1", name="standalone-atst-smoke",
               config=build_wandb_config_dict(Config), job_type="standalone_atst_smoke")
    best, since = -1.0, 0
    best_natural_combined, best_natural_f1 = 0.0, 0.0
    collapsed_epochs = 0
    try:
        for epoch in range(SMOKE_EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"StandaloneATST {epoch + 1}/{SMOKE_EPOCHS}"):
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
            # Collapse check: a degenerate all-silence or all-event predictor
            # would show up as near-zero probability variance -- catch it
            # explicitly rather than just reading a low eq_avg and guessing why.
            prob_std = float(np.std(np_)) if np_.size else 0.0
            is_collapsed = prob_std < 0.01
            if is_collapsed:
                collapsed_epochs += 1
            print(f"E{epoch + 1}: loss={tr:.4f} | natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"| pred_prob_std={prob_std:.4f}{' [COLLAPSED]' if is_collapsed else ''}", flush=True)
            wandb.log({"epoch": epoch + 1, "train/loss": tr,
                       "natural/dice": nd, "natural/f1": nf, "natural/combined": nc,
                       "synthetic/dice": sd, "synthetic/f1": sf, "synthetic/combined": sc,
                       "eq_avg": eq, "pred_prob_std": prob_std})
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
    print(f"Collapsed epochs: {collapsed_epochs}/{SMOKE_EPOCHS if since < SMOKE_PATIENCE else epoch+1}", flush=True)
    print(f"Decision rule: no exact matched baseline for a brand-new architecture. "
          f"Fused ATST-Frame Stage 1's own full-run ceiling was eq_avg={FUSED_ATST_S1_FULL_EQ_AVG:.4f}; "
          f"current real-best ensemble Natural F1 is {CURRENT_BEST_NATURAL_F1:.4f}. Proceed to a full, "
          f"uncapped run only if: (a) no collapse (pred_prob_std stays healthy across most epochs), "
          f"(b) eq_avg shows a clear improving trend, not flat/noisy, and (c) best natural_f1 is in a "
          f"plausible range to eventually contribute to the ensemble (not orders of magnitude below "
          f"{CURRENT_BEST_NATURAL_F1:.4f}) -- otherwise stop here, per this project's established discipline.",
          flush=True)


if __name__ == "__main__":
    main()
