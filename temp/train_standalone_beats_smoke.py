"""BEATs frozen-head smoke run (new backbone lineage), same recipe as the
Dasheng smoke run, trained on the 1.5427 ensemble's pseudo-labels. Its
checkpoint warm-starts temp/train_beats_stage2.py.

Usage: python -m temp.train_standalone_beats_smoke
"""
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import StandaloneBEATsSED
from temp.gen_ensemble_pseudo_labels import load_pseudo_labels
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import apply_mixup, build_wandb_config_dict, make_full_mix_loaders, make_gold_loaders

OUT = "./checkpoints/standalone_beats_smoke.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 5
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

    train_loader, _ = make_full_mix_loaders(
        config, None, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50,
        pseudo_labels=load_pseudo_labels())

    model = StandaloneBEATsSED(config).to(config.DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable(new head): {n_trainable/1e6:.2f}M  frozen(BEATs): {n_frozen/1e6:.2f}M  "
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

    wandb.init(project="vaani-track1", name="standalone-beats-smoke",
               config=build_wandb_config_dict(Config), job_type="standalone_beats_smoke")
    best, since = -1.0, 0
    best_natural_combined, best_natural_f1 = 0.0, 0.0
    collapsed_epochs = 0
    try:
        for epoch in range(SMOKE_EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"StandaloneBEATs {epoch + 1}/{SMOKE_EPOCHS}"):
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
    print(f"Collapsed epochs: {collapsed_epochs}", flush=True)
    print(f"Decision rule: no exact matched baseline. Standalone ATST-Frame's own smoke test peaked "
          f"eq_avg=1.4458 (natural_combined 1.0193) before its full run only modestly improved on it "
          f"(1.4499) and then failed the ensemble-swap test decisively. Current real-best ensemble "
          f"Natural F1 is {CURRENT_BEST_NATURAL_F1:.4f}. Proceed to a full run only if: (a) no collapse, "
          f"(b) eq_avg shows a clear improving trend, (c) natural_f1 is plausible -- otherwise stop here.",
          flush=True)


if __name__ == "__main__":
    main()
