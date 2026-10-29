"""Iteration 2: Phase H's exact from-scratch recipe (same capacity, data mix
gold 0.20 / silver+bronze 0.30 / synth 0.50, PEAK_LR 5e-4 OneCycle,
OfficialValidationDataset selection) with ONE change: Silver+Bronze
pseudo-labels come from the 1.5427 ensemble (data/ensemble_pseudo_labels_v1.npz)
instead of the Phase F teacher. Isolated test of whether better pseudo-labels
improve a member; candidate replacement for `phaseh` in the ensemble.

Usage: python -m temp.train_phaseh_i2
"""
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from temp.gen_ensemble_pseudo_labels import load_pseudo_labels
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import apply_mixup, build_wandb_config_dict, make_full_mix_loaders, make_gold_loaders

OUT = "./checkpoints/dualbranch_phaseh_i2.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
PEAK_LR = 5e-4
PHASE_H_REAL, PHASE_H_PROXY = 1.4815, 1.4729


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

    config.CNN_CH = (48, 64, 128, 256)
    config.CONF_DIM = 256
    config.CONF_HEADS = 8
    config.CONF_LAYERS = 6
    config.BATCH_SIZE = 8

    pseudo_labels = load_pseudo_labels()
    print(f"Loaded {len(pseudo_labels)} ensemble pseudo-labels (iteration 2)", flush=True)

    train_loader, _ = make_full_mix_loaders(
        config, None, full_ds, gold_train_loader.dataset,
        gold_frac=0.20, silver_bronze_frac=0.30, synth_frac=0.50,
        pseudo_labels=pseudo_labels)

    model = DualBranchSED(config).to(config.DEVICE)  # fresh init, matches Phase H exactly
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M  batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=PEAK_LR, steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="phaseh-i2-ensemble-pl",
               config=build_wandb_config_dict(Config), job_type="phaseh_i2")
    best, since = -1.0, 0
    best_natural_combined, best_natural_f1 = 0.0, 0.0
    print(f"tracking local best (starts unconstrained); reporting against Phase H's "
          f"real {PHASE_H_REAL:.4f} (proxy {PHASE_H_PROXY:.4f})", flush=True)
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"PhaseH-I2 {epoch + 1}/{config.EPOCHS}"):
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

            model.eval()
            np_, nt = cache_official(model, nat_loader, config.DEVICE)
            sp, st = cache_official(model, syn_loader, config.DEVICE)
            nd, nf, nc = score_official(np_, nt, hop)
            sd, sf, sc = score_official(sp, st, hop)
            eq = 0.5 * nc + 0.5 * sc
            tr = float(np.mean(losses))
            print(f"E{epoch + 1}: loss={tr:.4f} | natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(local best: {best:.4f}, Phase H proxy: {PHASE_H_PROXY:.4f})", flush=True)
            wandb.log({"epoch": epoch + 1, "train/loss": tr,
                       "natural/dice": nd, "natural/f1": nf, "natural/combined": nc,
                       "synthetic/dice": sd, "synthetic/f1": sf, "synthetic/combined": sc,
                       "eq_avg": eq})
            if eq > best:
                best, since = eq, 0
                best_natural_combined, best_natural_f1 = nc, nf
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
          f"(proxy {PHASE_H_PROXY:.4f}) (natural_combined={best_natural_combined:.4f}, "
          f"natural_f1={best_natural_f1:.4f}). Checkpoint -> {OUT}.", flush=True)


if __name__ == "__main__":
    main()
