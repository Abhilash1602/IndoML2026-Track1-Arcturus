"""Session 9 plan, Part 1 Step 5 control: isolates whether the category-aux
smoke test's failure to beat the gate (best eq_avg=1.3216 vs gate 1.4716,
see STRATEGY.md) is caused by the category head itself, or simply by
narrowing training data to Gold-only (away from Phase H's full
self-distillation mix) -- a confound this project already found harmful in
Phase C, independent of any other change.

Identical to temp/train_category_aux_head_smoke.py in every other respect
(warm-start, Gold-only loader, batch size, LR schedule, epoch cap, patience)
except `Config.USE_CATEGORY_HEAD = False`: plain SEDLoss only, no aux term,
no category_head weights trained at all. If this control also declines
similarly, the category head is exonerated and Part 1 should be retried on
the full self-distillation mix. If it holds much closer to the gate, the
category head itself is actively harmful.

Usage: python -m temp.train_category_aux_control_smoke
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
from train import apply_mixup, build_wandb_config_dict, make_gold_loaders

WARM_START = "./checkpoints/dualbranch_phaseh.pth"
OUT = "./checkpoints/dualbranch_category_aux_control_smoke.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 6


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
    config.BATCH_SIZE = 8  # match the category-aux smoke test's safety margin
    config.USE_CATEGORY_HEAD = False  # the only deliberate difference from the category-aux smoke test
    full_ds, train_loader, _ = make_gold_loaders(config)

    model = DualBranchSED(config).to(config.DEVICE)
    state_dict = torch.load(WARM_START, map_location=config.DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing, f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    print(f"warm-started from {WARM_START}, exact match (no category_head in this run)", flush=True)

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
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=SMOKE_EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="category-aux-control-smoke",
               config=build_wandb_config_dict(Config), job_type="category_aux_control_smoke")
    best, since = -1.0, 0
    print(f"tracking local best (starts unconstrained); reporting against Phase H's "
          f"own gate {gate:.4f} on OfficialValidationDataset (fixed decode 0.95/0.7)", flush=True)
    try:
        for epoch in range(SMOKE_EPOCHS):
            model.train()
            main_losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"CatControl {epoch + 1}/{SMOKE_EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                audio, masks = apply_mixup(audio, masks)
                optimizer.zero_grad()
                main_logits = model(audio)
                fr = min(main_logits.shape[1], masks.shape[1])
                main_loss = criterion(main_logits[:, :fr], masks[:, :fr])
                main_loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
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
            print(f"E{epoch + 1}: main_loss={tr_main:.4f} | "
                  f"natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(local best: {best:.4f}, gate: {gate:.4f})", flush=True)
            wandb.log({"epoch": epoch + 1, "train/main_loss": tr_main,
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
    if best > gate:
        print(f"Done. Gold-only fine-tune alone HELPS: eq_avg={best:.4f} (gate was {gate:.4f}) -> {OUT}.", flush=True)
    else:
        print(f"Done. Gold-only fine-tune alone did not beat the gate ({gate:.4f}) "
              f"(best eq_avg={best:.4f}, checkpoint saved to {OUT} regardless). "
              f"Compare against the category-aux smoke test's best (1.3216) to isolate the confound.", flush=True)


if __name__ == "__main__":
    main()
