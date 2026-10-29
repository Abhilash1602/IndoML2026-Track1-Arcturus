"""Session 9 plan, Part 1 Step 5: smoke test for the multi-task 7-category
auxiliary head. Warm-starts from Phase H (the current best CNN+Conformer
checkpoint, no pretrained-fusion entanglement, isolating "does the category
head help" as a single variable per the plan's own principle), trains
Gold-only first (Step 4a -- Silver's real per-event categories are an
explicit later stretch step, not bundled in here), and evaluates against
OfficialValidationDataset with BOTH loss terms logged separately -- the
diagnostic that would have caught the earlier NoiseCategory-as-boolean
collapse sooner if it had existed then.

Gate is computed live at epoch 0 (before any training): loading Phase H's
weights with a freshly-initialized (untrained) category_head does not change
main_logits at all (category_head branches off the same trunk output but
never feeds back into main_logits), so this script's own epoch-0 evaluation
IS Phase H's own real standalone score under this fixed-decode convention --
no separate lookup needed.

Usage: python -m temp.train_category_aux_head_smoke
"""
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import CATEGORY_LIST, OfficialValidationDataset
from src.loss import CategoryAuxLoss, SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import apply_mixup, build_wandb_config_dict, make_gold_loaders

WARM_START = "./checkpoints/dualbranch_phaseh.pth"
OUT = "./checkpoints/dualbranch_category_aux_smoke.pth"
VAL_ROOT = "./data/official_validation_raw/validation"
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 6

# Per-category inverse-frequency pos_weight, from temp/inspect_category_variance.py's
# Gold-tier frame-coverage numbers (documented in STRATEGY.md): human_non_speech
# 3378.4s, vehicle_traffic 6080.7s, animal 8495.7s (max), baby_child 2407.9s,
# phone_signal_alarm 572.1s, singing_music 830.6s, appliance_machine 722.6s.
# pos_weight[c] = max_coverage / coverage[c] -- up-weights rare categories
# relative to the most common one (animal), a standard, simple heuristic.
_GOLD_COVERAGE = {
    "human_non_speech": 3378.4, "vehicle_traffic": 6080.7, "animal": 8495.7,
    "baby_child": 2407.9, "phone_signal_alarm": 572.1, "singing_music": 830.6,
    "appliance_machine": 722.6,
}
_MAX_COVERAGE = max(_GOLD_COVERAGE.values())
CATEGORY_POS_WEIGHT = torch.tensor([_MAX_COVERAGE / _GOLD_COVERAGE[c] for c in CATEGORY_LIST])


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
    config.BATCH_SIZE = 8  # reduced for safety margin against shared-GPU contention spikes (2 OOMs at default 16)
    config.USE_CATEGORY_HEAD = True
    full_ds, train_loader, _ = make_gold_loaders(config)

    model = DualBranchSED(config).to(config.DEVICE)
    state_dict = torch.load(WARM_START, map_location=config.DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # category_head is new -- expected to be missing from Phase H's checkpoint.
    assert all(k.startswith("category_head.") for k in missing), f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    print(f"warm-started from {WARM_START}, category_head randomly initialized "
          f"(missing keys, as expected): {missing}", flush=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    # Epoch-0 gate: evaluate the just-warm-started model BEFORE any training.
    # main_logits doesn't depend on the freshly-initialized category_head at all
    # (see module docstring), so this IS Phase H's own real standalone score.
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
    category_criterion = CategoryAuxLoss(pos_weight=CATEGORY_POS_WEIGHT.to(config.DEVICE))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader),
        epochs=SMOKE_EPOCHS, pct_start=0.2)

    wandb.init(project="vaani-track1", name="category-aux-head-smoke",
               config=build_wandb_config_dict(Config), job_type="category_aux_head_smoke")
    best, since = -1.0, 0
    print(f"tracking local best (starts unconstrained); reporting against Phase H's "
          f"own gate {gate:.4f} on OfficialValidationDataset (fixed decode 0.95/0.7)", flush=True)
    try:
        for epoch in range(SMOKE_EPOCHS):
            model.train()
            main_losses, cat_losses = [], []
            for audio, masks, _, cat_masks in tqdm(train_loader, desc=f"CatAux {epoch + 1}/{SMOKE_EPOCHS}"):
                audio, masks, cat_masks = audio.to(config.DEVICE), masks.to(config.DEVICE), cat_masks.to(config.DEVICE)
                audio, masks, cat_masks = apply_mixup(audio, masks, extra=cat_masks)
                optimizer.zero_grad()
                main_logits, cat_logits = model(audio)
                fr = min(main_logits.shape[1], masks.shape[1])
                main_loss = criterion(main_logits[:, :fr], masks[:, :fr])
                cat_loss = category_criterion(cat_logits[:, :fr], cat_masks[:, :fr])
                loss = main_loss + config.CATEGORY_LOSS_WEIGHT * cat_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                optimizer.step()
                scheduler.step()
                main_losses.append(main_loss.item())
                cat_losses.append(cat_loss.item())

            model.eval()
            np_, nt = cache_official(model, nat_loader, config.DEVICE)
            sp, st = cache_official(model, syn_loader, config.DEVICE)
            nd, nf, nc = score_official(np_, nt, hop)
            sd, sf, sc = score_official(sp, st, hop)
            eq = 0.5 * nc + 0.5 * sc
            tr_main, tr_cat = float(np.mean(main_losses)), float(np.mean(cat_losses))
            print(f"E{epoch + 1}: main_loss={tr_main:.4f} cat_loss={tr_cat:.4f} | "
                  f"natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(local best: {best:.4f}, gate: {gate:.4f})", flush=True)
            wandb.log({"epoch": epoch + 1, "train/main_loss": tr_main, "train/cat_loss": tr_cat,
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
        print(f"Done. Category head HELPS: eq_avg={best:.4f} (gate was {gate:.4f}) -> {OUT}.", flush=True)
    else:
        print(f"Done. Category head did not beat the gate ({gate:.4f}) "
              f"(best eq_avg={best:.4f}, checkpoint saved to {OUT} regardless).", flush=True)


if __name__ == "__main__":
    main()
