"""Plan Step 3, Stage 2: unfreeze FusionSED's branch with the load-bearing atst_stage2
recipe (LLRD + Mean Teacher + ICT, Gold+Synth supervised at synth_frac 0.3), warm-
started from temp/train_fusion_stage1.py. Plain SEDLoss (OWBCE hurt the fused ATST
model after E1). EMA teacher built fresh and loaded from the student's state_dict
(BEATs uses weight_norm, which breaks deepcopy); both saved. Capped by MAX_EPOCHS.

Env: FUSION_BRANCH, STRONG_BACKBONE, BEATS_FREQ_RESHAPE, FUSION_TAG, MAX_EPOCHS.
Usage: FUSION_BRANCH=dasheng FUSION_TAG=dashengOwB MAX_EPOCHS=8 python -m temp.train_fusion_stage2
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.distributions import Beta
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import FusionSED
from src.config import Config
from src.dataset import BronzeInferenceDataset, OfficialValidationDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import (
    _cycle_batch, apply_mixup, build_wandb_config_dict, make_gold_synth_loaders,
    trainable_parameters,
)

import os
FUSION_TAG = os.environ.get("FUSION_TAG", "fusion")
WARM_START = os.environ.get("WARM_START", f"./checkpoints/fusion_{FUSION_TAG}_stage1.pth")
OUT = f"./checkpoints/fusion_{FUSION_TAG}_stage2.pth"
OUT_TEACHER = f"./checkpoints/fusion_{FUSION_TAG}_stage2_teacher.pth"
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "0"))  # 0 = no cap; a cap breaks the loop, LR schedule unchanged
VAL_ROOT = "./data/official_validation_raw/validation"
PATIENCE = 8  # standard; the 15 was a workaround for the since-fixed gate-init bug. (was 15, extended from 8): the first attempt was still climbing at E6-E7 (right as the
               # consistency weight reached full strength at E6) when the old gate-based patience
               # logic cut it off -- give the post-ramp-up recovery real room to develop before
               # concluding it's stalled, now that patience tracks genuine local improvement.


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
    # This loop does THREE student forward passes through the now-unfrozen 85M-param
    # transformer per step (gold, sb1, mixed_sb) plus two no_grad teacher passes --
    # more memory-hungry than WavLM Stage 2's single unfrozen forward pass, which
    # already needed BATCH_SIZE cut from 16 to 6 to avoid OOM. Cut further pre-emptively.
    config.BATCH_SIZE = 8  # GPU0 otherwise idle; 4 was chosen under heavy contention
    # Synthetic needs continued supervised signal during the unfreeze so it doesn't
    # drift while the recipe chases Natural (Phase E's own lesson: deliberately
    # trading Synthetic for Natural nets out a wash, not a win).
    full_ds, train_loader, _, _ = make_gold_synth_loaders(config, synth_frac=0.3)

    model = FusionSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(WARM_START, map_location=config.DEVICE, weights_only=True))
    model.unfreeze_branch()

    atst_groups_raw = model.separate_params()
    assert len(atst_groups_raw) == 12
    n_atst_grouped = sum(p.numel() for g in atst_groups_raw for p in g)
    n_atst_total = sum(p.numel() for p in model.branch_parameters())
    assert n_atst_grouped == n_atst_total, f"LLRD grouping dropped params: {n_atst_grouped} != {n_atst_total}"
    atst_ids = {id(p) for g in atst_groups_raw for p in g}
    rest_params = [p for p in model.parameters() if p.requires_grad and id(p) not in atst_ids]

    optimizer_groups = [
        {"params": g, "lr": config.ATST_BASE_LR * (config.LLRD_DECAY ** i)}
        for i, g in enumerate(atst_groups_raw)
    ] + [{"params": rest_params, "lr": config.DISTILLED_LR}]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS * len(train_loader))

    teacher = FusionSED(config)
    teacher.unfreeze_branch()
    teacher.load_state_dict(model.state_dict())
    teacher.to(config.DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    unlabeled_ds = BronzeInferenceDataset(full_ds, config)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=config.NUM_WORKERS, drop_last=True)

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)

    n_atst_trainable = sum(p.numel() for p in model.branch_parameters() if p.requires_grad)
    print(f"warm-started from {WARM_START}, branch unfrozen: {n_atst_trainable/1e6:.2f}M "
          f"(12 LLRD groups, base_lr={config.ATST_BASE_LR}, decay={config.LLRD_DECAY}), "
          f"rest: {sum(p.numel() for p in rest_params)/1e6:.2f}M @ lr={config.DISTILLED_LR}  "
          f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    wandb.init(project="vaani-track1", name=f"fusion-stage2-{FUSION_TAG}",
               config=build_wandb_config_dict(Config), job_type="fusion_stage2")
    GATE = 1.4759  # frozen Stage-1-full standalone eq_avg -- for reporting only, not the save/stop condition
    # Session 9 fix: the first attempt initialized `best` to GATE itself, so any epoch that
    # didn't immediately beat the frozen baseline both (a) never got its checkpoint saved and
    # (b) counted toward patience from epoch 1 onward -- even though E6-E7 showed a genuine,
    # still-climbing recovery as the consistency weight reached full strength, it was cut off
    # right as it got close (E7 eq_avg 1.4701, only 0.0058 below GATE) with no checkpoint saved
    # at all. Track genuine local improvement instead (starting from -inf) so the real best
    # epoch always gets saved and the patience clock only starts once progress actually stalls.
    best, since = -1.0, 0
    best_teacher = -1.0
    print(f"tracking local best (starts unconstrained); reporting against frozen-Stage-1 "
          f"standalone gate {GATE:.4f} on OfficialValidationDataset (fixed decode 0.95/0.7)", flush=True)
    unlabeled_iter = iter(unlabeled_loader)
    try:
        for epoch in range(config.EPOCHS):
            if MAX_EPOCHS and epoch >= MAX_EPOCHS:
                print(f"Stopping at MAX_EPOCHS={MAX_EPOCHS}.", flush=True)
                break
            rampup_phase = min(1.0, epoch / max(1, config.MEAN_TEACHER_RAMPUP_EPOCHS))
            consistency_weight = config.MEAN_TEACHER_CONSISTENCY_WEIGHT * float(
                np.exp(-5.0 * (1.0 - rampup_phase) ** 2))
            model.train()
            teacher.eval()
            losses, mt_losses, ict_losses = [], [], []
            for gold_audio, gold_masks, _ in tqdm(train_loader, desc=f"Fusion-S2-{FUSION_TAG} {epoch + 1}/{config.EPOCHS}"):
                gold_audio, gold_masks = gold_audio.to(config.DEVICE), gold_masks.to(config.DEVICE)
                gold_audio, gold_masks = apply_mixup(gold_audio, gold_masks)

                (sb1, _, _), unlabeled_iter = _cycle_batch(unlabeled_loader, unlabeled_iter)
                (sb2, _, _), unlabeled_iter = _cycle_batch(unlabeled_loader, unlabeled_iter)
                sb1, sb2 = sb1.to(config.DEVICE), sb2.to(config.DEVICE)

                optimizer.zero_grad()

                gold_logits = model(gold_audio)
                fr = min(gold_logits.shape[1], gold_masks.shape[1])
                supervised_loss = criterion(gold_logits[:, :fr], gold_masks[:, :fr])

                student_probs_1 = torch.sigmoid(model(sb1))
                with torch.no_grad():
                    teacher_probs_1 = torch.sigmoid(teacher(sb1))
                    teacher_probs_2 = torch.sigmoid(teacher(sb2))
                mt_loss = F.mse_loss(student_probs_1, teacher_probs_1)

                lam = float(Beta(config.ICT_ALPHA, config.ICT_ALPHA).sample())
                mixed_sb = lam * sb1 + (1 - lam) * sb2
                student_mixed_probs = torch.sigmoid(model(mixed_sb))
                teacher_interp = lam * teacher_probs_1 + (1 - lam) * teacher_probs_2
                ict_loss = F.mse_loss(student_mixed_probs, teacher_interp)

                loss = supervised_loss + consistency_weight * (mt_loss + ict_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), 5.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for teacher_param, student_param in zip(teacher.parameters(), model.parameters()):
                        teacher_param.data.mul_(config.MEAN_TEACHER_EMA_DECAY).add_(
                            student_param.data, alpha=1 - config.MEAN_TEACHER_EMA_DECAY)

                losses.append(supervised_loss.item())
                mt_losses.append(mt_loss.item())
                ict_losses.append(ict_loss.item())

            teacher.eval()
            tnp, tnt = cache_official(teacher, nat_loader, config.DEVICE)
            tsp, tst = cache_official(teacher, syn_loader, config.DEVICE)
            _, _, tnc = score_official(tnp, tnt, hop)
            _, _, tsc = score_official(tsp, tst, hop)
            teq = 0.5 * tnc + 0.5 * tsc
            if teq > best_teacher:
                best_teacher = teq
                torch.save(teacher.state_dict(), OUT_TEACHER)
            print(f"  EMA teacher E{epoch + 1}: natural={tnc:.4f} synthetic={tsc:.4f} eq_avg={teq:.4f} "
                  f"(teacher best {best_teacher:.4f})", flush=True)
            model.eval()
            np_, nt = cache_official(model, nat_loader, config.DEVICE)
            sp, st = cache_official(model, syn_loader, config.DEVICE)
            nd, nf, nc = score_official(np_, nt, hop)
            sd, sf, sc = score_official(sp, st, hop)
            eq = 0.5 * nc + 0.5 * sc
            tr = float(np.mean(losses))
            mmt, mict = float(np.mean(mt_losses)), float(np.mean(ict_losses))
            print(f"E{epoch + 1}: sup_loss={tr:.4f} mt={mmt:.4f} ict={mict:.4f} "
                  f"cw={consistency_weight:.3f} | natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(local best: {best:.4f}, gate: {GATE:.4f})",
                  flush=True)
            wandb.log({"epoch": epoch + 1, "train/supervised_loss": tr, "train/mt_loss": mmt,
                       "train/ict_loss": mict, "train/consistency_weight": consistency_weight,
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
    if best > GATE:
        print(f"Done. Beat the frozen Stage-1 standalone baseline: eq_avg={best:.4f} -> {OUT}", flush=True)
    else:
        print(f"Done. Did NOT beat the frozen Stage-1 standalone baseline ({GATE:.4f}) "
              f"(best eq_avg={best:.4f}, checkpoint saved to {OUT} regardless -- worth an "
              f"ensemble-swap test even short of the standalone gate, per this project's own "
              f"precedent that standalone gate-misses (Phase D, Phase G2) can still help an "
              f"ensemble).", flush=True)


if __name__ == "__main__":
    main()
