"""Plan Step 2, Stage 2: AudioSet-Strong backbone (StandaloneStrongSED) unfrozen with
LLRD + Mean Teacher + ICT + OWBCE -- identical recipe to
temp/train_dasheng_stage2_i2_owbce.py (which produced the v3/v4 members), warm-started
from temp/train_strong_smoke_i3.py. EMA teacher is built fresh and loaded from the
student's state_dict (PretrainedSED's BEATs uses weight_norm, which breaks deepcopy).
Env: STRONG_BACKBONE, STRONG_STAGE, WARM_START, OWBCE_HEIGHT/HALF_WIDTH/TAG, MAX_EPOCHS.
Reference: dasheng_owA_t (best single model, eq 1.4873 at the training decode).

Usage: STRONG_BACKBONE=ATST-F OWBCE_TAG=A MAX_EPOCHS=6 python -m temp.train_strong_stage2_owbce
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.distributions import Beta
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import StandaloneStrongSED
from src.config import Config
from src.dataset import BronzeInferenceDataset, OfficialValidationDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import (
    _cycle_batch, apply_mixup, build_wandb_config_dict, make_gold_synth_loaders,
    trainable_parameters,
)

BB = f"{Config.STRONG_BACKBONE}_{Config.STRONG_STAGE}".replace("-", "")
WARM_START = os.environ.get("WARM_START", f"./checkpoints/standalone_strong_smoke_i3_{BB}.pth")  # frozen-backbone smoke checkpoint, eq_avg 1.4769
OWBCE_HEIGHT = float(os.environ.get("OWBCE_HEIGHT", "4"))
OWBCE_HALF_WIDTH = int(os.environ.get("OWBCE_HALF_WIDTH", "10"))
TAG = os.environ.get("OWBCE_TAG", f"h{OWBCE_HEIGHT:g}w{OWBCE_HALF_WIDTH}")
OUT = f"./checkpoints/standalone_strong_stage2_{BB}_owbce{TAG}.pth"
OUT_TEACHER = f"./checkpoints/standalone_strong_stage2_{BB}_owbce{TAG}_teacher.pth"
REF_EQ, REF_NAT_F1 = 1.4873, 0.3452  # dasheng_owA_t (EMA teacher, E5)
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "0"))  # 0 = no cap; a cap breaks the loop, LR schedule unchanged  # dasheng_s2_i2 (no OWBCE), same fixed decode
VAL_ROOT = "./data/official_validation_raw/validation"
PATIENCE = 8  # standard; ATST's 15 was a workaround for the since-fixed gate-init bug. (was 15, extended from 8): the first attempt was still climbing at E6-E7 (right as the
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


def natural_miss_breakdown(probabilities, ground_truth, hop, p_stay_silence=0.95, p_stay_event=0.7,
                           tolerance_frac=0.20):
    """(missed, near_miss): reference events left unmatched by the official
    one-to-one matching (temp/official_metric.match_events), and how many of
    those overlap at least one prediction (detected but misplaced)."""
    missed = near = 0
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        preds = [(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)]
        refs = [(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)]
        candidates = []
        for ri, (r_on, r_off) in enumerate(refs):
            tol = max(tolerance_frac * (r_off - r_on), 0.05)
            for pi, (p_on, p_off) in enumerate(preds):
                if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                    candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
        matched_ref, matched_pred = set(), set()
        for _, ri, pi in sorted(candidates):
            if ri not in matched_ref and pi not in matched_pred:
                matched_ref.add(ri)
                matched_pred.add(pi)
        for ri, (r_on, r_off) in enumerate(refs):
            if ri in matched_ref:
                continue
            missed += 1
            if any(p_on < r_off and p_off > r_on for p_on, p_off in preds):
                near += 1
    return missed, near


def main():
    config = Config()
    # This loop does THREE student forward passes through the now-unfrozen 85M-param
    # transformer per step (gold, sb1, mixed_sb) plus two no_grad teacher passes --
    # more memory-hungry than WavLM Stage 2's single unfrozen forward pass, which
    # already needed BATCH_SIZE cut from 16 to 6 to avoid OOM. Cut further pre-emptively.
    config.BATCH_SIZE = 8  # GPUs idle right now; ATST's 4 was chosen under heavy contention
    config.OWBCE_HEIGHT, config.OWBCE_HALF_WIDTH = OWBCE_HEIGHT, OWBCE_HALF_WIDTH
    # Synthetic needs continued supervised signal during the unfreeze so it doesn't
    # drift while the recipe chases Natural (Phase E's own lesson: deliberately
    # trading Synthetic for Natural nets out a wash, not a win).
    full_ds, train_loader, _, _ = make_gold_synth_loaders(config, synth_frac=0.3)

    model = StandaloneStrongSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(WARM_START, map_location=config.DEVICE, weights_only=True))
    model.unfreeze_backbone()

    atst_groups_raw = model.separate_params()
    assert len(atst_groups_raw) == 12
    n_atst_grouped = sum(p.numel() for g in atst_groups_raw for p in g)
    n_atst_total = sum(p.numel() for p in model.backbone_parameters())
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

    teacher = StandaloneStrongSED(config)
    teacher.unfreeze_backbone()
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
    assert criterion.owbce_height == OWBCE_HEIGHT > 0, "OWBCE not active in SEDLoss"
    print(f"OWBCE on: height={OWBCE_HEIGHT} half_width={OWBCE_HALF_WIDTH} frames (tag {TAG})", flush=True)

    n_atst_trainable = sum(p.numel() for p in model.backbone_parameters() if p.requires_grad)
    print(f"warm-started from {WARM_START}, backbone {BB} unfrozen: {n_atst_trainable/1e6:.2f}M "
          f"(12 LLRD groups, base_lr={config.ATST_BASE_LR}, decay={config.LLRD_DECAY}), "
          f"rest: {sum(p.numel() for p in rest_params)/1e6:.2f}M @ lr={config.DISTILLED_LR}  "
          f"batches/epoch: {len(train_loader)}  -> {OUT}", flush=True)

    wandb.init(project="vaani-track1", name=f"strong-stage2-{BB}-owbce-{TAG}",
               config={**build_wandb_config_dict(Config), "OWBCE_HEIGHT": OWBCE_HEIGHT,
                       "OWBCE_HALF_WIDTH": OWBCE_HALF_WIDTH},
               job_type="strong_stage2_owbce")
    GATE = 1.4769  # frozen-backbone Dasheng smoke checkpoint eq_avg -- for reporting only, not the save/stop condition
    # Session 9 fix: the first attempt initialized `best` to GATE itself, so any epoch that
    # didn't immediately beat the frozen baseline both (a) never got its checkpoint saved and
    # (b) counted toward patience from epoch 1 onward -- even though E6-E7 showed a genuine,
    # still-climbing recovery as the consistency weight reached full strength, it was cut off
    # right as it got close (E7 eq_avg 1.4701, only 0.0058 below GATE) with no checkpoint saved
    # at all. Track genuine local improvement instead (starting from -inf) so the real best
    # epoch always gets saved and the patience clock only starts once progress actually stalls.
    best, since = -1.0, 0
    best_teacher = -1.0
    print(f"tracking local best (starts unconstrained); reporting against frozen-Dasheng-smoke "
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
            for gold_audio, gold_masks, _ in tqdm(train_loader, desc=f"Strong-S2-{BB}-{TAG} {epoch + 1}/{config.EPOCHS}"):
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

            model.eval()
            np_, nt = cache_official(model, nat_loader, config.DEVICE)
            sp, st = cache_official(model, syn_loader, config.DEVICE)
            nd, nf, nc = score_official(np_, nt, hop)
            sd, sf, sc = score_official(sp, st, hop)
            eq = 0.5 * nc + 0.5 * sc
            missed, near = natural_miss_breakdown(np_, nt, hop)
            tr = float(np.mean(losses))
            mmt, mict = float(np.mean(mt_losses)), float(np.mean(ict_losses))
            print(f"E{epoch + 1}: sup_loss={tr:.4f} mt={mmt:.4f} ict={mict:.4f} "
                  f"cw={consistency_weight:.3f} | natural d={nd:.4f} f1={nf:.4f} ({nc:.4f}) | "
                  f"synthetic d={sd:.4f} f1={sf:.4f} ({sc:.4f}) | eq_avg={eq:.4f} "
                  f"(local best: {best:.4f}, gate: {GATE:.4f})",
                  flush=True)
            print(f"  natural missed={missed} near_miss={near} | vs dasheng_owA_t: "
                  f"eq {eq - REF_EQ:+.4f}, natural f1 {nf - REF_NAT_F1:+.4f}", flush=True)
            wandb.log({"epoch": epoch + 1, "train/supervised_loss": tr, "train/mt_loss": mmt,
                       "train/ict_loss": mict, "train/consistency_weight": consistency_weight,
                       "natural/dice": nd, "natural/f1": nf, "natural/combined": nc,
                       "synthetic/dice": sd, "synthetic/f1": sf, "synthetic/combined": sc,
                       "eq_avg": eq, "natural/missed": missed, "natural/near_miss": near})
            teacher.eval()
            tnp, tnt = cache_official(teacher, nat_loader, config.DEVICE)
            tsp, tst = cache_official(teacher, syn_loader, config.DEVICE)
            _, tnf, tnc = score_official(tnp, tnt, hop)
            _, _, tsc = score_official(tsp, tst, hop)
            teq = 0.5 * tnc + 0.5 * tsc
            if teq > best_teacher:
                best_teacher = teq
                torch.save(teacher.state_dict(), OUT_TEACHER)
            print(f"  EMA teacher E{epoch + 1}: natural={tnc:.4f} (f1 {tnf:.4f}) synthetic={tsc:.4f} "
                  f"eq_avg={teq:.4f} (teacher best {best_teacher:.4f})", flush=True)
            wandb.log({"epoch": epoch + 1, "teacher/eq_avg": teq, "teacher/natural_f1": tnf})
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
        print(f"Done. Beat the frozen Dasheng smoke baseline: eq_avg={best:.4f} -> {OUT}", flush=True)
    else:
        print(f"Done. Did NOT beat the frozen Dasheng smoke baseline ({GATE:.4f}) "
              f"(best eq_avg={best:.4f}, checkpoint saved to {OUT} regardless -- worth an "
              f"ensemble-swap test even short of the standalone gate, per this project's own "
              f"precedent that standalone gate-misses (Phase D, Phase G2) can still help an "
              f"ensemble).", flush=True)


if __name__ == "__main__":
    main()
