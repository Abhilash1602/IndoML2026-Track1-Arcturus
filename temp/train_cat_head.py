"""Phase C: per-category model. A v4 member (warm start) gets a 7-way category head next to
its any-noise head, trained with the V1 recipe (Gold + our synthetic + official-validation
train70, LLRD + MT/ICT, OWBCE-A on the any-noise output, 3 epochs, last-epoch checkpoints)
plus a masked BCE on the per-category frame targets (Gold and validation carry categories;
our synthetic set has none, so its category loss is masked out).

Its category probabilities feed temp/eval_cat_decode.py, which tests category-aware event
construction on hold30.

Env: CAT_KIND (dasheng|beats), WARM_START, CAT_TAG, MAX_EPOCHS (3), CAT_WEIGHT (0.5),
     plus the V1 variables (VAL_TRAIN=train70, GOLD_ALL, ATST_BASE_LR, DISTILLED_LR, SYNTH_FRAC,
     VAL_NAT_SHARE, VAL_SYN_SHARE).
Usage: CAT_KIND=dasheng WARM_START=./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth CAT_TAG=cOwB \\
       VAL_TRAIN=train70 GOLD_ALL=1 python -m temp.train_cat_head
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.architecture import StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.dataset import BronzeInferenceDataset, SyntheticSEDDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import (SYNTH_TRAIN_DIR, VAL_SPLIT, _cycle_batch, apply_mixup, make_gold_loaders,
                   official_val_subset, trainable_parameters)

KIND = os.environ.get("CAT_KIND", "dasheng")
CLS = {"dasheng": StandaloneDashengSED, "beats": StandaloneBEATsSED}[KIND]
WARM_START = os.environ["WARM_START"]
TAG = os.environ.get("CAT_TAG", KIND)
OUT = f"./checkpoints/cat_{TAG}_last.pth"
OUT_TEACHER = f"./checkpoints/cat_{TAG}_last_teacher.pth"
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "3"))
CAT_WEIGHT = float(os.environ.get("CAT_WEIGHT", "0.5"))
SYNTH_FRAC = float(os.environ.get("SYNTH_FRAC", "0.3"))
VAL_NAT_SHARE = float(os.environ.get("VAL_NAT_SHARE", "0.3"))
VAL_SYN_SHARE = float(os.environ.get("VAL_SYN_SHARE", "0.5"))
N_CAT = 7


class WithCat(Dataset):
    """-> (audio, any-noise mask, category mask (T, 7) or zeros, category-valid flag)."""

    def __init__(self, base, has_cat, frames):
        self.base, self.has_cat, self.frames = base, has_cat, frames

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        cat = item[3] if (self.has_cat and len(item) > 3) else torch.zeros(self.frames, N_CAT)
        return item[0], item[1], cat, float(self.has_cat)


def build_config():
    config = Config()
    config.BATCH_SIZE = 8
    config.CAT_HEAD, config.OUTPUT_HEAD, config.USE_CATEGORY_HEAD = True, "anycat", True
    config.OWBCE_HEIGHT = float(os.environ.get("OWBCE_HEIGHT", "4"))
    config.OWBCE_HALF_WIDTH = int(os.environ.get("OWBCE_HALF_WIDTH", "10"))
    return config


def build_model(config):
    model = CLS(config)
    (model.unfreeze_dasheng if KIND == "dasheng" else model.unfreeze_beats)()
    return model


def load_warm(model, path):
    missing, unexpected = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    assert not unexpected and all(k.startswith("classifier_cat.") for k in missing), (missing, unexpected)


@torch.no_grad()
def evaluate(model, loaders, device, hop):
    model.eval()
    out = {}
    for sub, loader in loaders.items():
        probs, targets, cats, cat_t = [], [], [], []
        for batch in loader:
            audio, mask = batch[0], batch[1]
            p = torch.sigmoid(model(audio.to(device))).cpu().numpy()
            fr = min(p.shape[1], mask.shape[1])
            probs.append(p[:, :fr, 0]); targets.append(mask[:, :fr].numpy())
            if len(batch) > 3:
                cats.append(p[:, :fr, 1:]); cat_t.append(batch[3][:, :fr].numpy())
        P, T = np.concatenate(probs), np.concatenate(targets)
        pred = [[(e["onset"], e["offset"]) for e in extract_events_viterbi(viterbi_decode(x, p_stay_silence=0.9802, p_stay_event=0.9489), hop)] for x in P]
        gt = [[(e["onset"], e["offset"]) for e in extract_events_from_mask(t.astype(np.int32), hop)] for t in T]
        f1, _, _ = micro_event_f1(gt, pred)
        res = {"f1": f1, "dice": macro_segment_dice(gt, pred, hop_sec=hop)}
        if cats:
            Cp, Ct = np.concatenate(cats), np.concatenate(cat_t)
            act = Ct.sum((0, 1)) > 0
            # frame-level category accuracy on noise frames (argmax vs any true category)
            on = Ct.sum(-1) > 0
            hit = np.take_along_axis(Ct, Cp.argmax(-1)[..., None], -1)[..., 0] > 0
            res["cat_acc"] = float(hit[on].mean()) if on.any() else float("nan")
            res["active_cats"] = int(act.sum())
        out[sub] = res
    return out


def main():
    torch.manual_seed(0)
    config = build_config()
    device, hop = config.DEVICE, config.HOP_LENGTH / config.SR
    frames = config.target_samples // config.HOP_LENGTH
    part = os.environ.get("VAL_TRAIN", "train70")
    full_ds, gold_loader, _ = make_gold_loaders(config)
    parts = [WithCat(gold_loader.dataset, True, frames), WithCat(SyntheticSEDDataset(SYNTH_TRAIN_DIR, config), False, frames),
             WithCat(official_val_subset(config, "natural", part), True, frames),
             WithCat(official_val_subset(config, "synthetic", part), True, frames)]
    if part == "train70":
        hold = set(sum(json.load(open(VAL_SPLIT))["hold30"].values(), []))
        for d in parts[2:]:
            assert not any(d.base.dataset.rows[i]["segmentFileName"] in hold for i in d.base.indices), "hold30 leaked"
    shares = [(1 - SYNTH_FRAC) * (1 - VAL_NAT_SHARE), SYNTH_FRAC * (1 - VAL_SYN_SHARE),
              (1 - SYNTH_FRAC) * VAL_NAT_SHARE, SYNTH_FRAC * VAL_SYN_SHARE]
    weights = [w for d, s in zip(parts, shares) for w in [s / len(d)] * len(d)]
    train_loader = DataLoader(ConcatDataset(parts), batch_size=config.BATCH_SIZE, drop_last=True,
                              sampler=WeightedRandomSampler(weights, int(len(parts[0]) / (1 - SYNTH_FRAC)), replacement=True),
                              num_workers=config.NUM_WORKERS)
    unl_loader = DataLoader(BronzeInferenceDataset(full_ds, config), batch_size=config.BATCH_SIZE, shuffle=True,
                            num_workers=config.NUM_WORKERS, drop_last=True)
    eval_part = "hold30" if part == "train70" else "all"
    eval_loaders = {sub: DataLoader(official_val_subset(config, sub, eval_part), batch_size=8, num_workers=config.NUM_WORKERS)
                    for sub in ("natural", "synthetic")}

    model = build_model(config)
    load_warm(model, WARM_START)
    model = model.to(device)
    teacher = build_model(config)
    teacher.load_state_dict(model.state_dict())
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    groups = model.separate_params()
    ids = {id(p) for g in groups for p in g}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    opt = torch.optim.AdamW([{"params": g, "lr": config.ATST_BASE_LR * config.LLRD_DECAY ** i} for i, g in enumerate(groups)]
                            + [{"params": rest, "lr": config.DISTILLED_LR}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.EPOCHS * len(train_loader))
    criterion = SEDLoss(config).to(device)
    print(f"[{TAG}] {KIND} + category head, warm start {WARM_START}; {len(train_loader)} batches/epoch; "
          f"shares {[round(s, 3) for s in shares]}; cat weight {CAT_WEIGHT}", flush=True)
    unl_iter = iter(unl_loader)
    for epoch in range(MAX_EPOCHS):
        ramp = min(1.0, epoch / max(1, config.MEAN_TEACHER_RAMPUP_EPOCHS))
        cw = config.MEAN_TEACHER_CONSISTENCY_WEIGHT * float(np.exp(-5.0 * (1.0 - ramp) ** 2))
        model.train()
        la, lc = [], []
        for audio, mask, cat, valid in tqdm(train_loader, desc=f"Cat-{TAG} {epoch + 1}/{MAX_EPOCHS}"):
            audio, mask, cat, valid = audio.to(device), mask.to(device), cat.to(device), valid.to(device).float()
            pa, pm, pc, pv = [], [], [], []
            for v in (1.0, 0.0):  # mixup only among samples of the same label type
                sel = valid == v
                if sel.any():
                    a, m, c = apply_mixup(audio[sel], mask[sel], extra=cat[sel])
                    pa.append(a); pm.append(m); pc.append(c); pv.append(valid[sel])
            audio, mask, cat, valid = torch.cat(pa), torch.cat(pm), torch.cat(pc), torch.cat(pv)
            (sb1, _, _), unl_iter = _cycle_batch(unl_loader, unl_iter)
            (sb2, _, _), unl_iter = _cycle_batch(unl_loader, unl_iter)
            sb1, sb2 = sb1.to(device), sb2.to(device)
            opt.zero_grad()
            out = model(audio)
            fr = min(out.shape[1], mask.shape[1])
            loss_any = criterion(out[:, :fr, 0], mask[:, :fr])
            if valid.sum() > 0:
                bce = F.binary_cross_entropy_with_logits(out[:, :fr, 1:], cat[:, :fr], reduction="none").mean((1, 2))
                loss_cat = (bce * valid).sum() / valid.sum()
            else:
                loss_cat = out.sum() * 0
            s1 = torch.sigmoid(model(sb1)[..., 0])
            with torch.no_grad():
                t1, t2 = torch.sigmoid(teacher(sb1)[..., 0]), torch.sigmoid(teacher(sb2)[..., 0])
            lam = float(Beta(config.ICT_ALPHA, config.ICT_ALPHA).sample())
            sm = torch.sigmoid(model(lam * sb1 + (1 - lam) * sb2)[..., 0])
            loss = loss_any + CAT_WEIGHT * loss_cat + cw * (F.mse_loss(s1, t1) + F.mse_loss(sm, lam * t1 + (1 - lam) * t2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), 5.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), model.parameters()):
                    tp.data.mul_(config.MEAN_TEACHER_EMA_DECAY).add_(sp.data, alpha=1 - config.MEAN_TEACHER_EMA_DECAY)
            la.append(float(loss_any)); lc.append(float(loss_cat))
        r, rt = evaluate(model, eval_loaders, device, hop), evaluate(teacher, eval_loaders, device, hop)
        fmt = lambda x: " | ".join(f"{k}: F1 {v['f1']:.4f} D {v['dice']:.4f} catAcc {v.get('cat_acc', float('nan')):.3f}" for k, v in x.items())
        print(f"E{epoch + 1}: any={np.mean(la):.4f} cat={np.mean(lc):.4f} cw={cw:.3f}\n  student {fmt(r)}\n  teacher {fmt(rt)}", flush=True)
    torch.save(model.state_dict(), OUT)
    torch.save(teacher.state_dict(), OUT_TEACHER)
    print(f"Done [{TAG}] -> {OUT}, {OUT_TEACHER}", flush=True)


if __name__ == "__main__":
    main()
