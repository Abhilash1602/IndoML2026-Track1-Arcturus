"""Two-convention model (Sep 29 root cause, STRATEGY.md): Natural references are human
judgements (tighter edges), synthetic references are exact mix times (full acoustic
extent), and one output head trained on both learns the wide synthetic edges and
applies them to Natural. Here the shared trunk feeds two heads: `classifier` (Natural,
trained ONLY on Gold) and `classifier_syn` (trained ONLY on synthetic).

Recipe = the Stage-2 recipe behind the v3/v4 members (LLRD + Mean Teacher + ICT +
OWBCE-A, Gold + synthetic at synth_frac 0.3), warm-started from a v4 member with its
head copied into both heads, plus:
  * every supervised sample carries its domain; loss goes to its own head only
  * mixup only within a domain (a cross-domain mix would blend the two conventions)
  * MT/ICT consistency on the Natural head (Silver/Bronze are Natural recordings)
Each epoch: official validation scored with the Natural head on Natural clips and
the synthetic head on synthetic clips (oracle routing; the deployable router comes
from the prediction-sharpness gate), plus each head everywhere, for student and EMA teacher.

Env: DUAL_KIND (dasheng|beats), WARM_START, DUAL_TAG, MAX_EPOCHS (default 4), GOLD_ALL,
ATST_BASE_LR / DISTILLED_LR (backbone LLRD base / rest), OWBCE_HEIGHT/HALF_WIDTH (4/10).
Usage: DUAL_KIND=dasheng WARM_START=./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth \\
       DUAL_TAG=dOwB python -m temp.train_dual_head
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.architecture import StandaloneBEATsSED, StandaloneDashengSED, load_dual_warm_start
from src.config import Config
from src.dataset import BronzeInferenceDataset, OfficialValidationDataset, SyntheticSEDDataset
from src.loss import SEDLoss
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import SYNTH_TRAIN_DIR, _cycle_batch, apply_mixup, make_gold_loaders, trainable_parameters

KIND = os.environ.get("DUAL_KIND", "dasheng")
CLS = {"dasheng": StandaloneDashengSED, "beats": StandaloneBEATsSED}[KIND]
WARM_START = os.environ["WARM_START"]
TAG = os.environ.get("DUAL_TAG", KIND)
OUT = f"./checkpoints/dual_{TAG}.pth"
OUT_TEACHER = f"./checkpoints/dual_{TAG}_teacher.pth"
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "4"))
SYNTH_FRAC = float(os.environ.get("SYNTH_FRAC", "0.3"))
VAL_ROOT = "./data/official_validation_raw/validation"
NAT, SYN = 0, 1


class DomainTagged(Dataset):
    def __init__(self, base, domain):
        self.base, self.domain = base, domain

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        return item[0], item[1], self.domain


def build_config():
    config = Config()
    config.BATCH_SIZE = 8
    config.DUAL_HEAD, config.OUTPUT_HEAD = True, "both"
    config.OWBCE_HEIGHT = float(os.environ.get("OWBCE_HEIGHT", "4"))
    config.OWBCE_HALF_WIDTH = int(os.environ.get("OWBCE_HALF_WIDTH", "10"))
    return config


def build_model(config):
    model = CLS(config)
    if KIND == "dasheng":
        model.unfreeze_dasheng()
    else:
        model.unfreeze_beats()
    return model


@torch.no_grad()
def cache_both(model, loader, device):
    probs, targets = [], []
    for audio, masks, _ in loader:
        p = torch.sigmoid(model(audio.to(device)))  # (B, T, 2)
        fr = min(p.shape[1], masks.shape[1])
        probs.append(p[:, :fr].cpu().numpy())
        targets.append(masks[:, :fr].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score(probs, targets, hop, ps=0.9802, pe=0.9489):
    pred, gt = [], []
    for p, t in zip(probs, targets):
        pred.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(
            viterbi_decode(p, p_stay_silence=ps, p_stay_event=pe), hop)])
        gt.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(t.astype(np.int32), hop)])
    f1, _, _ = micro_event_f1(gt, pred)
    return f1, macro_segment_dice(gt, pred, hop_sec=hop)


def evaluate(model, nat_loader, syn_loader, device, hop):
    model.eval()
    npb, nt = cache_both(model, nat_loader, device)
    spb, st = cache_both(model, syn_loader, device)
    r = {}
    for h, name in ((NAT, "nat"), (SYN, "syn")):
        r[f"natural@{name}"] = score(npb[..., h], nt, hop)
        r[f"synthetic@{name}"] = score(spb[..., h], st, hop)
    nf, nd = r["natural@nat"]
    sf, sd = r["synthetic@syn"]
    r["oracle_eq"] = 0.5 * (nf + nd + sf + sd)
    return r


def fmt(r):
    return (f"oracle eq={r['oracle_eq']:.4f} | natural@nat F1={r['natural@nat'][0]:.4f} D={r['natural@nat'][1]:.4f} "
            f"| natural@syn F1={r['natural@syn'][0]:.4f} | synthetic@syn F1={r['synthetic@syn'][0]:.4f} "
            f"D={r['synthetic@syn'][1]:.4f} | synthetic@nat F1={r['synthetic@nat'][0]:.4f}")


def main():
    torch.manual_seed(0)
    config = build_config()
    device = config.DEVICE
    full_ds, gold_train_loader, _ = make_gold_loaders(config)
    gold = DomainTagged(gold_train_loader.dataset, NAT)
    synth = DomainTagged(SyntheticSEDDataset(SYNTH_TRAIN_DIR, config), SYN)
    n_g, n_s = len(gold), len(synth)
    weights = [(1 - SYNTH_FRAC) / n_g] * n_g + [SYNTH_FRAC / n_s] * n_s
    train_loader = DataLoader(ConcatDataset([gold, synth]), batch_size=config.BATCH_SIZE,
                              sampler=WeightedRandomSampler(weights, int(n_g / (1 - SYNTH_FRAC)), replacement=True),
                              num_workers=config.NUM_WORKERS, drop_last=True)
    unlabeled_loader = DataLoader(BronzeInferenceDataset(full_ds, config), batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
    nat_loader = DataLoader(OfficialValidationDataset(VAL_ROOT, config, subset="natural"), batch_size=8,
                            num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(OfficialValidationDataset(VAL_ROOT, config, subset="synthetic"), batch_size=8,
                            num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR

    model = build_model(config)
    copied = load_dual_warm_start(model, torch.load(WARM_START, map_location="cpu", weights_only=True))
    model = model.to(device)
    teacher = build_model(config)
    teacher.load_state_dict(model.state_dict())
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    groups = model.separate_params()
    ids = {id(p) for g in groups for p in g}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    optimizer = torch.optim.AdamW([{"params": g, "lr": config.ATST_BASE_LR * config.LLRD_DECAY ** i}
                                   for i, g in enumerate(groups)] + [{"params": rest, "lr": config.DISTILLED_LR}],
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * len(train_loader))
    criterion = SEDLoss(config).to(device)
    print(f"[{TAG}] {KIND} dual-head, warm start {WARM_START} (syn head copied from nat: {copied}); "
          f"gold {n_g} + synth {n_s} at synth_frac {SYNTH_FRAC}; {len(train_loader)} batches/epoch; "
          f"LR backbone {config.ATST_BASE_LR} rest {config.DISTILLED_LR}; OWBCE {config.OWBCE_HEIGHT}/{config.OWBCE_HALF_WIDTH}",
          flush=True)

    r0 = evaluate(model, nat_loader, syn_loader, device, hop)
    print(f"E0 (warm start): {fmt(r0)}", flush=True)
    best, best_t = -1.0, -1.0
    unl_iter = iter(unlabeled_loader)
    for epoch in range(MAX_EPOCHS):
        ramp = min(1.0, epoch / max(1, config.MEAN_TEACHER_RAMPUP_EPOCHS))
        cw = config.MEAN_TEACHER_CONSISTENCY_WEIGHT * float(np.exp(-5.0 * (1.0 - ramp) ** 2))
        model.train()
        sup, cons = [], []
        for audio, masks, dom in tqdm(train_loader, desc=f"Dual-{TAG} {epoch + 1}/{MAX_EPOCHS}"):
            audio, masks, dom = audio.to(device), masks.to(device), dom.to(device)
            parts_a, parts_m, parts_d = [], [], []
            for d in (NAT, SYN):
                sel = dom == d
                if sel.any():
                    a, m = apply_mixup(audio[sel], masks[sel])  # within-domain mixup only
                    parts_a.append(a); parts_m.append(m); parts_d.append(torch.full((a.shape[0],), d, device=device))
            audio, masks, dom = torch.cat(parts_a), torch.cat(parts_m), torch.cat(parts_d)
            (sb1, _, _), unl_iter = _cycle_batch(unlabeled_loader, unl_iter)
            (sb2, _, _), unl_iter = _cycle_batch(unlabeled_loader, unl_iter)
            sb1, sb2 = sb1.to(device), sb2.to(device)

            optimizer.zero_grad()
            out = model(audio)  # (B, T, 2)
            fr = min(out.shape[1], masks.shape[1])
            loss_sup = 0.0
            for d in (NAT, SYN):
                sel = dom == d
                if sel.any():
                    loss_sup = loss_sup + criterion(out[sel, :fr, d], masks[sel, :fr]) * sel.float().mean()
            s1 = torch.sigmoid(model(sb1)[..., NAT])
            with torch.no_grad():
                t1 = torch.sigmoid(teacher(sb1)[..., NAT])
                t2 = torch.sigmoid(teacher(sb2)[..., NAT])
            lam = float(Beta(config.ICT_ALPHA, config.ICT_ALPHA).sample())
            sm = torch.sigmoid(model(lam * sb1 + (1 - lam) * sb2)[..., NAT])
            loss_cons = F.mse_loss(s1, t1) + F.mse_loss(sm, lam * t1 + (1 - lam) * t2)
            loss = loss_sup + cw * loss_cons
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), 5.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), model.parameters()):
                    tp.data.mul_(config.MEAN_TEACHER_EMA_DECAY).add_(sp.data, alpha=1 - config.MEAN_TEACHER_EMA_DECAY)
            sup.append(float(loss_sup)); cons.append(float(loss_cons))

        r = evaluate(model, nat_loader, syn_loader, device, hop)
        rt = evaluate(teacher, nat_loader, syn_loader, device, hop)
        print(f"E{epoch + 1}: sup={np.mean(sup):.4f} cons={np.mean(cons):.4f} cw={cw:.3f}\n  student {fmt(r)}\n  teacher {fmt(rt)}",
              flush=True)
        if r["oracle_eq"] > best:
            best = r["oracle_eq"]
            torch.save(model.state_dict(), OUT)
            print(f"  -> student best, saved {OUT}", flush=True)
        if rt["oracle_eq"] > best_t:
            best_t = rt["oracle_eq"]
            torch.save(teacher.state_dict(), OUT_TEACHER)
            print(f"  -> teacher best, saved {OUT_TEACHER}", flush=True)
    print(f"Done [{TAG}]: warm-start oracle eq {r0['oracle_eq']:.4f} (natural@nat F1 {r0['natural@nat'][0]:.4f}); "
          f"best student {best:.4f}, best teacher {best_t:.4f}", flush=True)


if __name__ == "__main__":
    main()
