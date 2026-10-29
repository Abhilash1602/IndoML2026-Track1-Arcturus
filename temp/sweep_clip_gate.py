"""Test two things discovered from the org's official baseline repo
(nagshubhadip/noise-event-detection-and-removal), against our EXISTING best
checkpoints, with NO retraining required:

  1. How much does our local F1 change once scored with the official micro-pooled
     F1 + global closest-first matching (temp/official_metric.py), instead of our
     own macro-averaged-per-clip compute_event_f1 (src/utils.py)? This checks
     whether metric-fidelity is a real, quantifiable contributor to the persistent
     local/real gap, before spending any retraining effort on it.
  2. Does adding clip-level gating (emit zero events for a clip if its max frame
     probability is below a swept threshold -- our proxy for the official
     baseline's attention-pooled "clip probability" head, which we don't have)
     improve the corrected combined score? Their scorer rewards a correctly-silent
     clip with a free Dice=1.0, so gating low-confidence clips to silence can pay
     off even without any model change.

Usage: python -m temp.sweep_clip_gate --ckpt phased   (or: phasee)
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import SyntheticSEDDataset
from src.utils import compute_event_f1, compute_segment_dice, extract_events_from_mask, extract_events_viterbi, viterbi_decode
from train import SYNTH_VAL_DIR, make_gold_loaders
from temp.official_metric import micro_event_f1, macro_segment_dice

CKPTS = {
    "phased": dict(path="./checkpoints/dualbranch_big.pth",
                   overrides=dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6),
                   decode=(0.9986, 0.6944)),
    "phasee": dict(path="./checkpoints/dualbranch_phasee.pth",
                   overrides=dict(),
                   decode=(0.9963, 0.5257)),
}


def cache(model, loader, config):
    probs, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in tqdm(loader, desc="caching"):
            audio = audio.to(config.DEVICE, non_blocking=True)
            logits = model(audio)
            frames = min(logits.shape[1], masks.shape[1])
            probs.append(torch.sigmoid(logits[:, :frames]).cpu().numpy())
            targets.append(masks[:, :frames].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def decode_all(probabilities, p_stay_silence, p_stay_event, hop):
    states_list, events_list, clip_scores = [], [], []
    for p in probabilities:
        states = viterbi_decode(p, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        events_list.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        states_list.append(states)
        clip_scores.append(float(p.max()))
    return states_list, events_list, np.array(clip_scores)


def score(states_list, events_list, gt_events, gt_masks, clip_scores, gate):
    gated_events = [ev if cs >= gate else [] for ev, cs in zip(events_list, clip_scores)]
    gated_masks = [s if cs >= gate else np.zeros_like(s) for s, cs in zip(states_list, clip_scores)]
    f1, prec, rec = micro_event_f1(gt_events, gated_events)
    dice = macro_segment_dice(gt_events, gated_events, hop_sec=0.01)
    dice_direct = float(np.mean([compute_segment_dice(pm.astype(np.int32), gm.astype(np.int32))
                                 for pm, gm in zip(gated_masks, gt_masks)]))
    old_f1 = float(np.mean([
        compute_event_f1([{"onset": a, "offset": b} for a, b in ev],
                         [{"onset": a, "offset": b} for a, b in gt])
        for ev, gt in zip(gated_events, gt_events)]))
    return dict(f1=f1, prec=prec, rec=rec, dice=dice, dice_direct=dice_direct, old_f1=old_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", choices=list(CKPTS.keys()), required=True)
    args = ap.parse_args()
    spec = CKPTS[args.ckpt]

    config = Config()
    for k, v in spec["overrides"].items():
        setattr(config, k, v)
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(spec["path"], map_location=config.DEVICE, weights_only=True))
    model.eval()

    _, _, gold_val_loader = make_gold_loaders(config)
    synth_val = SyntheticSEDDataset(SYNTH_VAL_DIR, config, add_noise=False)
    synth_val_loader = DataLoader(synth_val, batch_size=config.BATCH_SIZE, shuffle=False,
                                  num_workers=config.NUM_WORKERS)
    hop = config.HOP_LENGTH / config.SR
    ps, pe = spec["decode"]

    for name, loader in [("gold", gold_val_loader), ("synth", synth_val_loader)]:
        print(f"\n=== {args.ckpt} / {name}-val ===", flush=True)
        probs, targets = cache(model, loader, config)
        states_list, events_list, clip_scores = decode_all(probs, ps, pe, hop)
        gt_events = [[(e["onset"], e["offset"]) for e in extract_events_from_mask(t.astype(np.int32), hop)]
                    for t in targets]

        base = score(states_list, events_list, gt_events, targets, clip_scores, gate=0.0)
        print(f"[gate=0.0 i.e. no gating] official micro-F1={base['f1']:.4f} "
              f"(P={base['prec']:.3f} R={base['rec']:.3f})  official-style dice={base['dice']:.4f}  "
              f"our OLD macro-F1={base['old_f1']:.4f}  our OLD dice(direct)={base['dice_direct']:.4f}  "
              f"combined(official F1+dice)={base['f1']+base['dice']:.4f}  "
              f"combined(our old)={base['old_f1']+base['dice_direct']:.4f}", flush=True)

        best = None
        for gate in np.arange(0.0, 0.95, 0.05):
            r = score(states_list, events_list, gt_events, targets, clip_scores, gate=float(gate))
            combined = r["f1"] + r["dice"]
            if best is None or combined > best[0]:
                best = (combined, gate, r)
        combined, gate, r = best
        print(f"[BEST gate={gate:.2f}] official micro-F1={r['f1']:.4f} (P={r['prec']:.3f} R={r['rec']:.3f})  "
              f"official-style dice={r['dice']:.4f}  combined={combined:.4f}  "
              f"(vs no-gate combined={base['f1']+base['dice']:.4f}, delta={combined-(base['f1']+base['dice']):+.4f})",
              flush=True)


if __name__ == "__main__":
    main()
