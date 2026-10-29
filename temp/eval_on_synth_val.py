"""Evaluate the current best ensemble on the new synthetic validation set, and
compare recall-by-category against the Gold-val diagnostic. If loud-event recall
is much higher here (cleaner, louder, separated events) it (a) explains why our
real Synthetic F1 > Natural F1, and (b) confirms synth_val is a usable
Synthetic-domain-predictive signal.
"""
from collections import defaultdict
import json, numpy as np, torch
from torch.utils.data import DataLoader
from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import SyntheticSEDDataset
from src.utils import (compute_segment_dice, compute_event_f1, extract_events_from_mask,
                       extract_events_viterbi, viterbi_decode)

P_SIL, P_EVT = 0.9986, 0.6944

def run(p_sil, p_evt, probs_all, masks_all, hop):
    dice, f1, tot_gt, tot_hit, tot_pred, tot_pm = [], [], 0, 0, 0, 0
    for probs, gt_mask in zip(probs_all, masks_all):
        states = viterbi_decode(probs, p_stay_silence=p_sil, p_stay_event=p_evt)
        preds = extract_events_viterbi(states, hop)
        gts = extract_events_from_mask(gt_mask.astype(np.int32), hop)
        dice.append(compute_segment_dice(states.astype(np.int32), gt_mask.astype(np.int32)))
        f1.append(compute_event_f1(preds, gts))
    return float(np.mean(dice)), float(np.mean(f1))

def main():
    config = Config()
    ds = SyntheticSEDDataset("data/synth_val", config, add_noise=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    models = []
    for path in ("./checkpoints/dualbranch_baseline.pth", "./checkpoints/dualbranch_distilled.pth"):
        m = DualBranchSED(config).to(config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True)); m.eval()
        models.append(m)
    hop = config.HOP_LENGTH / config.SR
    probs_all, masks_all = [], []
    with torch.no_grad():
        for audio, mask, _ in loader:
            audio = audio.to(config.DEVICE)
            p = torch.sigmoid(sum(m(audio) for m in models) / len(models)).cpu().numpy()
            for pi, mi in zip(p, mask.numpy()):
                f = min(len(pi), len(mi)); probs_all.append(pi[:f]); masks_all.append(mi[:f])

    print("=== current ensemble on synth_val (1200 clips) ===")
    for (ps, pe, lbl) in [(P_SIL, P_EVT, "F1-tuned (0.9986/0.6944)"),
                          (0.97, 0.90, "recall-tuned (0.97/0.90)"),
                          (0.95, 0.85, "recall-tuned (0.95/0.85)"),
                          (0.92, 0.80, "aggressive (0.92/0.80)")]:
        d, f = run(ps, pe, probs_all, masks_all, hop)
        print(f"  {lbl:32s}  Dice {d:.4f}  EventF1 {f:.4f}  Combined {d+f:.4f}")

if __name__ == "__main__":
    main()
