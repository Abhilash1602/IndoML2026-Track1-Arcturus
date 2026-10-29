"""Sweep Viterbi decode params on the Gold val split (COMPLETE labels, unlike
synth_val). Tests the 'recall-tuned decoding is a free win' hypothesis properly.
Also tries hysteresis + median-filter alternatives.
"""
import numpy as np, torch
from datasets import Audio
from torch.utils.data import DataLoader
from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import VaaniDataset, load_full_dataset
from src.utils import (compute_segment_dice, compute_event_f1, extract_events_from_mask,
                       extract_events_viterbi, viterbi_decode, extract_events_hysteresis)


def score_viterbi(probs_all, masks_all, hop, p_sil, p_evt):
    d, f = [], []
    for probs, gt in zip(probs_all, masks_all):
        st = viterbi_decode(probs, p_stay_silence=p_sil, p_stay_event=p_evt)
        d.append(compute_segment_dice(st.astype(np.int32), gt.astype(np.int32)))
        f.append(compute_event_f1(extract_events_viterbi(st, hop),
                                  extract_events_from_mask(gt.astype(np.int32), hop)))
    return float(np.mean(d)), float(np.mean(f))


def score_hyst(probs_all, masks_all, hop, hi, lo, gap):
    d, f = [], []
    for probs, gt in zip(probs_all, masks_all):
        ev = extract_events_hysteresis(probs, hop, high_thresh=hi, low_thresh=lo,
                                       min_duration_sec=0.05, max_gap_sec=gap)
        pm = np.zeros(len(probs), dtype=np.int32)
        for e in ev:
            pm[int(e["onset"]/hop):int(e["offset"]/hop)] = 1
        d.append(compute_segment_dice(pm, gt.astype(np.int32)))
        f.append(compute_event_f1(ev, extract_events_from_mask(gt.astype(np.int32), hop)))
    return float(np.mean(d)), float(np.mean(f))


def main():
    config = Config()
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold = full.filter(lambda r: r["annotationQuality"] == "verified_timestamps")
    split = gold.train_test_split(test_size=0.2, seed=42)
    vd = split["test"].cast_column("audio", Audio(decode=True))
    vr = split["test"].cast_column("audio", Audio(decode=False))
    loader = DataLoader(VaaniDataset(vd, vr, config), batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=config.NUM_WORKERS)
    models = []
    for p in ("./checkpoints/dualbranch_baseline.pth", "./checkpoints/dualbranch_distilled.pth"):
        m = DualBranchSED(config).to(config.DEVICE)
        m.load_state_dict(torch.load(p, map_location=config.DEVICE, weights_only=True)); m.eval()
        models.append(m)
    hop = config.HOP_LENGTH / config.SR
    probs_all, masks_all = [], []
    with torch.no_grad():
        for audio, mask, _ in loader:
            audio = audio.to(config.DEVICE)
            pr = torch.sigmoid(sum(m(audio) for m in models)/len(models)).cpu().numpy()
            for pi, mi in zip(pr, mask.numpy()):
                k = min(len(pi), len(mi)); probs_all.append(pi[:k]); masks_all.append(mi[:k])

    print("=== Viterbi sweep on Gold val (complete labels) ===")
    for p_sil in (0.9986, 0.995, 0.99, 0.98, 0.96, 0.94):
        for p_evt in (0.5, 0.6944, 0.85):
            d, f = score_viterbi(probs_all, masks_all, hop, p_sil, p_evt)
            print(f"  p_sil={p_sil:.4f} p_evt={p_evt:.4f}  Dice {d:.4f}  F1 {f:.4f}  Comb {d+f:.4f}")
    print("\n=== Hysteresis sweep ===")
    for hi in (0.5, 0.4, 0.3):
        for lo in (0.3, 0.2, 0.15):
            if lo >= hi: continue
            d, f = score_hyst(probs_all, masks_all, hop, hi, lo, 0.15)
            print(f"  hi={hi} lo={lo} gap=0.15  Dice {d:.4f}  F1 {f:.4f}  Comb {d+f:.4f}")


if __name__ == "__main__":
    main()
