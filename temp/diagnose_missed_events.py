"""Follow-up: WHY do we miss ~67% of events? For each GT event, record the
ensemble's mean and max probability inside its window, split by hit vs miss and
by tag family. If missed loud/long events have low model probability, it's a
model-capacity/imbalance problem (fixable by synthesis + reweighting). If they
have high probability but we still miss, it's a decoding problem.
"""
from collections import defaultdict

import numpy as np
import torch
from datasets import Audio
from torch.utils.data import DataLoader

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import VaaniDataset, load_full_dataset
from src.utils import extract_events_viterbi, viterbi_decode

P_SIL, P_EVT = 0.9986, 0.6944


def fam(tag):
    if tag in ("[breathing]", "[inhaling]"): return "breathing"
    if tag == "<horn>": return "horn"
    if tag.startswith("["): return "breathing"
    return "other_loud"


def main():
    config = Config()
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold = full.filter(lambda r: r["annotationQuality"] == "verified_timestamps")
    split = gold.train_test_split(test_size=0.2, seed=42)
    val_decode = split["test"].cast_column("audio", Audio(decode=True))
    val_raw = split["test"].cast_column("audio", Audio(decode=False))
    loader = DataLoader(VaaniDataset(val_decode, val_raw, config), batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=config.NUM_WORKERS)
    models = []
    for path in ("./checkpoints/dualbranch_baseline.pth", "./checkpoints/dualbranch_distilled.pth"):
        m = DualBranchSED(config).to(config.DEVICE); m.load_state_dict(
            torch.load(path, map_location=config.DEVICE, weights_only=True)); m.eval()
        models.append(m)
    hop = config.HOP_LENGTH / config.SR
    probs_all = []
    with torch.no_grad():
        for audio, _, _ in loader:
            audio = audio.to(config.DEVICE)
            probs_all.append(torch.sigmoid(sum(m(audio) for m in models) / len(models)).cpu().numpy())
    probs_all = np.concatenate(probs_all)
    raw_events = val_raw["NoiseSubCategoryTimeStamp"]

    stats = defaultdict(lambda: {"hit_meanp": [], "hit_maxp": [], "miss_meanp": [], "miss_maxp": []})
    frac_frames_over_half = defaultdict(list)

    for i, probs in enumerate(probs_all):
        states = viterbi_decode(probs, p_stay_silence=P_SIL, p_stay_event=P_EVT)
        preds = extract_events_viterbi(states, hop)
        gts = []
        for ev in (raw_events[i] or []):
            if "start" in ev and "end" in ev and float(ev["end"]) > float(ev["start"]):
                gts.append((float(ev["start"]), float(ev["end"]), ev.get("tag") or ev.get("category") or "?"))
        matched = set()
        for (s, e, tag) in gts:
            tol = max(0.2 * (e - s), 0.05)
            f0, f1 = int(s / hop), min(int(e / hop) + 1, len(probs))
            if f1 <= f0: continue
            win = probs[f0:f1]
            meanp, maxp = float(win.mean()), float(win.max())
            over = float((win > 0.5).mean())
            cands = [(abs(p["onset"] - s) + abs(p["offset"] - e), j) for j, p in enumerate(preds)
                     if j not in matched and abs(p["onset"] - s) <= tol and abs(p["offset"] - e) <= tol]
            k = fam(tag)
            frac_frames_over_half[k].append(over)
            if cands:
                _, j = min(cands); matched.add(j)
                stats[k]["hit_meanp"].append(meanp); stats[k]["hit_maxp"].append(maxp)
            else:
                stats[k]["miss_meanp"].append(meanp); stats[k]["miss_maxp"].append(maxp)

    print("\n=== Model probability inside GT event windows (hit vs miss) ===")
    for k in ("breathing", "horn", "other_loud"):
        s = stats[k]
        print(f"\n{k}:")
        for lbl in ("hit", "miss"):
            mp = np.mean(s[f"{lbl}_meanp"]) if s[f"{lbl}_meanp"] else float("nan")
            xp = np.mean(s[f"{lbl}_maxp"]) if s[f"{lbl}_maxp"] else float("nan")
            n = len(s[f"{lbl}_meanp"])
            print(f"  {lbl:5s} n={n:5d}  mean(mean prob)={mp:.3f}  mean(max prob)={xp:.3f}")
        fo = np.mean(frac_frames_over_half[k])
        print(f"  → avg fraction of event frames with prob>0.5: {fo:.3f}")


if __name__ == "__main__":
    main()
