"""Phase A diagnostic: where does the current best ensemble (non-BEATs, 1.0982)
lose on the Gold validation split? Break recall and boundary error down by event
tag/category and duration bucket. ~45% of events are [breathing]/[inhaling] --
if we're missing those, that's the gap.

Usage: python -m temp.diagnose_where_we_lose
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

BASELINE = "./checkpoints/dualbranch_baseline.pth"
DISTILLED = "./checkpoints/dualbranch_distilled.pth"
P_SIL, P_EVT = 0.9986, 0.6944


def duration_bucket(d):
    if d < 0.2: return "0.0-0.2"
    if d < 0.4: return "0.2-0.4"
    if d < 0.8: return "0.4-0.8"
    if d < 1.6: return "0.8-1.6"
    return "1.6+"


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
    for path in (BASELINE, DISTILLED):
        m = DualBranchSED(config).to(config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True))
        m.eval()
        models.append(m)

    hop = config.HOP_LENGTH / config.SR
    all_probs = []
    with torch.no_grad():
        for audio, _, _ in loader:
            audio = audio.to(config.DEVICE)
            p = torch.sigmoid(sum(m(audio) for m in models) / len(models)).cpu().numpy()
            all_probs.append(p)
    all_probs = np.concatenate(all_probs)

    # raw GT events (with tags) for each val clip, in dataset order
    raw_events = val_raw["NoiseSubCategoryTimeStamp"]

    by_tag = defaultdict(lambda: {"n": 0, "hit": 0, "on_err": [], "off_err": []})
    by_dur = defaultdict(lambda: {"n": 0, "hit": 0})
    total_gt = total_hit = total_pred = total_pred_matched = 0

    for i, probs in enumerate(all_probs):
        states = viterbi_decode(probs, p_stay_silence=P_SIL, p_stay_event=P_EVT)
        preds = extract_events_viterbi(states, hop)
        gts = []
        for ev in (raw_events[i] or []):
            if "start" in ev and "end" in ev:
                s, e = float(ev["start"]), float(ev["end"])
                if e > s:
                    tag = ev.get("tag") or ev.get("category") or "unknown"
                    gts.append((s, e, tag))
        total_gt += len(gts)
        total_pred += len(preds)
        matched_pred = set()
        for (s, e, tag) in gts:
            tol = max(0.2 * (e - s), 0.05)
            key = tag if tag in ("[breathing]", "[inhaling]", "<horn>") else \
                  ("breathing_family" if tag.startswith("[") else "other_named")
            by_tag[key]["n"] += 1
            db = duration_bucket(e - s)
            by_dur[db]["n"] += 1
            cands = [(abs(p["onset"] - s) + abs(p["offset"] - e), j)
                     for j, p in enumerate(preds) if j not in matched_pred
                     and abs(p["onset"] - s) <= tol and abs(p["offset"] - e) <= tol]
            if cands:
                _, j = min(cands)
                matched_pred.add(j)
                by_tag[key]["hit"] += 1
                by_dur[db]["hit"] += 1
                by_tag[key]["on_err"].append(abs(preds[j]["onset"] - s))
                by_tag[key]["off_err"].append(abs(preds[j]["offset"] - e))
                total_hit += 1
        total_pred_matched += len(matched_pred)

    recall = total_hit / max(total_gt, 1)
    precision = total_pred_matched / max(total_pred, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"\n=== Overall (Gold val, {len(all_probs)} clips) ===")
    print(f"GT events: {total_gt}  pred events: {total_pred}")
    print(f"Recall {recall:.3f}  Precision {precision:.3f}  Event F1 {f1:.3f}")

    print(f"\n=== Recall by tag family ===")
    for k, v in sorted(by_tag.items(), key=lambda kv: -kv[1]["n"]):
        r = v["hit"] / max(v["n"], 1)
        on = np.mean(v["on_err"]) if v["on_err"] else float("nan")
        off = np.mean(v["off_err"]) if v["off_err"] else float("nan")
        print(f"  {k:20s} n={v['n']:5d}  recall={r:.3f}  mean|onset err|={on:.3f}s  mean|offset err|={off:.3f}s")

    print(f"\n=== Recall by GT event duration ===")
    for k in ("0.0-0.2", "0.2-0.4", "0.4-0.8", "0.8-1.6", "1.6+"):
        v = by_dur[k]
        r = v["hit"] / max(v["n"], 1)
        print(f"  {k:10s} n={v['n']:5d}  recall={r:.3f}")


if __name__ == "__main__":
    main()
