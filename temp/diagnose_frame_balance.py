"""Fast, reusable sanity check: raw frame-level precision/recall and predicted-vs-true
positive rate on gold-val, pre-Viterbi. No retraining, one forward pass over gold-val.

Built to diagnose Phase G's regression (pos_weight=3.0 pushed predicted-positive rate
to 33% when truth is ~17.5%, crushing precision 0.655->0.429 for a smaller recall gain
0.688->0.816) -- use this as an early gate (e.g. after ~5 epochs) for any future
pos_weight/class-imbalance experiment instead of waiting a full run to notice.

Usage: python -m temp.diagnose_frame_balance <checkpoint_path> [<checkpoint_path> ...]
"""
import sys

import torch

from src.architecture import DualBranchSED
from src.config import Config
from train import make_gold_loaders


def frame_stats(ckpt_path, gold_val_loader, config, thresh=0.5, max_batches=20):
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True))
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for i, (audio, masks, _) in enumerate(gold_val_loader):
            if i >= max_batches:
                break
            audio = audio.to(config.DEVICE)
            logits = model(audio)
            frames = min(logits.shape[1], masks.shape[1])
            probs = torch.sigmoid(logits[:, :frames]).cpu().numpy()
            gt = masks[:, :frames].numpy().astype(int)
            pred = (probs >= thresh).astype(int)
            tp += int(((pred == 1) & (gt == 1)).sum())
            fp += int(((pred == 1) & (gt == 0)).sum())
            fn += int(((pred == 0) & (gt == 1)).sum())
            tn += int(((pred == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    pred_pos = (tp + fp) / (tp + fp + fn + tn)
    true_pos = (tp + fn) / (tp + fp + fn + tn)
    print(f"{ckpt_path}: precision={prec:.4f} recall={rec:.4f} "
          f"pred_pos_frac={pred_pos:.4f} true_pos_frac={true_pos:.4f}", flush=True)
    return dict(precision=prec, recall=rec, pred_pos=pred_pos, true_pos=true_pos)


if __name__ == "__main__":
    config = Config()
    _, _, gold_val_loader = make_gold_loaders(config)
    for ckpt in sys.argv[1:]:
        frame_stats(ckpt, gold_val_loader, config)
