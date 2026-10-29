"""Plan Step 1: ensemble Phase H (capacity+self-distillation) with WavLM full1-E4
(frozen-pretrained-fusion) -- structurally different models, more likely to be
complementary than the Phase F+G2 ensemble (which barely helped, being a
loss-hyperparameter tweak on an identical architecture).

Usage: python -m temp.eval_ensemble_phaseh_wavlm
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import micro_event_f1, macro_segment_dice

VAL_ROOT = "./data/official_validation_raw/validation"


def build_phaseh():
    config = Config()
    config.CNN_CH = (48, 64, 128, 256)
    config.CONF_DIM = 256
    config.CONF_HEADS = 8
    config.CONF_LAYERS = 6
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load("./checkpoints/dualbranch_phaseh.pth", map_location=config.DEVICE, weights_only=True))
    model.eval()
    return model, config


def build_wavlm_full1():
    config = Config()
    config.USE_WAVLM = True
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load("./checkpoints/dualbranch_wavlm_stage1_full.pth", map_location=config.DEVICE, weights_only=True))
    model.eval()
    return model, config


def cache_ensemble(models, loader, device):
    probs, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in tqdm(loader, desc="caching"):
            audio = audio.to(device, non_blocking=True)
            ensembled = torch.sigmoid(sum(m(audio) for m in models) / len(models))
            frames = min(ensembled.shape[1], masks.shape[1])
            probs.append(ensembled[:, :frames].cpu().numpy())
            targets.append(masks[:, :frames].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score(probabilities, ground_truth, p_stay_silence, p_stay_event, hop):
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, p, r = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, p, r


def main():
    phaseh_model, config_h = build_phaseh()
    wavlm_model, config_w = build_wavlm_full1()
    device = config_h.DEVICE
    hop = config_h.HOP_LENGTH / config_h.SR

    for name, ps, pe in [("phaseh params (0.9955/0.8222)", 0.9955, 0.8222),
                         ("wavlm full1 params (default 0.9986/0.6944)", 0.9986, 0.6944)]:
        results = {}
        for subset in ("natural", "synthetic"):
            ds = OfficialValidationDataset(VAL_ROOT, config_h, subset=subset)
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=config_h.NUM_WORKERS)
            probs, targets = cache_ensemble([phaseh_model, wavlm_model], loader, device)
            dice, f1, p, r = score(probs, targets, ps, pe, hop)
            results[subset] = (f1, dice)
            print(f"[{name}] [{subset}] F1={f1:.4f} (P={p:.3f} R={r:.3f}) Dice={dice:.4f} "
                  f"combined={f1+dice:.4f}", flush=True)
        eq_avg = 0.5 * (sum(results["natural"]) + sum(results["synthetic"]))
        print(f"[{name}] eq-weighted overall combined = {eq_avg:.4f}\n", flush=True)

    print("(reference: Phase H alone=1.4729-1.4734 (real 1.4815), WavLM full1 alone=1.4716)", flush=True)


if __name__ == "__main__":
    main()
