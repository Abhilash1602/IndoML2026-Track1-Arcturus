"""Re-tune Viterbi decode params for Phase F against the REAL organizer validation
set (OfficialValidationDataset), not our own gold_val/synth_val proxies. This is
Tier 1 item 1 of the post-Phase-F plan -- make sure the current best checkpoint is
decoded optimally against the most trustworthy signal we have before building on it.

Usage: python -m temp.tune_phasef_decode_realval
"""
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import micro_event_f1, macro_segment_dice

CKPT = "./checkpoints/dualbranch_phasef.pth"
VAL_ROOT = "./data/official_validation_raw/validation"


def cache_predictions(model, loader, config):
    probs, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in tqdm(loader, desc="caching"):
            audio = audio.to(config.DEVICE, non_blocking=True)
            logits = model(audio)
            frames = min(logits.shape[1], masks.shape[1])
            probs.append(torch.sigmoid(logits[:, :frames]).cpu().numpy())
            targets.append(masks[:, :frames].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score(probabilities, ground_truth, p_stay_silence, p_stay_event, config):
    hop = config.HOP_LENGTH / config.SR
    pred_events, gt_events = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, p, r = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice, f1, dice + f1


def main():
    config = Config()
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=config.DEVICE, weights_only=True))
    model.eval()

    nat_ds = OfficialValidationDataset(VAL_ROOT, config, subset="natural")
    syn_ds = OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")
    nat_loader = DataLoader(nat_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    syn_loader = DataLoader(syn_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    print("Caching natural predictions...", flush=True)
    np_, nt = cache_predictions(model, nat_loader, config)
    print("Caching synthetic predictions...", flush=True)
    sp, st = cache_predictions(model, syn_loader, config)

    for name, ps, pe in [("current (0.9989/0.8355)", 0.9989, 0.8355),
                         ("Phase E params (0.9963/0.5257)", 0.9963, 0.5257),
                         ("original default (0.9986/0.6944)", 0.9986, 0.6944)]:
        nd, nf, nc = score(np_, nt, ps, pe, config)
        sd, sf, sc = score(sp, st, ps, pe, config)
        eq = 0.5 * nc + 0.5 * sc
        print(f"[{name}] natural combined={nc:.4f} synthetic combined={sc:.4f} eq-avg={eq:.4f}", flush=True)

    def objective(trial):
        p_stay_silence = trial.suggest_float("p_stay_silence", 0.90, 0.999)
        p_stay_event = trial.suggest_float("p_stay_event", 0.4, 0.95)
        _, _, nc = score(np_, nt, p_stay_silence, p_stay_event, config)
        _, _, sc = score(sp, st, p_stay_silence, p_stay_event, config)
        eq = 0.5 * nc + 0.5 * sc
        trial.set_user_attr("natural_combined", nc)
        trial.set_user_attr("synthetic_combined", sc)
        return eq

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=80, show_progress_bar=True)
    best = study.best_trial
    print("\n=== BEST (on REAL validation set) ===", flush=True)
    print(f"p_stay_silence={best.params['p_stay_silence']:.4f} "
          f"p_stay_event={best.params['p_stay_event']:.4f} "
          f"natural_combined={best.user_attrs['natural_combined']:.4f} "
          f"synthetic_combined={best.user_attrs['synthetic_combined']:.4f} "
          f"eq_avg={best.value:.4f}", flush=True)


if __name__ == "__main__":
    main()
