"""Re-tune Viterbi decode params (p_stay_silence/p_stay_event) for the Phase E
checkpoint (E48, local sel=1.1140) before generating a submission -- decode
calibration has been checkpoint-specific in this project (BEATs needed very
different params than the plain architecture), so we never assume the training-time
proxy params (fixed 0.95/0.7, used only for cheap per-epoch checkpoint selection)
or an older tuned value (0.9986/0.6944, tuned against an earlier checkpoint)
transfer un-checked.

Tunes against gold-val (Natural proxy) + synth-val (Synthetic proxy) combined with
an EQUAL weight (0.5/0.5) matching the real scoring formula discovered this session
(Total Combined = equal-weighted mean of Natural-subset and Synthetic-subset
combined) -- NOT the 0.6/0.4 Natural-leaning weight used for training-time
checkpoint selection (that was a deliberate training-time bias, not the eval
formula).

Usage: python -m temp.tune_phasee_decode
"""
import numpy as np
import optuna
import torch
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.utils import compute_event_f1, compute_segment_dice, extract_events_from_mask, extract_events_viterbi, viterbi_decode
from train import SYNTH_VAL_DIR, make_gold_loaders

from src.dataset import SyntheticSEDDataset
from torch.utils.data import DataLoader

CKPT = "./checkpoints/dualbranch_phasee.pth"


def cache_predictions(model, loader, config):
    probs, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in tqdm(loader, desc="caching"):
            audio = audio.to(config.DEVICE, non_blocking=True)
            logits = model(audio)
            frames = min(logits.shape[1], masks.shape[1])
            p = torch.sigmoid(logits[:, :frames]).cpu().numpy()
            m = masks[:, :frames].numpy()
            probs.append(p)
            targets.append(m)
    return np.concatenate(probs), np.concatenate(targets)


def score(probabilities, ground_truth, p_stay_silence, p_stay_event, config):
    hop = config.HOP_LENGTH / config.SR
    dice_scores, f1_scores = [], []
    for probability, target in zip(probabilities, ground_truth):
        states = viterbi_decode(probability, p_stay_silence=p_stay_silence, p_stay_event=p_stay_event)
        predicted_events = extract_events_viterbi(states, hop)
        target_events = extract_events_from_mask(target.astype(np.int32), hop)
        dice_scores.append(compute_segment_dice(states.astype(np.int32), target.astype(np.int32)))
        f1_scores.append(compute_event_f1(predicted_events, target_events))
    dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return dice, f1, dice + f1


def main():
    config = Config()
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=config.DEVICE, weights_only=True))
    model.eval()

    _, _, gold_val_loader = make_gold_loaders(config)
    synth_val = SyntheticSEDDataset(SYNTH_VAL_DIR, config, add_noise=False)
    synth_val_loader = DataLoader(synth_val, batch_size=config.BATCH_SIZE, shuffle=False,
                                  num_workers=config.NUM_WORKERS)

    print("Caching gold-val predictions...", flush=True)
    gp, gt = cache_predictions(model, gold_val_loader, config)
    print("Caching synth-val predictions...", flush=True)
    sp, st = cache_predictions(model, synth_val_loader, config)

    # Baseline: current default params, un-tuned
    for name, ps, pe in [("current default (0.9986/0.6944)", 0.9986, 0.6944),
                         ("training-time proxy (0.95/0.7)", 0.95, 0.7)]:
        gd, gf, gc = score(gp, gt, ps, pe, config)
        sd, sf, sc = score(sp, st, ps, pe, config)
        eq = 0.5 * gc + 0.5 * sc
        print(f"[{name}] gold combined={gc:.4f} synth combined={sc:.4f} eq-avg={eq:.4f}", flush=True)

    def objective(trial):
        p_stay_silence = trial.suggest_float("p_stay_silence", 0.90, 0.999)
        p_stay_event = trial.suggest_float("p_stay_event", 0.4, 0.95)
        _, _, gc = score(gp, gt, p_stay_silence, p_stay_event, config)
        _, _, sc = score(sp, st, p_stay_silence, p_stay_event, config)
        eq = 0.5 * gc + 0.5 * sc
        trial.set_user_attr("gold_combined", gc)
        trial.set_user_attr("synth_combined", sc)
        return eq

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=60, show_progress_bar=True)
    best = study.best_trial
    print("\n=== BEST ===", flush=True)
    print(f"p_stay_silence={best.params['p_stay_silence']:.4f} "
          f"p_stay_event={best.params['p_stay_event']:.4f} "
          f"gold_combined={best.user_attrs['gold_combined']:.4f} "
          f"synth_combined={best.user_attrs['synth_combined']:.4f} "
          f"eq_avg={best.value:.4f}", flush=True)


if __name__ == "__main__":
    main()
