"""Ensemble search v2: adds the iteration-2 checkpoints (dasheng_s2_i2 =
Dasheng Stage 2 trained on ensemble pseudo-labels, standalone 1.4827 -- best
single model of the project; phaseh_i2, 1.4400; beats_smoke, frozen BEATs,
1.4606) to the six original members, searches every 3..7-member subset under
prob- and logit-averaging at the fixed reference decode, and compares against
the CURRENT REAL BEST (4-model logit, Codabench 1.5427). Only if the winner
beats it by >= SUBMIT_GATE does it decode-tune and write submission_i2.zip
(never overwrites earlier submission zips). Reuses cached validation logits.

Usage: python -m temp.search_ensemble_v2
"""
import glob
import itertools
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.architecture import DualBranchSED, StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config
from src.dataset import OfficialValidationDataset, pad_or_trim
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

VAL_ROOT = "./data/official_validation_raw/validation"
TEST_DIR = "./data/test_audio"
SCRATCH = "./work"
LOGIT_CACHE = os.path.join(SCRATCH, "dasheng_ens_val_logits.npz")  # 6 original members (reused)
LOGIT_CACHE_V2 = os.path.join(SCRATCH, "ens_v2_val_logits.npz")  # members added after
SUMMARY_JSON = os.path.join(SCRATCH, "ens_v2_summary.json")
OUT_JSONL = "predictions_i2.jsonl"
OUT_ZIP = "submission_i2.zip"

FIXED_PS, FIXED_PE = 0.9005, 0.8901  # current best ensemble's tuned decode (swap-test reference)
N_TRIALS = 80
N_WORKERS = 48
SUBMIT_GATE = 0.003  # plan gate: must beat the current real-best ensemble by this much
MAX_SUBSET = 7

BIG = dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)
MEMBERS = {
    "phaseh": dict(cls=DualBranchSED, path="./checkpoints/dualbranch_phaseh.pth", overrides=BIG),
    "wavlm_full1": dict(cls=DualBranchSED, path="./checkpoints/dualbranch_wavlm_stage1_full.pth",
                        overrides=dict(USE_WAVLM=True)),
    "phased": dict(cls=DualBranchSED, path="./checkpoints/dualbranch_big.pth", overrides=BIG),
    "atst_stage1": dict(cls=DualBranchSED, path="./checkpoints/dualbranch_atst_frame_stage1_full.pth",
                        overrides=dict(USE_ATST_FRAME=True)),
    "atst_stage2": dict(cls=DualBranchSED, path="./checkpoints/dualbranch_atst_frame_stage2.pth",
                        overrides=dict(USE_ATST_FRAME=True)),
    "dasheng_s2": dict(cls=StandaloneDashengSED, path="./checkpoints/standalone_dasheng_stage2.pth",
                       overrides=dict()),
    "dasheng_s2_i2": dict(cls=StandaloneDashengSED, path="./checkpoints/standalone_dasheng_stage2_i2.pth",
                          overrides=dict()),
    "phaseh_i2": dict(cls=DualBranchSED, path="./checkpoints/dualbranch_phaseh_i2.pth", overrides=BIG),
    "beats_smoke": dict(cls=StandaloneBEATsSED, path="./checkpoints/standalone_beats_smoke.pth",
                        overrides=dict()),
}
ORIGINAL6 = ("phaseh", "wavlm_full1", "phased", "atst_stage1", "atst_stage2", "dasheng_s2")
# Reference = the current REAL best (Codabench 1.5427): 4-model, logit-averaged.
REFERENCE = (("dasheng_s2", "phaseh", "wavlm_full1", "atst_stage2"), "logit")

# Set before any fork so worker processes inherit them without pickling.
G = {}


def load_member(name):
    spec = MEMBERS[name]
    config = Config()
    for k, v in spec["overrides"].items():
        setattr(config, k, v)
    model = spec["cls"](config).to(config.DEVICE)
    model.load_state_dict(torch.load(spec["path"], map_location=config.DEVICE, weights_only=True))
    model.eval()
    return model


def combine(logit_list, mode):
    """logit_list: list of same-shape arrays/tensors of raw logits."""
    if isinstance(logit_list[0], torch.Tensor):
        if mode == "prob":
            return torch.stack([torch.sigmoid(x) for x in logit_list]).mean(0)
        return torch.sigmoid(torch.stack(logit_list).mean(0))
    if mode == "prob":
        return np.mean([1.0 / (1.0 + np.exp(-x)) for x in logit_list], axis=0)
    return 1.0 / (1.0 + np.exp(-np.mean(logit_list, axis=0)))


def cache_val_logits():
    out = dict(np.load(LOGIT_CACHE))
    assert all(f"natural__{m}" in out for m in ORIGINAL6), "original logit cache incomplete"
    if os.path.exists(LOGIT_CACHE_V2):
        out.update(dict(np.load(LOGIT_CACHE_V2)))
    config = Config()
    new = {}
    for name in MEMBERS:
        if f"natural__{name}" in out:
            continue
        model = load_member(name)
        for subset in ("natural", "synthetic"):
            ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
            logits = []
            with torch.no_grad():
                for audio, _, _ in tqdm(loader, desc=f"{name}/{subset}"):
                    logits.append(model(audio.to(config.DEVICE)).float().cpu().numpy())
            new[f"{subset}__{name}"] = np.concatenate(logits)
        del model
        torch.cuda.empty_cache()
    if new:
        prev = dict(np.load(LOGIT_CACHE_V2)) if os.path.exists(LOGIT_CACHE_V2) else {}
        prev.update(new)
        np.savez(LOGIT_CACHE_V2, **prev)
        out.update(new)
    return out


def _frames(subset):
    n = min(G["cache"][f"{subset}__{m}"].shape[1] for m in MEMBERS)
    return min(n, G["cache"][f"{subset}__targets"].shape[1])


def score_probs(probs, targets, ps, pe, hop):
    pred_events, gt_events = [], []
    for probability, target in zip(probs, targets):
        states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    f1, _, _ = micro_event_f1(gt_events, pred_events)
    dice = macro_segment_dice(gt_events, pred_events, hop_sec=hop)
    return dice + f1


def eval_option(args):
    names, mode, ps, pe = args
    hop = G["hop"]
    result = {}
    for subset in ("natural", "synthetic"):
        n = _frames(subset)
        probs = combine([G["cache"][f"{subset}__{m}"][:, :n] for m in names], mode)
        result[subset] = score_probs(probs, G["cache"][f"{subset}__targets"][:, :n], ps, pe, hop)
    return names, mode, result["natural"], result["synthetic"], 0.5 * (result["natural"] + result["synthetic"])


def _score_chunk(args):
    subset, lo, hi, ps, pe = args
    probs = G["chosen_probs"][subset][lo:hi]
    targets = G["chosen_targets"][subset][lo:hi]
    hop = G["hop"]
    pred_events, gt_events = [], []
    for probability, target in zip(probs, targets):
        states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
        pred_events.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, hop)])
        gt_events.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(target.astype(np.int32), hop)])
    return subset, pred_events, gt_events


def parallel_score_chosen(pool, ps, pe):
    tasks = []
    for subset in ("natural", "synthetic"):
        n_clips = len(G["chosen_probs"][subset])
        step = max(1, n_clips // N_WORKERS + 1)
        tasks += [(subset, lo, min(lo + step, n_clips), ps, pe) for lo in range(0, n_clips, step)]
    collected = {"natural": ([], []), "synthetic": ([], [])}
    for subset, pred, gt in pool.map(_score_chunk, tasks):  # map preserves order
        collected[subset][0].extend(pred)
        collected[subset][1].extend(gt)
    combined = {}
    for subset, (pred, gt) in collected.items():
        f1, _, _ = micro_event_f1(gt, pred)
        combined[subset] = macro_segment_dice(gt, pred, hop_sec=G["hop"]) + f1
    return combined["natural"], combined["synthetic"], 0.5 * (combined["natural"] + combined["synthetic"])


def load_wav(path, config):
    wav, sample_rate = torchaudio.load(path)
    wav = wav.mean(dim=0) if wav.size(0) > 1 else wav.squeeze(0)
    if sample_rate != config.SR:
        wav = torchaudio.functional.resample(wav, sample_rate, config.SR)
    return wav


class ShortClipDataset(Dataset):
    def __init__(self, file_paths, config):
        self.file_paths, self.config = file_paths, config

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            wav = pad_or_trim(load_wav(path, self.config), self.config.target_samples)
        except Exception:
            wav = torch.zeros(self.config.target_samples)
        return wav, os.path.splitext(os.path.basename(path))[0]


@torch.no_grad()
def predict_long_clip_mode(models, wav, config, mode, window_sec=10.0, overlap_sec=2.0):
    """src/chunked_inference.predict_long_clip, with the ensemble combination
    rule parameterized so it matches the tuned `mode` exactly."""
    window_samples = int(window_sec * config.SR)
    hop = config.HOP_LENGTH
    total_frames = wav.numel() // hop
    if wav.numel() <= window_samples:
        x = pad_or_trim(wav, window_samples).unsqueeze(0).to(config.DEVICE)
        return combine([m(x) for m in models], mode)[0].cpu().numpy()[:total_frames]
    stride_samples = window_samples - int(overlap_sec * config.SR)
    starts = list(range(0, wav.numel() - window_samples + 1, stride_samples))
    if not starts or starts[-1] + window_samples < wav.numel():
        starts.append(wav.numel() - window_samples)
    prob_sum = np.zeros(total_frames, dtype=np.float64)
    weight_sum = np.zeros(total_frames, dtype=np.float64)
    for start in starts:
        chunk = wav[start:start + window_samples].unsqueeze(0).to(config.DEVICE)
        probs = combine([m(chunk) for m in models], mode)[0].cpu().numpy()
        frame_start = start // hop
        frame_end = min(frame_start + len(probs), total_frames)
        prob_sum[frame_start:frame_end] += probs[:frame_end - frame_start]
        weight_sum[frame_start:frame_end] += 1.0
    assert np.all(weight_sum > 0), "uncovered frame in chunked inference"
    return prob_sum / weight_sum


def main():
    config = Config()
    G["hop"] = config.HOP_LENGTH / config.SR
    G["cache"] = cache_val_logits()
    summary = {}

    # ---- 2. subset x mode search at the fixed reference decode ----
    names_all = list(MEMBERS)
    options = [(REFERENCE[0], REFERENCE[1])]
    for k in range(3, MAX_SUBSET + 1):
        for combo in itertools.combinations(names_all, k):
            for mode in ("prob", "logit"):
                if (combo, mode) != (REFERENCE[0], REFERENCE[1]):
                    options.append((combo, mode))
    print(f"Scoring {len(options)} subset/mode options at fixed decode "
          f"({FIXED_PS}/{FIXED_PE}) with {N_WORKERS} workers...", flush=True)
    ctx = __import__("multiprocessing").get_context("fork")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        results = list(ex.map(eval_option, [(n, m, FIXED_PS, FIXED_PE) for n, m in options], chunksize=4))
    results.sort(key=lambda r: -r[4])
    ref = next(r for r in results if tuple(r[0]) == REFERENCE[0] and r[1] == REFERENCE[1])
    print(f"\nReference (current real best, 1.5427 on Codabench): [{ref[1]}] {'+'.join(ref[0])} "
          f"eq={ref[4]:.4f} nat={ref[2]:.4f} syn={ref[3]:.4f}", flush=True)
    print("Top 15 options:", flush=True)
    for names, mode, nc, sc, eq in results[:15]:
        print(f"  {eq:.4f} (d_ref={eq - ref[4]:+.4f}) nat={nc:.4f} syn={sc:.4f} [{mode}] {'+'.join(names)}", flush=True)
    for cand in ("dasheng_s2_i2", "phaseh_i2", "beats_smoke"):
        best_with = next(r for r in results if cand in r[0])
        print(f"  best containing {cand}: {best_with[4]:.4f} (d_ref={best_with[4] - ref[4]:+.4f}) "
              f"[{best_with[1]}] {'+'.join(best_with[0])}", flush=True)
    summary["search"] = [dict(names=list(n), mode=m, natural=nc, synthetic=sc, eq=eq)
                         for n, m, nc, sc, eq in results[:200]]
    summary["reference"] = dict(names=list(ref[0]), mode=ref[1], eq=ref[4])

    top = results[0]
    if top[4] - ref[4] < SUBMIT_GATE:
        print(f"\nNO SUBMISSION: best option beats the current real best by only {top[4] - ref[4]:+.4f} "
              f"(< gate {SUBMIT_GATE}).", flush=True)
        with open(SUMMARY_JSON, "w") as fh:
            json.dump(summary, fh, indent=1)
        return
    chosen = top
    names, mode = tuple(chosen[0]), chosen[1]
    print(f"\nCHOSEN: [{mode}] {'+'.join(names)} eq={chosen[4]:.4f} (d_ref={chosen[4] - ref[4]:+.4f} >= gate)", flush=True)
    summary["chosen"] = dict(names=list(names), mode=mode, eq_fixed_decode=chosen[4])

    # ---- 3. decode tuning on the chosen option ----
    G["chosen_probs"], G["chosen_targets"] = {}, {}
    for subset in ("natural", "synthetic"):
        n = _frames(subset)
        G["chosen_probs"][subset] = combine([G["cache"][f"{subset}__{m}"][:, :n] for m in names], mode)
        G["chosen_targets"][subset] = G["cache"][f"{subset}__targets"][:, :n]
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
        def objective(trial):
            ps = trial.suggest_float("p_stay_silence", 0.90, 0.999)
            pe = trial.suggest_float("p_stay_event", 0.4, 0.95)
            nc, sc, eq = parallel_score_chosen(pool, ps, pe)
            trial.set_user_attr("natural", nc)
            trial.set_user_attr("synthetic", sc)
            return eq

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial({"p_stay_silence": FIXED_PS, "p_stay_event": FIXED_PE})
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best = study.best_trial
    ps, pe = best.params["p_stay_silence"], best.params["p_stay_event"]
    print(f"\n=== TUNED: p_stay_silence={ps:.4f} p_stay_event={pe:.4f} natural={best.user_attrs['natural']:.4f} "
          f"synthetic={best.user_attrs['synthetic']:.4f} eq_avg={best.value:.4f} "
          f"(current real-best 4-model at fixed decode: {ref[4]:.4f}; real Codabench 1.5427)",
          flush=True)
    summary["tuned"] = dict(p_stay_silence=ps, p_stay_event=pe, natural=best.user_attrs["natural"],
                            synthetic=best.user_attrs["synthetic"], eq=best.value,
                            reference_at_fixed_decode=ref[4])
    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---- 5. submission ----
    print("\nGenerating submission...", flush=True)
    models = [load_member(n) for n in names]
    hop = G["hop"]
    all_files = sorted(glob.glob(os.path.join(TEST_DIR, "**", "*.wav"), recursive=True))
    short_files, long_files = [], []
    for path in tqdm(all_files, desc="checking durations"):
        try:
            info = sf.info(path)
            duration = info.frames / info.samplerate
        except Exception:
            duration = 0.0
        (short_files if duration <= config.MAX_DURATION else long_files).append(path)
    print(f"{len(short_files)} short clips (batched), {len(long_files)} long clips (chunked)", flush=True)

    entries = []
    loader = DataLoader(ShortClipDataset(short_files, config), batch_size=8, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    with torch.no_grad():
        for audio, clip_ids in tqdm(loader, desc="short clips"):
            audio = audio.to(config.DEVICE)
            probabilities = combine([m(audio) for m in models], mode).cpu().numpy()
            for probability, clip_id in zip(probabilities, clip_ids):
                states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
                entries.append({"clip_id": clip_id, "events": extract_events_viterbi(states, hop)})
    for path in tqdm(long_files, desc="long clips (chunked)"):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = load_wav(path, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        probability = predict_long_clip_mode(models, wav, config, mode)
        states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
        entries.append({"clip_id": clip_id, "events": extract_events_viterbi(states, hop)})

    assert len(entries) == len(all_files), f"entry count mismatch: {len(entries)} vs {len(all_files)}"
    assert len({e["clip_id"] for e in entries}) == len(entries), "duplicate clip_ids"
    with open(OUT_JSONL, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUT_JSONL, arcname="predictions.jsonl")
    print(f"Done -> {OUT_ZIP} ({len(entries)} entries, {len(long_files)} chunked). "
          f"Members: {'+'.join(names)} [{mode}], decode {ps:.4f}/{pe:.4f}", flush=True)


if __name__ == "__main__":
    main()
