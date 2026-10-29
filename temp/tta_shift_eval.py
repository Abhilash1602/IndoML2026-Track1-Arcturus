"""Plan lever A: time-shift test-time augmentation for the v4 ensemble.

Ensembling pays because our Natural boundary errors are high-variance jitter, and
our backbones read coarse tokens (Dasheng/ATST 40 ms = 4 frames, BEATs 160 ms).
Shifting the input by a few 10 ms frames moves the token grid; averaging the
realigned predictions is a within-model ensemble over grid phases.

For each v4 member: logits on the official validation set for start-shifts of
SHIFTS frames (audio padded with s*160 zero samples at the start, cropped back to 10 s;
output frame t+s realigned to t, the last s frames fall back to the unshifted
prediction). Cached to scratchpad/tta_shift_logits.npz. Then a few pre-registered
shift sets are scored (member logit-mean over shifts, then the v4 logit-mean
ensemble, decode 0.9802/0.9489) against the unshifted v4, with a paired bootstrap.
Primary set, fixed in advance: {0,1,2,3} (the four sub-token phases of a 40 ms token).

Usage: python -m temp.tta_shift_eval
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import temp.search_ensemble_v2 as v2
import temp.search_ensemble_v3 as v3
from src.config import Config
from src.dataset import OfficialValidationDataset
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, match_events

MEMBERS = ("atst_stage2", "dasheng_owB", "beats_s2", "dasheng_owB_t", "beats_owA")
SHIFTS = (0, 1, 2, 3, 4, 6, 8, 12)
SETS = {"none": (0,), "primary {0,1,2,3}": (0, 1, 2, 3), "{0,2,4,6}": (0, 2, 4, 6),
        "{0,4,8,12}": (0, 4, 8, 12), "all 8": SHIFTS}
PS, PE, HOP, FRAMES = 0.9802, 0.9489, 0.01, 1000
CACHE = os.path.join(v2.SCRATCH, "tta_shift_logits.npz")
VAL_ROOT = "./data/official_validation_raw/validation"


def shift_audio(audio, s):
    if s == 0:
        return audio
    return torch.cat([audio.new_zeros(audio.shape[0], s * 160), audio], dim=1)[:, :audio.shape[1]]


def realign(logit_shifted, logit_base, s):
    """Output frame t + s of the shifted input is original frame t."""
    if s == 0:
        return logit_shifted
    out = logit_base.copy()
    out[:, :FRAMES - s] = logit_shifted[:, s:FRAMES]
    return out


def cache_logits():
    for name, (cls, path, overrides) in v3.NEW_MEMBERS.items():
        if os.path.exists(path):
            v2.MEMBERS[name] = dict(cls=cls, path=path, overrides=overrides)
    config = Config()
    out = {}
    for subset in ("natural", "synthetic"):
        ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS)
        masks = []
        for _, m, _ in loader:
            masks.append(m[:, :FRAMES].numpy())
        out[f"{subset}__targets"] = np.concatenate(masks).astype(np.uint8)
    for member in MEMBERS:
        model = v2.load_member(member)
        for subset in ("natural", "synthetic"):
            ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS)
            per_shift = {s: [] for s in SHIFTS}
            with torch.no_grad():
                for audio, _, _ in tqdm(loader, desc=f"{member}/{subset}"):
                    audio = audio.to(config.DEVICE)
                    for s in SHIFTS:
                        per_shift[s].append(model(shift_audio(audio, s)).float()[:, :FRAMES].cpu().numpy())
            base = np.concatenate(per_shift[0])
            for s in SHIFTS:
                raw = np.concatenate(per_shift[s])
                out[f"{subset}__{member}__s{s}"] = realign(raw, base, s).astype(np.float16)
        del model
        torch.cuda.empty_cache()
    np.savez(CACHE, **out)
    return out


def alignment_check(C):
    """Realigned shifted logits must correlate best with the unshifted ones at lag 0."""
    bad = []
    for member in MEMBERS:
        base = C[f"natural__{member}__s0"].astype(np.float32)[:, 50:900]
        for s in SHIFTS[1:]:
            x = C[f"natural__{member}__s{s}"].astype(np.float32)
            corr = {lag: float(np.mean(base * x[:, 50 + lag:900 + lag])) for lag in range(-6, 7)}
            best = max(corr, key=corr.get)
            if best != 0:
                bad.append((member, s, best))
    print(f"alignment check: {'PASS' if not bad else 'FAIL ' + str(bad)} (argmax lag 0 for every member x shift)", flush=True)
    return not bad


def per_clip(C, subset, shifts):
    logit = np.mean([np.mean([C[f"{subset}__{m}__s{s}"].astype(np.float32) for s in shifts], axis=0)
                     for m in MEMBERS], axis=0)
    prob = 1 / (1 + np.exp(-logit))
    rows = []
    for p, t in zip(prob, C[f"{subset}__targets"]):
        pred = [(e["onset"], e["offset"]) for e in extract_events_viterbi(
            viterbi_decode(p, p_stay_silence=PS, p_stay_event=PE), HOP)]
        ref = [(e["onset"], e["offset"]) for e in extract_events_from_mask(t.astype(np.int32), HOP)]
        tp, fp, fn = match_events(ref, pred)
        rows.append((tp, fp, fn, macro_segment_dice([ref], [pred], hop_sec=HOP)))
    return np.array(rows, float)


def f1(a):
    return 2 * a[:, 0].sum() / max(2 * a[:, 0].sum() + a[:, 1].sum() + a[:, 2].sum(), 1)


def pooled(nat, syn, i=None, j=None):
    a = np.concatenate([nat if i is None else nat[i], syn if j is None else syn[j]])
    return f1(a) + a[:, 3].mean()


def main():
    C = dict(np.load(CACHE)) if os.path.exists(CACHE) else cache_logits()
    alignment_check(C)
    res = {k: (per_clip(C, "natural", v), per_clip(C, "synthetic", v)) for k, v in SETS.items()}
    for k, (nat, syn) in res.items():
        print(f"[{k:18s}] pooled={pooled(nat, syn):.4f} natF1={f1(nat):.4f} natDice={nat[:, 3].mean():.4f} "
              f"synF1={f1(syn):.4f} synDice={syn[:, 3].mean():.4f} "
              f"eq_avg={0.5 * (f1(nat) + nat[:, 3].mean() + f1(syn) + syn[:, 3].mean()):.4f}", flush=True)
    rng = np.random.default_rng(0)
    nn_, ns = len(res["none"][0]), len(res["none"][1])
    boots = [(rng.integers(0, nn_, nn_), rng.integers(0, ns, ns)) for _ in range(1000)]
    b0 = np.array([pooled(*res["none"], i, j) for i, j in boots])
    n0 = np.array([f1(res["none"][0][i]) for i, _ in boots])
    for k in list(SETS)[1:]:
        d = np.array([pooled(*res[k], i, j) for i, j in boots]) - b0
        dn = np.array([f1(res[k][0][i]) for i, _ in boots]) - n0
        print(f"bootstrap {k:18s} vs none: pooled {d.mean():+.4f} 5%={np.percentile(d, 5):+.4f} "
              f"P(>0)={np.mean(d > 0):.3f} | natF1 {dn.mean():+.4f} P(>0)={np.mean(dn > 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
