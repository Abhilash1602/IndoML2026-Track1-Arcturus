"""Ensemble search v3 (Track 1 boundary-precision plan, Step 3).

Adds the Stage-2 checkpoints finished after v2 -- BEATs S2, ATST S2 i2 and the
OWBCE Dasheng S2 runs (A/B), each as student AND EMA teacher -- to v2's nine
members. Differences from v2, from the i2 lesson (the tuned 0.9802/0.9489
decode won on validation but lost real Natural F1 0.4123 -> 0.4020):
  * no Optuna decode tuning: every option is scored at three fixed decodes only,
    0.9065/0.8914 (the 1.5427 submission), 0.9802/0.9489 (the 1.5460 one) and
    0.998/0.98 (where the OWBCE checkpoints peak, fixed before the search);
  * Natural Event F1 is reported separately, and the gate requires it not to
    drop below the reference's;
  * with ~17 candidates v2's exhaustive 3..7 search is ~80k options, so the
    search is pruned: single members, reference +1 / -1 / swap-1 moves, then
    an exhaustive 2..6 search (logit) over the reference members plus the six
    best add-ons, and a prob-averaging check of the top 10.
Reference = the current real best: submission_i2 (atst_stage2 + dasheng_s2_i2
+ beats_smoke, logit, 0.9802/0.9489; Codabench 1.5460). Writes
submission_v3.zip only if the gate passes (never overwrites earlier zips).

Usage: python -m temp.search_ensemble_v3 [--no-submit]
"""
import glob
import itertools
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import temp.search_ensemble_v2 as v2
from src.architecture import DualBranchSED, FusionSED, StandaloneBEATsSED, StandaloneDashengSED, StandaloneStrongSED
from src.config import Config
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

SUMMARY_JSON = os.path.join(v2.SCRATCH, "ens_v3_summary.json")
OUT_JSONL = "predictions_v3.jsonl"
OUT_ZIP = "submission_v3.zip"
N_WORKERS = 48
SUBMIT_GATE = 0.003
DECODES = {"d1.5427": (0.9065, 0.8914), "d1.5460": (0.9802, 0.9489),
           "sticky": (0.998, 0.98)}  # OWBCE models peak here (owbce_decode_grid.py)
REFERENCE = (("atst_stage2", "dasheng_s2_i2", "beats_smoke"), "logit", "d1.5460")
POOL_ADDONS = 4  # ~35 candidates now; keeps stage 2 at ~4.4k options
# submission_v3 (generated from interim search 1; validation 1.5270). Its members are always in the
# stage-2 pool: beats_s2 is in every top set but only +0.0002 as a single add-on, so ranking
# add-ons by solo gain alone drops it.
# submission_v4 (real 1.5545, the current best): the bar for any new submission.
CURRENT_BEST = (("atst_stage2", "dasheng_owB", "beats_s2", "dasheng_owB_t", "beats_owA"),
                "logit", "d1.5460")

ATST = dict(USE_ATST_FRAME=True)
NEW_MEMBERS = {
    "beats_s2": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2.pth", {}),
    "beats_s2_t": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_teacher.pth", {}),
    "atst_s2_i2": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2.pth", ATST),
    "atst_s2_i2_t": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2_teacher.pth", ATST),
    "dasheng_owA": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceA.pth", {}),
    "dasheng_owA_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceA_teacher.pth", {}),
    "dasheng_owB": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth", {}),
    "dasheng_owB_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceB_teacher.pth", {}),
    "dasheng_owC": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceC.pth", {}),
    "dasheng_owC_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceC_teacher.pth", {}),
    "atst_owB": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2_owbceB.pth", ATST),
    "atst_owB_t": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2_owbceB_teacher.pth", ATST),
    "atst_owA": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2_owbceA.pth", ATST),
    "atst_owA_t": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2_i2_owbceA_teacher.pth", ATST),
    "beats_owA": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceA.pth", {}),
    "beats_owA_t": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceA_teacher.pth", {}),
    "beats_owC": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceC.pth", {}),
    "beats_owC_t": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceC_teacher.pth", {}),
    "dasheng_owAi3": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceAi3.pth", {}),
    "dasheng_owAi3_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceAi3_teacher.pth", {}),
    "beats_owAi3": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceAi3.pth", {}),
    "beats_owAi3_t": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceAi3_teacher.pth", {}),
    "beats_owAi3ff": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceAi3ff.pth",
                      dict(BEATS_FREQ_RESHAPE=True)),
    "beats_owAi3ff_t": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceAi3ff_teacher.pth",
                        dict(BEATS_FREQ_RESHAPE=True)),
    "beats_strong_owA": (StandaloneStrongSED, "./checkpoints/standalone_strong_stage2_BEATs_strong_owbceA.pth",
                dict(STRONG_BACKBONE="BEATs")),
    "beats_strong_owA_t": (StandaloneStrongSED, "./checkpoints/standalone_strong_stage2_BEATs_strong_owbceA_teacher.pth",
                  dict(STRONG_BACKBONE="BEATs")),
    "asit_strong_owA": (StandaloneStrongSED, "./checkpoints/standalone_strong_stage2_ASIT_strong_owbceA.pth",
                dict(STRONG_BACKBONE="ASIT")),
    "asit_strong_owA_t": (StandaloneStrongSED, "./checkpoints/standalone_strong_stage2_ASIT_strong_owbceA_teacher.pth",
                  dict(STRONG_BACKBONE="ASIT")),
    "fus_dashengOwB_s1": (FusionSED, "./checkpoints/fusion_dashengOwB_stage1.pth", dict(FUSION_BRANCH="dasheng")),
    "fus_dashengOwB_s2": (FusionSED, "./checkpoints/fusion_dashengOwB_stage2.pth", dict(FUSION_BRANCH="dasheng")),
    "fus_dashengOwB_s2_t": (FusionSED, "./checkpoints/fusion_dashengOwB_stage2_teacher.pth", dict(FUSION_BRANCH="dasheng")),
    "fus_dashengOwAt_s1": (FusionSED, "./checkpoints/fusion_dashengOwAt_stage1.pth", dict(FUSION_BRANCH="dasheng")),
    "fus_dashengOwAt_s2": (FusionSED, "./checkpoints/fusion_dashengOwAt_stage2.pth", dict(FUSION_BRANCH="dasheng")),
    "fus_dashengOwAt_s2_t": (FusionSED, "./checkpoints/fusion_dashengOwAt_stage2_teacher.pth", dict(FUSION_BRANCH="dasheng")),
    "d_FTg_owB": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceFTg_owB.pth", {}),
    "d_FTg_owB_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceFTg_owB_teacher.pth", {}),
    "d_FTg_owBt": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceFTg_owBt.pth", {}),
    "d_FTg_owBt_t": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceFTg_owBt_teacher.pth", {}),
    "b_FTg_bA": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceFTg_bA.pth", {}),
    "b_FTg_bA_t": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceFTg_bA_teacher.pth", {}),
    "b_FTg_bS2": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceFTg_bS2.pth", {}),
    "b_FTg_bS2_t": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceFTg_bS2_teacher.pth", {}),
}
G = v2.G  # shared worker state (set before fork)


def score_option(args):
    """-> (names, mode, decode_key, nat_f1, nat_dice, nat_comb, syn_comb, eq)."""
    names, mode, dkey = args
    ps, pe = DECODES[dkey]
    out = {}
    for subset in ("natural", "synthetic"):
        n = v2._frames(subset)
        probs = v2.combine([G["cache"][f"{subset}__{m}"][:, :n] for m in names], mode)
        pred, gt = [], []
        for p, t in zip(probs, G["cache"][f"{subset}__targets"][:, :n]):
            states = viterbi_decode(p, p_stay_silence=ps, p_stay_event=pe)
            pred.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(states, G["hop"])])
            gt.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(t.astype(np.int32), G["hop"])])
        f1, _, _ = micro_event_f1(gt, pred)
        out[subset] = (f1, macro_segment_dice(gt, pred, hop_sec=G["hop"]))
    nc, sc = sum(out["natural"]), sum(out["synthetic"])
    return tuple(names), mode, dkey, out["natural"][0], out["natural"][1], nc, sc, 0.5 * (nc + sc)


def run(ex, options):
    options = list(dict.fromkeys((tuple(n), m, d) for n, m, d in options))  # dedupe, keep order
    return list(ex.map(score_option, options, chunksize=2))


def fmt(r, ref=None):
    names, mode, dkey, nf, nd, nc, sc, eq = r
    delta = f" d_ref={eq - ref[7]:+.4f} dNatF1={nf - ref[3]:+.4f}" if ref else ""
    return f"{eq:.4f}{delta} natF1={nf:.4f} natDice={nd:.4f} syn={sc:.4f} [{mode},{dkey}] {'+'.join(names)}"


def generate_submission(names, mode, ps, pe):
    config = Config()
    models = [v2.load_member(n) for n in names]
    hop = config.HOP_LENGTH / config.SR
    all_files = sorted(glob.glob(os.path.join(v2.TEST_DIR, "**", "*.wav"), recursive=True))
    short_files, long_files = [], []
    for path in all_files:
        try:
            info = sf.info(path)
            duration = info.frames / info.samplerate
        except Exception:
            duration = 0.0
        (short_files if duration <= config.MAX_DURATION else long_files).append(path)
    entries = []
    loader = DataLoader(v2.ShortClipDataset(short_files, config), batch_size=8, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    with torch.no_grad():
        for audio, clip_ids in tqdm(loader, desc="short clips"):
            probabilities = v2.combine([m(audio.to(config.DEVICE)) for m in models], mode).cpu().numpy()
            for probability, clip_id in zip(probabilities, clip_ids):
                states = viterbi_decode(probability, p_stay_silence=ps, p_stay_event=pe)
                entries.append({"clip_id": clip_id, "events": extract_events_viterbi(states, hop)})
    for path in tqdm(long_files, desc="long clips (chunked)"):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = v2.load_wav(path, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        probability = v2.predict_long_clip_mode(models, wav, config, mode)
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


def main():
    submit = "--no-submit" not in sys.argv
    for name, (cls, path, overrides) in NEW_MEMBERS.items():
        if os.path.exists(path):
            v2.MEMBERS[name] = dict(cls=cls, path=path, overrides=overrides)
        else:
            print(f"skipping {name}: {path} not found", flush=True)
    # The v2 cache is keyed by member name only: drop entries whose checkpoint was
    # rewritten after the cache (e.g. a run that was still saving new bests).
    if os.path.exists(v2.LOGIT_CACHE_V2):
        cache_time = os.path.getmtime(v2.LOGIT_CACHE_V2)
        cached = dict(np.load(v2.LOGIT_CACHE_V2))
        stale = [n for n in NEW_MEMBERS if n in v2.MEMBERS and f"natural__{n}" in cached
                 and os.path.getmtime(v2.MEMBERS[n]["path"]) > cache_time]
        if stale:
            print(f"re-caching stale members: {stale}", flush=True)
            np.savez(v2.LOGIT_CACHE_V2, **{k: v for k, v in cached.items() if k.split("__")[1] not in stale})
    config = Config()
    G["hop"] = config.HOP_LENGTH / config.SR
    G["cache"] = v2.cache_val_logits()  # caches only members missing from the v1/v2 caches
    members = list(v2.MEMBERS)
    ref_names, ref_mode, ref_dec = REFERENCE
    summary = {"members": members}
    ctx = __import__("multiprocessing").get_context("fork")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        # 1. single members + reference moves (+1 / -1 / swap-1), logit, both decodes
        opts = [((m,), "logit", d) for m in members for d in DECODES]
        opts += [(ref_names, "logit", d) for d in DECODES]
        opts += [(CURRENT_BEST[0], "logit", d) for d in DECODES]
        others = [m for m in members if m not in ref_names]
        for d in DECODES:
            opts += [(ref_names + (m,), "logit", d) for m in others]
            opts += [(tuple(x for x in ref_names if x != drop), "logit", d) for drop in ref_names]
            opts += [(tuple(x for x in ref_names if x != drop) + (m,), "logit", d)
                     for drop in ref_names for m in others]
        print(f"Stage 1: {len(opts)} options (singles, reference +1/-1/swap)", flush=True)
        res1 = run(ex, opts)
        ref = next(r for r in res1 if r[0] == ref_names and r[1] == ref_mode and r[2] == ref_dec)
        print(f"\nREFERENCE (submission_i2, Codabench 1.5460): {fmt(ref)}", flush=True)
        cur = next(r for r in res1 if r[0] == CURRENT_BEST[0] and r[1] == CURRENT_BEST[1] and r[2] == CURRENT_BEST[2])
        print(f"CURRENT BEST (submission_v4 set): {fmt(cur, ref)}", flush=True)
        print("\nSingle members:", flush=True)
        for r in sorted((r for r in res1 if len(r[0]) == 1), key=lambda r: -r[7]):
            print(f"  {fmt(r, ref)}", flush=True)
        add_gain = {}
        for r in res1:
            if len(r[0]) == len(ref_names) + 1 and set(ref_names) <= set(r[0]):
                (m,) = set(r[0]) - set(ref_names)
                add_gain[m] = max(add_gain.get(m, -9), r[7])
        addons = sorted(add_gain, key=lambda m: -add_gain[m])[:POOL_ADDONS]
        print(f"\nBest reference+1 add-ons: " + ", ".join(f"{m} ({add_gain[m] - ref[7]:+.4f})" for m in addons),
              flush=True)

        # 2. exhaustive 2..6 over the reference members + best add-ons, logit
        pool = list(dict.fromkeys(list(ref_names) + list(CURRENT_BEST[0]) + addons))
        opts = [(c, "logit", d) for k in range(2, 7) for c in itertools.combinations(pool, k) for d in DECODES]
        print(f"\nStage 2: exhaustive over pool {pool}: {len(opts)} options", flush=True)
        res2 = run(ex, opts)

        # 3. prob-averaging check on the top 10 so far
        allres = sorted(res1 + res2, key=lambda r: -r[7])
        top10 = allres[:10]
        res3 = run(ex, [(r[0], "prob", r[2]) for r in top10])
    allres = sorted({(r[0], r[1], r[2]): r for r in res1 + res2 + res3}.values(), key=lambda r: -r[7])

    print(f"\nCURRENT BEST (submission_v4 set): {fmt(cur, ref)}", flush=True)
    print("\nTop 20 overall:", flush=True)
    for r in allres[:20]:
        print(f"  {fmt(r, ref)}", flush=True)
    eligible = [r for r in allres if r[7] - ref[7] >= SUBMIT_GATE and r[3] >= ref[3]]
    print(f"\nBest with Natural F1 >= reference ({ref[3]:.4f}):", flush=True)
    for r in [r for r in allres if r[3] >= ref[3]][:5]:
        print(f"  {fmt(r, ref)}", flush=True)
    for cand in [m for m in NEW_MEMBERS if m in members]:
        best_with = next(r for r in allres if cand in r[0]) if any(cand in r[0] for r in allres) else None
        if best_with:
            print(f"  best containing {cand}: {fmt(best_with, ref)}", flush=True)

    keys = ("names", "mode", "decode", "nat_f1", "nat_dice", "natural", "synthetic", "eq")
    summary["reference"] = dict(zip(keys, ref))
    summary["top"] = [dict(zip(keys, r)) for r in allres[:200]]
    if not eligible:
        best = allres[0]
        print(f"\nNO SUBMISSION: best option {best[7] - ref[7]:+.4f} vs reference; none clears "
              f"+{SUBMIT_GATE} with Natural F1 >= reference.", flush=True)
        summary["chosen"] = None
    else:
        chosen = eligible[0]
        print(f"\nCHOSEN (gate passed): {fmt(chosen, ref)}", flush=True)
        summary["chosen"] = dict(zip(keys, chosen))
    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=1, default=list)
    if eligible and submit:
        names, mode, dkey = eligible[0][:3]
        generate_submission(names, mode, *DECODES[dkey])


if __name__ == "__main__":
    main()
