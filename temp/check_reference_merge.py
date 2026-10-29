"""Plan Step 0a: does the reference convention matter?

Local scoring builds reference events from the merged any-noise mask (overlapping
category annotations become one event), as does the organisers' own local check.
The official scorer matches reference events exactly as listed, so if the hidden
references are the raw per-annotation spans (overlaps kept), merged predictions can
match at most one of each overlapping group.

Scores the i2 / v3 / v4 ensembles (cached validation logits, their submitted decode)
on the pooled validation set (Natural + Synthetic, micro Event F1 + macro Dice, the
Codabench definition) under both conventions, reports how much of the reference set
overlaps, and compares each convention's local deltas with the real Codabench deltas.

Usage: python -m temp.check_reference_merge
"""
import ast
import json

import numpy as np

from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1

SCRATCH = "./work"
META = "./data/official_validation_raw/validation/validationMetadata.json"
HOP, CLIP_SEC = 0.01, 10.0
SETS = {  # members, real Codabench combined
    "i2": (("atst_stage2", "dasheng_s2_i2", "beats_smoke"), 1.5460),
    "v3": (("atst_stage2", "dasheng_s2_i2", "beats_smoke", "dasheng_owA_t", "dasheng_owB", "beats_s2"), 1.5538),
    "v4": (("atst_stage2", "dasheng_owB", "beats_s2", "dasheng_owB_t", "beats_owA"), 1.5545),
}
PS, PE = 0.9802, 0.9489  # the decode all three were submitted with


def raw_events(row):
    v = row.get("NoiseSubCategoryTimeStamp") or "[]"
    v = ast.literal_eval(v) if isinstance(v, str) else v
    ev = set()
    for e in v:
        s, t = float(e["start"]), float(e["end"])
        if t > s and s < CLIP_SEC:
            ev.add((round(s, 3), round(min(t, CLIP_SEC), 3)))
    return sorted(ev)


def main():
    C = dict(np.load(f"{SCRATCH}/dasheng_ens_val_logits.npz"))
    C.update(dict(np.load(f"{SCRATCH}/ens_v2_val_logits.npz")))
    # frozen_v3 dasheng_owA_t is the cached dasheng_owA_t (same file at cache time)
    rows = json.load(open(META, encoding="utf-8"))
    subsets = {"natural": [r for r in rows if not r.get("syntheticData")],
               "synthetic": [r for r in rows if r.get("syntheticData")]}

    merged_ref, raw_ref = {}, {}
    for sub, rs in subsets.items():
        t = C[f"{sub}__targets"]
        assert len(rs) == len(t)
        merged_ref[sub] = [[(e["onset"], e["offset"]) for e in extract_events_from_mask(m.astype(np.int32), HOP)]
                           for m in t]
        raw_ref[sub] = [raw_events(r) for r in rs]
        n_m, n_r = sum(map(len, merged_ref[sub])), sum(map(len, raw_ref[sub]))
        overl = sum(len(r) > len(m) for r, m in zip(raw_ref[sub], merged_ref[sub]))
        print(f"{sub}: clips={len(rs)} merged refs={n_m} raw refs={n_r} (+{n_r - n_m}, "
              f"{(n_r - n_m) / max(n_m, 1):.1%}); clips where raw has more events than merged: {overl}", flush=True)

    results = {}
    for name, (members, real) in SETS.items():
        preds = {}
        for sub in subsets:
            n = min(min(C[f"{sub}__{m}"].shape[1] for m in members), C[f"{sub}__targets"].shape[1])
            p = 1 / (1 + np.exp(-np.mean([C[f"{sub}__{m}"][:, :n] for m in members], axis=0)))
            preds[sub] = [[(e["onset"], e["offset"]) for e in
                           extract_events_viterbi(viterbi_decode(x, p_stay_silence=PS, p_stay_event=PE), HOP)]
                          for x in p]
        out = {}
        for conv, ref in (("merged", merged_ref), ("raw", raw_ref)):
            all_ref = ref["natural"] + ref["synthetic"]
            all_pred = preds["natural"] + preds["synthetic"]
            f1, _, _ = micro_event_f1(all_ref, all_pred)
            dice = macro_segment_dice(all_ref, all_pred, hop_sec=HOP)
            nf1, _, _ = micro_event_f1(ref["natural"], preds["natural"])
            out[conv] = (f1, dice, f1 + dice, nf1)
        results[name] = (out, real)
        print(f"{name}: real {real:.4f} | merged F1={out['merged'][0]:.4f} Dice={out['merged'][1]:.4f} "
              f"comb={out['merged'][2]:.4f} natF1={out['merged'][3]:.4f} | raw F1={out['raw'][0]:.4f} "
              f"Dice={out['raw'][1]:.4f} comb={out['raw'][2]:.4f} natF1={out['raw'][3]:.4f}", flush=True)

    print("\nDeltas (local merged / local raw / REAL):", flush=True)
    for a, b in (("i2", "v3"), ("v3", "v4"), ("i2", "v4")):
        dm = results[b][0]["merged"][2] - results[a][0]["merged"][2]
        dr = results[b][0]["raw"][2] - results[a][0]["raw"][2]
        dreal = results[b][1] - results[a][1]
        print(f"  {a}->{b}: merged {dm:+.4f}  raw {dr:+.4f}  real {dreal:+.4f}", flush=True)
    print("\nLevels (local minus real):", flush=True)
    for name, (out, real) in results.items():
        print(f"  {name}: merged {out['merged'][2] - real:+.4f}  raw {out['raw'][2] - real:+.4f}", flush=True)


if __name__ == "__main__":
    main()
