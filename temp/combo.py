"""v8_combo recipe (real 1.5801), parameterised by the refiner ensemble.

On Natural-looking short test clips (v6-probability gate): merge adjacent v6 events when at least
--merge-need of the K refiners (temp.refiner_votes.REFINERS) have one event spanning both; drop
the real-proven drop set -- v6 events dropped by P1 (v7 refiner unsupported) or by P3 (gradient-
boosted false-positive filter); merged events are never dropped. Everything else is v6.

Usage: python -m temp.combo --merge-need 5 --out submission_v9c_combo9.zip
"""
import argparse
import json
import zipfile

from temp.refiner_votes import REFINERS, overlap, refiner_test_events


def load(p):
    rows = [json.loads(l) for l in zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines()]
    return {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}, [r["clip_id"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-need", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.out not in ("submission_v8_combo.zip", "submission_v8_d40_m60.zip")
    v6, order = load("submission_v6.zip")
    p1, _ = load("submission_probe_P1.zip")
    p3, _ = load("submission_probe_P3.zip")
    ids, R, gate = refiner_test_events()
    K = len(REFINERS)
    out, merges, drops = dict(v6), 0, 0
    for i, c in enumerate(ids):
        ev = v6[c]
        dropset = (set(ev) - set(p1[c])) | (set(ev) - set(p3[c]))
        if not gate[i] and not dropset:
            continue
        pv = [sum(any(overlap(ev[k], f) and overlap(ev[k + 1], f) for f in (R[p][i] or [])) for p in REFINERS)
              for k in range(len(ev) - 1)] if gate[i] else [0] * max(len(ev) - 1, 0)
        new, src = [], []
        for k, e in enumerate(ev):
            if new and pv[k - 1] >= args.merge_need:
                new[-1] = (new[-1][0], max(new[-1][1], e[1]))
                src[-1] = None
                merges += 1
            else:
                new.append(e)
                src.append(e)
        kept = [e for e, s in zip(new, src) if not (s is not None and s in dropset)]
        drops += len(new) - len(kept)
        out[c] = kept
    assert len(out) == 5517 and all(0 <= a < b for ev in out.values() for a, b in ev)
    print(f"K={K}, merge >= {args.merge_need}: merges {merges}, drops {drops}; events "
          f"{sum(map(len, v6.values()))} -> {sum(map(len, out.values()))}; clips changed {sum(out[c] != v6[c] for c in v6)}")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "".join(json.dumps({"clip_id": c, "events": [
            {"onset": a, "offset": b} for a, b in out[c]]}) + "\n" for c in order))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
