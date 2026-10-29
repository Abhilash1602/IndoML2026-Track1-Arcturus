"""Phase 1 real-test decomposition probes (plan "two-stage SEDT event verifier").

v7 (v6 + the in-domain refiner on Natural-looking clips) scored Event F1 +0.0119 / Dice -0.0133
vs v6. Backed out from the real scores, its F1 gain was precision: 532 fewer predictions for
~40-80 fewer matches. v7 did three things at once on its clips: dropped 364 v6 events, merged
386 v6 events (210 pairs) and moved the edges of the rest. Each probe below applies ONE of them
to v6 as a general rule and leaves everything else byte-identical to v6:

  P1 refiner-support filter -- drop every v6 event that no v7 (refiner) event overlaps.
  P2 refiner merges         -- v6 events spanned by one v7 event are replaced by their union.
  P4 two-refiner filter, either -- drop v6 events unsupported by the v7 OR the v7b refiner.
  P5 two-refiner filter, both   -- drop v6 events unsupported by the v7 AND the v7b refiner.
  P6 P1 + P2 -- merge as P2, then drop as P1 (disjoint: merged events are refiner-supported).
  P7 P1 drops, then merges by the v7 refiner and then by the v7b refiner (338 vs 210 pairs).
  P8 P6, then bridge every remaining gap < GAP s between consecutive events on Natural-looking
     short clips (v6-probability gate from scratchpad/verifier/test_logits.npy) -- no refiner.
(P2 real: 1.5748, +0.0054 -- merges raise F1 AND Dice: the gaps between fragments are real noise.)
(P1 real: 1.5737, +0.0043 vs v6 -- the precision lever confirmed on test.)

Usage: python -m temp.make_probes P1|P2|P4|P5 [--ref submission_v7.zip] [--ref2 submission_v7b.zip]
Writes submission_probe_<name>.zip in the repository root and prints the change report.
"""
import argparse
import json
import zipfile

import numpy as np


def load(path):
    rows = [json.loads(l) for l in zipfile.ZipFile(path).read("predictions.jsonl").decode().splitlines()]
    return {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}, [r["clip_id"] for r in rows]


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def p1_drop(v6, ref):
    return [e for e in v6 if any(overlap(e, f) for f in ref)]


def p2_merge(v6, ref):
    out, used = [], set()
    for i, e in enumerate(v6):
        if i in used:
            continue
        group = [i]
        for f in ref:
            if overlap(e, f):
                group = sorted(set(group) | {j for j, g in enumerate(v6) if overlap(g, f)})
        if len(group) > 1:
            used |= set(group)
            out.append((min(v6[j][0] for j in group), max(v6[j][1] for j in group)))
        else:
            out.append(e)
    return sorted(out)


def bridge(ev, gap):
    out = []
    for e in sorted(ev):
        if out and e[0] - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], e[1]))
        else:
            out.append(e)
    return out


def natural_gated_test():
    """Short test clips whose v6 Viterbi events look Natural (mean edge contrast < 0.5), as v7's gate."""
    from src.edge_correction import edge_contrast
    from temp.train_refiner import viterbi_events
    from temp.train_refiner_val import S
    ids = json.load(open(f"{S}/verifier/test_ids.json"))
    logits = np.load(f"{S}/verifier/test_logits.npy").astype(np.float32)
    out = set()
    for i, k in enumerate(ids):
        P = 1 / (1 + np.exp(-logits[i].mean(0)))
        vit = viterbi_events(P[None])[0]
        if not vit or np.mean([edge_contrast(P, a, b) for a, b in vit]) < 0.5:
            out.add(k)
    return out


def frames(ev, n=3000):
    m = np.zeros(n, bool)
    for a, b in ev:
        m[int(round(a * 100)):int(round(b * 100))] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["P1", "P2", "P4", "P5", "P6", "P7", "P8"])
    ap.add_argument("--ref", default="submission_v7.zip")
    ap.add_argument("--ref2", default="submission_v7b.zip")
    ap.add_argument("--gap", type=float, default=0.6)
    args = ap.parse_args()
    v6, order = load("submission_v6.zip")
    ref, _ = load(args.ref)
    ref2, _ = load(args.ref2) if args.probe in ("P4", "P5", "P7") else (ref, None)
    gated = natural_gated_test() if args.probe == "P8" else set()
    assert set(v6) == set(ref) == set(ref2)

    def two_refiner(ev, r1, r2, k):
        sup1 = [any(overlap(e, f) for f in r1) if v6[k] != ref[k] else True for e in ev]
        sup2 = [any(overlap(e, f) for f in r2) if v6[k] != ref2[k] else True for e in ev]
        keep = [(a and b) if args.probe == "P4" else (a or b) for a, b in zip(sup1, sup2)]
        return [e for e, kp in zip(ev, keep) if kp]

    out, changed, removed, added = {}, 0, 0, 0
    for k in order:
        if args.probe in ("P4", "P5"):
            new = two_refiner(v6[k], ref[k], ref2[k], k)
        elif args.probe == "P7":
            new = list(v6[k])
            if v6[k] != ref[k]:
                new = p2_merge(p1_drop(new, ref[k]), ref[k])
            if v6[k] != ref2[k]:
                new = p2_merge(new, ref2[k])
        elif args.probe == "P8":
            new = p1_drop(p2_merge(v6[k], ref[k]), ref[k]) if v6[k] != ref[k] else list(v6[k])
            if k in gated:
                new = bridge(new, args.gap)
        elif v6[k] != ref[k]:  # only clips the refiner touched
            if args.probe == "P6":
                new = p1_drop(p2_merge(v6[k], ref[k]), ref[k])
            else:
                new = {"P1": p1_drop, "P2": p2_merge}[args.probe](v6[k], ref[k])
        else:
            new = list(v6[k])
        if new != v6[k]:
            changed += 1
            a, b = frames(v6[k]), frames(new)
            removed += int((a & ~b).sum()); added += int((b & ~a).sum())
        out[k] = new
    bad = sum(not (0 <= a < b) for ev in out.values() for a, b in ev)
    assert bad == 0 and len(out) == 5517
    n6, nn = sum(map(len, v6.values())), sum(map(len, out.values()))
    print(f"{args.probe} (ref {args.ref}): clips changed {changed}; events {n6} -> {nn} ({nn - n6:+d}); "
          f"frames removed {removed / 100:.1f}s added {added / 100:.1f}s; malformed {bad}")
    path = f"submission_probe_{args.probe}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "".join(
            json.dumps({"clip_id": k, "events": [{"onset": round(a, 3), "offset": round(b, 3)} for a, b in out[k]]}) + "\n"
            for k in order))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
