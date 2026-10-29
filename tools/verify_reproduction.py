"""Compare a reproduced predictions file with a submitted one, clip by clip.

Usage: python tools/verify_reproduction.py <reproduced predictions.jsonl or .zip> <submitted .zip> [--tol 0.011]
Only clips present in the reproduced file are compared (so a subset run can be checked). An event
matches if both edges agree within --tol seconds (default: one 10 ms frame plus rounding).
"""
import argparse
import json
import zipfile


def load(path):
    text = zipfile.ZipFile(path).read("predictions.jsonl").decode() if path.endswith(".zip") else open(path).read()
    return {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in map(json.loads, text.splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reproduced")
    ap.add_argument("submitted")
    ap.add_argument("--tol", type=float, default=0.011)
    args = ap.parse_args()
    a, b = load(args.reproduced), load(args.submitted)
    same = near = differ = 0
    for k, ev in a.items():
        ref = b[k]
        if ev == ref:
            same += 1
        elif len(ev) == len(ref) and all(abs(x[0] - y[0]) <= args.tol and abs(x[1] - y[1]) <= args.tol for x, y in zip(ev, ref)):
            near += 1
        else:
            differ += 1
            print(f"DIFF {k}\n  reproduced {ev}\n  submitted  {ref}")
    print(f"{len(a)} clips: identical {same}, within {args.tol * 1000:.0f} ms {near}, different {differ}; "
          f"events {sum(map(len, a.values()))} vs {sum(len(b[k]) for k in a)}")


if __name__ == "__main__":
    main()
