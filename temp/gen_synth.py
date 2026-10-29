"""Generate synthetic strong-labelled SED clips: Silver-clip backgrounds + Gold-
extracted loud events at controlled SNR. Labels = added events + the background's
own pre-existing (approximate Silver) events, so the label set is complete and
synth_val scores are calibrated. Backgrounds are split so synth_train / synth_val
never share one. Val uses crop=False so background-event timestamps stay exact.

Usage: python -m temp.gen_synth --n_train 15000 --n_val 1200 [--val_only]
"""
import argparse, json, os
import numpy as np
import soundfile as sf
from src.synthesis import synthesize_clip, SR

EVENT_BANK, BG_DIR = "data/event_bank", "data/bg_silver"


def load_events():
    rows = [json.loads(l) for l in open(os.path.join(EVENT_BANK, "manifest.jsonl"))]
    return [os.path.join(EVENT_BANK, r["path"]) for r in rows]


def load_bgs():
    rows = [json.loads(l) for l in open(os.path.join(BG_DIR, "bg_labels.jsonl"))]
    return [(os.path.join(BG_DIR, r["path"]), r["events"]) for r in rows]


def gen(out_dir, bgs, event_paths, n_clips, seed, prefix, crop):
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []
    for i in range(n_clips):
        bg_path, bg_events = bgs[int(rng.integers(0, len(bgs)))]
        try:
            wav, labels = synthesize_clip(
                bg_path, event_paths, n_events=(1, 3), snr_db=(3.0, 20.0),
                min_gap_sec=0.15, background_events=bg_events, crop=crop, rng=rng)
        except Exception:
            continue
        cid = f"{prefix}_{i:06d}"
        sf.write(os.path.join(out_dir, "audio", f"{cid}.wav"), wav, SR, subtype="PCM_16")
        manifest.append({"clip_id": cid,
                         "events": [{"onset": round(a, 3), "offset": round(b, 3)} for a, b in labels]})
        if (i + 1) % 2000 == 0:
            print(f"  {prefix} {i+1}/{n_clips}", flush=True)
    with open(os.path.join(out_dir, "labels.jsonl"), "w", encoding="utf-8") as fh:
        for r in manifest:
            fh.write(json.dumps(r) + "\n")
    print(f"DONE {prefix}: {len(manifest)} clips -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=15000)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--val_only", action="store_true")
    args = ap.parse_args()
    event_paths = load_events()
    bgs = load_bgs()
    print(f"{len(event_paths)} events, {len(bgs)} backgrounds", flush=True)
    rng = np.random.default_rng(0)
    rng.shuffle(bgs)
    n_val_bg = max(800, len(bgs) // 6)
    val_bgs, train_bgs = bgs[:n_val_bg], bgs[n_val_bg:]
    gen("data/synth_val", val_bgs, event_paths, args.n_val, 1, "sval", crop=False)
    if not args.val_only:
        gen("data/synth_train", train_bgs, event_paths, args.n_train, 2, "strain", crop=True)


if __name__ == "__main__":
    main()
