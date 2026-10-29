"""Track 1 inference: raw audio folder -> predictions.jsonl (+ submission.zip), Phase 2 format.

Usage (from the repository root):
    python inference/infer.py <input_audio_dir> <output_dir> [--config configs/model_config.yaml] [--device cuda:0]

Each line of predictions.jsonl is {"clip_id": <wav file stem>, "events": [{"onset": s, "offset": s}, ...]}.
The checkpoints listed in CHECKPOINTS.md must be under ./checkpoints/.
"""
import argparse
import os
import shutil
import sys
import zipfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pipeline_args(cfg):
    """Translate configs/model_config.yaml into temp.pipeline_v11 arguments. Values the pipeline
    keeps as module constants (Viterbi, merge votes, verifier merge, edge table) are asserted."""
    e, d, x = cfg["ensemble"], cfg["decode"], cfg["event_decisions"]
    assert (d["viterbi_stay_background"], d["viterbi_stay_event"]) == (0.9802, 0.9489)
    assert x["merge_min_refiner_votes"] == 3 and x["verifier_merge_threshold"] == 0.6
    return ["--member-prefix", e["member_prefix"], "--no-gate", "--guard", str(x["confidence_guard"]),
            "--w", str(d["refiner_blend_weight"]), "--vdrop", str(x["verifier_drop_threshold"]),
            "--fpthr", str(x["fp_filter_drop_threshold"]), "--tag", "final"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="folder with the test .wav files (searched recursively)")
    ap.add_argument("output_dir")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "model_config.yaml"))
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: the GPU with most free memory)")
    ap.add_argument("--keep-cache", action="store_true", help="keep <output_dir>/features.pkl")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    in_dir, out_dir = os.path.abspath(args.input_dir), os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.device:
        os.environ["VAANI_DEVICE"] = args.device
    if not os.path.exists(os.path.join(ROOT, "checkpoints", "postproc_v11.pkl")):
        sys.exit("checkpoints missing: see checkpoints/README.md")

    os.chdir(ROOT)  # the pipeline reads ./checkpoints relative to the repository root
    sys.path.insert(0, ROOT)
    import release.v6_inference as v6
    v6.WINDOW_SEC = cfg["ensemble"]["long_clip_window_sec"]
    v6.OVERLAP_SEC = cfg["ensemble"]["long_clip_overlap_sec"]  # read at call time by temp.long_clips.windows
    import temp.pipeline_v11 as pipeline

    cache = os.path.join(out_dir, "features.pkl")
    sys.argv = ["pipeline_v11", "--test-dir", in_dir, "--out-dir", out_dir, "--cache", cache] + pipeline_args(cfg)
    pipeline.main()

    zpath = os.path.join(out_dir, "submission_final_w70.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extract("predictions.jsonl", out_dir)
    shutil.move(zpath, os.path.join(out_dir, "submission.zip"))
    if not args.keep_cache and os.path.exists(cache):
        os.remove(cache)
    print(f"wrote {os.path.join(out_dir, 'predictions.jsonl')} and {os.path.join(out_dir, 'submission.zip')}")


if __name__ == "__main__":
    main()
