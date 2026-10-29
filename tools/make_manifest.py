"""Write CHECKPOINTS.md: every file the final Track 1 system (main.py) loads, with size and sha256.
The list was confirmed with a Python audit hook on a full main.py run (20 files + mispeech/dasheng-base).

Usage (from the repository root): python tools/make_manifest.py
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from release.v6_inference import MEMBERS  # noqa: E402

PREFIX = "VS"  # round-1 self-trained students in the 5 validation-tuned positions
FIVE = ["refiner_val_all_x0.pth", "refiner_val_train70_x0.pth"] + [f"refiner_val_all_x0_s{s}.pth" for s in (1, 2, 3)]
OTHER = [
    ("FP filter + verifier (GBMs, PCAs)", "checkpoints/postproc_v11.pkl"),
    ("DETR-style detector (verifier evidence)", "checkpoints/sedt_gold_anc_s0_alone.pth"),
    ("gated edge-correction table", "checkpoints/edge_correction_v5.json"),
    ("backbone init (ATST-F)", "checkpoints/atst_frame/ATST-F_strong_1.pt"),
    ("backbone init (BEATs)", "checkpoints/beats/BEATs_iter3_plus_AS2M.pt"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = [(f"member {n}", f"checkpoints/{f.replace('owbceVA_', f'owbce{PREFIX}_')}") for n, _, f, _ in MEMBERS]
    rows += [(f"boundary refiner (voter {i})", f"checkpoints/{f}") for i, f in enumerate(FIVE)]
    rows += OTHER
    lines = ["# Checkpoints used by the final Track 1 system", "",
             "Loaded by `python inference/infer.py <input_dir> <output_dir>` (Phase 2 test: 1.5893, 8th place). Place them under `./checkpoints/` "
             "with these relative paths. `mispeech/dasheng-base` is downloaded from Hugging Face at first use.", "",
             "| role | file | size (MB) | sha256 |", "|---|---|---|---|"]
    missing = []
    for role, rel in rows:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        lines.append(f"| {role} | `{rel}` | {os.path.getsize(path) / 1e6:.1f} | `{sha256(path)}` |")
    assert not missing, f"missing checkpoints: {missing}"
    with open(os.path.join(ROOT, "CHECKPOINTS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(rows)} files -> CHECKPOINTS.md")


if __name__ == "__main__":
    main()
