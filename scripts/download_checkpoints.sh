#!/bin/bash
# Downloads the 20 checkpoint files into ./checkpoints/ and verifies them against CHECKPOINTS.md.
# Usage: bash scripts/download_checkpoints.sh            (from Hugging Face: asymptoticlearning/IndoML2026-Track1-Arcturus)
#        bash scripts/download_checkpoints.sh <path.tar>  (from a local Arcturus_Track1_checkpoints.tar)
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ge 1 && -f "$1" ]]; then
  tar -xf "$1"            # archive paths are checkpoints/...
else
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("asymptoticlearning/IndoML2026-Track1-Arcturus", allow_patterns=["checkpoints/*"], local_dir=".")
PY
fi
python - <<'PY'
import hashlib, re
ok = True
for path, digest in re.findall(r"`(checkpoints/[^`]+)` \| [0-9.]+ \| `([0-9a-f]{64})`", open("CHECKPOINTS.md").read()):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(("ok  " if h == digest else "BAD ") + path); ok &= h == digest
raise SystemExit(0 if ok else "checksum mismatch")
PY
