#!/bin/bash
# Usage: bash scripts/run_inference.sh <input_audio_dir> <output_dir> [device]
set -euo pipefail
cd "$(dirname "$0")/.."
python inference/infer.py "$1" "$2" ${3:+--device "$3"}
