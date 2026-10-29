#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
W=work
VAANI_DEVICE=cuda:1 python -m temp.cache_sedt_features > $W/log_sedt.log 2>&1 || exit 1
VAANI_DEVICE=cuda:1 python -m temp.cache_verifier_features > $W/log_verifier.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=1 python -m temp.pipeline_v11 --gated-blend --w 0.65 0.6 > $W/log_pipeline_g65.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=1 python -m temp.pipeline_v11 --no-gate --w 0.7 1.0 > $W/log_pipeline_ng.log 2>&1
