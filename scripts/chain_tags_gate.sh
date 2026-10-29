#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
until [ -f work/tagger_done.txt ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 python -m temp.verifier_tags > work/log_verifier_tags.log 2>&1
