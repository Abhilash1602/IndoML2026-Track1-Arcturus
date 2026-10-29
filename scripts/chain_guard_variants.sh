#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
python -m temp.pipeline_v11 --member-prefix VQ --no-gate --w 0.7 --guard 0.8 --tag v16ng_g80 > work/log_v16ng_g80.log 2>&1
python -m temp.pipeline_v11 --member-prefix VQ --no-gate --w 0.7 --guard 0.9 --band 0.05 0.95 --tag v16ng_g90b > work/log_v16ng_g90b.log 2>&1
python -m temp.pipeline_v11 --member-prefix VQ --no-gate --w 0.7 --guard 0.85 --band 0.05 0.95 --tag v16ng_g85b > work/log_v16ng_g85b.log 2>&1
