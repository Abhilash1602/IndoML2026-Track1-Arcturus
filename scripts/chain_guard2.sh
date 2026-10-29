#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
for g in 0.75 0.7 0.65; do
  python -m temp.pipeline_v11 --member-prefix VQ --no-gate --w 0.7 --guard $g --tag v16ng_g${g#0.} > work/log_v16ng_g${g#0.}.log 2>&1
done
python -m temp.pipeline_v11 --member-prefix VS --no-gate --w 0.7 --guard 0.8 --tag v14ng_g80 > work/log_v14ng_g80.log 2>&1
