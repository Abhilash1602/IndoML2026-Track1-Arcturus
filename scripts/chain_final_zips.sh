#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
python -m temp.pipeline_v11 --member-prefix VS --no-gate --guard 0.7 --w 0.7 --vdrop 0.30 --tag v14ng_g7_vd30 > work/log_v14ng_g7_vd30.log 2>&1
python -m temp.pipeline_v11 --member-prefix VQ --no-gate --guard 0.7 --w 0.7 --vdrop 0.30 --fpthr 0.4 --tag v16ng_g7_vd30_fp40 > work/log_v16ng_g7_vd30_fp40.log 2>&1
echo done > work/final_zips_done.txt
