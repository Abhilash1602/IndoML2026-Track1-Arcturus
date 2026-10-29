#!/bin/bash
# Oct 11: cheap 6th-place variants on top of v16ng_g7 (1.5883). CPU only.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
C="--member-prefix VQ --no-gate --guard 0.7"
python -m temp.pipeline_v11 $C --w 0.7 --tag v16ng_g7 --out-dir work/regress > work/log_regress_g7.log 2>&1
python -m temp.pipeline_v11 $C --w 0.7 --vdrop 0.30 --tag v16ng_g7_vd30 > work/log_v16ng_g7_vd30.log 2>&1
python -m temp.pipeline_v11 $C --w 0.7 --long-dice 1.5 0.3 1.0 --tag v16ng_g7_ld > work/log_v16ng_g7_ld.log 2>&1
python -m temp.pipeline_v11 $C --w 0.65 --tag v16ng_g7 > work/log_v16ng_g7_w65.log 2>&1
echo done > work/variants_done.txt
