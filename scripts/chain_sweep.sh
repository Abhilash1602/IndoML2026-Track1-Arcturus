#!/bin/bash
# Oct 13: guarded event-decision sweep on v14ng_g7_vd30 (1.5887). CPU only.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
C="--member-prefix VS --no-gate --guard 0.7 --w 0.7 --vdrop 0.30"
python -m temp.pipeline_v11 $C --tag v14ng_g7_vd30 --out-dir work/regress2 > work/log_sweep_regress.log 2>&1
python -m temp.pipeline_v11 --member-prefix VS --no-gate --guard 0.7 --w 0.7 --vdrop 0.25 --tag v14ng_g7_vd25 > work/log_sweep_vd25.log 2>&1
python -m temp.pipeline_v11 $C --vmerge 0.5 --tag v14ng_g7_vd30_vm50 > work/log_sweep_vm50.log 2>&1
python -m temp.pipeline_v11 $C --p1-votes 2 --tag v14ng_g7_vd30_p1v2 > work/log_sweep_p1v2.log 2>&1
python -m temp.pipeline_v11 $C --fpthr 0.3 --tag v14ng_g7_vd30_fp30 > work/log_sweep_fp30.log 2>&1
echo done > work/sweep_done.txt
