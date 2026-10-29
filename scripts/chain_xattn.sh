#!/bin/bash
# Lane A (Oct 11): event-query cross-attention boundary head, 5-fold CV of 4 configs on the labelled pool, GPU0.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=16
for cfg in full mean_member concat no_global; do
  python -m temp.train_boundary_xattn cv $cfg > work/log_xattn_cv_$cfg.log 2>&1
done
echo done > work/xattn_cv_done.txt
