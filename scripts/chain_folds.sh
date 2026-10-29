#!/bin/bash
# Oct 14: cross-fitted feature models for verifier v5. 2 Dasheng-base lanes x 5 folds over Gold-all + validation
# (fold k never seen by model k). Pretrained Dasheng + fresh head (no v4 warm start, no pseudo-labels). GPU1 only.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 VAL_TRAIN=fold FOLD_N=5 SYNTH_FRAC=0.3 OMP_NUM_THREADS=12
unset GOLD_ALL PSEUDO_LABELS PSEUDO_SHARE DASHENG_NAME CONTEXT
lane() {  # tag height half_width
  for k in 0 1 2 3 4; do
    FOLD_K=$k STAGE1_TAG=$1$k STAGE1_EPOCHS=3 BATCH_SIZE=16 OWBCE_HEIGHT=$2 OWBCE_HALF_WIDTH=$3 \
      python -m temp.train_dasheng_big_stage1 > work/log_fold_$1${k}_s1.log 2>&1 || return 1
    FOLD_K=$k MAX_EPOCHS=3 WARM_START=./checkpoints/standalone_dasheng_big_stage1_$1${k}_last.pth OWBCE_TAG=$1$k \
      OWBCE_HEIGHT=$2 OWBCE_HALF_WIDTH=$3 python -m temp.train_dasheng_stage2_i2_owbce > work/log_fold_$1${k}_s2.log 2>&1 || return 1
  done
}
lane FA 4 10 &
lane FB 12 20 &
wait
echo done > work/folds_done.txt
