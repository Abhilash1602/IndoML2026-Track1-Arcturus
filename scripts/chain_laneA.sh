#!/bin/bash
# Win-strategy Lane A (Oct 10): Dasheng-0.6B member on the student recipe, train70 (hold30 kept out
# for the gate). GPU0 only.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export DASHENG_NAME=mispeech/dasheng-0.6B VAL_TRAIN=train70 GOLD_ALL=1 SYNTH_FRAC=0.3 \
       PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 OMP_NUM_THREADS=16 VAANI_DEVICE=cuda:0
STAGE1_TAG=d06 STAGE1_EPOCHS=4 BATCH_SIZE=16 python -m temp.train_dasheng_big_stage1 > work/log_laneA_stage1.log 2>&1 || exit 1
ATST_BASE_LR=5e-5 DISTILLED_LR=2e-4 MAX_EPOCHS=4 WARM_START=./checkpoints/standalone_dasheng_big_stage1_d06.pth \
  OWBCE_TAG=VT_d06 OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 \
  python -m temp.train_dasheng_stage2_i2_owbce > work/log_laneA_stage2.log 2>&1
echo done > work/laneA_done.txt
