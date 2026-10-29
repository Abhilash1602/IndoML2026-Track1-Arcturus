#!/bin/bash
# Win-strategy Lane B (Oct 10): frozen Whisper-large-v3 encoder + 10 ms CNN (FusionSED), student data
# mix, train70 (hold30 kept out for the gate). Shares GPU0 with Lane A (7 GB); GPU1 untouched.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export VAL_TRAIN=train70 GOLD_ALL=1 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 \
       OMP_NUM_THREADS=12 VAANI_DEVICE=cuda:0
STAGE1_MODEL=fusion FUSION_BRANCH=whisper STAGE1_TAG=whisperVT STAGE1_EPOCHS=8 BATCH_SIZE=16 \
  python -m temp.train_dasheng_big_stage1 > work/log_laneB_stage1.log 2>&1
echo done > work/laneB_done.txt
