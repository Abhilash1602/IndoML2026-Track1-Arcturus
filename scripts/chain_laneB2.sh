#!/bin/bash
# Lane B (Oct 11): w2v-BERT 2.0 + LoRA (q/k/v/out, r16) + 10 ms CNN (FusionSED), student mix, train70. GPU1.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 VAL_TRAIN=train70 GOLD_ALL=1 SYNTH_FRAC=0.3 \
       PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 OMP_NUM_THREADS=12
STAGE1_MODEL=fusion FUSION_BRANCH=w2vbert STAGE1_TAG=w2vbertVT STAGE1_EPOCHS=6 BATCH_SIZE=16 LORA_LR=2e-4 \
  python -m temp.train_dasheng_big_stage1 > work/log_laneB2.log 2>&1
echo done > work/laneB2_done.txt
