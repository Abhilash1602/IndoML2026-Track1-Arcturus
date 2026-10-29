#!/bin/bash
# Oct 14: clean (Gold-train + synthetic only; no validation, no Gold held-out) versions of the new
# lineages, as feature providers for verifier v3. GPU1 only (Track 2 owns GPU0). Final-epoch checkpoints.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 SYNTH_FRAC=0.3 OMP_NUM_THREADS=12
unset VAL_TRAIN GOLD_ALL PSEUDO_LABELS PSEUDO_SHARE
# lane 1: w2v-BERT + LoRA + 10 ms CNN, then the pre-trained-context Dasheng
( STAGE1_MODEL=fusion FUSION_BRANCH=w2vbert STAGE1_TAG=w2vbertVG STAGE1_EPOCHS=6 BATCH_SIZE=16 LORA_LR=2e-4 \
    python -m temp.train_dasheng_big_stage1 > work/log_vg_w2v.log 2>&1
  CONTEXT=mae ATST_BASE_LR=2e-5 DISTILLED_LR=3e-4 MAX_EPOCHS=4 WARM_START=./checkpoints/ctx_mae_warm_clean.pth \
    OWBCE_TAG=VG_mae OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 python -m temp.train_dasheng_stage2_i2_owbce > work/log_vg_mae.log 2>&1 ) &
# lane 2: Dasheng-0.6B stage 1 -> stage 2
( DASHENG_NAME=mispeech/dasheng-0.6B STAGE1_TAG=d06VG STAGE1_EPOCHS=4 BATCH_SIZE=16 python -m temp.train_dasheng_big_stage1 > work/log_vg_d06_s1.log 2>&1
  DASHENG_NAME=mispeech/dasheng-0.6B ATST_BASE_LR=5e-5 DISTILLED_LR=2e-4 MAX_EPOCHS=4 \
    WARM_START=./checkpoints/standalone_dasheng_big_stage1_d06VG_last.pth OWBCE_TAG=VG_d06 OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 \
    python -m temp.train_dasheng_stage2_i2_owbce > work/log_vg_d06_s2.log 2>&1 ) &
wait
echo done > work/vg_done.txt
