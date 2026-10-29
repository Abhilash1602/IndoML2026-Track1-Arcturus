#!/bin/bash
# Oct 12: in-domain masked-reconstruction context network: pre-train (running) -> warm starts -> fine-tune
# pre-trained (VC_mae) and random-context control (VC_rand) in parallel on GPU1 -> hold30 gate.
cd "$(dirname "$0")/.." && source .venv/bin/activate
until grep -q "^done" work/log_ctx_pretrain.log; do sleep 60; grep -qi traceback work/log_ctx_pretrain.log && exit 1; done
python -m temp.build_ctx_warmstart ./checkpoints/context_mae_pretrained.pth > work/log_ctx_warm.log 2>&1 || exit 1
COMMON="CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 CONTEXT=mae VAL_TRAIN=train70 GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=3e-4 MAX_EPOCHS=4 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 OMP_NUM_THREADS=12"
env $COMMON WARM_START=./checkpoints/ctx_mae_warm.pth OWBCE_TAG=VC_mae python -m temp.train_dasheng_stage2_i2_owbce > work/log_ctx_ft_mae.log 2>&1 &
env $COMMON WARM_START=./checkpoints/ctx_rand_warm.pth OWBCE_TAG=VC_rand python -m temp.train_dasheng_stage2_i2_owbce > work/log_ctx_ft_rand.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=16 python -m temp.eval_ctx_gate > work/log_ctx_gate.log 2>&1
echo done > work/ctx_done.txt
