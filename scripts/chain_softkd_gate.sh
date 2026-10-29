#!/bin/bash
# Oct 12: soft-target KD round (VK) = VB's recipe with PSEUDO_SOFT_MIX=0.5; warm start VT teachers; train70.
cd "$(dirname "$0")/.." && source .venv/bin/activate
COMMON="VAL_TRAIN=train70 GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=1e-4 MAX_EPOCHS=3 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 PSEUDO_SOFT_MIX=0.5 OMP_NUM_THREADS=10"
run() {  # gpu script warm_start tag height half_width
  env $COMMON CUDA_VISIBLE_DEVICES=$1 VAANI_DEVICE=cuda:0 WARM_START=./checkpoints/$3 OWBCE_TAG=$4 OWBCE_HEIGHT=$5 OWBCE_HALF_WIDTH=$6 \
      python -m temp.$2 > work/log_gate_$4.log 2>&1
}
D=train_dasheng_stage2_i2_owbce; A=train_atst_frame_stage2_i2_owbce; Bt=train_beats_stage2_owbce
( run 0 $D standalone_dasheng_stage2_i2_owbceVT_owB_teacher.pth VK_owB 4 10; run 0 $A dualbranch_atst_frame_stage2_i2_owbceVT_atst_teacher.pth VK_atst 0 20 ) &
( run 0 $D standalone_dasheng_stage2_i2_owbceVT_owBt_teacher.pth VK_owBt 4 10 ) &
( run 1 $Bt standalone_beats_stage2_owbceVT_bS2_teacher.pth VK_bS2 4 10 ) &
( run 1 $Bt standalone_beats_stage2_owbceVT_bA_teacher.pth VK_bA 4 10 ) &
wait
echo done > work/softkd_gate_done.txt
