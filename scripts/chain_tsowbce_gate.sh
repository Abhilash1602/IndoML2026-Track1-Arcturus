#!/bin/bash
# Oct 11: tolerance-shaped OWBCE gate round. 5 families x {fixed OWBCE baseline (VB), tolerance-shaped (VW)},
# warm start = the train70 VT teachers, student data (train70 + Gold + v14 pseudo 0.4), 3 epochs. hold30 kept out.
cd "$(dirname "$0")/.." && source .venv/bin/activate
COMMON="VAL_TRAIN=train70 GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=1e-4 MAX_EPOCHS=3 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 OMP_NUM_THREADS=12"
run() {  # gpu script warm_start tag height half_width mode
  env $COMMON CUDA_VISIBLE_DEVICES=$1 VAANI_DEVICE=cuda:0 WARM_START=./checkpoints/$3 OWBCE_TAG=$4 OWBCE_HEIGHT=$5 OWBCE_HALF_WIDTH=$6 \
      OWBCE_MODE=$7 OWBCE_TOL_SCALE=1.5 python -m temp.$2 > work/log_gate_$4.log 2>&1
}
D=train_dasheng_stage2_i2_owbce; A=train_atst_frame_stage2_i2_owbce; Bt=train_beats_stage2_owbce
( run 0 $D standalone_dasheng_stage2_i2_owbceVT_owB_teacher.pth VW_owB 4 10 tol; run 0 $A dualbranch_atst_frame_stage2_i2_owbceVT_atst_teacher.pth VW_atst 4 20 tol ) &
( run 0 $D standalone_dasheng_stage2_i2_owbceVT_owB_teacher.pth VB_owB 4 10 fixed; run 0 $A dualbranch_atst_frame_stage2_i2_owbceVT_atst_teacher.pth VB_atst 0 20 fixed ) &
( run 0 $D standalone_dasheng_stage2_i2_owbceVT_owBt_teacher.pth VW_owBt 4 10 tol; run 0 $D standalone_dasheng_stage2_i2_owbceVT_owBt_teacher.pth VB_owBt 4 10 fixed ) &
( run 1 $Bt standalone_beats_stage2_owbceVT_bS2_teacher.pth VW_bS2 4 10 tol; run 1 $Bt standalone_beats_stage2_owbceVT_bA_teacher.pth VW_bA 4 10 tol ) &
( run 1 $Bt standalone_beats_stage2_owbceVT_bS2_teacher.pth VB_bS2 4 10 fixed; run 1 $Bt standalone_beats_stage2_owbceVT_bA_teacher.pth VB_bA 4 10 fixed ) &
wait
echo done > work/tsowbce_gate_done.txt
