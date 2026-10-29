#!/bin/bash
# Oct 15: the v4 half of the ensemble retrained as students (round-1 recipe that produced the VS members:
# Gold-all + all validation + v11 pseudo-labels 0.3, 3 epochs, EMA teacher). GPU0 (idle, user OK) + GPU1.
cd "$(dirname "$0")/.." && source .venv/bin/activate
COMMON="VAL_TRAIN=all GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=1e-4 MAX_EPOCHS=3 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v11 PSEUDO_SHARE=0.3 OMP_NUM_THREADS=10"
run() {  # gpu script warm_start tag height half_width
  env $COMMON CUDA_VISIBLE_DEVICES=$1 VAANI_DEVICE=cuda:0 WARM_START=./checkpoints/$3 OWBCE_TAG=$4 OWBCE_HEIGHT=$5 OWBCE_HALF_WIDTH=$6 \
      python -m temp.$2 > work/log_student_$4.log 2>&1
}
run 0 train_atst_frame_stage2_i2_owbce dualbranch_atst_frame_stage2.pth VV_atst 0 20 &
run 0 train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceB.pth VV_owB 4 10 &
run 0 train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceB_teacher.pth VV_owBt 4 10 &
run 1 train_beats_stage2_owbce standalone_beats_stage2.pth VV_bS2 4 10 &
run 1 train_beats_stage2_owbce standalone_beats_stage2_owbceA.pth VV_bA 4 10 &
wait
echo done > work/vv_done.txt
