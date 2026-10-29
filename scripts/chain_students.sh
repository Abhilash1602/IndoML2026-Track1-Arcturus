#!/bin/bash
# Iteration-4 self-training: 5 students = the v6 VA members fine-tuned with Gold + all validation +
# 30% of the Natural share from Silver+Bronze pseudo-labelled by the full v11g_w70 system.
cd "$(dirname "$0")/.." && source .venv/bin/activate
until [ $(ls work/pseudo_v11/shard*of4.npz 2>/dev/null | wc -l) -ge 4 ]; do sleep 60; done
COMMON="VAL_TRAIN=all GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=1e-4 MAX_EPOCHS=3 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v11 PSEUDO_SHARE=0.3 OMP_NUM_THREADS=12"
run() {  # gpu script warm_start tag owbce_height owbce_half_width
  env $COMMON VAANI_DEVICE=cuda:$1 WARM_START=./checkpoints/$3 OWBCE_TAG=$4 OWBCE_HEIGHT=$5 OWBCE_HALF_WIDTH=$6 \
      python -m temp.$2 > work/log_student_$4.log 2>&1
}
( run 0 train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceVA_owB_teacher.pth VS_owB 4 10
  run 0 train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceVA_owBt_teacher.pth VS_owBt 4 10 ) &
run 0 train_atst_frame_stage2_i2_owbce dualbranch_atst_frame_stage2_i2_owbceVA_atst_teacher.pth VS_atst 0 20 &
run 1 train_beats_stage2_owbce standalone_beats_stage2_owbceVA_bS2_teacher.pth VS_bS2 4 10 &
run 1 train_beats_stage2_owbce standalone_beats_stage2_owbceVA_bA_teacher.pth VS_bA 4 10 &
wait
ls -la checkpoints/*owbceVS_*teacher.pth > work/students_done.txt
# v14: the full recipe with the students in place of the VA members
python -m temp.pipeline_v11 --member-prefix VS --gated-blend --w 0.7 0.65 1.0 --tag v14g > work/log_pipeline_v14.log 2>&1
