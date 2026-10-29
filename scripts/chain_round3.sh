#!/bin/bash
# Round 3: + AudioSet-Strong (harmonised) -- GPU0 only, after round 2 finishes. Final form = UNGATED.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=12
until [ -f work/round2_students_done.txt ] && ! pgrep -f "chain_round2.sh" > /dev/null; do sleep 120; done
# ungated round-2 zips (cached features from the v15g build)
python -m temp.pipeline_v11 --member-prefix VR --no-gate --w 0.7 1.0 --tag v15ng > work/log_pipeline_v15ng.log 2>&1
# round-3 students: from round-1 students, labels v14 (0.4) + AudioSet-Strong harmonised (0.1)
COMMON="VAL_TRAIN=all GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=1e-4 MAX_EPOCHS=3 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4 AUDIOSET_SHARE=0.1"
run() {
  env $COMMON WARM_START=./checkpoints/$2 OWBCE_TAG=$3 OWBCE_HEIGHT=$4 OWBCE_HALF_WIDTH=$5 python -m temp.$1 > work/log_student_$3.log 2>&1
}
( run train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceVS_owB_teacher.pth VQ_owB 4 10
  run train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceVS_owBt_teacher.pth VQ_owBt 4 10
  run train_beats_stage2_owbce standalone_beats_stage2_owbceVS_bA_teacher.pth VQ_bA 4 10 ) &
( run train_atst_frame_stage2_i2_owbce dualbranch_atst_frame_stage2_i2_owbceVS_atst_teacher.pth VQ_atst 0 20
  run train_beats_stage2_owbce standalone_beats_stage2_owbceVS_bS2_teacher.pth VQ_bS2 4 10 ) &
wait
ls -la checkpoints/*owbceVQ_*teacher.pth > work/round3_students_done.txt
python -m temp.pipeline_v11 --member-prefix VQ --no-gate --w 0.7 1.0 --tag v16ng > work/log_pipeline_v16ng.log 2>&1
python -m temp.pipeline_v11 --member-prefix VQ --gated-blend --w 0.7 --tag v16g > work/log_pipeline_v16g.log 2>&1
