#!/bin/bash
# Self-training round 2 -- GPU0 only (GPU1 is running the user's Track 2 jobs; do not touch it).
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=12
# (a) relabel the Silver+Bronze pool with the v14 system (students VS), 3 shards
for k in 0 1 2; do
  SHARD=$k NSHARDS=3 MEMBER_PREFIX=VS PSEUDO_OUT=./work/pseudo_v14 python -m temp.gen_pseudo_v11 > work/log_pseudo14_$k.log 2>&1 &
done
# (b) meanwhile, the free test: 15 members = students VS + VA + v4
python -m temp.pipeline_v11 --member-prefix VS --extra-prefix VA --gated-blend --w 0.7 --tag v14xg > work/log_pipeline_v14x.log 2>&1
wait
# (c) round-2 students from the round-1 students, labels v14, pseudo share 0.4; two lanes on GPU0
COMMON="VAL_TRAIN=all GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=1e-4 MAX_EPOCHS=3 SYNTH_FRAC=0.3 PSEUDO_LABELS=./work/pseudo_v14 PSEUDO_SHARE=0.4"
run() {  # script warm_start tag owbce_height owbce_half_width
  env $COMMON WARM_START=./checkpoints/$2 OWBCE_TAG=$3 OWBCE_HEIGHT=$4 OWBCE_HALF_WIDTH=$5 python -m temp.$1 > work/log_student_$3.log 2>&1
}
( run train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceVS_owB_teacher.pth VR_owB 4 10
  run train_dasheng_stage2_i2_owbce standalone_dasheng_stage2_i2_owbceVS_owBt_teacher.pth VR_owBt 4 10
  run train_beats_stage2_owbce standalone_beats_stage2_owbceVS_bA_teacher.pth VR_bA 4 10 ) &
( run train_atst_frame_stage2_i2_owbce dualbranch_atst_frame_stage2_i2_owbceVS_atst_teacher.pth VR_atst 0 20
  run train_beats_stage2_owbce standalone_beats_stage2_owbceVS_bS2_teacher.pth VR_bS2 4 10 ) &
wait
ls -la checkpoints/*owbceVR_*teacher.pth > work/round2_students_done.txt
# (d) v15: round-2 students alone, and with the round-1 students as 5 extra members
python -m temp.pipeline_v11 --member-prefix VR --gated-blend --w 0.7 1.0 --tag v15g > work/log_pipeline_v15.log 2>&1
python -m temp.pipeline_v11 --member-prefix VR --extra-prefix VS --gated-blend --w 0.7 --tag v15xg > work/log_pipeline_v15x.log 2>&1
