#!/bin/bash
# Track 1 training chain, in the order the shipped checkpoints were produced.
# Run one step at a time:  bash train/run_training.sh <step>   (steps: data synth stage1 stage2 valft
#                                                               refiners sedt postproc edgetable pseudo students)
# Hyperparameters are documented in train/config.yaml; the scripts read them from src/config.py and
# the environment variables set here. GPU selection: VAANI_DEVICE=cuda:0 (default: GPU with most free memory).
# Note: GPU training is not bit-deterministic, so a retrain gives statistically similar members,
# not byte-identical checkpoints. The checkpoints used for the submission are listed in CHECKPOINTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
step=${1:?usage: bash train/run_training.sh <step>}

FT="VAL_TRAIN=all GOLD_ALL=1 ATST_BASE_LR=2e-5 DISTILLED_LR=1e-4 MAX_EPOCHS=3 SYNTH_FRAC=0.3"

case "$step" in
  data)       # fixed, seeded 70/30 split of the official validation set (measurement only)
              python -m temp.make_val_split ;;
  synth)      # Vaani-derived synthetic training data: event bank from Gold, 15,000 mixtures on Silver speech
              python -m temp.build_event_bank
              python -m temp.gen_synth ;;
  stage1)     # head training on frozen backbones (Gold + synthetic + pseudo-labelled Silver/Bronze)
              python -m temp.gen_ensemble_pseudo_labels          # pseudo-labels for Silver/Bronze
              python -m temp.train_standalone_dasheng_smoke_i2   # -> standalone_dasheng_smoke_i2.pth
              python -m temp.train_standalone_beats_smoke        # -> standalone_beats_smoke.pth
              python -m temp.train_atst_frame_stage1_full ;;     # -> dualbranch_atst_frame_stage1_full.pth
  stage2)     # backbone unfreezing: layer-wise LR decay + Mean Teacher + ICT on unlabelled audio
              python -m temp.train_atst_frame_stage2                                        # dualbranch_atst_frame_stage2
              OWBCE_HEIGHT=12 OWBCE_HALF_WIDTH=20 OWBCE_TAG=B python -m temp.train_dasheng_stage2_i2_owbce  # dasheng owbceB(+_teacher)
              python -m temp.train_beats_stage2                                             # standalone_beats_stage2
              OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 OWBCE_TAG=A python -m temp.train_beats_stage2_owbce ;;     # beats owbceA
  valft)      # fine-tuning with Gold + synthetic + all official validation (EMA teachers kept) -> owbceVA_*
              env $FT WARM_START=./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth OWBCE_TAG=VA_owB OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 python -m temp.train_dasheng_stage2_i2_owbce
              env $FT WARM_START=./checkpoints/standalone_dasheng_stage2_i2_owbceB_teacher.pth OWBCE_TAG=VA_owBt OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 python -m temp.train_dasheng_stage2_i2_owbce
              env $FT WARM_START=./checkpoints/dualbranch_atst_frame_stage2.pth OWBCE_TAG=VA_atst OWBCE_HEIGHT=0 OWBCE_HALF_WIDTH=20 python -m temp.train_atst_frame_stage2_i2_owbce
              env $FT WARM_START=./checkpoints/standalone_beats_stage2.pth OWBCE_TAG=VA_bS2 OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 python -m temp.train_beats_stage2_owbce
              env $FT WARM_START=./checkpoints/standalone_beats_stage2_owbceA.pth OWBCE_TAG=VA_bA OWBCE_HEIGHT=4 OWBCE_HALF_WIDTH=10 python -m temp.train_beats_stage2_owbce ;;
  refiners)   # 5 boundary refiners on the official validation set (inputs: frozen Stage-2 member logits + log-mel)
              python -m temp.cache_refiner_data
              python -m temp.train_refiner_val                                   # train70 refiner
              for s in 0 1 2 3; do REFINER_ALL=1 REFINER_SEED=$s python -m temp.train_refiner_val; done ;;
  sedt)       # DETR-style set detector on Gold (verifier evidence)
              python -m temp.cache_sedt_features
              SEDT_ANCHORS=1 python -m temp.train_sedt ;;                         # ship as sedt_gold_anc_s0_alone.pth
  postproc)   # false-positive filter + verifier (GBMs, PCAs) on Gold held-out + validation events
              python -m temp.cache_verifier_features
              rm -f checkpoints/postproc_v11.pkl
              python -c "import temp.pipeline_v11 as p; p.postproc_models()" ;;
  edgetable)  # gated edge-correction table, fitted on Gold held-out
              python tools/fit_edge_table.py --out checkpoints/edge_correction_v5.json ;;
  pseudo)     # self-training labels: the complete (teacher) system on 79,526 Silver + Bronze clips
              python -m temp.gen_pseudo_v11 ;;
  students)   # self-trained students -> owbceVS_* (the 5 members used with the Stage-2 members at inference)
              bash scripts/chain_students.sh ;;
  *) echo "unknown step: $step"; exit 1 ;;
esac
