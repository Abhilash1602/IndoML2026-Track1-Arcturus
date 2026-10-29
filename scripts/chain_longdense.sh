#!/bin/bash
# Oct 14: long-clip dense-window probe. GPU1 only (shares GPU1 with the user's vLLM server; ~9 GB free).
cd "$(dirname "$0")/.." && source .venv/bin/activate
for ov in 8 5; do
  CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=16 python -m temp.extract_long_dense $ov > work/log_longdense_$ov.log 2>&1 || continue
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 python -m temp.pipeline_v11 --member-prefix VS --no-gate --guard 0.7 --w 0.7 --vdrop 0.25 --fpthr 0.3 \
    --cache ./work/verifier/pipeline_v11_features_base_VS_ov$ov.pkl --tag v14ng_g7_vd25_fp30_ov$ov > work/log_longdense_zip_$ov.log 2>&1
done
echo done > work/longdense_done.txt
