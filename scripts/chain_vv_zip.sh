#!/bin/bash
# After the VV students: test extraction with the VV students replacing v4 in the ensemble mean (originals still give
# features), 5 s long-clip overlap as the best system, then the zip on the best settings + safety check.
cd "$(dirname "$0")/.." && source .venv/bin/activate
until [ -f work/vv_done.txt ]; do sleep 120; done
for t in atst owB owBt bS2 bA; do ls checkpoints/*owbceVV_${t}_teacher.pth > /dev/null || { echo "missing VV_$t" > work/vv_zip_done.txt; exit 1; }; done
used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
GPU=$([ "$used0" -lt 2000 ] && echo 0 || echo 1)
CUDA_VISIBLE_DEVICES=$GPU VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=16 python - > work/log_vv_extract.log 2>&1 <<'P'
import sys
import release.v6_inference as v6
v6.OVERLAP_SEC = 5.0
sys.argv = ["pipeline_v11", "--member-prefix", "VS", "--replace-v4", "VV", "--no-gate", "--guard", "0.7", "--w", "0.7",
            "--vdrop", "0.25", "--fpthr", "0.2", "--cache", "./work/verifier/pipeline_v11_features_base_VS_rVV_ov5.pkl",
            "--tag", "v14ng_ov5_fp20_vv"]
from temp.pipeline_v11 import main
main()
P
CUDA_VISIBLE_DEVICES="" python -m temp.sweep_safety submission_v14ng_ov5_fp20_w70.zip submission_v14ng_ov5_fp20_vv_w70.zip >> work/log_vv_extract.log 2>&1
echo done > work/vv_zip_done.txt
