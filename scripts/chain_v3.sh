#!/bin/bash
# Oct 14: after the clean VG models finish -> verifier v3 features, gate, and (if PASS) the test zip. GPU1 only.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=16
until [ -f work/vg_done.txt ]; do sleep 120; done
python -m temp.cache_vg_logits > work/log_v3_cache.log 2>&1 || exit 1
python -m temp.verifier3 > work/log_v3_gate.log 2>&1 || exit 1
grep -q "GATE: PASS" work/log_v3_gate.log || { echo "gate failed" > work/v3_done.txt; exit 0; }
python -m temp.verifier3 fit >> work/log_v3_gate.log 2>&1
python -m temp.cache_vg_test > work/log_v3_test.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES="" python -m temp.pipeline_v11 --member-prefix VS --no-gate --guard 0.7 --w 0.7 --vdrop 0.30 --fpthr 0.3 \
    --verifier3 --tag v14ng_g7_vd30_fp30_v3 > work/log_v3_zip.log 2>&1
CUDA_VISIBLE_DEVICES="" python -m temp.sweep_safety submission_v14ng_g7_vd30_fp30_w70.zip submission_v14ng_g7_vd30_fp30_v3_w70.zip >> work/log_v3_zip.log 2>&1
echo done > work/v3_done.txt
