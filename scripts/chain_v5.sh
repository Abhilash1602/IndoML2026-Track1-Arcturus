#!/bin/bash
# Oct 14: after the fold models -> OOF features -> verifier v5 gate -> (PASS) fit + test features + zip + safety. GPU1 only.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=16
until [ -f work/folds_done.txt ]; do sleep 120; done
python -m temp.cache_fold_feats oof > work/log_v5_oof.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=1 python -m temp.verifier5 > work/log_v5_gate.log 2>&1 || exit 1
MODE=""
for m in avg v5; do grep -q "GATE $m: PASS" work/log_v5_gate.log && [ -z "$MODE" ] && MODE=$m; done
if grep -q "GATE avg: PASS" work/log_v5_gate.log && grep -q "GATE v5: PASS" work/log_v5_gate.log; then
  MODE=$(python - <<'P'
import re
t = open("work/log_v5_gate.log").read()
sc = {m: float(re.search(rf"\[{m}\s*\] AUC [0-9.]+ \| drops \d+ \| pooled ([0-9.]+)", t).group(1)) for m in ("v5", "avg")}
print(max(sc, key=sc.get))
P
)
fi
[ -z "$MODE" ] && { echo "gate failed" > work/v5_done.txt; exit 0; }
python -m temp.verifier5 fit >> work/log_v5_gate.log 2>&1
python -m temp.cache_fold_feats test > work/log_v5_test.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES="" python -m temp.pipeline_v11 --member-prefix VS --no-gate --guard 0.7 --w 0.7 --vdrop 0.30 --fpthr 0.3 \
    --verifier5 $MODE --tag v14ng_g7_vd30_fp30_v5$MODE > work/log_v5_zip.log 2>&1
CUDA_VISIBLE_DEVICES="" python -m temp.sweep_safety submission_v14ng_g7_vd30_fp30_w70.zip submission_v14ng_g7_vd30_fp30_v5${MODE}_w70.zip >> work/log_v5_zip.log 2>&1
echo "done $MODE" > work/v5_done.txt
