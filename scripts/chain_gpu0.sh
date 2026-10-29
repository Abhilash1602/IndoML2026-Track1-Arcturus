#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
W=work
VAANI_DEVICE=cuda:0 python -m temp.cache_refiner_data > $W/log_cache_refiner.log 2>&1 || exit 1
p=0; while read J; do (CF_JOBS=$J VAANI_DEVICE=cuda:$((p%2)) OMP_NUM_THREADS=12 python -m temp.crossfit_refiner > $W/log_cf_$p.log 2>&1 &); p=$((p+1)); done < $W/cf_jobs.txt
