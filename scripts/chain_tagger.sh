#!/bin/bash
# Oct 14 idea #2: tagger training after token caching. GPU0 if idle (user OK), else GPU1 (shared with vLLM; never stop others).
cd "$(dirname "$0")/.." && source .venv/bin/activate
until grep -q "test done" work/log_tags_tok.log; do sleep 30; grep -qi traceback work/log_tags_tok.log && exit 1; done
used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
GPU=$([ "$used0" -lt 2000 ] && echo 0 || echo 1)
echo "tagger on GPU$GPU (GPU0 used ${used0} MiB)" > work/log_tagger.log
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=12 python -m temp.train_tagger >> work/log_tagger.log 2>&1
echo done > work/tagger_done.txt
