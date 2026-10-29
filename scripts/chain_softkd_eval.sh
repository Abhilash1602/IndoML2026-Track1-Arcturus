#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
until [ -f work/softkd_gate_done.txt ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 VAANI_DEVICE=cuda:0 OMP_NUM_THREADS=12 GATE_A=VK GATE_B=VB python -m temp.eval_tsowbce_gate _teacher > work/log_softkd_gate.log 2>&1
