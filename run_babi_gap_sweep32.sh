#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/aberconics-framework
export PYTHONPATH=code/python
python3 diagnose_babi_readout.py \
  --dim 32 \
  --seeds 20 \
  --epochs 10 \
  --lrs 0.005 \
  --hidden 32 \
  --variants full,no_slow,collapsed_gamma \
  --output-dir code/python/d2c/progress/babi_qa/gap_sweep32
