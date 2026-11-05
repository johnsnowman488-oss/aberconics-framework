#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/aberconics-framework
export PYTHONPATH=code/python
outdir=code/python/d2c/progress/babi_qa/sweep16
mkdir -p "$outdir"
for variant in full no_slow collapsed_gamma; do
  echo "START variant=$variant $(date -u +%FT%TZ)"
  python3 -m d2c.experiments.babi_qa \
    --task 1 \
    --variant "$variant" \
    --dim 16 \
    --seeds 20 \
    --output-dir "$outdir"
  echo "DONE variant=$variant $(date -u +%FT%TZ)"
done
