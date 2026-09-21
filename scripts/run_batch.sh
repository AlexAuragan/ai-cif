#!/usr/bin/env bash
set -e

for i in {4..10}; do
  uv run scripts/train_gpu.py \
    --wandb-name "crystal-$i" \
    --wandb-group crystal
done
