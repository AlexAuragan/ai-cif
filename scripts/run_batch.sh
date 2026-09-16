#!/usr/bin/env bash
set -e

for i in {1..10}; do
  uv run scripts/train.py \
    --wandb-name "blue-hp-$i" \
    --wandb-group blue \
    --set-model "seed=$i"
done
