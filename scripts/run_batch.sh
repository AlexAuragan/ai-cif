#!/usr/bin/env bash
set -e

# for i in {1..10}; do
#  uv run scripts/train.py \
#    --wandb-name "pearl-$i" \
#    --wandb-group pearl \
#    --starting-weight "data/models/crystal/pearl__01190.pt" \
#    --starting-iteration 11920
# done
#

uv run scripts/train.py \
    --wandb-name "pearl-1" \
    --wandb-group pearl-sweep \
    --starting-weights "checkpoints/pearl-next-1/iteration_02000.pt" \
    --starting-iteration 2000 \
    --set-training \
      iterations=10 \
    --set-ppo \
      entropy_coef="0.02" \
      gae_lambda="0.90" \
      epochs="6"

uv run scripts/train.py \
    --wandb-name "pearl-2" \
    --wandb-group pearl-sweep \
    --starting-weights "checkpoints/pearl-next-1/iteration_02000.pt" \
    --starting-iteration 2000 \
    --set-training \
      iterations=10 \
    --set-ppo \
      entropy_coef="0.04" \
      gae_lambda="0.95" \
      epochs="6"

uv run scripts/train.py \
    --wandb-name "pearl-3" \
    --wandb-group pearl-sweep \
    --starting-weights "checkpoints/pearl-next-1/iteration_02000.pt" \
    --starting-iteration 2000 \
    --set-training \
      iterations=10 \
    --set-ppo \
      entropy_coef="0.02" \
      gae_lambda="0.95" \
      epochs="6" \
      learning-rate="0.0004"

uv run scripts/train.py \
    --wandb-name "pearl-next-1-2" \
    --wandb-group pearl-sweep \
    --starting-weights "checkpoints/pearl-next-1-2/iteration_03000.pt" \
    --starting-iteration 3000 \
    --set-training \
      iterations=1000 \
    --set-ppo \
      entropy_coef="0.02" \
      gae_lambda="0.95" \
      epochs="6" \
      learning_rate="0.0005"
