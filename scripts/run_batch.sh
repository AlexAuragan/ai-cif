#!/usr/bin/env bash
set -e

for i in {1..10}; do
 uv run scripts/train.py \
   --wandb-name "ruby-$i" \
   --wandb-group ruby \
   --starting-weight "data/models/crystal/crystal_${i}_00200.pt" \
   --starting-iteration 200
done

# PREFIX="${1:-water}"
# PYTHON="${PYTHON:-python}"

# START_STEP=200
# LAST_STEP=300
# STEP_SIZE=10
# MODEL_COUNT=10

# ROUNDS=$(( (LAST_STEP - START_STEP) / STEP_SIZE ))

# echo "Population: ${PREFIX}"
# echo "Models: 1..${MODEL_COUNT}"
# echo "Training: ${START_STEP} -> ${LAST_STEP}"
# echo "Rounds: ${ROUNDS}"
# echo

# for round in $(seq 1 "${ROUNDS}"); do
#     target_step=$(( START_STEP + round * STEP_SIZE ))

#     echo "============================================================"
#     echo "Round ${round}/${ROUNDS} — target step ${target_step}"
#     echo "============================================================"

#     for model_index in $(seq 1 "${MODEL_COUNT}"); do
#         wandb_name="${PREFIX}-${model_index}"

#         echo
#         echo ">>> ${wandb_name}: advancing toward step ${target_step}"

#         "${PYTHON}" scripts/train_member_10_iter.py \
#             --wandb-name "${wandb_name}" \
#             --model-index "${model_index}" \
#             --wandb-group "${PREFIX}"
#     done
# done

# echo
# echo "All population members have been run through step ${LAST_STEP}."
