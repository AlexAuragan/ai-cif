import numpy as np
import torch
import torch.nn.functional as F

from ai_cif.vectorization.tensorizer import (
    CANT_REASON_VOCAB_SIZE,
    FIELD_NUMERIC_DIM,
    HISTORY_KIND_VOCAB_SIZE,
    HISTORY_NUMERIC_DIM,
    HISTORY_REF_VOCAB_SIZE,
    MOVE_CATEGORY_COUNT,
    MOVE_NUMERIC_DIM,
    MOVES_PER_POKEMON,
    POKEMON_NUMERIC_DIM,
    POKEMON_SLOTS,
    STATUS_VOCAB_SIZE,
    TYPE_COUNT,
    WEATHER_VOCAB_SIZE,
    BattleBatch,
    BattleTensors,
)

BASE_STATS_DIM = 6
ACTION_COUNT = 10


def tree_feature_dim(max_history: int) -> int:
    pokemon_features = (
        POKEMON_SLOTS * POKEMON_NUMERIC_DIM
        + POKEMON_SLOTS * TYPE_COUNT
        + POKEMON_SLOTS * BASE_STATS_DIM
        + POKEMON_SLOTS * MOVES_PER_POKEMON * TYPE_COUNT
        + POKEMON_SLOTS * MOVES_PER_POKEMON * MOVE_CATEGORY_COUNT
        + POKEMON_SLOTS * MOVES_PER_POKEMON * MOVE_NUMERIC_DIM
        + POKEMON_SLOTS
        + POKEMON_SLOTS * STATUS_VOCAB_SIZE
    )

    field_features = WEATHER_VOCAB_SIZE + FIELD_NUMERIC_DIM

    history_features = (
        max_history * HISTORY_NUMERIC_DIM
        + max_history * HISTORY_KIND_VOCAB_SIZE
        + max_history * HISTORY_REF_VOCAB_SIZE * 2
        + max_history * CANT_REASON_VOCAB_SIZE
        + max_history
        + 1
    )

    return pokemon_features + field_features + history_features + ACTION_COUNT


def _one_hot(values: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(values.long(), num_classes=classes).float()


def tree_features_batch(observations: BattleBatch) -> np.ndarray:
    batch_size = observations.batch_size

    pieces = [
        observations.pokemon_numeric.float().reshape(batch_size, -1),
        observations.pokemon_types.float().reshape(batch_size, -1),
        observations.pokemon_base_stats.float().reshape(batch_size, -1),
        observations.move_types.float().reshape(batch_size, -1),
        observations.move_categories.float().reshape(batch_size, -1),
        observations.move_numeric.float().reshape(batch_size, -1),
        observations.pokemon_mask.float().reshape(batch_size, -1),
        _one_hot(observations.status_ids, STATUS_VOCAB_SIZE).reshape(
            batch_size, -1
        ),
        _one_hot(observations.weather_id, WEATHER_VOCAB_SIZE).reshape(
            batch_size, -1
        ),
        observations.field_numeric.float().reshape(batch_size, -1),
        observations.history_numeric.float().reshape(batch_size, -1),
        _one_hot(observations.history_kind, HISTORY_KIND_VOCAB_SIZE).reshape(
            batch_size, -1
        ),
        _one_hot(observations.history_actor, HISTORY_REF_VOCAB_SIZE).reshape(
            batch_size, -1
        ),
        _one_hot(observations.history_target, HISTORY_REF_VOCAB_SIZE).reshape(
            batch_size, -1
        ),
        _one_hot(observations.history_reason, CANT_REASON_VOCAB_SIZE).reshape(
            batch_size, -1
        ),
        observations.history_mask.float().reshape(batch_size, -1),
        observations.history_length.float().reshape(batch_size, 1),
        observations.action_mask.float().reshape(batch_size, -1),
    ]

    features = torch.cat(pieces, dim=1)

    return features.detach().cpu().numpy().astype(np.float32, copy=False)


def tree_features(observation: BattleTensors) -> np.ndarray:
    return tree_features_batch(observation.batched())[0]
