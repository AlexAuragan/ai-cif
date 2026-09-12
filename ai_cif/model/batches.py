from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PokemonTensorBatch:
    # [batch, pokemon]
    species: Tensor
    item: Tensor
    ability: Tensor
    status: Tensor
    side: Tensor

    # [batch, pokemon, 4]
    moves: Tensor

    # [batch, pokemon, numeric_features]
    numeric: Tensor


@dataclass(frozen=True)
class TacticalHistoryTensorBatch:
    # [batch, history]
    event_type: Tensor
    move: Tensor

    # [batch, history, tactical_numeric_features]
    numeric: Tensor

    # True for real entries, False for padding.
    # [batch, history]
    mask: Tensor


@dataclass(frozen=True)
class BattleTensorBatch:
    pokemon: PokemonTensorBatch

    # [batch, field_numeric_features]
    field: Tensor

    history: TacticalHistoryTensorBatch

    # [batch, 10]
    legal_actions: Tensor
