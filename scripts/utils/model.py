from pathlib import Path

import torch

from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel
from ai_cif.vectorization.tensorizer import (
    FIELD_NUMERIC_DIM,
    HISTORY_NUMERIC_DIM,
    POKEMON_NUMERIC_DIM,
)


def create_model(
    device: torch.device,
    model_config: ModelConfig,
    starting_weights: Path | None = None,
) -> BattleModel:
    torch.manual_seed(model_config.seed)
    model = BattleModel(
        config=model_config,
        pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
        field_numeric_feature_count=FIELD_NUMERIC_DIM,
        tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
    )

    if starting_weights is not None:
        checkpoint = torch.load(
            starting_weights, map_location=device, weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Checkpoint {starting_weights} must contain a dict"
            )

        model_state = checkpoint.get("model", checkpoint)

        if not isinstance(model_state, dict):
            raise TypeError(
                f"Checkpoint {starting_weights} has invalid model state"
            )

        model.load_state_dict(model_state)
        print(f"Loaded starting weights from {starting_weights}")

    model.to(device)
    return model


def snapshot_model(model: BattleModel) -> dict[str, torch.Tensor]:
    """Create a stable CPU snapshot that can be sent to rollout processes."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_checkpoint(
    *,
    path: Path,
    model: BattleModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    *,
    path: Path,
    model: BattleModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Restore ``model`` and ``optimizer`` from a checkpoint.

    Accepts both the dict written by :func:`save_checkpoint` and a bare
    ``state_dict`` of model weights. The optimizer state is optional, so
    model-only checkpoints can be used as starting weights. Returns the
    iteration stored in the checkpoint, or ``0`` when it is absent.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} must contain a dict")

    model_state = checkpoint.get("model", checkpoint)

    if not isinstance(model_state, dict):
        raise TypeError(f"Checkpoint {path} has invalid model state")

    model.load_state_dict(model_state)

    optimizer_state = checkpoint.get("optimizer")

    if optimizer_state is not None:
        if not isinstance(optimizer_state, dict):
            raise TypeError(f"Checkpoint {path} has invalid optimizer state")

        optimizer.load_state_dict(optimizer_state)

    iteration = checkpoint.get("iteration", 0)

    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError(
            f"Checkpoint {path} has invalid iteration {iteration!r}"
        )

    return iteration
