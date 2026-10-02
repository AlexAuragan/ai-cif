from pathlib import Path
from typing import Literal

import torch

from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel, TransformerBattleModel
from ai_cif.vectorization.tensorizer import (
    FIELD_NUMERIC_DIM,
    HISTORY_NUMERIC_DIM,
    POKEMON_NUMERIC_DIM,
)


def create_model(
    device: torch.device,
    model_config: ModelConfig,
    model_type: Literal["mlp", "transfomer"],
    starting_weights: Path | None = None,
) -> BattleModel | TransformerBattleModel:
    torch.manual_seed(model_config.seed)
    if model_type == "mlp":
        model = BattleModel(
            config=model_config,
            pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
            field_numeric_feature_count=FIELD_NUMERIC_DIM,
            tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
        )
    elif model_type == "transfomer":
        model = TransformerBattleModel(
            config=model_config,
            pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
            field_numeric_feature_count=FIELD_NUMERIC_DIM,
            tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
        )
    else:
        raise ValueError()

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

        legacy_checkpoint = _load_model_state(model, model_state)

        if legacy_checkpoint:
            print(
                "Loaded legacy actor weights and initialized "
                "privileged critic from public critic"
            )

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
    model: BattleModel | TransformerBattleModel,
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
    model: BattleModel | TransformerBattleModel,
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


def _load_model_state(
    model: BattleModel | TransformerBattleModel,
    model_state: dict[str, torch.Tensor],
) -> bool:
    """Load model weights.

    Returns True when loading a legacy checkpoint that predates the
    privileged critic.
    """

    has_privileged_critic = any(
        name.startswith("critic_") for name in model_state
    )

    if has_privileged_critic:
        model.load_state_dict(model_state)
        return False

    result = model.load_state_dict(model_state, strict=False)

    unexpected = set(result.unexpected_keys)
    missing = set(result.missing_keys)

    expected_missing = {
        name for name in model.state_dict() if name.startswith("critic_")
    }

    if unexpected:
        raise RuntimeError(
            f"Legacy checkpoint has unexpected model keys: {sorted(unexpected)}"
        )

    if missing != expected_missing:
        raise RuntimeError(
            "Legacy checkpoint is missing unexpected model keys: "
            f"{sorted(missing - expected_missing)}"
        )

    model.reset_privileged_critic_from_public()

    return True
