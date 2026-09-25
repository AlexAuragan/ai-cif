from dataclasses import dataclass
from pathlib import Path


@dataclass
class DTPOConfig:
    learning_rate: float = 1.0
    clip_epsilon: float = 0.2
    gamma: float = 0.99

    max_depth: int | None = None
    max_leaf_nodes: int = 32
    policy_updates: int = 1

    value_hidden_dim: int = 128
    value_learning_rate: float = 3e-4
    value_epochs: int = 4
    value_minibatch_size: int = 256

    normalize_advantage: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")

        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")

        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be > 0")

        if self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be >= 2")

        if self.policy_updates <= 0:
            raise ValueError("policy_updates must be > 0")

        if self.value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be > 0")

        if self.value_learning_rate <= 0:
            raise ValueError("value_learning_rate must be > 0")

        if self.value_epochs <= 0:
            raise ValueError("value_epochs must be > 0")

        if self.value_minibatch_size <= 0:
            raise ValueError("value_minibatch_size must be > 0")


@dataclass
class DTPORunningConfig:
    url: str
    format: str

    checkpoint_dir: Path

    device: str

    wandb_enabled: bool
    wandb_project: str | None
    wandb_entity: str | None
    wandb_name: str
