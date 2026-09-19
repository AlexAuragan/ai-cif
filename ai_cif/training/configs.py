from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int
    rollout_battles: int
    eval_battles: int
    eval_interval: int
    team_seed: int


@dataclass
class RunningConfig:
    url: str
    format: str
    workers: int
    threads: int
    checkpoint_dir: Path
    wandb_project: str | None
    wandb_entity: str | None
