from dataclasses import dataclass
from pathlib import Path


@dataclass
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
    battle_lanes: int = 8
    gpu_batch_size: int = 32
    gpu_batch_wait_ms: float = 0.5
