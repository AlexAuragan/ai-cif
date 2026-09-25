from dataclasses import dataclass, field

import torch
from torch import Tensor

from ai_cif.training.rewards import RewardBreakdown
from ai_cif.vectorization.tensorizer import (
    BattleBatch,
    BattleTensors,
    collate_battles,
)


@dataclass(frozen=True)
class Decision:
    observation: BattleTensors
    action: int
    log_prob: float
    value: float


@dataclass
class Trajectory:
    decisions: list[Decision] = field(default_factory=list)
    outcome: float | None = None
    reward: float | None = None
    reward_breakdown: RewardBreakdown | None = None

    def clear(self) -> None:
        self.decisions.clear()
        self.outcome = None
        self.reward = None
        self.reward_breakdown = None


@dataclass(frozen=True)
class PackedRollout:
    observations: BattleBatch

    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor

    trajectory_lengths: Tensor
    outcomes: Tensor
    rewards: Tensor

    reward_breakdowns: tuple[RewardBreakdown, ...]

    def __post_init__(self) -> None:
        decision_count = self.observations.batch_size

        decision_tensors = {
            "actions": self.actions,
            "old_log_probs": self.old_log_probs,
            "old_values": self.old_values,
        }

        for name, tensor in decision_tensors.items():
            if tensor.ndim != 1:
                raise ValueError(
                    f"{name} must be 1-D, got shape {tuple(tensor.shape)}"
                )

            if tensor.shape[0] != decision_count:
                raise ValueError(
                    f"{name} has {tensor.shape[0]} rows, "
                    f"expected {decision_count}"
                )

        if self.trajectory_lengths.ndim != 1:
            raise ValueError("trajectory_lengths must be 1-D")

        battle_count = self.trajectory_lengths.shape[0]

        if self.outcomes.shape != (battle_count,):
            raise ValueError(
                f"outcomes has shape {tuple(self.outcomes.shape)}, "
                f"expected {(battle_count,)}"
            )

        if self.rewards.shape != (battle_count,):
            raise ValueError(
                f"rewards has shape {tuple(self.rewards.shape)}, "
                f"expected {(battle_count,)}"
            )

        if len(self.reward_breakdowns) != battle_count:
            raise ValueError(
                "reward_breakdowns count does not match trajectory_lengths"
            )

        if torch.any(self.trajectory_lengths <= 0):
            raise ValueError("Packed rollout contains an empty trajectory")

        if int(self.trajectory_lengths.sum().item()) != decision_count:
            raise ValueError(
                "Sum of trajectory_lengths does not match decision count"
            )

    @property
    def decision_count(self) -> int:
        return self.observations.batch_size

    @property
    def battle_count(self) -> int:
        return int(self.trajectory_lengths.shape[0])

    @classmethod
    def from_trajectories(cls, trajectories: list[Trajectory]) -> PackedRollout:
        if not trajectories:
            raise ValueError("Cannot pack an empty trajectory list")

        decisions: list[Decision] = []
        reward_breakdowns: list[RewardBreakdown] = []

        lengths: list[int] = []
        outcomes: list[float] = []
        rewards: list[float] = []

        for trajectory in trajectories:
            if not trajectory.decisions:
                raise ValueError("Cannot pack an empty trajectory")

            if trajectory.outcome is None:
                raise ValueError("All trajectories must have an outcome")

            if trajectory.reward is None:
                raise ValueError("All trajectories must have a reward")

            if trajectory.reward_breakdown is None:
                raise ValueError(
                    "All trajectories must have a reward breakdown"
                )

            decisions.extend(trajectory.decisions)

            lengths.append(len(trajectory.decisions))

            outcomes.append(float(trajectory.outcome))

            rewards.append(float(trajectory.reward))

            reward_breakdowns.append(trajectory.reward_breakdown)

        observations = collate_battles(
            [decision.observation for decision in decisions]
        )

        actions = torch.tensor(
            [decision.action for decision in decisions], dtype=torch.long
        )

        old_log_probs = torch.tensor(
            [decision.log_prob for decision in decisions], dtype=torch.float32
        )

        old_values = torch.tensor(
            [decision.value for decision in decisions], dtype=torch.float32
        )

        trajectory_lengths = torch.tensor(lengths, dtype=torch.long)

        outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return cls(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            trajectory_lengths=trajectory_lengths,
            outcomes=outcomes_tensor,
            rewards=rewards_tensor,
            reward_breakdowns=tuple(reward_breakdowns),
        )

    @classmethod
    def concat(cls, chunks: list[PackedRollout]) -> PackedRollout:
        if not chunks:
            raise ValueError("Cannot concatenate an empty rollout list")

        return cls(
            observations=BattleBatch.cat(
                [chunk.observations for chunk in chunks]
            ),
            actions=torch.cat([chunk.actions for chunk in chunks], dim=0),
            old_log_probs=torch.cat(
                [chunk.old_log_probs for chunk in chunks], dim=0
            ),
            old_values=torch.cat([chunk.old_values for chunk in chunks], dim=0),
            trajectory_lengths=torch.cat(
                [chunk.trajectory_lengths for chunk in chunks], dim=0
            ),
            outcomes=torch.cat([chunk.outcomes for chunk in chunks], dim=0),
            rewards=torch.cat([chunk.rewards for chunk in chunks], dim=0),
            reward_breakdowns=tuple(
                breakdown
                for chunk in chunks
                for breakdown in chunk.reward_breakdowns
            ),
        )


## Trajectories functions
def summarize_trajectories(
    trajectories: list[Trajectory] | PackedRollout,
) -> tuple[int, int, int, int]:
    if isinstance(trajectories, PackedRollout):
        wins = int((trajectories.outcomes == 1.0).sum().item())
        losses = int((trajectories.outcomes == -1.0).sum().item())
        ties = trajectories.battle_count - wins - losses
        return (wins, losses, ties, trajectories.decision_count)

    wins = 0
    losses = 0
    ties = 0
    decisions = 0

    for trajectory in trajectories:
        decisions += len(trajectory.decisions)

        if trajectory.outcome == 1.0:
            wins += 1
        elif trajectory.outcome == -1.0:
            losses += 1
        else:
            ties += 1

    return wins, losses, ties, decisions


def mean_trajectory_reward(
    trajectories: list[Trajectory] | PackedRollout,
) -> float:
    if isinstance(trajectories, PackedRollout):
        if trajectories.battle_count == 0:
            raise ValueError("Cannot summarize empty trajectories")

        return float(trajectories.rewards.mean().item())

    if not trajectories:
        raise ValueError("Cannot summarize empty trajectories")

    rewards: list[float] = []

    for trajectory in trajectories:
        if trajectory.reward is None:
            raise ValueError("All trajectories must have a reward")

        rewards.append(float(trajectory.reward))

    return sum(rewards) / len(rewards)
