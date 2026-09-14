from dataclasses import dataclass, field

from ai_cif.training.rewards import RewardBreakdown
from ai_cif.vectorization.tensorizer import BattleTensors


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


## Trajectories functions
def summarize_trajectories(trajectories: list[Trajectory]) -> tuple[int, int, int, int]:
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


def mean_trajectory_reward(trajectories: list[Trajectory]) -> float:
    if not trajectories:
        raise ValueError("Cannot summarize empty trajectories")

    rewards: list[float] = []

    for trajectory in trajectories:
        if trajectory.reward is None:
            raise ValueError("All trajectories must have a reward")

        rewards.append(float(trajectory.reward))

    return sum(rewards) / len(rewards)
