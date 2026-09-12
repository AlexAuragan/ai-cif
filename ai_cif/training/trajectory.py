from dataclasses import dataclass, field

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

    def clear(self) -> None:
        self.decisions.clear()
        self.outcome = None
