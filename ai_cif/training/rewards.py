from dataclasses import dataclass

from showdown_sdk.classes.dt import BattleResult
from torch import math


@dataclass(frozen=True)
class RewardConfig:
    outcome_weight: float
    own_hp_weight: float
    enemy_hp_weight: float
    speed_weight: float
    speed_scale: float

    def __post_init__(self) -> None:
        weights = (
            self.outcome_weight,
            self.own_hp_weight,
            self.enemy_hp_weight,
            self.speed_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ValueError("Reward weights must be non-negative")

        total_weight = sum(weights)

        if total_weight > 1.0:
            raise ValueError("Reward weights must sum to <= 1")

        if self.speed_scale <= 0:
            raise ValueError("speed_scale must be > 0")


DEFAULT_REWARD_CONFIG = RewardConfig(
    outcome_weight=0.80,
    own_hp_weight=0.075,
    enemy_hp_weight=0.075,
    speed_weight=0.05,
    speed_scale=40.0,
)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float

    outcome: float
    own_hp: float
    enemy_damage: float
    speed: float

    own_hp_fraction: float
    enemy_hp_fraction: float
    move_count: int


def _own_hp_fraction(result: BattleResult) -> float:
    team = result.final_state["team"]

    if not isinstance(team, list):
        raise TypeError("final_state['team'] must be a list")

    if not team:
        raise ValueError("Final state contains an empty team")

    total = 0.0

    for pokemon in team:
        if not isinstance(pokemon, dict):
            raise TypeError("Team entry must be a dict")

        curr_hp = pokemon["curr_hp"]
        max_hp = pokemon["max_hp"]

        if isinstance(curr_hp, bool) or not isinstance(curr_hp, int):
            raise TypeError("curr_hp must be an int")

        if isinstance(max_hp, bool) or not isinstance(max_hp, int):
            raise TypeError("max_hp must be an int")

        if max_hp <= 0:
            raise ValueError("max_hp must be positive")

        hp_fraction = curr_hp / max_hp
        total += max(0.0, min(1.0, hp_fraction))

    return total / len(team)


def _enemy_hp_fraction(result: BattleResult) -> float:
    enemy_team = result.final_state["enemy_team"]

    if not isinstance(enemy_team, list):
        raise TypeError("final_state['enemy_team'] must be a list")

    if not enemy_team:
        raise ValueError("Final state contains an empty enemy team")

    total = 0.0

    for pokemon in enemy_team:
        if not isinstance(pokemon, dict):
            raise TypeError("Enemy team entry must be a dict")

        hp_percent = pokemon["curr_hp_percent"]

        if isinstance(hp_percent, bool) or not isinstance(hp_percent, int):
            raise TypeError("curr_hp_percent must be an int")

        hp_fraction = hp_percent / 100.0
        total += max(0.0, min(1.0, hp_fraction))

    return total / len(enemy_team)


def breakdown_for(
    result: BattleResult,
    outcome: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> RewardBreakdown:
    own_hp_fraction = _own_hp_fraction(result)
    enemy_hp_fraction = _enemy_hp_fraction(result)

    enemy_damage_fraction = 1.0 - enemy_hp_fraction

    speed_score = 0.0

    if outcome > 0.0:
        speed_score = math.exp(-result.move_count / config.speed_scale)

    outcome_reward = config.outcome_weight * outcome

    own_hp_reward = config.own_hp_weight * own_hp_fraction

    enemy_damage_reward = config.enemy_hp_weight * enemy_damage_fraction

    speed_reward = config.speed_weight * speed_score

    total = outcome_reward + own_hp_reward + enemy_damage_reward + speed_reward

    return RewardBreakdown(
        total=total,
        outcome=outcome_reward,
        own_hp=own_hp_reward,
        enemy_damage=enemy_damage_reward,
        speed=speed_reward,
        own_hp_fraction=own_hp_fraction,
        enemy_hp_fraction=enemy_hp_fraction,
        move_count=result.move_count,
    )
