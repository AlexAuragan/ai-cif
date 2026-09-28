"""Observation recording and ranked-action resolution.

Battle generation uses only ``AsyncSemiRandomCombatHandler``-derived policies.
The handler returns its ranking unchanged. It does NOT decide which fallback
action should count as the training label.

After a battle finishes, dataset generation resolves each recorded ranking
outside the handler: try action 1, then action 2, etc. and keep the first action
allowed by the tensorizer action mask. This mirrors the intended SDK fallback
semantics without making ``async_select_top_actions`` responsible for retry
logic.
"""

from typing import override

from showdown_sdk.features import battle_to_features

from ai_cif.training.combat_handler import AsyncSemiRandomCombatHandler
from ai_cif.vectorization.tensorizer import BattleTensorizer, BattleTensors


def encode_action(action: tuple[str, int]) -> int:
    kind, slot = action

    if kind == "move" and 1 <= slot <= 4:
        return slot - 1

    if kind == "switch" and 1 <= slot <= 6:
        return slot + 3

    raise ValueError(f"Unsupported action: {action!r}")


class RecordingSemiRandomHandler(AsyncSemiRandomCombatHandler):
    """Semi-random policy that records observations and returned rankings.

    It intentionally does not decide which action in the ranking becomes the
    training target. That is resolved later by dataset generation.
    """

    def __init__(
        self, tensorizer: BattleTensorizer, random_share: float
    ) -> None:
        super().__init__(random_share=random_share)
        self.tensorizer = tensorizer
        self.random_share = random_share
        self.reset()

    def reset(self) -> None:
        self.observations: list[BattleTensors] = []
        self.rankings: list[list[tuple[str, int]]] = []

    @override
    async def async_select_top_actions(self, battle_state):
        observation = self.tensorizer.tensorize(
            battle_to_features(battle_state)
        )

        ranking = list(await super().async_select_top_actions(battle_state))

        if not ranking:
            raise RuntimeError("Semi-random handler returned an empty ranking")

        self.observations.append(observation)
        self.rankings.append(ranking)

        # Important: do not filter, reorder, retry, or choose here.
        return ranking


def resolve_ranked_action(
    observation: BattleTensors, ranking: list[tuple[str, int]]
) -> int:
    """Return the first tensorizer-legal action from a handler ranking.

    Retry semantics live here, outside RecordingSemiRandomHandler:
    top action first, then the second, then the third, and so on.
    """

    for action in ranking:
        action_index = encode_action(action)

        if bool(observation.action_mask[action_index]):
            return action_index

    raise RuntimeError(
        "Semi-random ranking contains no action accepted by the tensorizer. "
        f"ranking={ranking!r}, mask={observation.action_mask.tolist()}"
    )


def resolved_examples(
    handler: RecordingSemiRandomHandler,
) -> tuple[list[BattleTensors], list[int]]:
    if len(handler.observations) != len(handler.rankings):
        raise RuntimeError(
            "Recorder observations/rankings became desynchronized"
        )

    observations: list[BattleTensors] = []
    labels: list[int] = []

    for observation, ranking in zip(
        handler.observations, handler.rankings, strict=True
    ):
        observations.append(observation)
        labels.append(resolve_ranked_action(observation, ranking))

    return observations, labels
