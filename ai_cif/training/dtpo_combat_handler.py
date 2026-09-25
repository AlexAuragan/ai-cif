from typing import override

import numpy as np
import torch
from showdown_sdk.classes.combat_handler.base_handler import BaseCombatHandler
from showdown_sdk.features import battle_to_features
from showdown_sdk.models.sdk import BattleState

from ai_cif.dtpo.features import tree_features
from ai_cif.dtpo.policy import DecisionTreePolicy
from ai_cif.dtpo.value import BattleValueModel
from ai_cif.inference.combat_handler import Action
from ai_cif.training.rewards import RewardBreakdown
from ai_cif.training.trajectory import Decision, Trajectory
from ai_cif.vectorization.tensorizer import BattleTensorizer


class DTPOCombatHandler(BaseCombatHandler):
    def __init__(
        self,
        *,
        policy: DecisionTreePolicy,
        tensorizer: BattleTensorizer,
        value_model: BattleValueModel | None = None,
        device: str | torch.device = "cpu",
        sample: bool = False,
        record_trajectory: bool = False,
    ) -> None:
        if record_trajectory and value_model is None:
            raise ValueError(
                "value_model is required when record_trajectory=True"
            )

        self.policy = policy
        self.tensorizer = tensorizer
        self.value_model = value_model
        self.device = torch.device(device)
        self.sample = sample
        self.record_trajectory = record_trajectory
        self.trajectory = Trajectory()

        if self.value_model is not None:
            self.value_model.to(self.device)
            self.value_model.eval()

    def start_battle(self) -> None:
        self.trajectory = Trajectory()

    def finish_battle(
        self, outcome: float, reward_breakdown: RewardBreakdown
    ) -> Trajectory:
        if not self.record_trajectory:
            raise RuntimeError("finish_battle requires record_trajectory=True")

        reward = reward_breakdown.total

        if outcome not in {-1.0, 0.0, 1.0}:
            raise ValueError(f"Outcome must be -1, 0, or +1, got {outcome}")

        if not -1.0 <= reward <= 1.0:
            raise ValueError(f"Reward must be in [-1, 1], got {reward}")

        self.trajectory.outcome = outcome
        self.trajectory.reward = reward
        self.trajectory.reward_breakdown = reward_breakdown

        return self.trajectory

    @override
    def select_top_actions(self, battle_state: BattleState) -> list[Action]:
        features = battle_to_features(battle_state)
        tensors = self.tensorizer.tensorize(features)

        tree_input = tree_features(tensors)
        action_mask = tensors.action_mask.detach().cpu().numpy().astype(bool)

        probabilities = self.policy.probabilities(tree_input, action_mask)

        if self.sample:
            first_action, log_prob = self.policy.sample(tree_input, action_mask)
        else:
            first_action = int(np.argmax(probabilities))
            log_prob = float(np.log(max(probabilities[first_action], 1e-8)))

        if self.record_trajectory:
            if self.value_model is None:
                raise RuntimeError("Missing value model")

            value_input = torch.from_numpy(tree_input).to(
                self.device, dtype=torch.float32
            )
            value = self.value_model.predict_one(value_input)

            self.trajectory.decisions.append(
                Decision(
                    observation=tensors,
                    action=first_action,
                    log_prob=log_prob,
                    value=value,
                )
            )

        legal_indices = np.flatnonzero(action_mask)
        remaining = legal_indices[legal_indices != first_action]
        remaining = remaining[np.argsort(probabilities[remaining])[::-1]]

        ranked = [first_action, *[int(index) for index in remaining]]

        return [self._decode_action(index) for index in ranked]

    @staticmethod
    def _decode_action(index: int) -> Action:
        if 0 <= index < 4:
            return ("move", index + 1)

        if 4 <= index < 10:
            return ("switch", index - 3)

        raise ValueError(f"Invalid action index: {index}")

    @classmethod
    @override
    def select_team_order(cls) -> list[int]:
        return [1, 2, 3, 4, 5, 6]
