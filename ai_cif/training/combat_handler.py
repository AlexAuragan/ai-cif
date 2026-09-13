from typing import override

import torch
from showdown_sdk.features import battle_to_features
from showdown_sdk.models.sdk import BattleState

from ai_cif.inference.combat_handler import Action, NeuralCombatHandler
from ai_cif.model.model import BattleModel
from ai_cif.training.rewards import RewardBreakdown
from ai_cif.training.trajectory import Decision, Trajectory
from ai_cif.vectorization.tensorizer import BattleTensorizer


class TrainingCombatHandler(NeuralCombatHandler):
    """Stochastic policy used to collect RL trajectories."""

    def __init__(
        self,
        model: BattleModel,
        tensorizer: BattleTensorizer,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(model=model, tensorizer=tensorizer, device=device)

        self.trajectory = Trajectory()

    def start_battle(self) -> None:
        """Reset trajectory state before starting a new battle."""
        self.trajectory = Trajectory()

    def finish_battle(
        self, outcome: float, reward_breakdown: RewardBreakdown
    ) -> Trajectory:
        """Attach the terminal outcome and return the completed trajectory."""
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

        # Keep the stored observation on CPU.
        tensors = self.tensorizer.tensorize(features)

        # Only move the temporary inference batch to the model device.
        batch = tensors.batched().to(self.device)

        with torch.inference_mode():
            logits, value = self.model(batch)

        if logits.shape != (1, 10):
            raise RuntimeError(
                "Expected policy logits shape (1, 10), "
                f"got {tuple(logits.shape)}"
            )

        if value.shape != (1,):
            raise RuntimeError(
                f"Expected value shape (1,), got {tuple(value.shape)}"
            )

        legal_mask = batch.action_mask[0]
        legal_indices = torch.where(legal_mask)[0]

        if legal_indices.numel() == 0:
            raise RuntimeError("Model received a state with no legal actions")

        # Sample only among legal actions.
        legal_logits = logits[0, legal_indices]

        distribution = torch.distributions.Categorical(logits=legal_logits)

        sampled_position = distribution.sample()
        sampled_index = legal_indices[sampled_position]

        log_prob = distribution.log_prob(sampled_position)

        action_index = int(sampled_index.item())

        self.trajectory.decisions.append(
            Decision(
                observation=tensors,
                action=action_index,
                log_prob=float(log_prob.item()),
                value=float(value[0].item()),
            )
        )

        # The sampled action is tried first.
        #
        # If Showdown rejects it, the SDK can continue through the rest
        # of this ranking without calling the policy again.
        remaining_mask = legal_indices != sampled_index
        remaining_indices = legal_indices[remaining_mask]

        if remaining_indices.numel() > 0:
            remaining_scores = logits[0, remaining_indices]

            order = torch.argsort(remaining_scores, descending=True)

            remaining_indices = remaining_indices[order]

        ranked_indices = [
            action_index,
            *[int(index.item()) for index in remaining_indices],
        ]

        return [self._decode_action(index) for index in ranked_indices]
