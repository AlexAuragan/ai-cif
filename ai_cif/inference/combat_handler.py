from typing import override

import torch
from showdown_sdk.classes.combat_handler.base_handler import BaseCombatHandler
from showdown_sdk.features import battle_to_features
from showdown_sdk.models.sdk import BattleState

from ai_cif.model.model import BattleModel
from ai_cif.vectorization.tensorizer import BattleTensorizer

Action = tuple[str, int]


class NeuralCombatHandler(BaseCombatHandler):
    """Use a BattleModel to rank every legal Showdown action."""

    def __init__(
        self,
        model: BattleModel,
        tensorizer: BattleTensorizer,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tensorizer = tensorizer

        self.model.eval()

    @override
    def select_top_actions(
        self,
        battle_state: BattleState,
    ) -> list[Action]:
        features = battle_to_features(battle_state)

        tensors = self.tensorizer.tensorize(
            features,
            device=self.device,
        )
        batch = tensors.batched()

        with torch.inference_mode():
            logits, _ = self.model(batch)

        if logits.shape != (1, 10):
            raise RuntimeError(
                f"Expected policy logits shape (1, 10), got {tuple(logits.shape)}"
            )

        legal_mask = batch.action_mask[0]
        legal_indices = torch.where(legal_mask)[0]

        if legal_indices.numel() == 0:
            raise RuntimeError("Model received a state with no legal actions")

        scores = logits[0, legal_indices]

        ranking = torch.argsort(
            scores,
            descending=True,
        )

        ranked_indices = legal_indices[ranking]

        return [
            self._decode_action(int(index.item()))
            for index in ranked_indices
        ]

    @staticmethod
    def _decode_action(index: int) -> Action:
        if 0 <= index < 4:
            # Neural action 0..3 -> Showdown move 1..4.
            return ("move", index + 1)

        if 4 <= index < 10:
            # Neural action 4..9 -> Showdown party slot 1..6.
            return ("switch", index - 3)

        raise ValueError(f"Invalid action index: {index}")

    @staticmethod
    @override
    def select_team_order() -> list[int]:
        return [1, 2, 3, 4, 5, 6]
