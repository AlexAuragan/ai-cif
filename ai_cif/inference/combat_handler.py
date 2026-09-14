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
        verbose: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tensorizer = tensorizer
        self.verbose = verbose
        self.model.eval()

    @override
    def select_top_actions(self, battle_state: BattleState) -> list[Action]:
        features = battle_to_features(battle_state)

        tensors = self.tensorizer.tensorize(features, device=self.device)
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
        probabilities = torch.softmax(scores, dim=0)

        ranking = torch.argsort(scores, descending=True)

        ranked_indices = legal_indices[ranking]

        if self.verbose:
            print(f"\nTurn {battle_state.turn} action probabilities:")

            for index, probability in zip(
                legal_indices.tolist(), probabilities.tolist(), strict=True
            ):
                if index < 4:
                    move_index = index

                    if move_index < len(battle_state.available_moves):
                        name = battle_state.available_moves[move_index].name
                    else:
                        name = f"move {move_index + 1}"

                    label = f"MOVE   {name}"

                else:
                    party_index = index - 4

                    if party_index < len(battle_state.team):
                        pokemon = battle_state.team[party_index]
                        label = f"SWITCH {pokemon.id}"
                    else:
                        label = f"SWITCH slot {party_index + 1}"

                print(
                    f"  {label:<30} "
                    f"{probability * 100:6.2f}% "
                    f"(logit={logits[0, index].item():.4f})"
                )

            selected_index = int(ranked_indices[0].item())

            if selected_index < 4:
                selected_name = battle_state.available_moves[selected_index].name
                print(f"Selected: MOVE {selected_name}")
            else:
                pokemon = battle_state.team[selected_index - 4]
                print(f"Selected: SWITCH {pokemon.id}")

        return [self._decode_action(int(index.item())) for index in ranked_indices]

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
