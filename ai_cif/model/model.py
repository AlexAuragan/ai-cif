import torch
from torch import Tensor, nn

from ai_cif.model.batches import BattleTensorBatch
from ai_cif.model.config import ModelConfig
from ai_cif.model.encoders import (
    FieldEncoder,
    PokemonEncoder,
    TacticalHistoryEncoder,
)


class BattleModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        pokemon_numeric_feature_count: int,
        field_numeric_feature_count: int,
        tactical_numeric_feature_count: int,
    ) -> None:
        super().__init__()

        self.config = config

        self.pokemon_encoder = PokemonEncoder(
            config,
            pokemon_numeric_feature_count,
        )

        self.field_encoder = FieldEncoder(
            config,
            field_numeric_feature_count,
        )

        self.history_encoder = TacticalHistoryEncoder(
            config,
            tactical_numeric_feature_count,
        )

        state_dim = (
            12 * config.pokemon_output_dim
            + config.field_output_dim
            + config.history_hidden_dim
        )

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, config.trunk_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                config.trunk_hidden_dim,
                config.trunk_output_dim,
            ),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(
            config.trunk_output_dim,
            config.action_count,
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.trunk_output_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        batch: BattleTensorBatch,
    ) -> tuple[Tensor, Tensor]:
        pokemon = self.pokemon_encoder(batch.pokemon)
        pokemon = pokemon.flatten(start_dim=1)

        field = self.field_encoder(batch.field)
        history = self.history_encoder(batch.history)

        state = torch.cat(
            (
                pokemon,
                field,
                history,
            ),
            dim=-1,
        )

        hidden = self.trunk(state)

        logits = self.policy_head(hidden)

        logits = logits.masked_fill(
            ~batch.legal_actions.bool(),
            torch.finfo(logits.dtype).min,
        )

        value = self.value_head(hidden).squeeze(-1)

        return logits, value

if __name__ == "__main__":
    config = ModelConfig(
        species_count=200,
        move_count=300,
        item_count=32,
        ability_count=32,
        status_count=16,
        tactical_event_type_count=16,
    )

    model = BattleModel(
        config=config,
        pokemon_numeric_feature_count=16,
        field_numeric_feature_count=32,
        tactical_numeric_feature_count=16,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"{parameter_count:,} parameters")
