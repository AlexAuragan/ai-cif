from typing import Literal

import torch
from torch import Tensor, nn

from ai_cif.model.config import ModelConfig
from ai_cif.model.encoders import (
    FieldEncoder,
    PokemonEncoder,
    SimplifiedHistoryEncoder,
)
from ai_cif.vectorization.tensorizer import (
    CANT_REASON_VOCAB_SIZE,
    FIELD_NUMERIC_DIM,
    HISTORY_KIND_VOCAB_SIZE,
    HISTORY_NUMERIC_DIM,
    HISTORY_REF_VOCAB_SIZE,
    POKEMON_NUMERIC_DIM,
    STATUS_VOCAB_SIZE,
    WEATHER_VOCAB_SIZE,
    BattleBatch,
    BattleTensorizer,
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
            config, pokemon_numeric_feature_count
        )

        self.field_encoder = FieldEncoder(config, field_numeric_feature_count)

        self.history_encoder = SimplifiedHistoryEncoder(
            config, tactical_numeric_feature_count
        )

        state_dim = (
            12 * config.pokemon_output_dim
            + config.field_output_dim
            + config.history_hidden_dim
        )

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, config.trunk_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.trunk_hidden_dim, config.trunk_output_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(
            config.trunk_output_dim, config.action_count
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.trunk_output_dim, 1), nn.Tanh()
        )

    def forward(self, batch: BattleBatch) -> tuple[Tensor, Tensor]:
        pokemon = self.pokemon_encoder(
            base_species=batch.base_species_ids,
            species=batch.species_ids,
            form=batch.form_ids,
            pokemon_types=batch.pokemon_types,
            pokemon_base_stats=batch.pokemon_base_stats,
            moves=batch.move_ids,
            move_types=batch.move_types,
            move_categories=batch.move_categories,
            move_numeric=batch.move_numeric,
            item=batch.item_ids,
            ability=batch.ability_ids,
            status=batch.status_ids,
            numeric=batch.pokemon_numeric,
        )

        pokemon = pokemon.flatten(start_dim=1)

        field = self.field_encoder(
            weather=batch.weather_id, numeric=batch.field_numeric
        )

        history = self.history_encoder(
            kind=batch.history_kind,
            move=batch.history_move,
            species=batch.history_species,
            form=batch.history_form,
            actor=batch.history_actor,
            target=batch.history_target,
            reason=batch.history_reason,
            numeric=batch.history_numeric,
            length=batch.history_length,
        )

        state = torch.cat((pokemon, field, history), dim=-1)

        hidden = self.trunk(state)

        logits = self.policy_head(hidden)

        logits = logits.masked_fill(
            ~batch.action_mask.bool(), torch.finfo(logits.dtype).min
        )

        value = self.value_head(hidden).squeeze(-1)

        return logits, value


class TransformerBattleModel(nn.Module):
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
            config, pokemon_numeric_feature_count
        )

        self.field_encoder = FieldEncoder(config, field_numeric_feature_count)

        self.history_encoder = SimplifiedHistoryEncoder(
            config, tactical_numeric_feature_count
        )

        d_model = config.transformer_dim

        # Existing encoders produce representations with different widths.
        # Project them all into the common Transformer token dimension.
        self.pokemon_projection = nn.Linear(config.pokemon_output_dim, d_model)

        self.field_projection = nn.Linear(config.field_output_dim, d_model)

        self.history_projection = nn.Linear(config.history_hidden_dim, d_model)

        # One learned token whose final representation summarizes the battle.
        self.state_token = nn.Parameter(torch.zeros(1, 1, d_model))

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=config.transformer_layers
        )

        self.output_norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Linear(d_model, config.action_count)

        self.value_head = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())

        nn.init.normal_(self.state_token, mean=0.0, std=0.02)

    def forward(self, batch: BattleBatch) -> tuple[Tensor, Tensor]:
        pokemon = self.pokemon_encoder(
            base_species=batch.base_species_ids,
            species=batch.species_ids,
            form=batch.form_ids,
            pokemon_types=batch.pokemon_types,
            pokemon_base_stats=batch.pokemon_base_stats,
            moves=batch.move_ids,
            move_types=batch.move_types,
            move_categories=batch.move_categories,
            move_numeric=batch.move_numeric,
            item=batch.item_ids,
            ability=batch.ability_ids,
            status=batch.status_ids,
            numeric=batch.pokemon_numeric,
        )

        # Before:
        #
        #     pokemon.shape == [B, 12, pokemon_output_dim]
        #     pokemon = pokemon.flatten(start_dim=1)
        #
        # Now each Pokémon remains its own token.
        pokemon_tokens = self.pokemon_projection(pokemon)

        field = self.field_encoder(
            weather=batch.weather_id, numeric=batch.field_numeric
        )

        field_token = self.field_projection(field).unsqueeze(1)

        history = self.history_encoder(
            kind=batch.history_kind,
            move=batch.history_move,
            species=batch.history_species,
            form=batch.history_form,
            actor=batch.history_actor,
            target=batch.history_target,
            reason=batch.history_reason,
            numeric=batch.history_numeric,
            length=batch.history_length,
        )

        history_token = self.history_projection(history).unsqueeze(1)

        batch_size = pokemon.shape[0]

        state_token = self.state_token.expand(batch_size, -1, -1)

        # Shape:
        #
        # [B, 15, d_model]
        #
        # 1 state token
        # 12 Pokémon tokens
        # 1 field token
        # 1 history token
        tokens = torch.cat(
            (state_token, pokemon_tokens, field_token, history_token), dim=1
        )

        encoded = self.transformer(tokens)

        # The state token has attended to all Pokémon,
        # field information and battle history.
        hidden = self.output_norm(encoded[:, 0])

        logits = self.policy_head(hidden)

        logits = logits.masked_fill(
            ~batch.action_mask.bool(), torch.finfo(logits.dtype).min
        )

        value = self.value_head(hidden).squeeze(-1)

        return logits, value


def create_battle_model(
    *,
    device: str | torch.device = "cpu",
    max_history: int = 32,
    vocab_gen: int = 4,
    model_type: Literal["normal", "transformer"] = "normal",
) -> tuple[BattleModel | TransformerBattleModel, BattleTensorizer]:
    tensorizer = BattleTensorizer(max_history=max_history, vocab_gen=vocab_gen)

    config = ModelConfig(
        species_count=tensorizer.species_vocab_size,
        form_count=tensorizer.form_vocab_size,
        move_count=tensorizer.move_vocab_size,
        item_count=tensorizer.item_vocab_size,
        ability_count=tensorizer.ability_vocab_size,
        status_count=STATUS_VOCAB_SIZE,
        weather_count=WEATHER_VOCAB_SIZE,
        tactical_event_type_count=HISTORY_KIND_VOCAB_SIZE,
        history_ref_count=HISTORY_REF_VOCAB_SIZE,
        history_reason_count=CANT_REASON_VOCAB_SIZE,
    )

    if model_type == "transformer":
        model = TransformerBattleModel(
            config=config,
            pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
            field_numeric_feature_count=FIELD_NUMERIC_DIM,
            tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
        )
    else:
        model = BattleModel(
            config=config,
            pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
            field_numeric_feature_count=FIELD_NUMERIC_DIM,
            tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
        )

    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Parameters: {parameter_count:,}")
    print(f"Trainable: {trainable_parameter_count:,}")

    return model, tensorizer
