import torch
from torch import Tensor, nn

from ai_cif.model.config import ModelConfig


class PokemonEncoder(nn.Module):
    def __init__(self, config: ModelConfig, numeric_feature_count: int) -> None:
        super().__init__()

        self.species_embedding = nn.Embedding(
            config.species_count, config.species_embedding_dim, padding_idx=0
        )
        self.form_embedding = nn.Embedding(
            config.form_count, config.form_embedding_dim, padding_idx=0
        )
        self.move_embedding = nn.Embedding(
            config.move_count, config.move_embedding_dim, padding_idx=0
        )
        self.item_embedding = nn.Embedding(
            config.item_count, config.item_embedding_dim, padding_idx=0
        )
        self.ability_embedding = nn.Embedding(
            config.ability_count, config.ability_embedding_dim, padding_idx=0
        )
        self.status_embedding = nn.Embedding(
            config.status_count, config.status_embedding_dim, padding_idx=0
        )

        self.side_embedding = nn.Embedding(2, 4)

        input_dim = (
            2 * config.species_embedding_dim
            + config.form_embedding_dim
            + 4 * config.move_embedding_dim
            + config.item_embedding_dim
            + config.ability_embedding_dim
            + config.status_embedding_dim
            + 4
            + numeric_feature_count
        )

        self.network = nn.Sequential(
            nn.Linear(input_dim, config.pokemon_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.pokemon_hidden_dim, config.pokemon_output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        *,
        base_species: Tensor,
        species: Tensor,
        form: Tensor,
        moves: Tensor,
        item: Tensor,
        ability: Tensor,
        status: Tensor,
        numeric: Tensor,
    ) -> Tensor:
        batch_size = species.shape[0]

        side = torch.tensor(
            [0] * 6 + [1] * 6, dtype=torch.long, device=species.device
        )
        side = side.unsqueeze(0).expand(batch_size, -1)

        base_species_emb = self.species_embedding(base_species)
        species_emb = self.species_embedding(species)
        form_emb = self.form_embedding(form)

        moves_emb = self.move_embedding(moves)
        moves_emb = moves_emb.flatten(start_dim=-2)

        item_emb = self.item_embedding(item)
        ability_emb = self.ability_embedding(ability)
        status_emb = self.status_embedding(status)
        side_emb = self.side_embedding(side)

        x = torch.cat(
            (
                base_species_emb,
                species_emb,
                form_emb,
                moves_emb,
                item_emb,
                ability_emb,
                status_emb,
                side_emb,
                numeric,
            ),
            dim=-1,
        )

        return self.network(x)


class SimplifiedHistoryEncoder(nn.Module):
    def __init__(self, config: ModelConfig, numeric_feature_count: int) -> None:
        super().__init__()

        self.kind_embedding = nn.Embedding(
            config.tactical_event_type_count,
            config.event_type_embedding_dim,
            padding_idx=0,
        )
        self.move_embedding = nn.Embedding(
            config.move_count, config.move_embedding_dim, padding_idx=0
        )
        self.species_embedding = nn.Embedding(
            config.species_count, config.species_embedding_dim, padding_idx=0
        )
        self.form_embedding = nn.Embedding(
            config.form_count, config.form_embedding_dim, padding_idx=0
        )
        self.ref_embedding = nn.Embedding(
            config.history_ref_count,
            config.history_ref_embedding_dim,
            padding_idx=0,
        )
        self.reason_embedding = nn.Embedding(
            config.history_reason_count,
            config.history_reason_embedding_dim,
            padding_idx=0,
        )

        input_dim = (
            config.event_type_embedding_dim
            + config.move_embedding_dim
            + config.species_embedding_dim
            + config.form_embedding_dim
            + 2 * config.history_ref_embedding_dim
            + config.history_reason_embedding_dim
            + numeric_feature_count
        )

        self.entry_encoder = nn.Sequential(
            nn.Linear(input_dim, config.tactical_entry_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                config.tactical_entry_hidden_dim,
                config.tactical_entry_output_dim,
            ),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=config.tactical_entry_output_dim,
            hidden_size=config.history_hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        *,
        kind: Tensor,
        move: Tensor,
        species: Tensor,
        form: Tensor,
        actor: Tensor,
        target: Tensor,
        reason: Tensor,
        numeric: Tensor,
        length: Tensor,
    ) -> Tensor:
        x = torch.cat(
            (
                self.kind_embedding(kind),
                self.move_embedding(move),
                self.species_embedding(species),
                self.form_embedding(form),
                self.ref_embedding(actor),
                self.ref_embedding(target),
                self.reason_embedding(reason),
                numeric,
            ),
            dim=-1,
        )

        x = self.entry_encoder(x)

        # pack_padded_sequence cannot accept zero lengths.
        actual_lengths = length.long()
        safe_lengths = actual_lengths.clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, safe_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, hidden = self.gru(packed)

        result = hidden[-1]

        # A truly empty history should have no learned GRU state.
        empty = actual_lengths == 0
        result = result.masked_fill(empty.unsqueeze(-1), 0.0)

        return result


class FieldEncoder(nn.Module):
    def __init__(self, config: ModelConfig, numeric_feature_count: int) -> None:
        super().__init__()

        self.weather_embedding = nn.Embedding(
            config.weather_count, config.weather_embedding_dim, padding_idx=0
        )

        self.network = nn.Sequential(
            nn.Linear(
                numeric_feature_count + config.weather_embedding_dim,
                config.field_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(config.field_hidden_dim, config.field_output_dim),
            nn.ReLU(),
        )

    def forward(self, *, weather: Tensor, numeric: Tensor) -> Tensor:
        weather_emb = self.weather_embedding(weather)

        x = torch.cat((weather_emb, numeric), dim=-1)

        return self.network(x)
