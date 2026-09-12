import torch
from torch import Tensor, nn

from ai_cif.model.batches import PokemonTensorBatch, TacticalHistoryTensorBatch
from ai_cif.model.config import ModelConfig


class PokemonEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        numeric_feature_count: int,
    ) -> None:
        super().__init__()

        self.species_embedding = nn.Embedding(
            config.species_count,
            config.species_embedding_dim,
        )
        self.move_embedding = nn.Embedding(
            config.move_count,
            config.move_embedding_dim,
        )
        self.item_embedding = nn.Embedding(
            config.item_count,
            config.item_embedding_dim,
        )
        self.ability_embedding = nn.Embedding(
            config.ability_count,
            config.ability_embedding_dim,
        )
        self.status_embedding = nn.Embedding(
            config.status_count,
            config.status_embedding_dim,
        )

        # self / opponent
        self.side_embedding = nn.Embedding(2, 4)

        input_dim = (
            config.species_embedding_dim
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
            nn.Linear(
                config.pokemon_hidden_dim,
                config.pokemon_output_dim,
            ),
            nn.ReLU(),
        )

    def forward(self, batch: PokemonTensorBatch) -> Tensor:
        species = self.species_embedding(batch.species)

        moves = self.move_embedding(batch.moves)
        moves = moves.flatten(start_dim=-2)

        item = self.item_embedding(batch.item)
        ability = self.ability_embedding(batch.ability)
        status = self.status_embedding(batch.status)
        side = self.side_embedding(batch.side)

        x = torch.cat(
            (
                species,
                moves,
                item,
                ability,
                status,
                side,
                batch.numeric,
            ),
            dim=-1,
        )

        return self.network(x)

class TacticalHistoryEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        numeric_feature_count: int,
    ) -> None:
        super().__init__()

        self.event_type_embedding = nn.Embedding(
            config.tactical_event_type_count,
            config.event_type_embedding_dim,
        )
        self.move_embedding = nn.Embedding(
            config.move_count,
            config.move_embedding_dim,
        )

        input_dim = (
            config.event_type_embedding_dim
            + config.move_embedding_dim
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

    def forward(self, batch: TacticalHistoryTensorBatch) -> Tensor:
        event_type = self.event_type_embedding(batch.event_type)
        move = self.move_embedding(batch.move)

        x = torch.cat(
            (
                event_type,
                move,
                batch.numeric,
            ),
            dim=-1,
        )

        x = self.entry_encoder(x)

        lengths = batch.mask.sum(dim=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)

        return hidden[-1]

class FieldEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        numeric_feature_count: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                numeric_feature_count,
                config.field_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                config.field_hidden_dim,
                config.field_output_dim,
            ),
            nn.ReLU(),
        )

    def forward(self, field: Tensor) -> Tensor:
        return self.network(field)
