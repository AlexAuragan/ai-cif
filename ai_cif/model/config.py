from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    species_count: int
    form_count: int
    move_count: int
    item_count: int
    ability_count: int
    status_count: int
    weather_count: int

    tactical_event_type_count: int
    history_ref_count: int
    history_reason_count: int

    species_embedding_dim: int = 16
    form_embedding_dim: int = 8
    move_embedding_dim: int = 16
    item_embedding_dim: int = 8
    ability_embedding_dim: int = 8
    status_embedding_dim: int = 8
    weather_embedding_dim: int = 8

    event_type_embedding_dim: int = 8
    history_ref_embedding_dim: int = 8
    history_reason_embedding_dim: int = 8

    pokemon_hidden_dim: int = 128
    pokemon_output_dim: int = 64

    field_hidden_dim: int = 64
    field_output_dim: int = 32

    tactical_entry_hidden_dim: int = 96
    tactical_entry_output_dim: int = 64
    history_hidden_dim: int = 128

    trunk_hidden_dim: int = 256
    trunk_output_dim: int = 128

    action_count: int = 10
