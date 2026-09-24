from pathlib import Path

PPO_TYPES = {
    "learning_rate": float,
    "clip_epsilon": float,
    "value_coef": float,
    "entropy_coef": float,
    "max_grad_norm": float,
    "epochs": int,
    "minibatch_size": int,
    "kl_target": float,
}
TRAINING_TYPES = {
    "iterations": int,
    "rollout_battles": int,
    "eval_battles": int,
    "eval_interval": int,
    "team_seed": int,
}

REWARD_TYPES = {
    "outcome_weight": float,
    "own_hp_weight": float,
    "enemy_hp_weight": float,
    "speed_weight": float,
    "speed_scale": float,
}

RUNNING_TYPES = {
    "url": str,
    "format": str,
    "workers": int,
    "threads": int,
    "checkpoint_dir": Path,
    "wandb_project": str,
    "wandb_entity": str,
    "battle_lanes": int,
    "gpu_batch_size": int,
    "gpu_batch_wait_ms": float,
}
MODEL_TYPES = {
    "species_count": int,
    "form_count": int,
    "move_count": int,
    "item_count": int,
    "ability_count": int,
    "status_count": int,
    "weather_count": int,
    "tactical_event_type_count": int,
    "history_ref_count": int,
    "history_reason_count": int,
    "species_embedding_dim": int,
    "form_embedding_dim": int,
    "move_embedding_dim": int,
    "item_embedding_dim": int,
    "ability_embedding_dim": int,
    "status_embedding_dim": int,
    "weather_embedding_dim": int,
    "event_type_embedding_dim": int,
    "history_ref_embedding_dim": int,
    "history_reason_embedding_dim": int,
    "pokemon_hidden_dim": int,
    "pokemon_output_dim": int,
    "field_hidden_dim": int,
    "field_output_dim": int,
    "tactical_entry_hidden_dim": int,
    "tactical_entry_output_dim": int,
    "history_hidden_dim": int,
    "trunk_hidden_dim": int,
    "trunk_output_dim": int,
    "action_count": int,
    "seed": int,
}

POOL_TYPES = {
    "semi_random_share": float,
    "win_rate_threshold": float,
    "random_share_increment": float,
}
