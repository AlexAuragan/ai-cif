"""Dataset manifest construction, validation, and lookup."""

import json
import shutil

import torch

from ai_cif.rl.config import (
    DATA_DIR,
    DATASET_SCHEMA,
    FORMAT,
    MAX_HISTORY,
    SEED,
    SEMI_RANDOM_SHARES,
    VOCAB_GEN,
)


def build_manifest() -> dict:
    return {
        "schema": DATASET_SCHEMA,
        "format": FORMAT,
        "source": "live_semi_random_mixture",
        "max_history": MAX_HISTORY,
        "vocab_gen": VOCAB_GEN,
        "seed": SEED,
        "semi_random_shares": SEMI_RANDOM_SHARES,
        "train_battles_per_share": 0,
        "validation_battles_per_share": 0,
        "train_battles": 0,
        "validation_battles": 0,
        "train": [],
        "validation": [],
        "decisions": 0,
    }


def manifest_matches_configuration(manifest: dict) -> bool:
    expected = {
        "schema": DATASET_SCHEMA,
        "format": FORMAT,
        "source": "live_semi_random_mixture",
        "max_history": MAX_HISTORY,
        "vocab_gen": VOCAB_GEN,
        "seed": SEED,
        "semi_random_shares": SEMI_RANDOM_SHARES,
    }

    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            return False

    for split in ("train", "validation"):
        names = manifest.get(split)

        if not isinstance(names, list):
            return False

        for name in names:
            if not (DATA_DIR / name).is_file():
                return False

    return True


def load_cached_manifest() -> dict | None:
    manifest_path = DATA_DIR / "manifest.json"

    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not manifest_matches_configuration(manifest):
        raise RuntimeError(
            f"Existing dataset at {DATA_DIR} is incompatible "
            "with the current configuration."
        )

    return manifest


def next_battle_id(manifest: dict) -> int:
    maximum = -1

    for split in ("train", "validation"):
        for name in manifest[split]:
            payload = torch.load(
                DATA_DIR / name, map_location="cpu", weights_only=True
            )

            battle_ids = payload["battle_ids"]

            if battle_ids.numel() == 0:
                continue

            maximum = max(maximum, int(battle_ids.max().item()))

    return maximum + 1


def reset_dataset_directory() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
