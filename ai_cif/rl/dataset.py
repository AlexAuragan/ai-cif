"""Battle dataset generation across all semi-random shares."""

import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from ai_cif.rl.collection import collect_shard
from ai_cif.rl.config import (
    DATA_DIR,
    FORMAT,
    SEED,
    SEMI_RANDOM_SHARES,
    SHARD_BATTLES,
    THREADS_PER_WORKER,
    TRAIN_BATTLES,
    TRAIN_BATTLES_PER_SHARE,
    VALIDATION_BATTLES,
    VALIDATION_BATTLES_PER_SHARE,
    WEBSOCKET_URL,
    WORKERS,
)
from ai_cif.rl.manifest import (
    build_manifest,
    load_cached_manifest,
    next_battle_id,
)


def generate_battles() -> None:
    """Generate a balanced dataset across all semi-random shares."""

    if FORMAT != "gen4randombattle":
        raise ValueError("This script is configured for gen4randombattle.")

    if WORKERS <= 0:
        raise ValueError("WORKERS must be positive")

    if SHARD_BATTLES <= 0:
        raise ValueError("SHARD_BATTLES must be positive")

    torch.set_num_threads(max(1, THREADS_PER_WORKER))

    random.seed(SEED)
    torch.manual_seed(SEED)

    print()
    print("Battle generation")
    print("-----------------")
    print(f"Server: {WEBSOCKET_URL}")
    print(f"Format: {FORMAT}")
    print(f"Dataset: {DATA_DIR}")
    print(
        "Semi-random shares: "
        + ", ".join(f"{share:.1f}" for share in SEMI_RANDOM_SHARES)
    )

    print(
        f"Per share: "
        f"{TRAIN_BATTLES_PER_SHARE} train + "
        f"{VALIDATION_BATTLES_PER_SHARE} validation"
    )

    print(f"Total: {TRAIN_BATTLES + VALIDATION_BATTLES} battles")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_cached_manifest()

    if manifest is None:
        manifest = build_manifest()

    existing_train_per_share = int(manifest.get("train_battles_per_share", 0))

    existing_validation_per_share = int(
        manifest.get("validation_battles_per_share", 0)
    )

    if existing_train_per_share > TRAIN_BATTLES_PER_SHARE:
        raise ValueError(
            "Existing dataset has more training "
            "battles per share than requested."
        )

    if existing_validation_per_share > VALIDATION_BATTLES_PER_SHARE:
        raise ValueError(
            "Existing dataset has more validation "
            "battles per share than requested."
        )

    missing_train_per_share = TRAIN_BATTLES_PER_SHARE - existing_train_per_share

    missing_validation_per_share = (
        VALIDATION_BATTLES_PER_SHARE - existing_validation_per_share
    )

    if missing_train_per_share == 0 and missing_validation_per_share == 0:
        print(
            "Dataset already contains all "
            f"{TRAIN_BATTLES + VALIDATION_BATTLES} battles."
        )
        return

    print(
        f"Existing per share: "
        f"{existing_train_per_share} train + "
        f"{existing_validation_per_share} validation"
    )

    print(
        f"Generating per share: "
        f"{missing_train_per_share} train + "
        f"{missing_validation_per_share} validation"
    )

    battle_id = next_battle_id(manifest)

    jobs: list[dict] = []

    new_train_shards: list[str] = []
    new_validation_shards: list[str] = []

    for share_index, random_share in enumerate(SEMI_RANDOM_SHARES):
        share_tag = f"{round(random_share * 100):03d}"

        for split, existing_count, target_count, new_shards in (
            (
                "train",
                existing_train_per_share,
                TRAIN_BATTLES_PER_SHARE,
                new_train_shards,
            ),
            (
                "validation",
                existing_validation_per_share,
                VALIDATION_BATTLES_PER_SHARE,
                new_validation_shards,
            ),
        ):
            for start in range(existing_count, target_count, SHARD_BATTLES):
                stop = min(start + SHARD_BATTLES, target_count)

                count = stop - start

                name = f"{split}_share_{share_tag}_{start:06d}.pt"

                path = DATA_DIR / name

                if path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing shard: {path}"
                    )

                battle_ids = list(range(battle_id, battle_id + count))

                battle_id += count

                new_shards.append(name)

                jobs.append(
                    {
                        "path": str(path.resolve()),
                        "battle_ids": battle_ids,
                        "random_share": random_share,
                        "seed": (SEED + share_index * 100_000 + start),
                    }
                )

    random.Random(SEED).shuffle(jobs)

    total_missing = len(SEMI_RANDOM_SHARES) * (
        missing_train_per_share + missing_validation_per_share
    )

    finished_battles = 0
    new_decisions = 0

    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = [pool.submit(collect_shard, job) for job in jobs]

        try:
            for future in as_completed(futures):
                (battle_count, state_count) = future.result()

                finished_battles += battle_count
                new_decisions += state_count

                print(
                    f"Collected "
                    f"{finished_battles}/{total_missing} "
                    f"new battles; "
                    f"{new_decisions} decisions",
                    flush=True,
                )

        except BaseException:
            for future in futures:
                future.cancel()

            pool.terminate_workers()
            raise

    manifest["train"].extend(new_train_shards)

    manifest["validation"].extend(new_validation_shards)

    manifest["train_battles_per_share"] = TRAIN_BATTLES_PER_SHARE

    manifest["validation_battles_per_share"] = VALIDATION_BATTLES_PER_SHARE

    manifest["train_battles"] = TRAIN_BATTLES

    manifest["validation_battles"] = VALIDATION_BATTLES

    manifest["decisions"] = int(manifest.get("decisions", 0)) + new_decisions

    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Dataset complete.")

    for random_share in SEMI_RANDOM_SHARES:
        print(
            f"  share={random_share:.1f}: "
            f"{TRAIN_BATTLES_PER_SHARE} train + "
            f"{VALIDATION_BATTLES_PER_SHARE} validation"
        )

    print(f"Total: {TRAIN_BATTLES + VALIDATION_BATTLES} battles")
