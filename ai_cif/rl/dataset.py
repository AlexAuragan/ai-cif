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

    # First recover shards that were completely written by an earlier
    # interrupted run but never made it into manifest.json.
    for random_share in SEMI_RANDOM_SHARES:
        share_tag = f"{round(random_share * 100):03d}"

        for split, target_count in (
            ("train", TRAIN_BATTLES_PER_SHARE),
            ("validation", VALIDATION_BATTLES_PER_SHARE),
        ):
            for start in range(0, target_count, SHARD_BATTLES):
                stop = min(start + SHARD_BATTLES, target_count)
                expected_battles = stop - start

                name = f"{split}_share_{share_tag}_{start:06d}.pt"
                path = DATA_DIR / name

                if name in manifest[split]:
                    continue

                if not path.is_file():
                    continue

                payload = torch.load(
                    path, map_location="cpu", weights_only=True
                )

                battle_ids = payload["battle_ids"]
                labels = payload["labels"]

                actual_battles = int(torch.unique(battle_ids).numel())

                if actual_battles != expected_battles:
                    raise RuntimeError(
                        f"Existing orphan shard {path} contains "
                        f"{actual_battles} battles, expected {expected_battles}"
                    )

                print(f"Recovering completed shard: {name}")

                manifest[split].append(name)
                manifest["decisions"] = int(manifest.get("decisions", 0)) + int(
                    labels.numel()
                )

                battle_key = f"{split}_battles"
                manifest[battle_key] = (
                    int(manifest.get(battle_key, 0)) + actual_battles
                )

    for split in ("train", "validation"):
        manifest[split].sort()

    save_manifest(manifest)

    battle_id = next_battle_id(manifest)

    jobs: list[dict] = []

    for share_index, random_share in enumerate(SEMI_RANDOM_SHARES):
        share_tag = f"{round(random_share * 100):03d}"

        for split, target_count in (
            ("train", TRAIN_BATTLES_PER_SHARE),
            ("validation", VALIDATION_BATTLES_PER_SHARE),
        ):
            for start in range(0, target_count, SHARD_BATTLES):
                stop = min(start + SHARD_BATTLES, target_count)
                count = stop - start

                name = f"{split}_share_{share_tag}_{start:06d}.pt"

                # This exact shard is already complete.
                if name in manifest[split]:
                    continue

                path = DATA_DIR / name

                if path.exists():
                    raise RuntimeError(
                        f"Shard exists but was not recovered: {path}"
                    )

                battle_ids = list(range(battle_id, battle_id + count))
                battle_id += count

                jobs.append(
                    {
                        "path": str(path.resolve()),
                        "name": name,
                        "split": split,
                        "battle_ids": battle_ids,
                        "random_share": random_share,
                        "seed": (SEED + share_index * 100_000 + start),
                    }
                )

    random.Random(SEED).shuffle(jobs)

    total_missing = sum(len(job["battle_ids"]) for job in jobs)
    finished_battles = 0
    new_decisions = 0

    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = {pool.submit(collect_shard, job): job for job in jobs}

        try:
            for future in as_completed(futures):
                job = futures[future]

                path, battle_count, state_count = future.result()

                finished_battles += battle_count
                new_decisions += state_count

                split = job["split"]
                name = job["name"]

                manifest[split].append(name)
                manifest[split].sort()

                manifest["decisions"] = (
                    int(manifest.get("decisions", 0)) + state_count
                )

                battle_key = f"{split}_battles"

                manifest[battle_key] = (
                    int(manifest.get(battle_key, 0)) + battle_count
                )

                # Commit progress immediately.
                save_manifest(manifest)

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

    manifest["train_battles_per_share"] = TRAIN_BATTLES_PER_SHARE
    manifest["validation_battles_per_share"] = VALIDATION_BATTLES_PER_SHARE

    manifest["train_battles"] = TRAIN_BATTLES
    manifest["validation_battles"] = VALIDATION_BATTLES

    save_manifest(manifest)

    print()
    print("Dataset complete.")

    for random_share in SEMI_RANDOM_SHARES:
        print(
            f"  share={random_share:.1f}: "
            f"{TRAIN_BATTLES_PER_SHARE} train + "
            f"{VALIDATION_BATTLES_PER_SHARE} validation"
        )

    print(f"Total: {TRAIN_BATTLES + VALIDATION_BATTLES} battles")


def save_manifest(manifest: dict) -> None:
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
