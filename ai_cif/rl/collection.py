"""Battle collection: connecting clients and building dataset shards."""

import asyncio
import random
import uuid
from pathlib import Path

import torch
from showdown_sdk.classes.client import Client

from ai_cif.rl.config import (
    FORMAT,
    MAX_HISTORY,
    THREADS_PER_WORKER,
    VOCAB_GEN,
    WEBSOCKET_URL,
)
from ai_cif.rl.recording import RecordingSemiRandomHandler, resolved_examples
from ai_cif.rl.storage import atomic_save
from ai_cif.vectorization.tensorizer import (
    BattleTensorizer,
    BattleTensors,
    collate_battles,
)
from scripts.utils.battles import run_battle


async def connect_pair(
    first_handler: RecordingSemiRandomHandler,
    second_handler: RecordingSemiRandomHandler,
) -> list[Client]:
    clients = [
        Client(WEBSOCKET_URL, combat_handler=first_handler),
        Client(WEBSOCKET_URL, combat_handler=second_handler),
    ]

    for client in clients:
        client.log_manager.disable()

    tag = uuid.uuid4().hex[:12]

    try:
        await asyncio.gather(*(client.connect() for client in clients))

        await asyncio.gather(
            clients[0].login(f"BC{tag}A"), clients[1].login(f"BC{tag}B")
        )

    except BaseException:
        await asyncio.gather(
            *(client.close() for client in clients), return_exceptions=True
        )
        raise

    return clients


async def collect_shard_async(job: dict) -> tuple[str, int, int]:
    tensorizer = BattleTensorizer(max_history=MAX_HISTORY, vocab_gen=VOCAB_GEN)

    random_share = float(job["random_share"])

    handlers = [
        RecordingSemiRandomHandler(tensorizer, random_share=random_share),
        RecordingSemiRandomHandler(tensorizer, random_share=random_share),
    ]

    clients = await connect_pair(*handlers)

    shard_observations: list[BattleTensors] = []
    shard_labels: list[int] = []
    shard_battle_ids: list[int] = []

    try:
        for battle_id in job["battle_ids"]:
            for handler in handlers:
                handler.reset()

            await run_battle(
                *clients,
                fmt=FORMAT,
                team_generator_1=None,
                team_generator_2=None,
            )

            for handler in handlers:
                observations, labels = resolved_examples(handler)

                shard_observations.extend(observations)
                shard_labels.extend(labels)
                shard_battle_ids.extend([battle_id] * len(labels))

        if not shard_labels:
            raise RuntimeError("Collected shard contains no decisions")

        batch = collate_battles(shard_observations)

        atomic_save(
            {
                "observations": vars(batch),
                "labels": torch.tensor(shard_labels, dtype=torch.long),
                "battle_ids": torch.tensor(shard_battle_ids, dtype=torch.long),
            },
            Path(job["path"]),
        )

        return (job["path"], len(job["battle_ids"]), len(shard_labels))

    finally:
        await asyncio.gather(
            *(client.close() for client in clients), return_exceptions=True
        )


def collect_shard(job: dict) -> tuple[str, int, int]:
    torch.set_num_threads(THREADS_PER_WORKER)

    random.seed(job["seed"])
    torch.manual_seed(job["seed"])

    return asyncio.run(collect_shard_async(job))
