import asyncio
from pathlib import Path

import torch
from showdown_sdk.classes.client import Client
from showdown_sdk.exceptions import UserNotFoundError

from ai_cif.inference.combat_handler import NeuralCombatHandler
from scripts.train import MODEL_CONFIG, TENSORIZER, create_model

WEBSOCKET_URL = "ws://127.0.0.1:8000/showdown/websocket"

BOT_NAME = "AI-cif"
TARGET_NAME = "AlexAuragan"
FORMAT = "gen1randombattle"

CHECKPOINT_PATH = Path("data/models/gen1randombattle/red-hp-1-150-best.pt")


def load_bot() -> NeuralCombatHandler:
    device = torch.device("cpu")

    model = create_model(device, MODEL_CONFIG)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return NeuralCombatHandler(
        model=model, tensorizer=TENSORIZER, device=device, verbose=True
    )


async def main() -> None:
    handler = load_bot()

    client = Client(WEBSOCKET_URL, combat_handler=handler)

    try:
        await client.connect()
        await client.login(BOT_NAME)

        print(f"Loaded {CHECKPOINT_PATH}")
        print(f"Logged in as {BOT_NAME}")

        while True:
            try:
                await client.challenge(TARGET_NAME, FORMAT, timeout=60)
                break
            except UserNotFoundError:
                await asyncio.sleep(10)

        print(f"Challenge sent to {TARGET_NAME}")

        result = await client.wait_for_battle_end(timeout=300)

        print(result)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
