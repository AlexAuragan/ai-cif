"""Configuration values for the semi-random behavior-cloning pipeline.

As with the other training scripts in this repository, there is deliberately no
CLI configuration: the important values are defined here as module constants.
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

WEBSOCKET_URL = (
    os.environ.get("DEFAULT_WEBSOCKET_URL")
    or "ws://127.0.0.1:8000/showdown/websocket"
)

FORMAT = "gen4randombattle"

DATA_DIR = Path("data/bc-semi-random-gen4")
OUTPUT_DIR = Path("checkpoints/bc-semi-random-gen4")

SEMI_RANDOM_SHARES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

TRAIN_BATTLES_PER_SHARE = 2000
VALIDATION_BATTLES_PER_SHARE = 500

TRAIN_BATTLES = len(SEMI_RANDOM_SHARES) * TRAIN_BATTLES_PER_SHARE

VALIDATION_BATTLES = len(SEMI_RANDOM_SHARES) * VALIDATION_BATTLES_PER_SHARE

WORKERS = 24
SHARD_BATTLES = 16
THREADS_PER_WORKER = 1

EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
PATIENCE = 4
TARGET_AGREEMENT = 0.90

MAX_HISTORY = 32
VOCAB_GEN = 4
SEED = 42

STARTING_WEIGHTS: Path | None = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_SCHEMA = 2
