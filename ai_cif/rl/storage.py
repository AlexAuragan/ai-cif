"""Atomic checkpoint / dataset persistence helpers."""

from pathlib import Path

import torch


def atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    torch.save(payload, temporary)
    temporary.replace(path)
