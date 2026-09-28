"""Behavior-cloning training loop over the generated dataset."""

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ai_cif.model.model import create_battle_model
from ai_cif.rl.config import (
    BATCH_SIZE,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    FORMAT,
    LEARNING_RATE,
    MAX_HISTORY,
    OUTPUT_DIR,
    PATIENCE,
    SEED,
    STARTING_WEIGHTS,
    TARGET_AGREEMENT,
    VOCAB_GEN,
)
from ai_cif.rl.manifest import load_cached_manifest
from ai_cif.rl.metrics import Metrics
from ai_cif.rl.storage import atomic_save
from ai_cif.vectorization.tensorizer import BattleBatch


def epoch_pass(
    model, paths: list[Path], *, optimizer: torch.optim.Optimizer | None = None
) -> dict:
    training = optimizer is not None

    model.train(training)

    paths = list(paths)

    if training:
        random.shuffle(paths)

    metrics = Metrics()

    with torch.set_grad_enabled(training):
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=True)

            batch = BattleBatch(**payload["observations"]).to(DEVICE)

            labels = payload["labels"].to(DEVICE)

            if batch.batch_size != labels.numel():
                raise ValueError(
                    f"Observation/label count mismatch in {path}: "
                    f"{batch.batch_size} != {labels.numel()}"
                )

            if training:
                order = torch.randperm(labels.numel(), device=DEVICE)
            else:
                order = torch.arange(labels.numel(), device=DEVICE)

            for indices in order.split(BATCH_SIZE):
                inputs = batch.index_select(indices)
                targets = labels[indices]

                target_is_legal = inputs.action_mask.gather(
                    1, targets[:, None]
                ).squeeze(1)

                if not bool(target_is_legal.all()):
                    raise ValueError(
                        f"Dataset contains an illegal label in {path}"
                    )

                logits, _ = model(inputs)

                ce = torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none"
                )

                choices = inputs.action_mask.sum(-1) > 1

                if bool(choices.any()):
                    policy_loss = ce[choices].mean()
                else:
                    policy_loss = ce.mean()

                if not bool(torch.isfinite(policy_loss)):
                    raise FloatingPointError(f"Non-finite BC loss in {path}")

                if training:
                    optimizer.zero_grad(set_to_none=True)

                    policy_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()

                metrics.add(
                    logits.detach().argmax(-1),
                    targets,
                    inputs.action_mask,
                    inputs.field_numeric[:, 3].bool(),
                    ce.detach(),
                )

    return metrics.result()


def load_starting_weights(model) -> None:
    if STARTING_WEIGHTS is None:
        return

    if not STARTING_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {STARTING_WEIGHTS}"
        )

    checkpoint = torch.load(
        STARTING_WEIGHTS, map_location="cpu", weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Starting checkpoint must contain a dict")

    model_state = checkpoint.get("model", checkpoint)

    if not isinstance(model_state, dict):
        raise TypeError("Starting checkpoint has invalid model state")

    model.load_state_dict(model_state, strict=True)

    print(f"Loaded starting weights from {STARTING_WEIGHTS}")


def save_best_checkpoint(model, *, epoch: int, validation: dict) -> None:
    atomic_save(
        {
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "iteration": 0,
            "bc_epoch": epoch,
            "model_config": asdict(model.config),
            "bc_validation": validation,
            "bc_format": FORMAT,
            "bc_policy": "semi_random",
            "source_weights": (
                str(STARTING_WEIGHTS) if STARTING_WEIGHTS is not None else None
            ),
            "note": (
                "Model-only behavior-cloning checkpoint; "
                "optimizer state intentionally omitted."
            ),
        },
        OUTPUT_DIR / "best.pt",
    )


def train_model() -> None:
    """Train the BattleModel from the already generated dataset."""

    print()
    print("Behavior cloning")
    print("----------------")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    manifest = load_cached_manifest()

    if manifest is None:
        raise FileNotFoundError(
            f"No complete generated dataset at {DATA_DIR}. "
            "generate_battles() must complete first."
        )

    random.seed(SEED)
    torch.manual_seed(SEED)

    model, _ = create_battle_model(
        device=DEVICE, max_history=MAX_HISTORY, vocab_gen=VOCAB_GEN
    )

    load_starting_weights(model)

    # There are no value targets in this dataset.
    for parameter in model.value_head.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
    )

    train_paths = [DATA_DIR / name for name in manifest["train"]]

    validation_paths = [DATA_DIR / name for name in manifest["validation"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "metrics.jsonl"

    metrics_path.write_text("", encoding="utf-8")

    best_key = (-1.0, float("-inf"))

    stale_epochs = 0

    # Epoch zero gives us the untrained / starting-weight baseline.
    for epoch in range(EPOCHS + 1):
        started = time.perf_counter()

        train_metrics = None

        if epoch > 0:
            train_metrics = epoch_pass(model, train_paths, optimizer=optimizer)

        validation = epoch_pass(model, validation_paths)

        row = {
            "epoch": epoch,
            "seconds": (time.perf_counter() - started),
            "train": train_metrics,
            "validation": validation,
        }

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        print(json.dumps(row), flush=True)

        agreement = validation["choice_agreement"]

        if agreement is None:
            raise RuntimeError(
                "Validation set contains no genuine multi-action choices"
            )

        key = (agreement, -validation["choice_ce"])

        if key > best_key:
            best_key = key
            stale_epochs = 0

            save_best_checkpoint(model, epoch=epoch, validation=validation)
        else:
            stale_epochs += 1

        if epoch > 0 and agreement >= TARGET_AGREEMENT:
            print("Reached target held-out agreement.", flush=True)
            break

        if stale_epochs >= PATIENCE:
            print("Validation stopped improving.", flush=True)
            break

    print(f"Best checkpoint: {OUTPUT_DIR / 'best.pt'}")
