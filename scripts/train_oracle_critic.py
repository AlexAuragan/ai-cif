import argparse
import asyncio
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch import Tensor

from scripts.train_population import (
    MODEL_CONFIG,
    REWARD_CONFIG,
    RUNNING_CONFIG,
    TENSORIZER,
    OpponentChoice,
    PopulationGpuInferenceBroker,
    TRAINING_MODEL_KEY,
    collect_trajectories_multiprocess,
    population_gpu_worker_initializer,
)

from ai_cif.model.model import BattleModel, TransformerBattleModel
from ai_cif.training.trajectory import PackedRollout
from scripts.utils.gpu import SharedBattleBuffer
from scripts.utils.model import create_model

ORACLE_ENV = "SHOWDOWN_USE_REQUEST_STATE"
OPPONENT_MODEL_KEY = "critic_warmup_opponent"

DEFAULT_INPUT = Path("data/models/platinium/platinium_00300_privileged_critic.pt")
DEFAULT_OUTPUT = Path(
    "data/models/platinium/platinium_00300_privileged_critic.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up only the privileged critic from a frozen Platinium actor."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--battles", type=int, default=5000)
    parser.add_argument("--batch-battles", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--team-seed", type=int, default=42)
    return parser.parse_args()


def _critic_parameters(
    model: BattleModel | TransformerBattleModel,
) -> list[torch.nn.Parameter]:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("critic_")
    ]

    if not parameters:
        raise RuntimeError("Model has no privileged critic parameters")

    return parameters


def _freeze_non_critic(model: BattleModel | TransformerBattleModel) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("critic_"))


def _snapshot_non_critic_state(
    model: BattleModel | TransformerBattleModel,
) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("critic_")
    }


def _assert_non_critic_unchanged(
    model: BattleModel | TransformerBattleModel, before: dict[str, Tensor]
) -> None:
    current = model.state_dict()

    for name, old_tensor in before.items():
        new_tensor = current[name].detach().cpu()

        if not torch.equal(old_tensor, new_tensor):
            raise RuntimeError(
                f"Non-critic parameter changed during critic warmup: {name}"
            )


def _terminal_targets(rollout: PackedRollout) -> Tensor:
    """Monte-Carlo value targets for critic warmup."""

    targets = torch.empty(rollout.decision_count, dtype=torch.float32)

    offset = 0

    for trajectory_index, length_tensor in enumerate(
        rollout.trajectory_lengths
    ):
        length = int(length_tensor.item())
        end = offset + length
        targets[offset:end] = rollout.rewards[trajectory_index]
        offset = end

    if offset != rollout.decision_count:
        raise RuntimeError("Terminal target construction lost decisions")

    return targets


def _critic_values(
    *,
    model: BattleModel | TransformerBattleModel,
    rollout: PackedRollout,
    device: torch.device,
    minibatch_size: int,
) -> Tensor:
    if rollout.oracle_observations is None:
        raise RuntimeError(
            "Critic warmup rollout has no oracle observations. "
            "SHOWDOWN_USE_REQUEST_STATE must be enabled."
        )

    values: list[Tensor] = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, rollout.decision_count, minibatch_size):
            end = min(start + minibatch_size, rollout.decision_count)
            indices = torch.arange(start, end, dtype=torch.long)

            oracle_batch = rollout.oracle_observations.index_select(indices).to(
                device
            )

            batch_values = model._privileged_value(oracle_batch)
            values.append(batch_values.detach().cpu())

    return torch.cat(values, dim=0)


def _explained_variance(predictions: Tensor, targets: Tensor) -> float:
    target_variance = torch.var(targets, unbiased=False)

    if target_variance <= 1e-8:
        return 0.0

    residual_variance = torch.var(targets - predictions, unbiased=False)

    return float((1.0 - residual_variance / target_variance).item())


def _train_critic(
    *,
    model: BattleModel | TransformerBattleModel,
    optimizer: torch.optim.Optimizer,
    rollout: PackedRollout,
    device: torch.device,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
) -> tuple[float, float, float, float]:
    if rollout.oracle_observations is None:
        raise RuntimeError("Critic warmup rollout has no oracle observations")

    targets_cpu = _terminal_targets(rollout)

    before_values = _critic_values(
        model=model,
        rollout=rollout,
        device=device,
        minibatch_size=minibatch_size,
    )
    before_loss = float(
        torch.nn.functional.mse_loss(before_values, targets_cpu).item()
    )
    before_ev = _explained_variance(before_values, targets_cpu)

    critic_parameters = _critic_parameters(model)

    model.train()

    for _ in range(epochs):
        permutation = torch.randperm(rollout.decision_count)

        for start in range(0, rollout.decision_count, minibatch_size):
            indices = permutation[start : start + minibatch_size]

            oracle_batch = rollout.oracle_observations.index_select(indices).to(
                device
            )
            targets = targets_cpu[indices].to(device)

            values = model._privileged_value(oracle_batch)
            loss = torch.nn.functional.mse_loss(values, targets)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm)
            optimizer.step()

    model.eval()

    after_values = _critic_values(
        model=model,
        rollout=rollout,
        device=device,
        minibatch_size=minibatch_size,
    )
    after_loss = float(
        torch.nn.functional.mse_loss(after_values, targets_cpu).item()
    )
    after_ev = _explained_variance(after_values, targets_cpu)

    return before_loss, after_loss, before_ev, after_ev


async def main() -> None:
    args = parse_args()

    if args.battles <= 0:
        raise ValueError("--battles must be positive")
    if args.batch_battles <= 0:
        raise ValueError("--batch-battles must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.minibatch_size <= 0:
        raise ValueError("--minibatch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if not torch.cuda.is_available():
        raise RuntimeError("This warmup script expects a CUDA/ROCm device")

    device = torch.device("cuda")

    model = create_model(device, MODEL_CONFIG, "transfomer", args.input)
    model.eval()

    _freeze_non_critic(model)
    actor_before = _snapshot_non_critic_state(model)

    critic_parameters = _critic_parameters(model)
    optimizer = torch.optim.Adam(critic_parameters, lr=args.learning_rate)

    print(f"Critic parameters: {sum(p.numel() for p in critic_parameters):,}")
    print(
        f"Frozen non-critic parameters: "
        f"{sum(p.numel() for n, p in model.named_parameters() if not n.startswith('critic_')):,}"
    )

    previous_oracle_env = os.environ.get(ORACLE_ENV)
    os.environ[ORACLE_ENV] = "1"

    context = multiprocessing.get_context("spawn")

    active_slot_count = RUNNING_CONFIG.workers * RUNNING_CONFIG.battle_lanes
    slot_count = active_slot_count * 2

    request_queue = context.Queue(
        maxsize=max(slot_count * 2, RUNNING_CONFIG.gpu_batch_size * 4)
    )
    response_queues = [
        context.Queue(maxsize=max(RUNNING_CONFIG.battle_lanes * 4, 16))
        for _ in range(RUNNING_CONFIG.workers)
    ]

    shared_buffer = SharedBattleBuffer.create(
        slot_count=slot_count, max_history=TENSORIZER.max_history
    )

    inference_broker = PopulationGpuInferenceBroker(
        models={TRAINING_MODEL_KEY: model, OPPONENT_MODEL_KEY: model},
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        shared_buffer=shared_buffer,
        max_batch_size=RUNNING_CONFIG.gpu_batch_size,
        batch_wait_ms=RUNNING_CONFIG.gpu_batch_wait_ms,
    )

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False

    inference_broker.start()

    battles_done = 0
    phase_id = 90_000_000

    try:
        pool = ProcessPoolExecutor(
            max_workers=RUNNING_CONFIG.workers,
            mp_context=context,
            initializer=population_gpu_worker_initializer,
            initargs=(
                RUNNING_CONFIG.threads,
                request_queue,
                response_queues,
                shared_buffer,
            ),
        )

        with tempfile.TemporaryDirectory(
            prefix="ai-cif-critic-warmup-"
        ) as temporary_directory_string:
            temporary_directory = Path(temporary_directory_string)

            while battles_done < args.battles:
                batch_battles = min(
                    args.batch_battles, args.battles - battles_done
                )

                opponent_choices = [
                    OpponentChoice(
                        group="critic_warmup_selfplay",
                        model_key=OPPONENT_MODEL_KEY,
                        model_name="platinium_00300",
                    )
                    for _ in range(batch_battles)
                ]

                rollout = await collect_trajectories_multiprocess(
                    pool=pool,
                    url=RUNNING_CONFIG.url,
                    fmt=RUNNING_CONFIG.format,
                    team_seed=args.team_seed,
                    worker_count=RUNNING_CONFIG.workers,
                    battle_lanes=RUNNING_CONFIG.battle_lanes,
                    phase_id=phase_id,
                    temporary_directory=temporary_directory,
                    reward_config=REWARD_CONFIG,
                    tensorizer=TENSORIZER,
                    opponent_choices=opponent_choices,
                    active_slot_count=active_slot_count,
                )

                if rollout.oracle_observations is None:
                    raise RuntimeError(
                        "Collected rollout without oracle observations"
                    )

                (before_loss, after_loss, before_ev, after_ev) = _train_critic(
                    model=model,
                    optimizer=optimizer,
                    rollout=rollout,
                    device=device,
                    epochs=args.epochs,
                    minibatch_size=args.minibatch_size,
                    max_grad_norm=args.max_grad_norm,
                )

                battles_done += batch_battles
                phase_id += 1

                print(
                    f"battles={battles_done}/{args.battles} "
                    f"decisions={rollout.decision_count} "
                    f"loss={before_loss:.5f}->{after_loss:.5f} "
                    f"EV={before_ev:+.4f}->{after_ev:+.4f}"
                )

                del rollout

    except BaseException:
        if pool is not None:
            pool_terminated = True
            pool.terminate_workers()
        raise

    finally:
        if pool is not None and not pool_terminated:
            pool.shutdown(wait=True, cancel_futures=True)

        inference_broker.stop()

        request_queue.close()
        request_queue.join_thread()

        for response_queue in response_queues:
            response_queue.close()
            response_queue.join_thread()

        if previous_oracle_env is None:
            os.environ.pop(ORACLE_ENV, None)
        else:
            os.environ[ORACLE_ENV] = previous_oracle_env

    _assert_non_critic_unchanged(model, actor_before)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "critic_warmup": {
                "source": str(args.input),
                "battles": args.battles,
                "batch_battles": args.batch_battles,
                "epochs": args.epochs,
                "minibatch_size": args.minibatch_size,
                "learning_rate": args.learning_rate,
                "max_grad_norm": args.max_grad_norm,
                "team_seed": args.team_seed,
                "reward": {
                    "outcome_weight": REWARD_CONFIG.outcome_weight,
                    "own_hp_weight": REWARD_CONFIG.own_hp_weight,
                    "enemy_hp_weight": REWARD_CONFIG.enemy_hp_weight,
                    "speed_weight": REWARD_CONFIG.speed_weight,
                },
            },
        },
        args.output,
    )

    print(f"Saved warmed checkpoint to {args.output}")
    print("Verified: all non-critic weights are bit-for-bit unchanged.")


if __name__ == "__main__":
    asyncio.run(main())
