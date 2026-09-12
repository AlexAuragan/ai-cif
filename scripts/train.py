import argparse
import asyncio
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import RandomMoveCombatHandler
from showdown_sdk.classes.dt import BattleResult
from showdown_sdk.models.sdk import (
    SampleTeamGenerator,
    TeamSet,
    print_reproduction_teams,
)

from ai_cif.inference.combat_handler import NeuralCombatHandler
from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel
from ai_cif.training.combat_handler import TrainingCombatHandler
from ai_cif.training.ppo import PPOConfig, PPOMetrics, ppo_update
from ai_cif.training.trajectory import Trajectory
from ai_cif.vectorization.tensorizer import (
    CANT_REASON_VOCAB_SIZE,
    FIELD_NUMERIC_DIM,
    HISTORY_KIND_VOCAB_SIZE,
    HISTORY_NUMERIC_DIM,
    HISTORY_REF_VOCAB_SIZE,
    POKEMON_NUMERIC_DIM,
    STATUS_VOCAB_SIZE,
    WEATHER_VOCAB_SIZE,
    BattleTensorizer,
)

DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/showdown/websocket"


def create_model(device: torch.device) -> tuple[BattleModel, BattleTensorizer]:
    tensorizer = BattleTensorizer(max_history=32, vocab_gen=4)

    config = ModelConfig(
        species_count=tensorizer.species_vocab_size,
        form_count=tensorizer.form_vocab_size,
        move_count=tensorizer.move_vocab_size,
        item_count=tensorizer.item_vocab_size,
        ability_count=tensorizer.ability_vocab_size,
        status_count=STATUS_VOCAB_SIZE,
        weather_count=WEATHER_VOCAB_SIZE,
        tactical_event_type_count=HISTORY_KIND_VOCAB_SIZE,
        history_ref_count=HISTORY_REF_VOCAB_SIZE,
        history_reason_count=CANT_REASON_VOCAB_SIZE,
    )

    model = BattleModel(
        config=config,
        pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
        field_numeric_feature_count=FIELD_NUMERIC_DIM,
        tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
    )

    model.to(device)

    return model, tensorizer


def snapshot_model(model: BattleModel) -> dict[str, torch.Tensor]:
    """Create a stable CPU snapshot that can be sent to rollout processes."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def split_battles(battles: int, worker_count: int) -> list[int]:
    base = battles // worker_count
    remainder = battles % worker_count

    return [
        base + (1 if index < remainder else 0) for index in range(worker_count)
    ]


async def run_battle(
    client_1: Client,
    client_2: Client,
    *,
    fmt: str,
    team_generator: SampleTeamGenerator | None,
) -> tuple[BattleResult, BattleResult]:
    await asyncio.gather(
        client_1.ensure_connected(), client_2.ensure_connected()
    )

    if client_1.username is None:
        raise RuntimeError("Client 1 is not logged in")

    if client_2.username is None:
        raise RuntimeError("Client 2 is not logged in")

    team_1: TeamSet | None = None
    team_2: TeamSet | None = None

    if team_generator is not None:
        team_1 = await team_generator.generate(
            fmt, lambda team: client_1.validate_team(fmt, team)
        )

        team_2 = await team_generator.generate(
            fmt, lambda team: client_2.validate_team(fmt, team)
        )

    await client_1.challenge(client_2.username, fmt, timeout=60, team=team_1)

    await client_2.accept_challenge(client_1.username, team=team_2)

    try:
        await asyncio.gather(
            client_1.battle_manager.room_ready.wait(),
            client_2.battle_manager.room_ready.wait(),
        )

        return await asyncio.gather(
            client_1.wait_for_battle_end(timeout=300),
            client_2.wait_for_battle_end(timeout=300),
        )
    except BaseException:
        print_reproduction_teams(team_1, team_2)
        raise


def outcome_for(result: BattleResult, username: str) -> float:
    if result.winner is None:
        return 0.0

    if result.winner == username:
        return 1.0

    return -1.0


async def collect_trajectories(
    *,
    neural_client: Client,
    random_client: Client,
    handler: TrainingCombatHandler,
    fmt: str,
    team_generator: SampleTeamGenerator | None,
    battles: int,
) -> list[Trajectory]:
    if neural_client.username is None:
        raise RuntimeError("Neural client has no username")

    neural_client.combat_handler = handler

    trajectories: list[Trajectory] = []

    for _ in range(battles):
        handler.start_battle()

        result, _ = await run_battle(
            neural_client, random_client, fmt=fmt, team_generator=team_generator
        )

        outcome = outcome_for(result, neural_client.username)

        trajectory = handler.finish_battle(outcome)

        if not trajectory.decisions:
            raise RuntimeError("Collected empty trajectory")

        trajectories.append(trajectory)

    return trajectories


async def evaluate(
    *,
    neural_client: Client,
    random_client: Client,
    handler: NeuralCombatHandler,
    fmt: str,
    team_generator: SampleTeamGenerator | None,
    battles: int,
) -> tuple[int, int, int]:
    if neural_client.username is None:
        raise RuntimeError("Neural client has no username")

    neural_client.combat_handler = handler

    wins = 0
    losses = 0
    ties = 0

    for _ in range(battles):
        result, _ = await run_battle(
            neural_client, random_client, fmt=fmt, team_generator=team_generator
        )

        outcome = outcome_for(result, neural_client.username)

        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            ties += 1

    return wins, losses, ties


def _worker_initializer(torch_threads: int) -> None:
    # Every rollout process already gives us CPU parallelism. Letting every
    # process also create a large PyTorch thread pool usually oversubscribes
    # the machine and hurts throughput for this small model.
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)


async def _rollout_worker_async(
    *,
    model_state: dict[str, torch.Tensor],
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    phase_id: int,
    output_path: str,
) -> str:
    device = torch.device("cpu")
    model, tensorizer = create_model(device)
    model.load_state_dict(model_state)
    model.eval()

    handler = TrainingCombatHandler(
        model=model, tensorizer=tensorizer, device=device
    )

    neural_client = Client(url, combat_handler=handler)

    random_client = Client(url, combat_handler=RandomMoveCombatHandler())

    neural_client.log_manager.disable()
    random_client.log_manager.disable()

    team_generator: SampleTeamGenerator | None = None

    if "randombattle" not in fmt:
        team_generator = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index
        )

    neural_name = f"A{phase_id}N{worker_index}"
    random_name = f"A{phase_id}R{worker_index}"

    try:
        await asyncio.gather(neural_client.connect(), random_client.connect())

        await asyncio.gather(
            neural_client.login(neural_name), random_client.login(random_name)
        )

        trajectories = await collect_trajectories(
            neural_client=neural_client,
            random_client=random_client,
            handler=handler,
            fmt=fmt,
            team_generator=team_generator,
            battles=battles,
        )

        # Do not push thousands of small tensors through the multiprocessing
        # result queue. Serialize one trajectory chunk per process instead.
        torch.save(trajectories, output_path)

        return output_path

    finally:
        await asyncio.gather(
            neural_client.close(), random_client.close(), return_exceptions=True
        )


def _rollout_worker(
    model_state: dict[str, torch.Tensor],
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    phase_id: int,
    output_path: str,
) -> str:
    return asyncio.run(
        _rollout_worker_async(
            model_state=model_state,
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            battles=battles,
            worker_index=worker_index,
            phase_id=phase_id,
            output_path=output_path,
        )
    )


async def _evaluation_worker_async(
    *,
    model_state: dict[str, torch.Tensor],
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    phase_id: int,
) -> tuple[int, int, int]:
    device = torch.device("cpu")
    model, tensorizer = create_model(device)
    model.load_state_dict(model_state)
    model.eval()

    handler = NeuralCombatHandler(
        model=model, tensorizer=tensorizer, device=device
    )

    neural_client = Client(url, combat_handler=handler)

    random_client = Client(url, combat_handler=RandomMoveCombatHandler())

    neural_client.log_manager.disable()
    random_client.log_manager.disable()

    team_generator: SampleTeamGenerator | None = None

    if "randombattle" not in fmt:
        team_generator = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index
        )

    neural_name = f"A{phase_id}N{worker_index}"
    random_name = f"A{phase_id}R{worker_index}"

    try:
        await asyncio.gather(neural_client.connect(), random_client.connect())

        await asyncio.gather(
            neural_client.login(neural_name), random_client.login(random_name)
        )

        return await evaluate(
            neural_client=neural_client,
            random_client=random_client,
            handler=handler,
            fmt=fmt,
            team_generator=team_generator,
            battles=battles,
        )

    finally:
        await asyncio.gather(
            neural_client.close(), random_client.close(), return_exceptions=True
        )


def _evaluation_worker(
    model_state: dict[str, torch.Tensor],
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    phase_id: int,
) -> tuple[int, int, int]:
    return asyncio.run(
        _evaluation_worker_async(
            model_state=model_state,
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            battles=battles,
            worker_index=worker_index,
            phase_id=phase_id,
        )
    )


async def collect_trajectories_multiprocess(
    *,
    pool: ProcessPoolExecutor,
    model: BattleModel,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_count: int,
    phase_id: int,
    temporary_directory: Path,
) -> list[Trajectory]:
    counts = split_battles(battles, worker_count)

    model_state = snapshot_model(model)
    loop = asyncio.get_running_loop()

    tasks = []

    for worker_index, count in enumerate(counts):
        if count <= 0:
            continue

        output_path = (
            temporary_directory
            / f"phase_{phase_id:05d}_worker_{worker_index:03d}.pt"
        )

        tasks.append(
            loop.run_in_executor(
                pool,
                partial(
                    _rollout_worker,
                    model_state,
                    url,
                    fmt,
                    team_seed,
                    count,
                    worker_index,
                    phase_id,
                    str(output_path),
                ),
            )
        )

    output_paths = await asyncio.gather(*tasks)

    trajectories: list[Trajectory] = []

    for output_path_string in output_paths:
        output_path = Path(output_path_string)

        worker_trajectories = torch.load(
            output_path, map_location="cpu", weights_only=False
        )

        if not isinstance(worker_trajectories, list):
            raise TypeError(
                "Rollout worker returned a non-list trajectory chunk"
            )

        for trajectory in worker_trajectories:
            if not isinstance(trajectory, Trajectory):
                raise TypeError(
                    "Rollout worker returned an invalid trajectory object"
                )

        trajectories.extend(worker_trajectories)
        output_path.unlink()

    return trajectories


async def evaluate_multiprocess(
    *,
    pool: ProcessPoolExecutor,
    model: BattleModel,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_count: int,
    phase_id: int,
) -> tuple[int, int, int]:
    counts = split_battles(battles, worker_count)

    model_state = snapshot_model(model)
    loop = asyncio.get_running_loop()

    tasks = [
        loop.run_in_executor(
            pool,
            partial(
                _evaluation_worker,
                model_state,
                url,
                fmt,
                team_seed,
                count,
                worker_index,
                phase_id,
            ),
        )
        for worker_index, count in enumerate(counts)
        if count > 0
    ]

    results = await asyncio.gather(*tasks)

    wins = sum(result[0] for result in results)
    losses = sum(result[1] for result in results)
    ties = sum(result[2] for result in results)

    return wins, losses, ties


def summarize_training(
    trajectories: list[Trajectory],
) -> tuple[int, int, int, int]:
    wins = 0
    losses = 0
    ties = 0
    decisions = 0

    for trajectory in trajectories:
        decisions += len(trajectory.decisions)

        if trajectory.outcome == 1.0:
            wins += 1
        elif trajectory.outcome == -1.0:
            losses += 1
        else:
            ties += 1

    return wins, losses, ties, decisions


def save_checkpoint(
    *,
    path: Path,
    model: BattleModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def print_metrics(metrics: PPOMetrics) -> None:
    print(
        f"policy={metrics.policy_loss:+.4f} "
        f"value={metrics.value_loss:.4f} "
        f"entropy={metrics.entropy:.4f} "
        f"kl={metrics.approx_kl:.5f} "
        f"clip={metrics.clip_fraction:.3f}"
    )

    print(
        f"mean_value={metrics.mean_value:+.3f} "
        f"mean_return={metrics.mean_return:+.3f}"
    )


async def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training device: {device}")
    print("Rollout device: cpu")
    print(f"Rollout processes: {args.workers}")
    print(f"PyTorch threads per rollout process: {args.worker_torch_threads}")

    model, _ = create_model(device)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model parameters: {parameter_count:,}")

    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        epochs=4,
        minibatch_size=256,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo_config.learning_rate
    )

    context = multiprocessing.get_context("spawn")

    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(args.worker_torch_threads,),
    )

    pool_terminated = False
    phase_id = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix="ai-cif-rollouts-"
        ) as temporary_directory_string:
            temporary_directory = Path(temporary_directory_string)

            print()
            print("Initial evaluation")
            print("------------------")

            phase_id += 1
            evaluation_start = perf_counter()

            wins, losses, ties = await evaluate_multiprocess(
                pool=pool,
                model=model,
                url=args.url,
                fmt=args.fmt,
                team_seed=args.team_seed,
                battles=args.eval_battles,
                worker_count=args.workers,
                phase_id=phase_id,
            )

            evaluation_seconds = perf_counter() - evaluation_start

            print(
                f"wins={wins} "
                f"losses={losses} "
                f"ties={ties} "
                f"win_rate={wins / args.eval_battles:.1%}"
            )

            print(
                f"evaluation_time={evaluation_seconds:.2f}s "
                f"battles/s="
                f"{args.eval_battles / evaluation_seconds:.2f}"
            )

            for iteration in range(1, args.iterations + 1):
                phase_id += 1
                rollout_start = perf_counter()

                trajectories = await collect_trajectories_multiprocess(
                    pool=pool,
                    model=model,
                    url=args.url,
                    fmt=args.fmt,
                    team_seed=args.team_seed,
                    battles=args.rollout_battles,
                    worker_count=args.workers,
                    phase_id=phase_id,
                    temporary_directory=temporary_directory,
                )

                rollout_seconds = perf_counter() - rollout_start

                wins, losses, ties, decisions = summarize_training(trajectories)

                ppo_start = perf_counter()

                metrics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    trajectories=trajectories,
                    config=ppo_config,
                    device=device,
                )

                ppo_seconds = perf_counter() - ppo_start

                print()
                print(
                    f"iteration={iteration} "
                    f"battles={len(trajectories)} "
                    f"decisions={decisions}"
                )

                print(f"train wins={wins} losses={losses} ties={ties}")

                print(
                    f"rollout_time={rollout_seconds:.2f}s "
                    f"battles/s="
                    f"{len(trajectories) / rollout_seconds:.2f} "
                    f"decisions/s="
                    f"{decisions / rollout_seconds:.1f}"
                )

                print(f"ppo_time={ppo_seconds:.2f}s")

                print_metrics(metrics)

                if iteration % args.eval_interval == 0:
                    phase_id += 1
                    evaluation_start = perf_counter()

                    (
                        eval_wins,
                        eval_losses,
                        eval_ties,
                    ) = await evaluate_multiprocess(
                        pool=pool,
                        model=model,
                        url=args.url,
                        fmt=args.fmt,
                        team_seed=args.team_seed,
                        battles=args.eval_battles,
                        worker_count=args.workers,
                        phase_id=phase_id,
                    )

                    evaluation_seconds = perf_counter() - evaluation_start

                    win_rate = eval_wins / args.eval_battles

                    print(
                        f"EVAL iteration={iteration} "
                        f"wins={eval_wins} "
                        f"losses={eval_losses} "
                        f"ties={eval_ties} "
                        f"win_rate={win_rate:.1%}"
                    )

                    print(
                        f"eval_time={evaluation_seconds:.2f}s "
                        f"battles/s="
                        f"{args.eval_battles / evaluation_seconds:.2f}"
                    )

                    checkpoint = (
                        args.checkpoint_dir / f"iteration_{iteration:05d}.pt"
                    )

                    save_checkpoint(
                        path=checkpoint,
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                    )

                    save_checkpoint(
                        path=(args.checkpoint_dir / "latest.pt"),
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                    )

    except BaseException:
        # Python 3.14: do not leave long-running rollout children alive after
        # Ctrl-C or a worker failure.
        pool_terminated = True
        pool.terminate_workers()
        raise

    finally:
        if not pool_terminated:
            pool.shutdown(wait=True, cancel_futures=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", default=DEFAULT_WEBSOCKET_URL)

    parser.add_argument("--format", dest="fmt", default="gen1randombattle")

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=("Number of independent rollout/evaluation processes"),
    )

    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=1,
        help=("PyTorch intra-op threads used by each rollout process"),
    )

    parser.add_argument("--iterations", type=int, default=100)

    parser.add_argument("--rollout-battles", type=int, default=32)

    parser.add_argument("--eval-battles", type=int, default=100)

    parser.add_argument("--eval-interval", type=int, default=10)

    parser.add_argument("--learning-rate", type=float, default=3e-4)

    parser.add_argument("--team-seed", type=int, default=42)

    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints")
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.worker_torch_threads < 1:
        parser.error("--worker-torch-threads must be at least 1")

    if args.rollout_battles < 1:
        parser.error("--rollout-battles must be at least 1")

    if args.eval_battles < 1:
        parser.error("--eval-battles must be at least 1")

    return args


async def main() -> None:
    args = parse_args()
    await train(args)


if __name__ == "__main__":
    t0 = perf_counter()
    asyncio.run(main())
    print(f"took {perf_counter() - t0}")
