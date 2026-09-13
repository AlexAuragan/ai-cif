import argparse
import asyncio
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import RandomMoveCombatHandler
from showdown_sdk.classes.dt import BattleResult
from showdown_sdk.exceptions import BattleLifecycleError
from showdown_sdk.models.sdk import (
    SampleTeamGenerator,
    TeamSet,
    print_reproduction_teams,
)

import wandb
from ai_cif.inference.combat_handler import NeuralCombatHandler
from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel
from ai_cif.training.combat_handler import TrainingCombatHandler
from ai_cif.training.ppo import PPOConfig, PPOMetrics, ppo_update
from ai_cif.training.rewards import RewardConfig, breakdown_for
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

REWARD_CONFIG = RewardConfig(
    outcome_weight=0,
    own_hp_weight=0.5,
    enemy_hp_weight=0.5,
    speed_weight=0.0,
    speed_scale=40.0,
)


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

    battle_waiter_1: asyncio.Task[BattleResult] | None = None
    battle_waiter_2: asyncio.Task[BattleResult] | None = None

    try:
        await asyncio.gather(
            client_1.battle_manager.room_ready.wait(),
            client_2.battle_manager.room_ready.wait(),
        )

        battle_waiter_1 = asyncio.create_task(
            client_1.wait_for_battle_end(timeout=300)
        )
        battle_waiter_2 = asyncio.create_task(
            client_2.wait_for_battle_end(timeout=300)
        )

        result_1, result_2 = await asyncio.gather(
            battle_waiter_1, battle_waiter_2
        )

        return result_1, result_2

    except BaseException as error:
        waiters = [
            waiter
            for waiter in (battle_waiter_1, battle_waiter_2)
            if waiter is not None
        ]

        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()

        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)

        client_1.battle_manager.abandon_battle(error)
        client_2.battle_manager.abandon_battle(error)

        if team_generator is not None:
            print_reproduction_teams(team_1, team_2)
        else:
            print("\n========== TEAM 1 ==========")
            for pokemon in client_1.battle_manager.battle_state.team:
                print(pokemon)

            print("\n========== TEAM 2 ==========")

            for pokemon in client_2.battle_manager.battle_state.team:
                print(pokemon)

            print("============================\n")

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

    while len(trajectories) < battles:
        handler.start_battle()

        try:
            result, _ = await run_battle(
                neural_client,
                random_client,
                fmt=fmt,
                team_generator=team_generator,
            )
        except BattleLifecycleError as error:
            print(f"Discarding failed battle and retrying: {error!r}")

            await asyncio.gather(
                neural_client.close(),
                random_client.close(),
                return_exceptions=True,
            )

            await asyncio.gather(
                neural_client.ensure_connected(),
                random_client.ensure_connected(),
            )

            continue

        outcome = outcome_for(result, neural_client.username)
        breakdown = breakdown_for(result, outcome, config=REWARD_CONFIG)

        trajectory = handler.finish_battle(outcome, breakdown)

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


def mean_trajectory_reward(trajectories: list[Trajectory]) -> float:
    if not trajectories:
        raise ValueError("Cannot summarize empty trajectories")

    rewards: list[float] = []

    for trajectory in trajectories:
        if trajectory.reward is None:
            raise ValueError("All trajectories must have a reward")

        rewards.append(float(trajectory.reward))

    return sum(rewards) / len(rewards)


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

    wandb_run = None

    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "format": args.fmt,
                "iterations": args.iterations,
                "rollout_battles": (args.rollout_battles),
                "eval_battles": (args.eval_battles),
                "eval_interval": (args.eval_interval),
                "learning_rate": (args.learning_rate),
                "workers": args.workers,
                "worker_torch_threads": (args.worker_torch_threads),
                "team_seed": args.team_seed,
                "checkpoint_dir": str(args.checkpoint_dir),
                "training_device": str(device),
                "parameter_count": parameter_count,
                "ppo_clip_epsilon": (ppo_config.clip_epsilon),
                "ppo_value_coef": (ppo_config.value_coef),
                "ppo_entropy_coef": (ppo_config.entropy_coef),
                "ppo_epochs": (ppo_config.epochs),
                "ppo_minibatch_size": (ppo_config.minibatch_size),
                "reward_config": asdict(REWARD_CONFIG),
            },
        )

    context = multiprocessing.get_context("spawn")

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False
    phase_id = 0

    try:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(args.worker_torch_threads,),
        )

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

            initial_win_rate = wins / args.eval_battles

            best_eval_win_rate = initial_win_rate

            print(
                f"wins={wins} "
                f"losses={losses} "
                f"ties={ties} "
                f"win_rate="
                f"{initial_win_rate:.1%}"
            )

            print(
                f"evaluation_time="
                f"{evaluation_seconds:.2f}s "
                f"battles/s="
                f"{args.eval_battles / evaluation_seconds:.2f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/wins": wins,
                        "eval/losses": losses,
                        "eval/ties": ties,
                        "eval/win_rate": (initial_win_rate),
                        "eval/best_win_rate": (best_eval_win_rate),
                        "eval/seconds": (evaluation_seconds),
                        "eval/battles_per_second": (
                            args.eval_battles / evaluation_seconds
                        ),
                    },
                    step=0,
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
                    battles=(args.rollout_battles),
                    worker_count=(args.workers),
                    phase_id=phase_id,
                    temporary_directory=(temporary_directory),
                )

                rollout_seconds = perf_counter() - rollout_start

                (wins, losses, ties, decisions) = summarize_training(
                    trajectories
                )

                mean_reward = mean_trajectory_reward(trajectories)

                ppo_start = perf_counter()

                metrics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    trajectories=trajectories,
                    config=ppo_config,
                    device=device,
                )

                ppo_seconds = perf_counter() - ppo_start

                battle_count = len(trajectories)

                train_win_rate = wins / battle_count

                mean_decisions = decisions / battle_count

                breakdowns = [
                    trajectory.reward_breakdown for trajectory in trajectories
                ]

                if any(breakdown is None for breakdown in breakdowns):
                    raise RuntimeError("Missing reward breakdown")

                reward_breakdowns = [
                    breakdown
                    for breakdown in breakdowns
                    if breakdown is not None
                ]

                print()
                print(
                    f"iteration={iteration} "
                    f"battles={battle_count} "
                    f"decisions={decisions}"
                )

                print(f"train wins={wins} losses={losses} ties={ties}")

                print(
                    f"rollout_time="
                    f"{rollout_seconds:.2f}s "
                    f"battles/s="
                    f"{battle_count / rollout_seconds:.2f} "
                    f"decisions/s="
                    f"{decisions / rollout_seconds:.1f}"
                )

                print(f"ppo_time={ppo_seconds:.2f}s")

                print_metrics(metrics)

                log_data = {
                    "train/battles": battle_count,
                    "train/decisions": decisions,
                    "train/wins": wins,
                    "train/losses": losses,
                    "train/ties": ties,
                    "train/win_rate": (train_win_rate),
                    "train/mean_reward": (mean_reward),
                    "train/mean_decisions_per_battle": (mean_decisions),
                    "rollout/seconds": (rollout_seconds),
                    "rollout/battles_per_second": (
                        battle_count / rollout_seconds
                    ),
                    "rollout/decisions_per_second": (
                        decisions / rollout_seconds
                    ),
                    "ppo/seconds": ppo_seconds,
                    "ppo/policy_loss": (metrics.policy_loss),
                    "ppo/value_loss": (metrics.value_loss),
                    "ppo/entropy": (metrics.entropy),
                    "ppo/total_loss": (metrics.total_loss),
                    "ppo/approx_kl": (metrics.approx_kl),
                    "ppo/clip_fraction": (metrics.clip_fraction),
                    "ppo/mean_value": (metrics.mean_value),
                    "ppo/mean_return": (metrics.mean_return),
                    "optimizer/learning_rate": (
                        optimizer.param_groups[0]["lr"]
                    ),
                    "reward/total": sum(
                        item.total for item in reward_breakdowns
                    )
                    / battle_count,
                    "reward/outcome": sum(
                        item.outcome for item in reward_breakdowns
                    )
                    / battle_count,
                    "reward/own_hp": sum(
                        item.own_hp for item in reward_breakdowns
                    )
                    / battle_count,
                    "reward/enemy_damage": sum(
                        item.enemy_damage for item in reward_breakdowns
                    )
                    / battle_count,
                    "reward/speed": sum(
                        item.speed for item in reward_breakdowns
                    )
                    / battle_count,
                    "battle/own_hp_fraction": sum(
                        item.own_hp_fraction for item in reward_breakdowns
                    )
                    / battle_count,
                    "battle/enemy_hp_fraction": sum(
                        item.enemy_hp_fraction for item in reward_breakdowns
                    )
                    / battle_count,
                    "battle/mean_moves": sum(
                        item.move_count for item in reward_breakdowns
                    )
                    / battle_count,
                }

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
                        team_seed=(args.team_seed),
                        battles=(args.eval_battles),
                        worker_count=(args.workers),
                        phase_id=phase_id,
                    )

                    evaluation_seconds = perf_counter() - evaluation_start

                    win_rate = eval_wins / args.eval_battles

                    best_eval_win_rate = max(best_eval_win_rate, win_rate)

                    print(
                        f"EVAL "
                        f"iteration={iteration} "
                        f"wins={eval_wins} "
                        f"losses={eval_losses} "
                        f"ties={eval_ties} "
                        f"win_rate={win_rate:.1%}"
                    )

                    print(
                        f"eval_time="
                        f"{evaluation_seconds:.2f}s "
                        f"battles/s="
                        f"{args.eval_battles / evaluation_seconds:.2f}"
                    )

                    log_data.update(
                        {
                            "eval/wins": (eval_wins),
                            "eval/losses": (eval_losses),
                            "eval/ties": (eval_ties),
                            "eval/win_rate": (win_rate),
                            "eval/best_win_rate": (best_eval_win_rate),
                            "eval/seconds": (evaluation_seconds),
                            "eval/battles_per_second": (
                                args.eval_battles / evaluation_seconds
                            ),
                        }
                    )

                    checkpoint = (
                        args.checkpoint_dir
                        / args.wandb_name
                        / (f"iteration_{iteration:05d}.pt")
                    )

                    save_checkpoint(
                        path=checkpoint,
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                    )

                    save_checkpoint(
                        path=(
                            args.checkpoint_dir / args.wandb_name / "latest.pt"
                        ),
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                    )

                if wandb_run is not None:
                    #
                    # Exactly one log call for each
                    # training iteration. Evaluation
                    # values are included when this
                    # was an evaluation iteration.
                    #
                    wandb_run.log(log_data, step=iteration)

    except BaseException:
        if pool is not None:
            pool_terminated = True
            pool.terminate_workers()

        raise

    finally:
        if pool is not None and not pool_terminated:
            pool.shutdown(wait=True, cancel_futures=True)

        if wandb_run is not None:
            wandb_run.finish()


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

    parser.add_argument("--wandb-project", default="ai-cif")

    parser.add_argument("--wandb-entity", default=None)

    parser.add_argument("--wandb-name", default=None)

    parser.add_argument("--no-wandb", action="store_true")

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
