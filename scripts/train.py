import argparse
import asyncio
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
import wandb
from dotenv import load_dotenv
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import RandomMoveCombatHandler
from showdown_sdk.exceptions import BattleLifecycleError
from showdown_sdk.models.sdk import SampleTeamGenerator
from showdown_sdk.models.sdk.team_generators.team_generator import (
    BaseTeamGenerator,
)

from ai_cif.inference.combat_handler import NeuralCombatHandler
from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel
from ai_cif.training.combat_handler import TrainingCombatHandler
from ai_cif.training.configs import RunningConfig, TrainingConfig
from ai_cif.training.ppo import PPOConfig, ppo_update
from ai_cif.training.rewards import RewardConfig, breakdown_for
from ai_cif.training.trajectory import (
    Trajectory,
    mean_trajectory_reward,
    summarize_trajectories,
)
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
from scripts.utils.battles import outcome_for, run_battle
from scripts.utils.config import (
    MODEL_TYPES,
    PPO_TYPES,
    REWARD_TYPES,
    RUNNING_TYPES,
    TRAINING_TYPES,
)
from scripts.utils.metrics import print_initial_metrics, print_metrics
from scripts.utils.model import save_checkpoint, snapshot_model
from scripts.utils.multithreading import split_battles, worker_initializer

load_dotenv()

DEFAULT_WEBSOCKET_URL = (
    os.environ.get("DEFAULT_WEBSOCKET_URL")
    or "ws://127.0.0.1:8000/showdown/websocket"
)

REWARD_CONFIG = RewardConfig(
    outcome_weight=0.0,
    own_hp_weight=0.5,
    enemy_hp_weight=0.5,
    speed_weight=0.0,
    speed_scale=40.0,
)


PPO_CONFIG = PPOConfig(
    learning_rate=3e-4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    epochs=4,
    minibatch_size=256,
    kl_target=0.02,
    kl_ratio_threshold=None,
)

TRAINING_CONFIG = TrainingConfig(
    iterations=200,
    rollout_battles=100,
    eval_battles=200,
    eval_interval=10,
    team_seed=42,
)

RUNNING_CONFIG = RunningConfig(
    url=DEFAULT_WEBSOCKET_URL,
    format="gen1randombattle",
    workers=20,
    threads=1,
    checkpoint_dir=Path("checkpoints"),
    wandb_project="ai-cif",
    wandb_entity=None,
    battle_lanes=8,
)

TENSORIZER = BattleTensorizer(max_history=32, vocab_gen=4)

MODEL_CONFIG = ModelConfig(
    species_count=TENSORIZER.species_vocab_size,
    form_count=TENSORIZER.form_vocab_size,
    move_count=TENSORIZER.move_vocab_size,
    item_count=TENSORIZER.item_vocab_size,
    ability_count=TENSORIZER.ability_vocab_size,
    status_count=STATUS_VOCAB_SIZE,
    weather_count=WEATHER_VOCAB_SIZE,
    tactical_event_type_count=HISTORY_KIND_VOCAB_SIZE,
    history_ref_count=HISTORY_REF_VOCAB_SIZE,
    history_reason_count=CANT_REASON_VOCAB_SIZE,
)


def create_model(
    device: torch.device, model_config: ModelConfig
) -> BattleModel:
    torch.manual_seed(model_config.seed)
    model = BattleModel(
        config=model_config,
        pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
        field_numeric_feature_count=FIELD_NUMERIC_DIM,
        tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
    )

    model.to(device)

    return model


async def collect_trajectories(
    *,
    neural_client: Client,
    random_client: Client,
    handler: TrainingCombatHandler,
    fmt: str,
    team_generator_1: BaseTeamGenerator | None,
    team_generator_2: BaseTeamGenerator | None,
    battles: int,
    reward_config: RewardConfig,
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
                team_generator_1=team_generator_1,
                team_generator_2=team_generator_2,
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
        breakdown = breakdown_for(result, outcome, config=reward_config)

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
    team_generator_1: BaseTeamGenerator | None,
    team_generator_2: BaseTeamGenerator | None,
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
            neural_client,
            random_client,
            fmt=fmt,
            team_generator_1=team_generator_1,
            team_generator_2=team_generator_2,
        )

        outcome = outcome_for(result, neural_client.username)

        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            ties += 1

    return wins, losses, ties


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
    reward_config: RewardConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
) -> str:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    device = torch.device("cpu")
    model = create_model(device, model_config)
    model.load_state_dict(model_state)
    model.eval()

    handler = TrainingCombatHandler(
        model=model, tensorizer=tensorizer, device=device
    )

    neural_client = Client(url, combat_handler=handler)

    random_client = Client(url, combat_handler=RandomMoveCombatHandler())

    neural_client.log_manager.disable()
    random_client.log_manager.disable()

    team_generator_1: SampleTeamGenerator | None = None
    team_generator_2: SampleTeamGenerator | None = None

    if "randombattle" not in fmt:
        team_generator_1 = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index
        )
        team_generator_2 = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index + 1
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
            team_generator_1=team_generator_1,
            team_generator_2=team_generator_2,
            battles=battles,
            reward_config=reward_config,
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
    reward_config: RewardConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
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
            reward_config=reward_config,
            model_config=model_config,
            tensorizer=tensorizer,
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
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
) -> tuple[int, int, int]:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    device = torch.device("cpu")
    model = create_model(device, model_config)
    model.load_state_dict(model_state)
    model.eval()

    handler = NeuralCombatHandler(
        model=model, tensorizer=tensorizer, device=device
    )

    neural_client = Client(url, combat_handler=handler)

    random_client = Client(url, combat_handler=RandomMoveCombatHandler())

    neural_client.log_manager.disable()
    random_client.log_manager.disable()

    team_generator_1: SampleTeamGenerator | None = None
    team_generator_2: SampleTeamGenerator | None = None

    if "randombattle" not in fmt:
        team_generator_1 = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index
        )
        team_generator_2 = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index + 1
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
            team_generator_1=team_generator_1,
            team_generator_2=team_generator_2,
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
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
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
            model_config=model_config,
            tensorizer=tensorizer,
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
    reward_config: RewardConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
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
                    reward_config,
                    model_config,
                    tensorizer,
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
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
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
                model_config,
                tensorizer,
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


async def train(
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    reward_config: RewardConfig,
    running_config: RunningConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training device: {device}")
    print(f"Rollout device: {device}")
    print(f"Rollout processes: {running_config.workers}")
    print(f"PyTorch threads per rollout process: {running_config.threads}")

    model = create_model(device, model_config)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model parameters: {parameter_count:,}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo_config.learning_rate
    )

    wandb_run = None

    if not args.no_wandb:
        _running_config = asdict(running_config)
        _running_config["checkpoint_dir"] = str(
            _running_config["checkpoint_dir"]
        )
        wandb_run = wandb.init(
            project=running_config.wandb_project,
            entity=running_config.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                "running": _running_config,
                "training": asdict(training_config),
                "ppo": asdict(ppo_config),
                "reward": asdict(reward_config),
                "training_device": str(device),
                "parameter_count": parameter_count,
            },
        )

    context = multiprocessing.get_context("spawn")

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False
    phase_id = 0

    try:
        pool = ProcessPoolExecutor(
            max_workers=running_config.workers,
            mp_context=context,
            initializer=worker_initializer,
            initargs=(running_config.threads,),
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
                url=running_config.url,
                fmt=running_config.format,
                team_seed=training_config.team_seed,
                battles=training_config.eval_battles,
                worker_count=running_config.workers,
                phase_id=phase_id,
                model_config=model_config,
                tensorizer=tensorizer,
            )

            evaluation_seconds = perf_counter() - evaluation_start

            initial_win_rate = wins / training_config.eval_battles

            best_eval_win_rate = initial_win_rate

            eval_battles = training_config.eval_battles
            print_initial_metrics(
                wins, losses, ties, eval_battles, evaluation_seconds
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
                            training_config.eval_battles / evaluation_seconds
                        ),
                    },
                    step=0,
                )

            for iteration in range(1, training_config.iterations + 1):
                phase_id += 1

                rollout_start = perf_counter()

                trajectories = await collect_trajectories_multiprocess(
                    pool=pool,
                    model=model,
                    url=running_config.url,
                    fmt=running_config.format,
                    team_seed=training_config.team_seed,
                    battles=(training_config.rollout_battles),
                    worker_count=(running_config.workers),
                    phase_id=phase_id,
                    temporary_directory=(temporary_directory),
                    reward_config=reward_config,
                    model_config=model_config,
                    tensorizer=tensorizer,
                )

                rollout_seconds = perf_counter() - rollout_start

                (wins, losses, ties, decisions) = summarize_trajectories(
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
                print(datetime.now(tz=UTC).strftime("%c"))  # type:ignore
                print(
                    f"iteration={iteration} "
                    + f"battles={battle_count} "
                    + f"decisions={decisions}"
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

                if iteration % training_config.eval_interval == 0:
                    phase_id += 1

                    evaluation_start = perf_counter()

                    (
                        eval_wins,
                        eval_losses,
                        eval_ties,
                    ) = await evaluate_multiprocess(
                        pool=pool,
                        model=model,
                        url=running_config.url,
                        fmt=running_config.format,
                        team_seed=(training_config.team_seed),
                        battles=(training_config.eval_battles),
                        worker_count=(running_config.workers),
                        phase_id=phase_id,
                        model_config=model_config,
                        tensorizer=tensorizer,
                    )

                    evaluation_seconds = perf_counter() - evaluation_start

                    win_rate = eval_wins / training_config.eval_battles

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
                        f"{training_config.eval_battles / evaluation_seconds:.2f}"
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
                                training_config.eval_battles
                                / evaluation_seconds
                            ),
                        }
                    )

                    checkpoint = (
                        running_config.checkpoint_dir
                        / (args.wandb_name or "local")
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
                            running_config.checkpoint_dir
                            / args.wandb_name
                            / "latest.pt"
                        ),
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                    )

                if wandb_run is not None:
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


def apply_overrides(
    config, overrides: list[str], types: dict[str, type]
) -> None:
    for override in overrides:
        key, value = override.split("=", 1)

        if key not in types:
            raise ValueError(f"Unknown config field: {key}")

        value_type = types[key]

        if value_type is bool:
            if value.lower() in {"true", "1", "yes"}:
                parsed_value = True
            elif value.lower() in {"false", "0", "no"}:
                parsed_value = False
            else:
                raise ValueError(f"Invalid boolean value: {value}")
        else:
            parsed_value = value_type(value)

        setattr(config, key, parsed_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb-name", default=None)

    parser.add_argument("--no-wandb", action="store_true")

    parser.add_argument("--wandb-group", default=None)

    parser.add_argument("--set-training", nargs="*", default=[])
    parser.add_argument("--set-ppo", nargs="*", default=[])
    parser.add_argument("--set-reward", nargs="*", default=[])
    parser.add_argument("--set-running", nargs="*", default=[])
    parser.add_argument("--set-model", nargs="*", default=[])

    args = parser.parse_args()

    if args.wandb_name is None and not args.no_wandb:
        raise ValueError(
            "--wandb-name must be set unless --no-wandb flag is active"
        )

    return args


async def main() -> None:

    ppo_config = copy(PPO_CONFIG)
    training_config = copy(TRAINING_CONFIG)
    reward_config = copy(REWARD_CONFIG)
    running_config = copy(RUNNING_CONFIG)
    model_config = copy(MODEL_CONFIG)
    tensorizer = copy(TENSORIZER)
    args = parse_args()

    apply_overrides(ppo_config, args.set_ppo, PPO_TYPES)
    apply_overrides(training_config, args.set_training, TRAINING_TYPES)
    apply_overrides(reward_config, args.set_reward, REWARD_TYPES)
    apply_overrides(running_config, args.set_running, RUNNING_TYPES)
    apply_overrides(model_config, args.set_model, MODEL_TYPES)

    await train(
        args,
        ppo_config,
        training_config,
        reward_config,
        running_config,
        model_config,
        tensorizer,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    asyncio.run(main())
    print(f"took {perf_counter() - t0}")
