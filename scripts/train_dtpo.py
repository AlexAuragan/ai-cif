import argparse
import asyncio
import multiprocessing
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import SimpleHeuristicsCombatHandler
from showdown_sdk.exceptions import (
    BattleLifecycleError,
    BattleReproductionError,
    SDKTimeoutError,
)
from showdown_sdk.models.sdk import SampleTeamGenerator

import wandb
from ai_cif.dtpo.config import DTPOConfig
from ai_cif.dtpo.features import tree_feature_dim
from ai_cif.dtpo.policy import DecisionTreePolicy
from ai_cif.dtpo.training import dtpo_update
from ai_cif.dtpo.value import BattleValueModel
from ai_cif.training.configs import RunningConfig, TrainingConfig
from ai_cif.training.dtpo_combat_handler import DTPOCombatHandler
from ai_cif.training.rewards import RewardConfig, breakdown_for
from ai_cif.training.trajectory import (
    PackedRollout,
    Trajectory,
    mean_trajectory_reward,
    summarize_trajectories,
)
from ai_cif.vectorization.tensorizer import BattleTensorizer
from scripts.utils.battles import outcome_for, run_battle
from scripts.utils.config import REWARD_TYPES, RUNNING_TYPES, TRAINING_TYPES
from scripts.utils.multithreading import split_battles, worker_initializer

load_dotenv()

DEFAULT_WEBSOCKET_URL = (
    os.environ.get("DEFAULT_WEBSOCKET_URL")
    or "ws://127.0.0.1:8000/showdown/websocket"
)

DTPO_TYPES = {
    "learning_rate": float,
    "clip_epsilon": float,
    "gamma": float,
    "max_depth": int,
    "max_leaf_nodes": int,
    "policy_updates": int,
    "value_hidden_dim": int,
    "value_learning_rate": float,
    "value_epochs": int,
    "value_minibatch_size": int,
    "normalize_advantage": bool,
}

DTPO_CONFIG = DTPOConfig(
    learning_rate=1.0,
    clip_epsilon=0.2,
    gamma=0.99,
    max_depth=None,
    max_leaf_nodes=32,
    policy_updates=1,
    value_hidden_dim=128,
    value_learning_rate=3e-4,
    value_epochs=4,
    value_minibatch_size=256,
    normalize_advantage=True,
)

REWARD_CONFIG = RewardConfig(
    outcome_weight=0.2,
    own_hp_weight=0.3,
    enemy_hp_weight=0.3,
    speed_weight=0.0,
    speed_scale=40.0,
)

TRAINING_CONFIG = TrainingConfig(
    iterations=1000,
    rollout_battles=1000,
    eval_battles=1000,
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
    battle_lanes=10,
    gpu_batch_size=128,
    gpu_batch_wait_ms=0.0,
)

TENSORIZER = BattleTensorizer(max_history=32, vocab_gen=4)


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


def _team_generators(
    *, fmt: str, team_seed: int, phase_id: int, slot_index: int
) -> tuple[SampleTeamGenerator | None, SampleTeamGenerator | None]:
    if "randombattle" in fmt:
        return None, None

    seed = team_seed + phase_id * 100_000 + slot_index * 2
    return SampleTeamGenerator(seed), SampleTeamGenerator(seed + 1)


def _copy_policy(
    policy: DecisionTreePolicy, *, seed: int
) -> DecisionTreePolicy:
    copied = DecisionTreePolicy(
        action_count=policy.action_count,
        max_depth=policy.max_depth,
        max_leaf_nodes=policy.max_leaf_nodes,
        seed=seed,
    )

    if policy.tree is not None:
        copied.replace_tree(policy.tree)

    return copied


async def _collect_lane(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    lane_index: int,
    battle_lanes: int,
    phase_id: int,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
    value_model: BattleValueModel,
) -> list[Trajectory]:
    slot_index = worker_index * battle_lanes + lane_index
    lane_seed = team_seed + phase_id * 1_000_003 + slot_index * 10_007

    training_policy = _copy_policy(policy, seed=lane_seed)
    opponent_policy = _copy_policy(policy, seed=lane_seed + 1)

    training_handler = DTPOCombatHandler(
        policy=training_policy,
        tensorizer=tensorizer,
        value_model=value_model,
        device="cpu",
        sample=True,
        record_trajectory=True,
    )
    opponent_handler = DTPOCombatHandler(
        policy=opponent_policy,
        tensorizer=tensorizer,
        sample=True,
        record_trajectory=False,
    )

    training_client = Client(url, combat_handler=training_handler)
    opponent_client = Client(url, combat_handler=opponent_handler)

    training_client.log_manager.disable()
    opponent_client.log_manager.disable()

    team_generator_1, team_generator_2 = _team_generators(
        fmt=fmt, team_seed=team_seed, phase_id=phase_id, slot_index=slot_index
    )

    training_name = f"D{phase_id}T{slot_index}"
    opponent_name = f"D{phase_id}O{slot_index}"

    trajectories: list[Trajectory] = []
    discarded_battles = 0

    try:
        await asyncio.gather(
            training_client.connect(), opponent_client.connect()
        )
        await asyncio.gather(
            training_client.login(training_name),
            opponent_client.login(opponent_name),
        )

        if training_client.username is None:
            raise RuntimeError("Training client has no username")

        while len(trajectories) < battles:
            training_handler.start_battle()

            try:
                result, _ = await run_battle(
                    training_client,
                    opponent_client,
                    fmt=fmt,
                    team_generator_1=team_generator_1,
                    team_generator_2=team_generator_2,
                )
            except (
                BattleLifecycleError,
                BattleReproductionError,
                SDKTimeoutError,
                RuntimeError,
            ) as error:
                discarded_battles += 1
                print(
                    f"worker={worker_index} lane={lane_index} "
                    "discarding failed battle: "
                    f"{type(error).__name__}: {error}"
                )

                await asyncio.gather(
                    training_client.close(),
                    opponent_client.close(),
                    return_exceptions=True,
                )
                await asyncio.gather(
                    training_client.connect(), opponent_client.connect()
                )
                await asyncio.gather(
                    training_client.login(training_name),
                    opponent_client.login(opponent_name),
                )
                continue

            outcome = outcome_for(result, training_client.username)
            reward_breakdown = breakdown_for(result, outcome, reward_config)
            trajectory = training_handler.finish_battle(
                outcome, reward_breakdown
            )

            if not trajectory.decisions:
                raise RuntimeError("Collected empty trajectory")

            trajectories.append(trajectory)

        if discarded_battles > 0:
            print(
                f"worker={worker_index} lane={lane_index} "
                f"discarded={discarded_battles}"
            )

        return trajectories

    finally:
        await asyncio.gather(
            training_client.close(),
            opponent_client.close(),
            return_exceptions=True,
        )


async def _rollout_worker_async(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    output_path: str,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
    value_state_dict: dict[str, torch.Tensor],
    feature_count: int,
    value_hidden_dim: int,
) -> str:
    value_model = BattleValueModel(
        feature_count=feature_count, hidden_dim=value_hidden_dim
    )
    value_model.load_state_dict(value_state_dict)
    value_model.eval()

    lane_counts = split_battles(battles, battle_lanes)

    tasks = [
        asyncio.create_task(
            _collect_lane(
                url=url,
                fmt=fmt,
                team_seed=team_seed,
                battles=count,
                worker_index=worker_index,
                lane_index=lane_index,
                battle_lanes=battle_lanes,
                phase_id=phase_id,
                reward_config=reward_config,
                tensorizer=tensorizer,
                policy=policy,
                value_model=value_model,
            )
        )
        for lane_index, count in enumerate(lane_counts)
        if count > 0
    ]

    try:
        chunks = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    trajectories = [trajectory for chunk in chunks for trajectory in chunk]
    rollout = PackedRollout.from_trajectories(trajectories)

    torch.save(rollout, output_path)
    return output_path


def _rollout_worker(
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    output_path: str,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
    value_state_dict: dict[str, torch.Tensor],
    feature_count: int,
    value_hidden_dim: int,
) -> str:
    return asyncio.run(
        _rollout_worker_async(
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            battles=battles,
            worker_index=worker_index,
            battle_lanes=battle_lanes,
            phase_id=phase_id,
            output_path=output_path,
            reward_config=reward_config,
            tensorizer=tensorizer,
            policy=policy,
            value_state_dict=value_state_dict,
            feature_count=feature_count,
            value_hidden_dim=value_hidden_dim,
        )
    )


async def collect_self_play_multiprocess(
    *,
    pool: ProcessPoolExecutor,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_count: int,
    battle_lanes: int,
    phase_id: int,
    temporary_directory: Path,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
    value_model: BattleValueModel,
    feature_count: int,
    value_hidden_dim: int,
) -> PackedRollout:
    counts = split_battles(battles, worker_count)
    loop = asyncio.get_running_loop()

    value_state_dict = {
        key: value.detach().cpu()
        for key, value in value_model.state_dict().items()
    }

    tasks = []

    for worker_index, count in enumerate(counts):
        if count <= 0:
            continue

        output_path = temporary_directory / (
            f"phase_{phase_id:05d}_worker_{worker_index:03d}.pt"
        )

        tasks.append(
            loop.run_in_executor(
                pool,
                partial(
                    _rollout_worker,
                    url,
                    fmt,
                    team_seed,
                    count,
                    worker_index,
                    battle_lanes,
                    phase_id,
                    str(output_path),
                    reward_config,
                    tensorizer,
                    policy,
                    value_state_dict,
                    feature_count,
                    value_hidden_dim,
                ),
            )
        )

    worker_results = await asyncio.gather(*tasks)
    chunks: list[PackedRollout] = []

    for output_path_string in worker_results:
        output_path = Path(output_path_string)
        rollout = torch.load(
            output_path, map_location="cpu", weights_only=False
        )

        if not isinstance(rollout, PackedRollout):
            raise TypeError("Rollout worker returned an invalid PackedRollout")

        chunks.append(rollout)
        output_path.unlink()

    return PackedRollout.concat(chunks)


async def _evaluation_lane(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    lane_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
) -> tuple[int, int, int]:
    slot_index = worker_index * battle_lanes + lane_index
    lane_seed = team_seed + phase_id * 1_000_003 + slot_index * 10_007

    evaluation_policy = _copy_policy(policy, seed=lane_seed)
    evaluation_handler = DTPOCombatHandler(
        policy=evaluation_policy,
        tensorizer=tensorizer,
        sample=False,
        record_trajectory=False,
    )
    heuristic_handler = SimpleHeuristicsCombatHandler()

    evaluation_client = Client(url, combat_handler=evaluation_handler)
    heuristic_client = Client(url, combat_handler=heuristic_handler)

    evaluation_client.log_manager.disable()
    heuristic_client.log_manager.disable()

    team_generator_1, team_generator_2 = _team_generators(
        fmt=fmt, team_seed=team_seed, phase_id=phase_id, slot_index=slot_index
    )

    evaluation_name = f"D{phase_id}E{slot_index}"
    heuristic_name = f"D{phase_id}H{slot_index}"

    wins = 0
    losses = 0
    ties = 0

    try:
        await asyncio.gather(
            evaluation_client.connect(), heuristic_client.connect()
        )
        await asyncio.gather(
            evaluation_client.login(evaluation_name),
            heuristic_client.login(heuristic_name),
        )

        if evaluation_client.username is None:
            raise RuntimeError("Evaluation client has no username")

        for _ in range(battles):
            result, _ = await run_battle(
                evaluation_client,
                heuristic_client,
                fmt=fmt,
                team_generator_1=team_generator_1,
                team_generator_2=team_generator_2,
            )

            outcome = outcome_for(result, evaluation_client.username)

            if outcome > 0.0:
                wins += 1
            elif outcome < 0.0:
                losses += 1
            else:
                ties += 1

        return wins, losses, ties

    finally:
        await asyncio.gather(
            evaluation_client.close(),
            heuristic_client.close(),
            return_exceptions=True,
        )


async def _evaluation_worker_async(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
) -> tuple[int, int, int]:
    lane_counts = split_battles(battles, battle_lanes)

    tasks = [
        asyncio.create_task(
            _evaluation_lane(
                url=url,
                fmt=fmt,
                team_seed=team_seed,
                battles=count,
                worker_index=worker_index,
                lane_index=lane_index,
                battle_lanes=battle_lanes,
                phase_id=phase_id,
                tensorizer=tensorizer,
                policy=policy,
            )
        )
        for lane_index, count in enumerate(lane_counts)
        if count > 0
    ]

    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    wins = sum(result[0] for result in results)
    losses = sum(result[1] for result in results)
    ties = sum(result[2] for result in results)

    return wins, losses, ties


def _evaluation_worker(
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
) -> tuple[int, int, int]:
    return asyncio.run(
        _evaluation_worker_async(
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            battles=battles,
            worker_index=worker_index,
            battle_lanes=battle_lanes,
            phase_id=phase_id,
            tensorizer=tensorizer,
            policy=policy,
        )
    )


async def evaluate_multiprocess(
    *,
    pool: ProcessPoolExecutor,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_count: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    policy: DecisionTreePolicy,
) -> tuple[int, int, int]:
    counts = split_battles(battles, worker_count)
    loop = asyncio.get_running_loop()

    tasks = [
        loop.run_in_executor(
            pool,
            partial(
                _evaluation_worker,
                url,
                fmt,
                team_seed,
                count,
                worker_index,
                battle_lanes,
                phase_id,
                tensorizer,
                policy,
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


def save_checkpoint(
    *,
    path: Path,
    iteration: int,
    policy: DecisionTreePolicy,
    value_model: BattleValueModel,
    value_optimizer: torch.optim.Optimizer,
    dtpo_config: DTPOConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "iteration": iteration,
        "policy": policy,
        "value_model": value_model.state_dict(),
        "value_optimizer": value_optimizer.state_dict(),
        "dtpo_config": asdict(dtpo_config),
    }

    with path.open("wb") as file:
        pickle.dump(payload, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", default=None)

    parser.add_argument("--set-training", nargs="*", default=[])
    parser.add_argument("--set-dtpo", nargs="*", default=[])
    parser.add_argument("--set-reward", nargs="*", default=[])
    parser.add_argument("--set-running", nargs="*", default=[])

    args = parser.parse_args()

    if args.wandb_name is None and not args.no_wandb:
        raise ValueError(
            "--wandb-name must be set unless --no-wandb flag is active"
        )

    return args


async def train(
    args: argparse.Namespace,
    dtpo_config: DTPOConfig,
    training_config: TrainingConfig,
    reward_config: RewardConfig,
    running_config: RunningConfig,
    tensorizer: BattleTensorizer,
) -> None:
    torch.manual_seed(training_config.team_seed)

    device = torch.device("cuda")
    feature_count = tree_feature_dim(tensorizer.max_history)

    policy = DecisionTreePolicy(
        action_count=10,
        max_depth=dtpo_config.max_depth,
        max_leaf_nodes=dtpo_config.max_leaf_nodes,
        seed=training_config.team_seed,
    )

    value_model = BattleValueModel(
        feature_count=feature_count, hidden_dim=dtpo_config.value_hidden_dim
    ).to(device)

    value_optimizer = torch.optim.Adam(
        value_model.parameters(), lr=dtpo_config.value_learning_rate
    )

    max_rollout_concurrency = min(
        training_config.rollout_battles,
        running_config.workers * running_config.battle_lanes,
    )

    print(f"Training device: {device}")
    print(f"Format: {running_config.format}")
    print(f"Tree feature count: {feature_count}")
    print(
        f"Tree limits: depth={dtpo_config.max_depth} "
        f"leaves={dtpo_config.max_leaf_nodes}"
    )
    print(f"Rollout processes: {running_config.workers}")
    print(f"Battle lanes per process: {running_config.battle_lanes}")
    print(f"Max rollout battles in flight: {max_rollout_concurrency}")
    print(f"PyTorch threads per rollout process: {running_config.threads}")

    value_parameter_count = sum(
        parameter.numel() for parameter in value_model.parameters()
    )
    print(f"Value model parameters: {value_parameter_count:,}")

    wandb_run = None

    if not args.no_wandb:
        running_dict = asdict(running_config)
        running_dict["checkpoint_dir"] = str(running_dict["checkpoint_dir"])

        wandb_run = wandb.init(
            project=running_config.wandb_project,
            entity=running_config.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                "running": running_dict,
                "training": asdict(training_config),
                "dtpo": asdict(dtpo_config),
                "reward": asdict(reward_config),
                "training_device": str(device),
                "rollout_inference_device": "cpu",
                "tree_feature_count": feature_count,
                "value_parameter_count": value_parameter_count,
            },
        )

    checkpoint_dir = running_config.checkpoint_dir / (
        args.wandb_name or "local"
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
            prefix="ai-cif-dtpo-rollouts-"
        ) as temporary_directory_string:
            temporary_directory = Path(temporary_directory_string)

            for iteration in range(1, training_config.iterations + 1):
                phase_id += 1

                timestamp = datetime.now(tz=UTC).strftime("%c")
                print()
                print(timestamp)
                print(
                    f"ROLLOUT iteration={iteration} "
                    f"battles={training_config.rollout_battles} "
                    f"workers={running_config.workers} "
                    f"lanes={running_config.battle_lanes} "
                    f"max_in_flight={max_rollout_concurrency}"
                )

                rollout_start = perf_counter()

                trajectories = await collect_self_play_multiprocess(
                    pool=pool,
                    url=running_config.url,
                    fmt=running_config.format,
                    team_seed=training_config.team_seed,
                    battles=training_config.rollout_battles,
                    worker_count=running_config.workers,
                    battle_lanes=running_config.battle_lanes,
                    phase_id=phase_id,
                    temporary_directory=temporary_directory,
                    reward_config=reward_config,
                    tensorizer=tensorizer,
                    policy=policy,
                    value_model=value_model,
                    feature_count=feature_count,
                    value_hidden_dim=dtpo_config.value_hidden_dim,
                )

                rollout_seconds = perf_counter() - rollout_start

                wins, losses, ties, decisions = summarize_trajectories(
                    trajectories
                )
                mean_reward = mean_trajectory_reward(trajectories)

                update_start = perf_counter()

                metrics = dtpo_update(
                    policy=policy,
                    value_model=value_model,
                    value_optimizer=value_optimizer,
                    trajectories=trajectories,
                    config=dtpo_config,
                    device=device,
                )

                update_seconds = perf_counter() - update_start

                battle_count = trajectories.battle_count
                train_win_rate = wins / battle_count
                mean_decisions = decisions / battle_count
                reward_breakdowns = trajectories.reward_breakdowns

                print()
                print(
                    f"iteration={iteration} "
                    f"battles={battle_count} "
                    f"decisions={decisions}"
                )
                print(f"train wins={wins} losses={losses} ties={ties}")
                print(
                    f"rollout_time={rollout_seconds:.2f}s "
                    f"battles/s={battle_count / rollout_seconds:.2f} "
                    f"decisions/s={decisions / rollout_seconds:.1f}"
                )
                print(f"dtpo_time={update_seconds:.2f}s")
                print(
                    f"objective="
                    f"{metrics.policy_objective_before:.4f}"
                    f"->{metrics.policy_objective_after:.4f} "
                    f"entropy={metrics.entropy:.4f} "
                    f"value_loss={metrics.value_loss:.6f}"
                )
                print(
                    f"tree depth={metrics.tree_depth} "
                    f"leaves={metrics.leaf_count} "
                    f"updated={metrics.tree_updated}"
                )

                log_data: dict[str, float | int] = {
                    "iteration/index": iteration,
                    "train/battles": battle_count,
                    "train/decisions": decisions,
                    "train/wins": wins,
                    "train/losses": losses,
                    "train/ties": ties,
                    "train/win_rate": train_win_rate,
                    "train/mean_reward": mean_reward,
                    "train/mean_decisions_per_battle": mean_decisions,
                    "rollout/seconds": rollout_seconds,
                    "rollout/battles_per_second": (
                        battle_count / rollout_seconds
                    ),
                    "rollout/decisions_per_second": (
                        decisions / rollout_seconds
                    ),
                    "dtpo/seconds": update_seconds,
                    "dtpo/policy_objective_before": (
                        metrics.policy_objective_before
                    ),
                    "dtpo/policy_objective_after": (
                        metrics.policy_objective_after
                    ),
                    "dtpo/value_loss": metrics.value_loss,
                    "dtpo/entropy": metrics.entropy,
                    "dtpo/mean_advantage": metrics.mean_advantage,
                    "dtpo/mean_return": metrics.mean_return,
                    "dtpo/tree_depth": metrics.tree_depth,
                    "dtpo/leaf_count": metrics.leaf_count,
                    "dtpo/tree_updated": int(metrics.tree_updated),
                    "reward/total": (
                        sum(item.total for item in reward_breakdowns)
                        / battle_count
                    ),
                    "reward/outcome": (
                        sum(item.outcome for item in reward_breakdowns)
                        / battle_count
                    ),
                    "reward/own_hp": (
                        sum(item.own_hp for item in reward_breakdowns)
                        / battle_count
                    ),
                    "reward/enemy_damage": (
                        sum(item.enemy_damage for item in reward_breakdowns)
                        / battle_count
                    ),
                    "reward/speed": (
                        sum(item.speed for item in reward_breakdowns)
                        / battle_count
                    ),
                    "battle/own_hp_fraction": (
                        sum(item.own_hp_fraction for item in reward_breakdowns)
                        / battle_count
                    ),
                    "battle/enemy_hp_fraction": (
                        sum(
                            item.enemy_hp_fraction for item in reward_breakdowns
                        )
                        / battle_count
                    ),
                    "battle/mean_moves": (
                        sum(item.move_count for item in reward_breakdowns)
                        / battle_count
                    ),
                }

                if iteration % training_config.eval_interval == 0:
                    phase_id += 1
                    eval_concurrency = min(
                        training_config.eval_battles,
                        running_config.workers * running_config.battle_lanes,
                    )

                    print()
                    print(
                        f"EVALUATION iteration={iteration} "
                        f"battles={training_config.eval_battles} "
                        f"max_in_flight={eval_concurrency}"
                    )

                    eval_start = perf_counter()

                    (
                        eval_wins,
                        eval_losses,
                        eval_ties,
                    ) = await evaluate_multiprocess(
                        pool=pool,
                        url=running_config.url,
                        fmt=running_config.format,
                        team_seed=training_config.team_seed,
                        battles=training_config.eval_battles,
                        worker_count=running_config.workers,
                        battle_lanes=running_config.battle_lanes,
                        phase_id=phase_id,
                        tensorizer=tensorizer,
                        policy=policy,
                    )

                    eval_seconds = perf_counter() - eval_start
                    eval_total = eval_wins + eval_losses + eval_ties
                    win_rate = eval_wins / eval_total

                    print(
                        f"EVAL iteration={iteration} "
                        f"wins={eval_wins} "
                        f"losses={eval_losses} "
                        f"ties={eval_ties} "
                        f"win_rate={win_rate:.1%}"
                    )
                    print(
                        f"eval_time={eval_seconds:.2f}s "
                        f"battles/s={eval_total / eval_seconds:.2f}"
                    )

                    log_data.update(
                        {
                            "eval/wins": eval_wins,
                            "eval/losses": eval_losses,
                            "eval/ties": eval_ties,
                            "eval/win_rate": win_rate,
                            "eval/seconds": eval_seconds,
                            "eval/battles_per_second": (
                                eval_total / eval_seconds
                            ),
                        }
                    )

                    save_checkpoint(
                        path=(
                            checkpoint_dir / f"iteration_{iteration:05d}.pkl"
                        ),
                        iteration=iteration,
                        policy=policy,
                        value_model=value_model,
                        value_optimizer=value_optimizer,
                        dtpo_config=dtpo_config,
                    )
                    save_checkpoint(
                        path=checkpoint_dir / "latest.pkl",
                        iteration=iteration,
                        policy=policy,
                        value_model=value_model,
                        value_optimizer=value_optimizer,
                        dtpo_config=dtpo_config,
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


async def main() -> None:
    dtpo_config = copy(DTPO_CONFIG)
    training_config = copy(TRAINING_CONFIG)
    reward_config = copy(REWARD_CONFIG)
    running_config = copy(RUNNING_CONFIG)
    tensorizer = copy(TENSORIZER)

    args = parse_args()

    apply_overrides(dtpo_config, args.set_dtpo, DTPO_TYPES)
    apply_overrides(training_config, args.set_training, TRAINING_TYPES)
    apply_overrides(reward_config, args.set_reward, REWARD_TYPES)
    apply_overrides(running_config, args.set_running, RUNNING_TYPES)

    await train(
        args,
        dtpo_config,
        training_config,
        reward_config,
        running_config,
        tensorizer,
    )


if __name__ == "__main__":
    started = perf_counter()
    asyncio.run(main())
    print(f"took {perf_counter() - started:.2f}s")
