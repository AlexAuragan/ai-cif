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
from typing import override

import torch
from dotenv import load_dotenv
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import RandomMoveCombatHandler
from showdown_sdk.classes.combat_handler.base_handler import (
    AsyncBaseCombatHandler,
)
from showdown_sdk.classes.combat_handler.utils import Action
from showdown_sdk.exceptions import (
    BattleLifecycleError,
    BattleReproductionError,
    SDKTimeoutError,
)
from showdown_sdk.features import battle_to_features
from showdown_sdk.models.sdk import BattleState, SampleTeamGenerator
from showdown_sdk.models.sdk.team_generators.team_generator import (
    BaseTeamGenerator,
)

import wandb
from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel
from ai_cif.training.configs import RunningConfig, TrainingConfig
from ai_cif.training.ppo import PPOConfig, ppo_update
from ai_cif.training.rewards import RewardBreakdown, RewardConfig, breakdown_for
from ai_cif.training.trajectory import (
    Decision,
    PackedRollout,
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
    BattleTensors,
)
from scripts.utils.battles import outcome_for, run_battle
from scripts.utils.config import (
    MODEL_TYPES,
    PPO_TYPES,
    REWARD_TYPES,
    RUNNING_TYPES,
    TRAINING_TYPES,
)
from scripts.utils.gpu import (
    AsyncInferenceFn,
    BatchedGpuInferenceBroker,
    GpuInferenceStats,
    PendingInference,
    SharedBattleBuffer,
    WorkerInferenceStats,
    gpu_worker_initializer,
    make_remote_infer,
    response_pump,
    stop_response_pump,
)
from scripts.utils.model import save_checkpoint
from scripts.utils.multithreading import split_battles

load_dotenv()

DEFAULT_WEBSOCKET_URL = (
    os.environ.get("DEFAULT_WEBSOCKET_URL")
    or "ws://127.0.0.1:8000/showdown/websocket"
)

REWARD_CONFIG = RewardConfig(
    outcome_weight=0.2,
    own_hp_weight=0.4,
    enemy_hp_weight=0.4,
    speed_weight=0.0,
    speed_scale=40.0,
)

PPO_CONFIG = PPOConfig(
    learning_rate=3e-4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    epochs=2,
    minibatch_size=256,
    kl_target=0.02,
    kl_ratio_threshold=None,
)

TRAINING_CONFIG = TrainingConfig(
    iterations=200,
    rollout_battles=1000,
    eval_battles=400,
    eval_interval=10,
    team_seed=42,
)

RUNNING_CONFIG = RunningConfig(
    url=DEFAULT_WEBSOCKET_URL,
    format="gen1randombattle",
    workers=30,
    threads=1,
    checkpoint_dir=Path("checkpoints"),
    wandb_project="ai-cif",
    wandb_entity=None,
    battle_lanes=4,
    gpu_batch_size=32,
    gpu_batch_wait_ms=0.5,
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



class AsyncNeuralCombatHandler(AsyncBaseCombatHandler):
    """Generic async neural policy.

    It knows how to tensorize a battle and consume an async inference callable.
    It does not know anything about multiprocessing, shared memory, or CUDA.
    """

    def __init__(
        self,
        *,
        tensorizer: BattleTensorizer,
        infer: AsyncInferenceFn,
    ) -> None:
        self.tensorizer = tensorizer
        self.infer = infer

    async def _infer(
        self, battle_state: BattleState
    ) -> tuple[BattleTensors, torch.Tensor, float]:
        features = battle_to_features(battle_state)
        tensors = self.tensorizer.tensorize(features)


        logits, value = await self.infer(tensors)

        if logits.shape != (10,):
            raise RuntimeError(
                f"Expected policy logits shape (10,), got {tuple(logits.shape)}"
            )

        return tensors, logits, value

    @override
    async def async_select_top_actions(
        self, battle_state: BattleState
    ) -> list[Action]:
        tensors, logits, _ = await self._infer(battle_state)

        legal_indices = torch.where(tensors.action_mask)[0]

        if legal_indices.numel() == 0:
            raise RuntimeError("Model received a state with no legal actions")

        scores = logits[legal_indices]
        ranking = torch.argsort(scores, descending=True)
        ranked_indices = legal_indices[ranking]

        actions = [self._decode_action(int(index)) for index in ranked_indices]

        return actions

    @staticmethod
    @override
    async def async_select_team_order() -> list[int]:
        return [1, 2, 3, 4, 5, 6]

    @staticmethod
    def _decode_action(index: int) -> Action:
        if 0 <= index < 4:
            return ("move", index + 1)

        if 4 <= index < 10:
            return ("switch", index - 3)

        raise ValueError(f"Invalid action index: {index}")


class AsyncTrainingCombatHandler(AsyncNeuralCombatHandler):
    """Stochastic async policy used to collect PPO trajectories."""

    def __init__(
        self,
        *,
        tensorizer: BattleTensorizer,
        infer: AsyncInferenceFn,
    ) -> None:
        super().__init__(tensorizer=tensorizer, infer=infer)
        self.trajectory = Trajectory()

    def start_battle(self) -> None:
        self.trajectory = Trajectory()

    def finish_battle(
        self, outcome: float, reward_breakdown: RewardBreakdown
    ) -> Trajectory:
        reward = reward_breakdown.total

        if outcome not in {-1.0, 0.0, 1.0}:
            raise ValueError(f"Outcome must be -1, 0, or +1, got {outcome}")

        if not -1.0 <= reward <= 1.0:
            raise ValueError(f"Reward must be in [-1, 1], got {reward}")

        self.trajectory.outcome = outcome
        self.trajectory.reward = reward
        self.trajectory.reward_breakdown = reward_breakdown
        return self.trajectory

    @override
    async def async_select_top_actions(
        self, battle_state: BattleState
    ) -> list[Action]:
        tensors, logits, value = await self._infer(battle_state)


        legal_indices = torch.where(tensors.action_mask)[0]

        if legal_indices.numel() == 0:
            raise RuntimeError("Model received a state with no legal actions")

        legal_logits = logits[legal_indices]

        distribution = torch.distributions.Categorical(logits=legal_logits)

        sampled_position = distribution.sample()
        sampled_index = legal_indices[sampled_position]
        log_prob = distribution.log_prob(sampled_position)

        action_index = int(sampled_index.item())

        self.trajectory.decisions.append(
            Decision(
                observation=tensors,
                action=action_index,
                log_prob=float(log_prob.item()),
                value=value,
            )
        )

        remaining_indices = legal_indices[legal_indices != sampled_index]

        if remaining_indices.numel() > 0:
            remaining_scores = logits[remaining_indices]

            order = torch.argsort(remaining_scores, descending=True)

            remaining_indices = remaining_indices[order]

        ranked_indices = [
            action_index,
            *[int(index.item()) for index in remaining_indices],
        ]

        actions = [self._decode_action(index) for index in ranked_indices]

        return actions


def print_gpu_inference_stats(label: str, stats: GpuInferenceStats) -> None:
    print(
        f"{label} "
        f"requests={stats.requests} "
        f"batches={stats.batches} "
        f"mean_batch={stats.mean_batch_size:.2f} "
        f"max_batch={stats.max_batch_size} "
        f"batch_wait={stats.total_batch_wait_seconds:.3f}s "
        f"gather={stats.total_gather_seconds:.3f}s "
        f"gpu_roundtrip={stats.total_inference_seconds:.3f}s "
        f"dispatch={stats.total_dispatch_seconds:.3f}s"
    )


async def collect_trajectories(
    *,
    neural_client: Client,
    random_client: Client,
    handler: AsyncTrainingCombatHandler,
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
    discarded_battles = 0

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

        except (
            BattleLifecycleError,
            BattleReproductionError,
            SDKTimeoutError,
            RuntimeError,
        ) as error:
            discarded_battles += 1

            print(
                "Discarding failed battle and retrying: "
                f"{type(error).__name__}: {error}"
            )

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

    if discarded_battles > 0:
        print(
            f"Discarded {discarded_battles} failed battles "
            f"while collecting {battles} trajectories"
        )

    return trajectories


async def evaluate(
    *,
    neural_client: Client,
    random_client: Client,
    handler: AsyncNeuralCombatHandler,
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


def _team_generators(
    *, fmt: str, team_seed: int, phase_id: int, slot_index: int
) -> tuple[SampleTeamGenerator | None, SampleTeamGenerator | None]:
    if "randombattle" in fmt:
        return None, None

    seed = team_seed + phase_id * 100_000 + slot_index * 2
    return SampleTeamGenerator(seed), SampleTeamGenerator(seed + 1)


async def _rollout_lane(
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
    pending: PendingInference,
    inference_stats: WorkerInferenceStats,
) -> list[Trajectory]:
    slot_index = worker_index * battle_lanes + lane_index
    infer = make_remote_infer(
        worker_index=worker_index,
        slot_index=slot_index,
        pending=pending,
        stats=inference_stats,
    )

    handler = AsyncTrainingCombatHandler(
        tensorizer=tensorizer, infer=infer
    )

    neural_client = Client(url, combat_handler=handler)
    random_client = Client(url, combat_handler=RandomMoveCombatHandler())

    neural_client.log_manager.disable()
    random_client.log_manager.disable()

    team_generator_1, team_generator_2 = _team_generators(
        fmt=fmt, team_seed=team_seed, phase_id=phase_id, slot_index=slot_index
    )

    neural_name = f"A{phase_id}N{slot_index}"
    random_name = f"A{phase_id}R{slot_index}"

    try:

        await asyncio.gather(neural_client.connect(), random_client.connect())

        await asyncio.gather(
            neural_client.login(neural_name), random_client.login(random_name)
        )


        return await collect_trajectories(
            neural_client=neural_client,
            random_client=random_client,
            handler=handler,
            fmt=fmt,
            team_generator_1=team_generator_1,
            team_generator_2=team_generator_2,
            battles=battles,
            reward_config=reward_config,
        )

    finally:
        await asyncio.gather(
            neural_client.close(), random_client.close(), return_exceptions=True
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
) -> tuple[str, dict]:
    lane_counts = split_battles(battles, battle_lanes)

    pending: PendingInference = {}

    inference_stats = WorkerInferenceStats()

    pump_task = asyncio.create_task(
        response_pump(worker_index=worker_index, pending=pending)
    )

    try:
        tasks = [
            asyncio.create_task(
                _rollout_lane(
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
                    pending=pending,
                    inference_stats=inference_stats,
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

        if pending:
            raise RuntimeError(
                f"Worker {worker_index} finished with "
                f"{len(pending)} pending inference requests"
            )

        # Convert the large Python object graph into a small number
        # of batched tensors before crossing the process boundary.
        rollout = PackedRollout.from_trajectories(trajectories)

        # The original Decision/BattleTensors objects are no longer
        # required after packing.
        del trajectories
        del chunks

        torch.save(rollout, output_path)

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

        print(
            f"worker_save "
            f"worker={worker_index} "
            f"battles={rollout.battle_count} "
            f"decisions={rollout.decision_count} "
            f"size={file_size_mb:.1f}MB"
        )

        return (output_path, asdict(inference_stats))

    finally:
        stop_response_pump(worker_index)

        await pump_task


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
) -> tuple[str, dict[str, int | float]]:
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
        )
    )


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
    pending: PendingInference,
) -> tuple[int, int, int]:
    slot_index = worker_index * battle_lanes + lane_index
    infer = make_remote_infer(
        worker_index=worker_index, slot_index=slot_index, pending=pending
    )

    handler = AsyncNeuralCombatHandler(tensorizer=tensorizer, infer=infer)

    neural_client = Client(url, combat_handler=handler)
    random_client = Client(url, combat_handler=RandomMoveCombatHandler())

    neural_client.log_manager.disable()
    random_client.log_manager.disable()

    team_generator_1, team_generator_2 = _team_generators(
        fmt=fmt, team_seed=team_seed, phase_id=phase_id, slot_index=slot_index
    )

    neural_name = f"A{phase_id}N{slot_index}"
    random_name = f"A{phase_id}R{slot_index}"

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
) -> tuple[int, int, int]:
    lane_counts = split_battles(battles, battle_lanes)
    pending: PendingInference = {}
    pump_task = asyncio.create_task(
        response_pump(worker_index=worker_index, pending=pending)
    )

    try:
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
                    pending=pending,
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

        if pending:
            raise RuntimeError(
                f"Worker {worker_index} finished with pending inference requests"
            )

        wins = sum(result[0] for result in results)
        losses = sum(result[1] for result in results)
        ties = sum(result[2] for result in results)
        return wins, losses, ties

    finally:
        stop_response_pump(worker_index)
        await pump_task


def _evaluation_worker(
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
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
        )
    )


async def collect_trajectories_multiprocess(
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
) -> PackedRollout:
    counts = split_battles(battles, worker_count)

    loop = asyncio.get_running_loop()
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
                ),
            )
        )

    worker_results = await asyncio.gather(*tasks)

    output_paths: list[str] = []
    profiles: list[dict] = []
    inference_profiles: list[dict] = []

    for output_path, profile, inference_profile in worker_results:
        output_paths.append(output_path)
        profiles.append(profile)
        inference_profiles.append(inference_profile)

    rollout_chunks: list[PackedRollout] = []

    for output_path_string in output_paths:
        output_path = Path(output_path_string)

        worker_rollout = torch.load(
            output_path, map_location="cpu", weights_only=False
        )

        if not isinstance(worker_rollout, PackedRollout):
            raise TypeError("Rollout worker returned an invalid packed rollout")

        rollout_chunks.append(worker_rollout)
        output_path.unlink()

    rollout = PackedRollout.concat(rollout_chunks)

    del rollout_chunks

    ## Aggregate rollout profiling
    decisions = sum(profile["decisions"] for profile in profiles)

    completed_battles = sum(profile["battles"] for profile in profiles)
    inference_requests = sum(
        profile["requests"] for profile in inference_profiles
    )

    if inference_requests != decisions:
        print(
            "WARNING: profiling request/decision mismatch: "
            f"decisions={decisions} "
            f"inference_requests={inference_requests}"
        )

    if rollout.decision_count != decisions:
        print(
            "WARNING: packed rollout decision mismatch: "
            f"profiled={decisions} "
            f"packed={rollout.decision_count}"
        )

    if rollout.battle_count != completed_battles:
        print(
            "WARNING: packed rollout battle mismatch: "
            f"profiled={completed_battles} "
            f"packed={rollout.battle_count}"
        )
    return rollout


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
    if not torch.cuda.is_available():
        raise RuntimeError("train_gpu.py requires a CUDA/ROCm PyTorch device")

    device = torch.device("cuda")
    max_rollout_concurrency = min(
        training_config.rollout_battles,
        running_config.workers * running_config.battle_lanes,
    )

    print(f"Training device: {device}")
    print("Rollout inference: cuda (central async batched broker)")
    print(f"Rollout processes: {running_config.workers}")
    print(f"Battle lanes per process: {running_config.battle_lanes}")
    print(f"Max rollout battles in flight: {max_rollout_concurrency}")
    print(f"PyTorch threads per rollout process: {running_config.threads}")
    print(f"GPU inference max batch: {running_config.gpu_batch_size}")
    print(
        f"GPU inference batch wait: {running_config.gpu_batch_wait_ms:.3f} ms"
    )

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
                "rollout_inference_device": str(device),
                "parameter_count": parameter_count,
            },
        )

    context = multiprocessing.get_context("spawn")
    slot_count = running_config.workers * running_config.battle_lanes

    request_queue = context.Queue(
        maxsize=max(slot_count * 2, running_config.gpu_batch_size * 2)
    )
    response_queues = [
        context.Queue(maxsize=max(running_config.battle_lanes * 2, 8))
        for _ in range(running_config.workers)
    ]

    shared_buffer = SharedBattleBuffer.create(
        slot_count=slot_count, max_history=tensorizer.max_history
    )

    inference_broker = BatchedGpuInferenceBroker(
        model=model,
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        shared_buffer=shared_buffer,
        max_batch_size=running_config.gpu_batch_size,
        batch_wait_ms=running_config.gpu_batch_wait_ms,
    )

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False
    phase_id = 0

    inference_broker.start()

    try:
        pool = ProcessPoolExecutor(
            max_workers=running_config.workers,
            mp_context=context,
            initializer=gpu_worker_initializer,
            initargs=(
                running_config.threads,
                request_queue,
                response_queues,
                shared_buffer,
            ),
        )

        with tempfile.TemporaryDirectory(
            prefix="ai-cif-rollouts-"
        ) as temporary_directory_string:
            temporary_directory = Path(temporary_directory_string)

            print()
            print("Initial evaluation")
            print("------------------")

            phase_id += 1
            inference_broker.reset_stats()
            evaluation_start = perf_counter()

            wins, losses, ties = await evaluate_multiprocess(
                pool=pool,
                url=running_config.url,
                fmt=running_config.format,
                team_seed=training_config.team_seed,
                battles=training_config.eval_battles,
                worker_count=running_config.workers,
                battle_lanes=running_config.battle_lanes,
                phase_id=phase_id,
                tensorizer=tensorizer,
            )

            evaluation_seconds = perf_counter() - evaluation_start
            initial_inference_stats = inference_broker.snapshot_stats()
            initial_win_rate = wins / training_config.eval_battles
            best_eval_win_rate = initial_win_rate

            print_gpu_inference_stats("gpu_inference", initial_inference_stats)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/wins": wins,
                        "eval/losses": losses,
                        "eval/ties": ties,
                        "eval/win_rate": initial_win_rate,
                        "eval/best_win_rate": best_eval_win_rate,
                        "eval/seconds": evaluation_seconds,
                        "eval/battles_per_second": (
                            training_config.eval_battles / evaluation_seconds
                        ),
                        "gpu_inference/requests": initial_inference_stats.requests,
                        "gpu_inference/batches": initial_inference_stats.batches,
                        "gpu_inference/mean_batch_size": (
                            initial_inference_stats.mean_batch_size
                        ),
                        "gpu_inference/max_batch_size": (
                            initial_inference_stats.max_batch_size
                        ),
                        "gpu_inference/seconds": (
                            initial_inference_stats.total_inference_seconds
                        ),
                    },
                    step=0,
                )

            for iteration in range(1, training_config.iterations + 1):
                phase_id += 1
                inference_broker.reset_stats()
                rollout_start = perf_counter()

                trajectories = await collect_trajectories_multiprocess(
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
                )

                rollout_seconds = perf_counter() - rollout_start
                rollout_inference_stats = inference_broker.snapshot_stats()

                wins, losses, ties, decisions = summarize_trajectories(
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

                battle_count = trajectories.battle_count

                train_win_rate = wins / battle_count

                mean_decisions = decisions / battle_count

                reward_breakdowns = trajectories.reward_breakdowns

                timestamp = datetime.now(tz=UTC).strftime("%c")
                print()
                print(timestamp)
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
                print(f"ppo_time={ppo_seconds:.2f}s")
                print_gpu_inference_stats(
                    "gpu_inference", rollout_inference_stats
                )

                log_data = {
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
                    "gpu_inference/requests": rollout_inference_stats.requests,
                    "gpu_inference/batches": rollout_inference_stats.batches,
                    "gpu_inference/mean_batch_size": (
                        rollout_inference_stats.mean_batch_size
                    ),
                    "gpu_inference/max_batch_size": (
                        rollout_inference_stats.max_batch_size
                    ),
                    "gpu_inference/seconds": (
                        rollout_inference_stats.total_inference_seconds
                    ),
                    "gpu_inference/requests_per_inference_second": (
                        rollout_inference_stats.requests_per_inference_second
                    ),
                    "ppo/seconds": ppo_seconds,
                    "ppo/policy_loss": metrics.policy_loss,
                    "ppo/value_loss": metrics.value_loss,
                    "ppo/entropy": metrics.entropy,
                    "ppo/total_loss": metrics.total_loss,
                    "ppo/approx_kl": metrics.approx_kl,
                    "ppo/clip_fraction": metrics.clip_fraction,
                    "ppo/mean_value": metrics.mean_value,
                    "ppo/mean_return": metrics.mean_return,
                    "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
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
                    inference_broker.reset_stats()
                    evaluation_start = perf_counter()

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
                    )

                    evaluation_seconds = perf_counter() - evaluation_start
                    eval_inference_stats = inference_broker.snapshot_stats()
                    win_rate = eval_wins / training_config.eval_battles
                    best_eval_win_rate = max(best_eval_win_rate, win_rate)

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
                        f"{training_config.eval_battles / evaluation_seconds:.2f}"
                    )
                    print_gpu_inference_stats(
                        "eval_gpu_inference", eval_inference_stats
                    )

                    log_data.update(
                        {
                            "eval/wins": eval_wins,
                            "eval/losses": eval_losses,
                            "eval/ties": eval_ties,
                            "eval/win_rate": win_rate,
                            "eval/best_win_rate": best_eval_win_rate,
                            "eval/seconds": evaluation_seconds,
                            "eval/battles_per_second": (
                                training_config.eval_battles
                                / evaluation_seconds
                            ),
                            "eval_gpu_inference/requests": (
                                eval_inference_stats.requests
                            ),
                            "eval_gpu_inference/batches": (
                                eval_inference_stats.batches
                            ),
                            "eval_gpu_inference/mean_batch_size": (
                                eval_inference_stats.mean_batch_size
                            ),
                            "eval_gpu_inference/max_batch_size": (
                                eval_inference_stats.max_batch_size
                            ),
                            "eval_gpu_inference/seconds": (
                                eval_inference_stats.total_inference_seconds
                            ),
                        }
                    )

                    checkpoint = (
                        running_config.checkpoint_dir
                        / (args.wandb_name or "local")
                        / f"iteration_{iteration:05d}.pt"
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
                            / (args.wandb_name or "local")
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

        inference_broker.stop()

        request_queue.close()
        request_queue.join_thread()

        for response_queue in response_queues:
            response_queue.close()
            response_queue.join_thread()

        if wandb_run is not None:
            wandb_run.finish()


def apply_overrides(
    config, overrides: list[str], types: dict[str, type]
) -> None:
    for over in overrides:
        key, value = over.split("=", 1)

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
