"""
This one is not trained from scratch but from crystal-x-00200 and against the SimpleHeuristic handler
"""

import argparse
import asyncio
import multiprocessing
import os
import queue
import random
import tempfile
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import (
    AsyncSimpleHeuristicsCombatHandler,
    SimpleHeuristicsCombatHandler,
)
from showdown_sdk.classes.combat_handler.base_handler import (
    AsyncBaseCombatHandler,
)
from showdown_sdk.exceptions import (
    BattleLifecycleError,
    BattleReproductionError,
    SDKTimeoutError,
)
from showdown_sdk.models.sdk import SampleTeamGenerator
from showdown_sdk.models.sdk.team_generators.team_generator import (
    BaseTeamGenerator,
)

import wandb
from ai_cif.inference.combat_handler import AsyncNeuralCombatHandler
from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel
from ai_cif.training.configs import RunningConfig, TrainingConfig
from ai_cif.training.ppo import PPOConfig, ppo_update
from ai_cif.training.rewards import RewardConfig, breakdown_for
from ai_cif.training.trajectory import (
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
from scripts.train_simple_heuristics import AsyncTrainingCombatHandler
from scripts.utils.battles import outcome_for, run_battle
from scripts.utils.config import (
    MODEL_TYPES,
    PPO_TYPES,
    REWARD_TYPES,
    RUNNING_TYPES,
    TRAINING_TYPES,
)
from scripts.utils.gpu import (
    GpuInferenceStats,
    SharedBattleBuffer,
    WorkerInferenceStats,
)
from scripts.utils.model import save_checkpoint
from scripts.utils.multithreading import split_battles, worker_initializer

type AsyncInferenceFn = Callable[
    [BattleTensors], Awaitable[tuple[torch.Tensor, float]]
]
type PendingInference = dict[
    tuple[int, int], asyncio.Future[tuple[torch.Tensor, float]]
]

load_dotenv()

DEFAULT_WEBSOCKET_URL = (
    os.environ.get("DEFAULT_WEBSOCKET_URL")
    or "ws://127.0.0.1:8000/showdown/websocket"
)

REWARD_CONFIG = RewardConfig(
    outcome_weight=0.2,
    own_hp_weight=0.3,
    enemy_hp_weight=0.3,
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
    battle_lanes=14,
    gpu_batch_size=128,
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

# Population training
#
# Each member starts from data/models/crystal/crystal_${i}_00200.pt.
# One member is trained for POPULATION_BLOCK_ITERATIONS, then the script
# rotates to the next member. During a block, every other population member
# is frozen and acts as a "current" opponent at its most recent weights.
POPULATION_SIZE = 10
POPULATION_BLOCK_ITERATIONS = 10
POPULATION_INITIAL_WEIGHTS = {
    index: Path(f"data/models/crystal/crystal_{index}_00200.pt")
    for index in range(1, POPULATION_SIZE + 1)
}

# Exact battle split for BOTH rollout and evaluation.
# "current" is sampled uniformly across the other 9 population members.
# These values must sum to 1.0.
OPPONENT_WEIGHTS = {"simple_heuristics": 0.0, "current": 0.5, "historical": 0.5}

HISTORICAL_GROUP = "historical"
SIMPLE_HEURISTICS_GROUP = "simple_heuristics"
CURRENT_GROUP = "current"
TRAINING_MODEL_KEY = "training"

HISTORICAL_POOL_SIZE = 5

# Single-member invocation semantics.
# A process resumes one member, trains at most STEPS_PER_RUN PPO iterations,
# evaluates once, saves, and exits. Run the script repeatedly for each member.
INITIAL_STEP = 200
LAST_STEP = 300
STEPS_PER_RUN = 10
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def historical_model_key(member_index: int, iteration: int) -> str:
    return f"historical/model_{member_index:02d}/iteration_{iteration:05d}"


def make_historical_pool(
    *, current_iteration: int, team_seed: int
) -> list[tuple[int, int]]:
    available_iterations = list(
        range(INITIAL_STEP, current_iteration, STEPS_PER_RUN)
    )

    if not available_iterations:
        available_iterations = [INITIAL_STEP]

    candidates = [
        (member_index, iteration)
        for iteration in available_iterations
        for member_index in range(1, POPULATION_SIZE + 1)
    ]

    rng = random.Random(team_seed + current_iteration * 1_000_003)
    rng.shuffle(candidates)

    return candidates[:HISTORICAL_POOL_SIZE]


def load_historical_opponents(
    *,
    device: torch.device,
    model_config: ModelConfig,
    running_config: RunningConfig,
    population_prefix: str,
    historical_pool: list[tuple[int, int]],
) -> dict[str, BattleModel]:
    models: dict[str, BattleModel] = {}

    for member_index, iteration in historical_pool:
        weights = member_weights_at_iteration(
            running_config=running_config,
            population_prefix=population_prefix,
            member_index=member_index,
            iteration=iteration,
        )

        model = create_model(device, model_config, weights)
        model.eval()
        model.requires_grad_(False)

        key = historical_model_key(member_index, iteration)

        models[key] = model

        print(
            f"Historical opponent: "
            f"model={member_index:02d} "
            f"iteration={iteration} "
            f"weights={weights}"
        )

    return models


def create_model(
    device: torch.device,
    model_config: ModelConfig,
    starting_weights: Path | None = None,
) -> BattleModel:
    torch.manual_seed(model_config.seed)
    model = BattleModel(
        config=model_config,
        pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
        field_numeric_feature_count=FIELD_NUMERIC_DIM,
        tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
    )

    if starting_weights is not None:
        checkpoint = torch.load(
            starting_weights, map_location=device, weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Checkpoint {starting_weights} must contain a dict"
            )

        model_state = checkpoint.get("model", checkpoint)

        if not isinstance(model_state, dict):
            raise TypeError(
                f"Checkpoint {starting_weights} has invalid model state"
            )

        model.load_state_dict(model_state)
        print(f"Loaded starting weights from {starting_weights}")

    model.to(device)
    return model


@dataclass(frozen=True)
class OpponentChoice:
    group: str
    model_key: str | None
    model_name: str


@dataclass(frozen=True)
class PopulationInferenceRequest:
    worker_index: int
    slot_index: int
    request_id: int
    model_key: str


@dataclass(frozen=True)
class PopulationInferenceResponse:
    slot_index: int
    request_id: int
    logits: tuple[float, ...] | None
    value: float | None
    error: str | None = None


_REQUEST_QUEUE: ProcessQueue | None = None
_RESPONSE_QUEUES: list[ProcessQueue] | None = None
_SHARED_BUFFER: SharedBattleBuffer | None = None


def population_gpu_worker_initializer(
    torch_threads: int,
    request_queue: ProcessQueue,
    response_queues: list[ProcessQueue],
    shared_buffer: SharedBattleBuffer,
) -> None:
    worker_initializer(torch_threads)

    global _REQUEST_QUEUE
    global _RESPONSE_QUEUES
    global _SHARED_BUFFER

    _REQUEST_QUEUE = request_queue
    _RESPONSE_QUEUES = response_queues
    _SHARED_BUFFER = shared_buffer


def _worker_transport(
    worker_index: int,
) -> tuple[ProcessQueue, ProcessQueue, SharedBattleBuffer]:
    request_queue = _REQUEST_QUEUE
    response_queues = _RESPONSE_QUEUES
    shared_buffer = _SHARED_BUFFER

    if (
        request_queue is None
        or response_queues is None
        or shared_buffer is None
    ):
        raise RuntimeError("GPU inference transport was not initialized")

    if not 0 <= worker_index < len(response_queues):
        raise IndexError(f"No response queue for worker {worker_index}")

    return request_queue, response_queues[worker_index], shared_buffer


async def response_pump(
    *, worker_index: int, pending: PendingInference
) -> None:
    _, response_queue, _ = _worker_transport(worker_index)

    while True:
        response = await asyncio.to_thread(response_queue.get)

        if response is None:
            return

        if not isinstance(response, PopulationInferenceResponse):
            raise TypeError(
                "GPU inference broker returned invalid response "
                f"of type {type(response).__name__}"
            )

        key = (response.slot_index, response.request_id)
        future = pending.pop(key, None)

        if future is None:
            raise RuntimeError(
                "Received GPU inference response with no pending request: "
                f"slot={response.slot_index} request={response.request_id}"
            )

        if response.error is not None:
            future.set_exception(RuntimeError(response.error))
            continue

        if response.logits is None or response.value is None:
            future.set_exception(
                RuntimeError("GPU inference broker returned an empty result")
            )
            continue

        if len(response.logits) != 10:
            future.set_exception(
                RuntimeError(
                    f"Expected 10 policy logits, got {len(response.logits)}"
                )
            )
            continue

        future.set_result(
            (torch.tensor(response.logits, dtype=torch.float32), response.value)
        )


def stop_response_pump(worker_index: int) -> None:
    _, response_queue, _ = _worker_transport(worker_index)
    response_queue.put(None)


def make_remote_infer(
    *,
    worker_index: int,
    slot_index: int,
    model_key: str,
    pending: PendingInference,
    stats: WorkerInferenceStats | None = None,
    timeout_seconds: float = 120.0,
) -> AsyncInferenceFn:
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be > 0")

    request_queue, _, shared_buffer = _worker_transport(worker_index)

    if not 0 <= slot_index < shared_buffer.slot_count:
        raise IndexError(f"No shared inference slot {slot_index}")

    next_request_id = 0

    async def infer(observation: BattleTensors) -> tuple[torch.Tensor, float]:
        nonlocal next_request_id

        request_id = next_request_id
        next_request_id += 1

        key = (slot_index, request_id)
        loop = asyncio.get_running_loop()

        future: asyncio.Future[tuple[torch.Tensor, float]] = (
            loop.create_future()
        )

        if key in pending:
            raise RuntimeError(
                "Duplicate pending inference request: "
                f"slot={slot_index} request={request_id}"
            )

        pending[key] = future

        write_start = perf_counter()
        shared_buffer.write(slot_index, observation)
        write_end = perf_counter()

        request_queue.put(
            PopulationInferenceRequest(
                worker_index=worker_index,
                slot_index=slot_index,
                request_id=request_id,
                model_key=model_key,
            )
        )
        put_end = perf_counter()

        if stats is not None:
            stats.requests += 1
            stats.shared_write_seconds += write_end - write_start
            stats.queue_put_seconds += put_end - write_end

        try:
            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=timeout_seconds
            )

            if stats is not None:
                stats.response_wait_seconds += perf_counter() - put_end

            return result

        except BaseException:
            if stats is not None:
                stats.response_wait_seconds += perf_counter() - put_end

            pending.pop(key, None)

            if not future.done():
                future.cancel()

            raise

    return infer


class PopulationGpuInferenceBroker:
    def __init__(
        self,
        *,
        models: dict[str, BattleModel],
        device: torch.device,
        request_queue: ProcessQueue,
        response_queues: list[ProcessQueue],
        shared_buffer: SharedBattleBuffer,
        max_batch_size: int,
        batch_wait_ms: float,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                f"PopulationGpuInferenceBroker requires CUDA/ROCm, got {device}"
            )

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")

        if batch_wait_ms < 0.0:
            raise ValueError("batch_wait_ms must be >= 0")

        if TRAINING_MODEL_KEY not in models:
            raise ValueError(
                f"models must contain the {TRAINING_MODEL_KEY!r} model"
            )

        self.models = models
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.shared_buffer = shared_buffer
        self.max_batch_size = max_batch_size
        self.batch_wait_seconds = batch_wait_ms / 1000.0

        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()

        self._requests = 0
        self._batches = 0
        self._max_observed_batch_size = 0
        self._total_batch_wait_seconds = 0.0
        self._total_gather_seconds = 0.0
        self._total_inference_seconds = 0.0
        self._total_dispatch_seconds = 0.0

    def set_training_model(self, model: BattleModel) -> None:
        # The caller only switches models between fully awaited rollout/eval phases.
        self.models[TRAINING_MODEL_KEY] = model

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU inference broker is already running")

        self._thread = threading.Thread(
            target=self._run, name="population-gpu-inference", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread

        if thread is None:
            return

        self.request_queue.put(None)
        thread.join(timeout=30.0)

        if thread.is_alive():
            raise RuntimeError("GPU inference broker did not stop cleanly")

        self._thread = None

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._requests = 0
            self._batches = 0
            self._max_observed_batch_size = 0
            self._total_batch_wait_seconds = 0.0
            self._total_gather_seconds = 0.0
            self._total_inference_seconds = 0.0
            self._total_dispatch_seconds = 0.0

    def snapshot_stats(self) -> GpuInferenceStats:
        with self._stats_lock:
            return GpuInferenceStats(
                requests=self._requests,
                batches=self._batches,
                max_batch_size=self._max_observed_batch_size,
                total_batch_wait_seconds=self._total_batch_wait_seconds,
                total_gather_seconds=self._total_gather_seconds,
                total_inference_seconds=self._total_inference_seconds,
                total_dispatch_seconds=self._total_dispatch_seconds,
            )

    def _run(self) -> None:
        stop_after_batch = False

        while True:
            item = self.request_queue.get()

            if item is None:
                return

            if not isinstance(item, PopulationInferenceRequest):
                raise TypeError(
                    "GPU inference request queue received invalid object "
                    f"of type {type(item).__name__}"
                )

            requests = [item]
            batch_wait_start = perf_counter()

            if self.batch_wait_seconds > 0.0:
                deadline = perf_counter() + self.batch_wait_seconds

                while len(requests) < self.max_batch_size:
                    remaining = deadline - perf_counter()

                    if remaining <= 0.0:
                        break

                    try:
                        next_item = self.request_queue.get(timeout=remaining)
                    except queue.Empty:
                        break

                    if next_item is None:
                        stop_after_batch = True
                        break

                    if not isinstance(next_item, PopulationInferenceRequest):
                        raise TypeError(
                            "GPU inference request queue received invalid object "
                            f"of type {type(next_item).__name__}"
                        )

                    requests.append(next_item)
            else:
                while len(requests) < self.max_batch_size:
                    try:
                        next_item = self.request_queue.get_nowait()
                    except queue.Empty:
                        break

                    if next_item is None:
                        stop_after_batch = True
                        break

                    if not isinstance(next_item, PopulationInferenceRequest):
                        raise TypeError(
                            "GPU inference request queue received invalid object "
                            f"of type {type(next_item).__name__}"
                        )

                    requests.append(next_item)

            batch_wait_seconds = perf_counter() - batch_wait_start

            with self._stats_lock:
                self._total_batch_wait_seconds += batch_wait_seconds

            self._process_batch(requests)

            if stop_after_batch:
                return

    def _process_batch(
        self, requests: list[PopulationInferenceRequest]
    ) -> None:
        by_model: dict[str, list[PopulationInferenceRequest]] = {}

        for request in requests:
            model = self.models.get(request.model_key)

            if model is None:
                self._send_errors(
                    [request], f"Unknown inference model {request.model_key!r}"
                )
                continue

            by_model.setdefault(request.model_key, []).append(request)

        for model_key, model_requests in by_model.items():
            model = self.models[model_key]

            gather_start = perf_counter()
            cpu_batch = self.shared_buffer.batch(
                [request.slot_index for request in model_requests]
            )
            gather_seconds = perf_counter() - gather_start

            inference_start = perf_counter()
            gpu_batch = cpu_batch.to(self.device)

            with torch.inference_mode():
                logits, values = model(gpu_batch)

            logits_cpu = logits.detach().cpu()
            values_cpu = values.detach().cpu()
            inference_seconds = perf_counter() - inference_start

            dispatch_start = perf_counter()

            for index, request in enumerate(model_requests):
                self.response_queues[request.worker_index].put(
                    PopulationInferenceResponse(
                        slot_index=request.slot_index,
                        request_id=request.request_id,
                        logits=tuple(
                            float(value) for value in logits_cpu[index].tolist()
                        ),
                        value=float(values_cpu[index].item()),
                    )
                )

            dispatch_seconds = perf_counter() - dispatch_start
            count = len(model_requests)

            with self._stats_lock:
                self._requests += count
                self._batches += 1
                self._max_observed_batch_size = max(
                    self._max_observed_batch_size, count
                )
                self._total_gather_seconds += gather_seconds
                self._total_inference_seconds += inference_seconds
                self._total_dispatch_seconds += dispatch_seconds

    def _send_errors(
        self, requests: list[PopulationInferenceRequest], error_text: str
    ) -> None:
        for request in requests:
            self.response_queues[request.worker_index].put(
                PopulationInferenceResponse(
                    slot_index=request.slot_index,
                    request_id=request.request_id,
                    logits=None,
                    value=None,
                    error=error_text,
                )
            )


def population_model_key(member_index: int) -> str:
    return f"current/model_{member_index:02d}"


def validate_population_config() -> None:
    if POPULATION_SIZE < 2:
        raise ValueError("POPULATION_SIZE must be at least 2")

    if POPULATION_BLOCK_ITERATIONS <= 0:
        raise ValueError("POPULATION_BLOCK_ITERATIONS must be positive")

    total_weight = sum(OPPONENT_WEIGHTS.values())

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            f"OPPONENT_WEIGHTS must sum to 1.0, got {total_weight}"
        )

    if any(weight < 0.0 for weight in OPPONENT_WEIGHTS.values()):
        raise ValueError("Opponent weights must be non-negative")

    supported_groups = {
        SIMPLE_HEURISTICS_GROUP,
        CURRENT_GROUP,
        HISTORICAL_GROUP,
    }

    unknown_groups = set(OPPONENT_WEIGHTS) - supported_groups

    if unknown_groups:
        raise ValueError(
            "Unknown opponent groups: " + ", ".join(sorted(unknown_groups))
        )

    if OPPONENT_WEIGHTS.get(CURRENT_GROUP, 0.0) > 0.0 and POPULATION_SIZE < 2:
        raise ValueError(
            "current opponents require at least two population members"
        )

    for member_index, path in POPULATION_INITIAL_WEIGHTS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing initial weights for model {member_index}: {path}"
            )


def _weighted_group_counts(battles: int) -> dict[str, int]:
    exact = {
        group: battles * weight for group, weight in OPPONENT_WEIGHTS.items()
    }

    counts = {group: int(value) for group, value in exact.items()}

    remaining = battles - sum(counts.values())

    order = sorted(
        exact, key=lambda group: (-(exact[group] - counts[group]), group)
    )

    for group in order[:remaining]:
        counts[group] += 1

    return counts


def make_opponent_schedule(
    *,
    battles: int,
    active_member_index: int,
    phase_id: int,
    team_seed: int,
    historical_pool: list[tuple[int, int]],
) -> list[OpponentChoice]:
    counts = _weighted_group_counts(battles)

    rng = random.Random(
        team_seed + phase_id * 1_000_003 + active_member_index * 10_007
    )

    schedule: list[OpponentChoice] = []

    heuristic_count = counts.get(SIMPLE_HEURISTICS_GROUP, 0)

    schedule.extend(
        OpponentChoice(
            group=SIMPLE_HEURISTICS_GROUP,
            model_key=None,
            model_name=SIMPLE_HEURISTICS_GROUP,
        )
        for _ in range(heuristic_count)
    )

    current_count = counts.get(CURRENT_GROUP, 0)

    if current_count > 0:
        opponents = [
            member_index
            for member_index in range(1, POPULATION_SIZE + 1)
            if member_index != active_member_index
        ]

        rng.shuffle(opponents)

        for index in range(current_count):
            opponent_index = opponents[index % len(opponents)]

            schedule.append(
                OpponentChoice(
                    group=CURRENT_GROUP,
                    model_key=population_model_key(opponent_index),
                    model_name=f"model_{opponent_index:02d}",
                )
            )

    historical_count = counts.get(HISTORICAL_GROUP, 0)

    if historical_count > 0:
        if not historical_pool:
            raise RuntimeError(
                "Historical opponents requested but "
                "no historical checkpoints are available"
            )

        historical_choices = list(historical_pool)
        rng.shuffle(historical_choices)

        for index in range(historical_count):
            member_index, iteration = historical_choices[
                index % len(historical_choices)
            ]

            schedule.append(
                OpponentChoice(
                    group=HISTORICAL_GROUP,
                    model_key=historical_model_key(member_index, iteration),
                    model_name=(
                        f"model_{member_index:02d}_iteration_{iteration:05d}"
                    ),
                )
            )

    rng.shuffle(schedule)

    if len(schedule) != battles:
        raise RuntimeError(
            f"Opponent schedule has {len(schedule)} entries, expected {battles}"
        )

    return schedule


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


def _team_generators(
    *, fmt: str, team_seed: int, phase_id: int, slot_index: int
) -> tuple[SampleTeamGenerator | None, SampleTeamGenerator | None]:
    if "randombattle" in fmt:
        return None, None

    seed = team_seed + phase_id * 100_000 + slot_index * 2
    return SampleTeamGenerator(seed), SampleTeamGenerator(seed + 1)


def _opponent_handler(
    *,
    choice: OpponentChoice,
    tensorizer: BattleTensorizer,
    worker_index: int,
    opponent_slot_index: int,
    pending: PendingInference,
) -> AsyncBaseCombatHandler:
    if choice.model_key is None:
        return AsyncSimpleHeuristicsCombatHandler()

    infer = make_remote_infer(
        worker_index=worker_index,
        slot_index=opponent_slot_index,
        model_key=choice.model_key,
        pending=pending,
    )

    return AsyncNeuralCombatHandler(tensorizer=tensorizer, infer=infer)


async def collect_trajectories(
    *,
    neural_client: Client,
    opponent_client: Client,
    handler: AsyncTrainingCombatHandler,
    tensorizer: BattleTensorizer,
    worker_index: int,
    opponent_slot_index: int,
    pending: PendingInference,
    fmt: str,
    team_generator_1: BaseTeamGenerator | None,
    team_generator_2: BaseTeamGenerator | None,
    opponent_choices: list[OpponentChoice],
    reward_config: RewardConfig,
) -> list[Trajectory]:
    if neural_client.username is None:
        raise RuntimeError("Neural client has no username")

    neural_client.combat_handler = handler

    trajectories: list[Trajectory] = []
    discarded_battles = 0

    for choice in opponent_choices:
        opponent_client.combat_handler = _opponent_handler(
            choice=choice,
            tensorizer=tensorizer,
            worker_index=worker_index,
            opponent_slot_index=opponent_slot_index,
            pending=pending,
        )

        while True:
            handler.start_battle()

            try:
                result, _ = await run_battle(
                    neural_client,
                    opponent_client,
                    fmt=fmt,
                    team_generator_1=team_generator_1,
                    team_generator_2=team_generator_2,
                )
                break

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
                    opponent_client.close(),
                    return_exceptions=True,
                )

                await asyncio.gather(
                    neural_client.ensure_connected(),
                    opponent_client.ensure_connected(),
                )

        outcome = outcome_for(result, neural_client.username)
        breakdown = breakdown_for(result, outcome, config=reward_config)

        trajectory = handler.finish_battle(outcome, breakdown)

        if not trajectory.decisions:
            raise RuntimeError("Collected empty trajectory")

        trajectories.append(trajectory)

    if discarded_battles > 0:
        print(
            f"Discarded {discarded_battles} failed battles "
            f"while collecting {len(opponent_choices)} trajectories"
        )

    return trajectories


async def evaluate(
    *,
    neural_client: Client,
    opponent_client: Client,
    handler: AsyncNeuralCombatHandler,
    tensorizer: BattleTensorizer,
    worker_index: int,
    opponent_slot_index: int,
    pending: PendingInference,
    fmt: str,
    team_generator_1: BaseTeamGenerator | None,
    team_generator_2: BaseTeamGenerator | None,
    opponent_choices: list[OpponentChoice],
) -> tuple[
    int,
    int,
    int,
    dict[str, tuple[int, int, int]],
    dict[str, tuple[int, int, int]],
]:
    if neural_client.username is None:
        raise RuntimeError("Neural client has no username")

    neural_client.combat_handler = handler

    wins = 0
    losses = 0
    ties = 0

    by_group: dict[str, tuple[int, int, int]] = {}
    by_model: dict[str, tuple[int, int, int]] = {}

    for choice in opponent_choices:
        opponent_client.combat_handler = _opponent_handler(
            choice=choice,
            tensorizer=tensorizer,
            worker_index=worker_index,
            opponent_slot_index=opponent_slot_index,
            pending=pending,
        )

        result, _ = await run_battle(
            neural_client,
            opponent_client,
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

        group_wins, group_losses, group_ties = by_group.get(
            choice.group, (0, 0, 0)
        )

        model_wins, model_losses, model_ties = by_model.get(
            choice.model_name, (0, 0, 0)
        )

        if outcome > 0:
            group_wins += 1
            model_wins += 1
        elif outcome < 0:
            group_losses += 1
            model_losses += 1
        else:
            group_ties += 1
            model_ties += 1

        by_group[choice.group] = (group_wins, group_losses, group_ties)

        by_model[choice.model_name] = (model_wins, model_losses, model_ties)

    return wins, losses, ties, by_group, by_model


async def _rollout_lane(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    worker_index: int,
    lane_index: int,
    battle_lanes: int,
    phase_id: int,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    pending: PendingInference,
    inference_stats: WorkerInferenceStats,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> list[Trajectory]:
    slot_index = worker_index * battle_lanes + lane_index
    opponent_slot_index = active_slot_count + slot_index

    infer = make_remote_infer(
        worker_index=worker_index,
        slot_index=slot_index,
        model_key=TRAINING_MODEL_KEY,
        pending=pending,
        stats=inference_stats,
    )

    handler = AsyncTrainingCombatHandler(tensorizer=tensorizer, infer=infer)

    neural_client = Client(url, combat_handler=handler)
    opponent_client = Client(
        url, combat_handler=SimpleHeuristicsCombatHandler()
    )

    neural_client.log_manager.disable()
    opponent_client.log_manager.disable()

    team_generator_1, team_generator_2 = _team_generators(
        fmt=fmt, team_seed=team_seed, phase_id=phase_id, slot_index=slot_index
    )

    neural_name = f"A{phase_id}N{slot_index}"
    opponent_name = f"A{phase_id}O{slot_index}"

    try:
        await asyncio.gather(neural_client.connect(), opponent_client.connect())

        await asyncio.gather(
            neural_client.login(neural_name),
            opponent_client.login(opponent_name),
        )

        return await collect_trajectories(
            neural_client=neural_client,
            opponent_client=opponent_client,
            handler=handler,
            tensorizer=tensorizer,
            worker_index=worker_index,
            opponent_slot_index=opponent_slot_index,
            pending=pending,
            fmt=fmt,
            team_generator_1=team_generator_1,
            team_generator_2=team_generator_2,
            opponent_choices=opponent_choices,
            reward_config=reward_config,
        )

    finally:
        await asyncio.gather(
            neural_client.close(),
            opponent_client.close(),
            return_exceptions=True,
        )


async def _rollout_worker_async(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    output_path: str,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> str:
    lane_counts = split_battles(len(opponent_choices), battle_lanes)

    pending: PendingInference = {}
    inference_stats = WorkerInferenceStats()

    pump_task = asyncio.create_task(
        response_pump(worker_index=worker_index, pending=pending)
    )

    lane_choices: list[list[OpponentChoice]] = []
    offset = 0

    for count in lane_counts:
        lane_choices.append(opponent_choices[offset : offset + count])
        offset += count

    try:
        tasks = [
            asyncio.create_task(
                _rollout_lane(
                    url=url,
                    fmt=fmt,
                    team_seed=team_seed,
                    worker_index=worker_index,
                    lane_index=lane_index,
                    battle_lanes=battle_lanes,
                    phase_id=phase_id,
                    reward_config=reward_config,
                    tensorizer=tensorizer,
                    pending=pending,
                    inference_stats=inference_stats,
                    opponent_choices=choices,
                    active_slot_count=active_slot_count,
                )
            )
            for lane_index, choices in enumerate(lane_choices)
            if choices
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

        rollout = PackedRollout.from_trajectories(trajectories)

        torch.save(rollout, output_path)
        return output_path

    finally:
        stop_response_pump(worker_index)
        await pump_task


def _rollout_worker(
    url: str,
    fmt: str,
    team_seed: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    output_path: str,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> str:
    return asyncio.run(
        _rollout_worker_async(
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            worker_index=worker_index,
            battle_lanes=battle_lanes,
            phase_id=phase_id,
            output_path=output_path,
            reward_config=reward_config,
            tensorizer=tensorizer,
            opponent_choices=opponent_choices,
            active_slot_count=active_slot_count,
        )
    )


async def _evaluation_lane(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    worker_index: int,
    lane_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    pending: PendingInference,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> tuple[
    int,
    int,
    int,
    dict[str, tuple[int, int, int]],
    dict[str, tuple[int, int, int]],
]:
    slot_index = worker_index * battle_lanes + lane_index
    opponent_slot_index = active_slot_count + slot_index

    infer = make_remote_infer(
        worker_index=worker_index,
        slot_index=slot_index,
        model_key=TRAINING_MODEL_KEY,
        pending=pending,
    )

    handler = AsyncNeuralCombatHandler(tensorizer=tensorizer, infer=infer)

    neural_client = Client(url, combat_handler=handler)
    opponent_client = Client(
        url, combat_handler=SimpleHeuristicsCombatHandler()
    )

    neural_client.log_manager.disable()
    opponent_client.log_manager.disable()

    team_generator_1, team_generator_2 = _team_generators(
        fmt=fmt, team_seed=team_seed, phase_id=phase_id, slot_index=slot_index
    )

    neural_name = f"E{phase_id}N{slot_index}"
    opponent_name = f"E{phase_id}O{slot_index}"

    try:
        await asyncio.gather(neural_client.connect(), opponent_client.connect())

        await asyncio.gather(
            neural_client.login(neural_name),
            opponent_client.login(opponent_name),
        )

        return await evaluate(
            neural_client=neural_client,
            opponent_client=opponent_client,
            handler=handler,
            tensorizer=tensorizer,
            worker_index=worker_index,
            opponent_slot_index=opponent_slot_index,
            pending=pending,
            fmt=fmt,
            team_generator_1=team_generator_1,
            team_generator_2=team_generator_2,
            opponent_choices=opponent_choices,
        )

    finally:
        await asyncio.gather(
            neural_client.close(),
            opponent_client.close(),
            return_exceptions=True,
        )


async def _evaluation_worker_async(
    *,
    url: str,
    fmt: str,
    team_seed: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> tuple[
    int,
    int,
    int,
    dict[str, tuple[int, int, int]],
    dict[str, tuple[int, int, int]],
]:
    lane_counts = split_battles(len(opponent_choices), battle_lanes)

    pending: PendingInference = {}

    pump_task = asyncio.create_task(
        response_pump(worker_index=worker_index, pending=pending)
    )

    lane_choices: list[list[OpponentChoice]] = []
    offset = 0

    for count in lane_counts:
        lane_choices.append(opponent_choices[offset : offset + count])
        offset += count

    try:
        tasks = [
            asyncio.create_task(
                _evaluation_lane(
                    url=url,
                    fmt=fmt,
                    team_seed=team_seed,
                    worker_index=worker_index,
                    lane_index=lane_index,
                    battle_lanes=battle_lanes,
                    phase_id=phase_id,
                    tensorizer=tensorizer,
                    pending=pending,
                    opponent_choices=choices,
                    active_slot_count=active_slot_count,
                )
            )
            for lane_index, choices in enumerate(lane_choices)
            if choices
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
                f"Worker {worker_index} finished with "
                f"{len(pending)} pending inference requests"
            )

        return _merge_evaluation_results(results)

    finally:
        stop_response_pump(worker_index)
        await pump_task


def _evaluation_worker(
    url: str,
    fmt: str,
    team_seed: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> tuple[
    int,
    int,
    int,
    dict[str, tuple[int, int, int]],
    dict[str, tuple[int, int, int]],
]:
    return asyncio.run(
        _evaluation_worker_async(
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            worker_index=worker_index,
            battle_lanes=battle_lanes,
            phase_id=phase_id,
            tensorizer=tensorizer,
            opponent_choices=opponent_choices,
            active_slot_count=active_slot_count,
        )
    )


def _merge_count_maps(
    maps: list[dict[str, tuple[int, int, int]]],
) -> dict[str, tuple[int, int, int]]:
    merged: dict[str, tuple[int, int, int]] = {}

    for mapping in maps:
        for key, counts in mapping.items():
            wins, losses, ties = merged.get(key, (0, 0, 0))

            merged[key] = (
                wins + counts[0],
                losses + counts[1],
                ties + counts[2],
            )

    return merged


def _merge_evaluation_results(
    results: list[
        tuple[
            int,
            int,
            int,
            dict[str, tuple[int, int, int]],
            dict[str, tuple[int, int, int]],
        ]
    ],
) -> tuple[
    int,
    int,
    int,
    dict[str, tuple[int, int, int]],
    dict[str, tuple[int, int, int]],
]:
    return (
        sum(result[0] for result in results),
        sum(result[1] for result in results),
        sum(result[2] for result in results),
        _merge_count_maps([result[3] for result in results]),
        _merge_count_maps([result[4] for result in results]),
    )


async def collect_trajectories_multiprocess(
    *,
    pool: ProcessPoolExecutor,
    url: str,
    fmt: str,
    team_seed: int,
    worker_count: int,
    battle_lanes: int,
    phase_id: int,
    temporary_directory: Path,
    reward_config: RewardConfig,
    tensorizer: BattleTensorizer,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> PackedRollout:
    counts = split_battles(len(opponent_choices), worker_count)

    loop = asyncio.get_running_loop()
    tasks = []

    offset = 0

    for worker_index, count in enumerate(counts):
        if count <= 0:
            continue

        worker_choices = opponent_choices[offset : offset + count]
        offset += count

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
                    worker_index,
                    battle_lanes,
                    phase_id,
                    str(output_path),
                    reward_config,
                    tensorizer,
                    worker_choices,
                    active_slot_count,
                ),
            )
        )

    worker_results = await asyncio.gather(*tasks)

    rollout_chunks: list[PackedRollout] = []

    for output_path_string in worker_results:
        output_path = Path(output_path_string)

        worker_rollout = torch.load(
            output_path, map_location="cpu", weights_only=False
        )

        if not isinstance(worker_rollout, PackedRollout):
            raise TypeError("Rollout worker returned an invalid packed rollout")

        rollout_chunks.append(worker_rollout)
        output_path.unlink()

    return PackedRollout.concat(rollout_chunks)


async def evaluate_multiprocess(
    *,
    pool: ProcessPoolExecutor,
    url: str,
    fmt: str,
    team_seed: int,
    worker_count: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    opponent_choices: list[OpponentChoice],
    active_slot_count: int,
) -> tuple[
    int,
    int,
    int,
    dict[str, tuple[int, int, int]],
    dict[str, tuple[int, int, int]],
]:
    counts = split_battles(len(opponent_choices), worker_count)

    loop = asyncio.get_running_loop()
    tasks = []

    offset = 0

    for worker_index, count in enumerate(counts):
        if count <= 0:
            continue

        worker_choices = opponent_choices[offset : offset + count]
        offset += count

        tasks.append(
            loop.run_in_executor(
                pool,
                partial(
                    _evaluation_worker,
                    url,
                    fmt,
                    team_seed,
                    worker_index,
                    battle_lanes,
                    phase_id,
                    tensorizer,
                    worker_choices,
                    active_slot_count,
                ),
            )
        )

    results = await asyncio.gather(*tasks)

    return _merge_evaluation_results(results)


def add_eval_breakdown_logs(
    *,
    log_data: dict[str, float | int],
    prefix: str,
    by_group: dict[str, tuple[int, int, int]],
    by_model: dict[str, tuple[int, int, int]],
) -> None:
    for group, counts in by_group.items():
        wins, losses, ties = counts
        battles = wins + losses + ties

        log_data[f"{prefix}/by_group/{group}/battles"] = battles
        log_data[f"{prefix}/by_group/{group}/wins"] = wins
        log_data[f"{prefix}/by_group/{group}/losses"] = losses
        log_data[f"{prefix}/by_group/{group}/ties"] = ties
        log_data[f"{prefix}/by_group/{group}/win_rate"] = wins / battles

    for model_name, counts in by_model.items():
        wins, losses, ties = counts
        battles = wins + losses + ties

        log_data[f"{prefix}/by_model/{model_name}/battles"] = battles
        log_data[f"{prefix}/by_model/{model_name}/wins"] = wins
        log_data[f"{prefix}/by_model/{model_name}/losses"] = losses
        log_data[f"{prefix}/by_model/{model_name}/ties"] = ties
        log_data[f"{prefix}/by_model/{model_name}/win_rate"] = wins / battles


def infer_population_prefix(*, wandb_name: str, model_index: int) -> str:
    suffix = f"-{model_index}"

    if not wandb_name.endswith(suffix):
        raise ValueError(
            f"--wandb-name must end with {suffix!r} for "
            f"--model-index {model_index}; got {wandb_name!r}"
        )

    prefix = wandb_name[: -len(suffix)]

    if not prefix:
        raise ValueError(
            "--wandb-name must contain a population prefix, "
            f"for example fire-{model_index}"
        )

    return prefix


def member_run_name(*, population_prefix: str, member_index: int) -> str:
    return f"{population_prefix}-{member_index}"


def member_checkpoint_dir(
    *, running_config: RunningConfig, population_prefix: str, member_index: int
) -> Path:
    return running_config.checkpoint_dir / member_run_name(
        population_prefix=population_prefix, member_index=member_index
    )


def member_weights_at_iteration(
    *,
    running_config: RunningConfig,
    population_prefix: str,
    member_index: int,
    iteration: int,
) -> Path:
    if iteration == INITIAL_STEP:
        initial = POPULATION_INITIAL_WEIGHTS[member_index]

        if not initial.is_file():
            raise FileNotFoundError(
                f"Missing initial weights for member {member_index}: {initial}"
            )

        return initial

    checkpoint = (
        member_checkpoint_dir(
            running_config=running_config,
            population_prefix=population_prefix,
            member_index=member_index,
        )
        / f"iteration_{iteration:05d}.pt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing frozen opponent checkpoint for member "
            f"{member_index} at iteration {iteration}: {checkpoint}"
        )

    return checkpoint


def load_active_member(
    *,
    device: torch.device,
    model_config: ModelConfig,
    ppo_config: PPOConfig,
    checkpoint_dir: Path,
    member_index: int,
) -> tuple[BattleModel, torch.optim.Optimizer, int]:
    latest = checkpoint_dir / "latest.pt"

    if latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint {latest} must contain a dict")

        model_state = checkpoint.get("model")

        if not isinstance(model_state, dict):
            raise TypeError(f"Checkpoint {latest} has no valid model state")

        optimizer_state = checkpoint.get("optimizer")

        if not isinstance(optimizer_state, dict):
            raise TypeError(f"Checkpoint {latest} has no valid optimizer state")

        iteration = checkpoint.get("iteration")

        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TypeError(
                f"Checkpoint {latest} has invalid iteration {iteration!r}"
            )

        model = create_model(device, model_config)
        model.load_state_dict(model_state)
        model.eval()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=ppo_config.learning_rate
        )
        optimizer.load_state_dict(optimizer_state)

        # Keep the configured learning rate authoritative in case it changed.
        for param_group in optimizer.param_groups:
            param_group["lr"] = ppo_config.learning_rate

        print(
            f"Resumed model {member_index:02d} from {latest} "
            f"at iteration {iteration}"
        )

        return model, optimizer, iteration

    initial_weights = POPULATION_INITIAL_WEIGHTS[member_index]

    model = create_model(device, model_config, initial_weights)
    model.eval()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo_config.learning_rate
    )

    print(
        f"Starting model {member_index:02d} from "
        f"{initial_weights} at iteration {INITIAL_STEP}"
    )

    return model, optimizer, INITIAL_STEP


def load_frozen_opponents(
    *,
    device: torch.device,
    model_config: ModelConfig,
    running_config: RunningConfig,
    population_prefix: str,
    active_member_index: int,
    opponent_iteration: int,
) -> dict[str, BattleModel]:
    models: dict[str, BattleModel] = {}

    for member_index in range(1, POPULATION_SIZE + 1):
        if member_index == active_member_index:
            continue

        weights = member_weights_at_iteration(
            running_config=running_config,
            population_prefix=population_prefix,
            member_index=member_index,
            iteration=opponent_iteration,
        )

        model = create_model(device, model_config, weights)
        model.eval()
        model.requires_grad_(False)

        model_key = population_model_key(member_index)
        models[model_key] = model

        print(
            f"Opponent model {member_index:02d}: "
            f"iteration={opponent_iteration} "
            f"weights={weights}"
        )

    return models


def init_wandb_run(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    running_config: RunningConfig,
    training_config: TrainingConfig,
    ppo_config: PPOConfig,
    reward_config: RewardConfig,
    model_index: int,
    population_prefix: str,
    current_iteration: int,
    parameter_count: int,
):
    if args.no_wandb:
        return None

    run_id_path = checkpoint_dir / WANDB_RUN_ID_FILENAME

    if run_id_path.is_file():
        run_id = run_id_path.read_text().strip()

        if not run_id:
            raise ValueError(f"W&B run ID file is empty: {run_id_path}")

        run = wandb.init(
            project=running_config.wandb_project,
            entity=running_config.wandb_entity,
            id=run_id,
            resume="must",
        )

        print(f"Resumed W&B run {run_id} for {args.wandb_name}")
        return run

    running_dict = asdict(running_config)
    running_dict["checkpoint_dir"] = str(running_dict["checkpoint_dir"])

    run = wandb.init(
        project=running_config.wandb_project,
        entity=running_config.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name,
        config={
            "running": running_dict,
            "training": asdict(training_config),
            "ppo": asdict(ppo_config),
            "reward": asdict(reward_config),
            "population_size": POPULATION_SIZE,
            "population_prefix": population_prefix,
            "model_index": model_index,
            "initial_step": INITIAL_STEP,
            "last_step": LAST_STEP,
            "steps_per_run": STEPS_PER_RUN,
            "opponent_weights": OPPONENT_WEIGHTS,
            "initial_weights": str(POPULATION_INITIAL_WEIGHTS[model_index]),
            "parameter_count": parameter_count,
            "resumed_from_iteration": current_iteration,
        },
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id_path.write_text(run.id)

    print(f"Created W&B run {run.id} for {args.wandb_name}")

    return run


async def train_member(
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    reward_config: RewardConfig,
    running_config: RunningConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
) -> None:
    validate_population_config()

    if not 1 <= args.model_index <= POPULATION_SIZE:
        raise ValueError(
            f"--model-index must be between 1 and {POPULATION_SIZE}"
        )

    if STEPS_PER_RUN <= 0:
        raise ValueError("STEPS_PER_RUN must be positive")

    if LAST_STEP < INITIAL_STEP:
        raise ValueError("LAST_STEP must be >= INITIAL_STEP")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "train_member.py requires a CUDA/ROCm PyTorch device"
        )

    if args.wandb_name is None:
        raise ValueError("--wandb-name is required")

    population_prefix = infer_population_prefix(
        wandb_name=args.wandb_name, model_index=args.model_index
    )

    checkpoint_dir = member_checkpoint_dir(
        running_config=running_config,
        population_prefix=population_prefix,
        member_index=args.model_index,
    )

    device = torch.device("cuda")

    model, optimizer, current_iteration = load_active_member(
        device=device,
        model_config=model_config,
        ppo_config=ppo_config,
        checkpoint_dir=checkpoint_dir,
        member_index=args.model_index,
    )

    if current_iteration > LAST_STEP:
        raise ValueError(
            f"Checkpoint is already at iteration {current_iteration}, "
            f"past LAST_STEP={LAST_STEP}"
        )

    if current_iteration == LAST_STEP:
        print(
            f"{args.wandb_name} is already complete at "
            f"iteration {LAST_STEP}; nothing to do."
        )
        return

    target_iteration = min(current_iteration + STEPS_PER_RUN, LAST_STEP)

    print(f"Training device: {device}")
    print(f"Population: {population_prefix}")
    print(f"Active member: {args.model_index:02d}")
    print(f"Iterations: {current_iteration + 1}..{target_iteration}")
    print(f"Opponent split: {OPPONENT_WEIGHTS}")

    opponent_models = load_frozen_opponents(
        device=device,
        model_config=model_config,
        running_config=running_config,
        population_prefix=population_prefix,
        active_member_index=args.model_index,
        opponent_iteration=current_iteration,
    )
    historical_pool = make_historical_pool(
        current_iteration=current_iteration, team_seed=training_config.team_seed
    )

    historical_models = load_historical_opponents(
        device=device,
        model_config=model_config,
        running_config=running_config,
        population_prefix=population_prefix,
        historical_pool=historical_pool,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    wandb_run = init_wandb_run(
        args=args,
        checkpoint_dir=checkpoint_dir,
        running_config=running_config,
        training_config=training_config,
        ppo_config=ppo_config,
        reward_config=reward_config,
        model_index=args.model_index,
        population_prefix=population_prefix,
        current_iteration=current_iteration,
        parameter_count=parameter_count,
    )

    context = multiprocessing.get_context("spawn")

    active_slot_count = running_config.workers * running_config.battle_lanes

    # A lane may have simultaneous inference requests from both battle sides.
    slot_count = active_slot_count * 2

    request_queue = context.Queue(
        maxsize=max(slot_count * 2, running_config.gpu_batch_size * 4)
    )

    response_queues = [
        context.Queue(maxsize=max(running_config.battle_lanes * 4, 16))
        for _ in range(running_config.workers)
    ]

    shared_buffer = SharedBattleBuffer.create(
        slot_count=slot_count, max_history=tensorizer.max_history
    )

    broker_models = {
        TRAINING_MODEL_KEY: model,
        **opponent_models,
        **historical_models,
    }

    inference_broker = PopulationGpuInferenceBroker(
        models=broker_models,
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        shared_buffer=shared_buffer,
        max_batch_size=running_config.gpu_batch_size,
        batch_wait_ms=running_config.gpu_batch_wait_ms,
    )

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False
    phase_id = current_iteration * 2

    inference_broker.start()

    try:
        pool = ProcessPoolExecutor(
            max_workers=running_config.workers,
            mp_context=context,
            initializer=population_gpu_worker_initializer,
            initargs=(
                running_config.threads,
                request_queue,
                response_queues,
                shared_buffer,
            ),
        )

        with tempfile.TemporaryDirectory(
            prefix="ai-cif-member-rollouts-"
        ) as temporary_directory_string:
            temporary_directory = Path(temporary_directory_string)

            final_train_log: dict[str, float | int] | None = None

            for iteration in range(current_iteration + 1, target_iteration + 1):
                phase_id += 1

                opponent_choices = make_opponent_schedule(
                    battles=training_config.rollout_battles,
                    active_member_index=args.model_index,
                    phase_id=phase_id,
                    team_seed=training_config.team_seed,
                    historical_pool=historical_pool,
                )

                inference_broker.reset_stats()
                rollout_start = perf_counter()

                trajectories = await collect_trajectories_multiprocess(
                    pool=pool,
                    url=running_config.url,
                    fmt=running_config.format,
                    team_seed=training_config.team_seed,
                    worker_count=running_config.workers,
                    battle_lanes=running_config.battle_lanes,
                    phase_id=phase_id,
                    temporary_directory=temporary_directory,
                    reward_config=reward_config,
                    tensorizer=tensorizer,
                    opponent_choices=opponent_choices,
                    active_slot_count=active_slot_count,
                )

                rollout_seconds = perf_counter() - rollout_start

                rollout_inference_stats = inference_broker.snapshot_stats()

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

                battle_count = trajectories.battle_count
                train_win_rate = wins / battle_count
                mean_decisions = decisions / battle_count
                reward_breakdowns = trajectories.reward_breakdowns

                timestamp = datetime.now(tz=UTC).strftime("%c")

                print()
                print(timestamp)
                print(
                    f"model={args.model_index:02d} "
                    f"iteration={iteration} "
                    f"battles={battle_count} "
                    f"decisions={decisions}"
                )
                print(f"train wins={wins} losses={losses} ties={ties}")
                print(
                    f"rollout_time={rollout_seconds:.2f}s "
                    f"battles/s="
                    f"{battle_count / rollout_seconds:.2f} "
                    f"decisions/s="
                    f"{decisions / rollout_seconds:.1f}"
                )
                print(f"ppo_time={ppo_seconds:.2f}s")
                print_gpu_inference_stats(
                    "gpu_inference", rollout_inference_stats
                )

                log_data: dict[str, float | int] = {
                    "iteration/index": iteration,
                    "population/model_index": (args.model_index),
                    "train/battles": battle_count,
                    "train/decisions": decisions,
                    "train/wins": wins,
                    "train/losses": losses,
                    "train/ties": ties,
                    "train/win_rate": train_win_rate,
                    "train/mean_reward": mean_reward,
                    "train/mean_decisions_per_battle": (mean_decisions),
                    "rollout/seconds": rollout_seconds,
                    "rollout/battles_per_second": (
                        battle_count / rollout_seconds
                    ),
                    "rollout/decisions_per_second": (
                        decisions / rollout_seconds
                    ),
                    "gpu_inference/requests": (
                        rollout_inference_stats.requests
                    ),
                    "gpu_inference/batches": (rollout_inference_stats.batches),
                    "gpu_inference/mean_batch_size": (
                        rollout_inference_stats.mean_batch_size
                    ),
                    "gpu_inference/max_batch_size": (
                        rollout_inference_stats.max_batch_size
                    ),
                    "gpu_inference/seconds": (
                        rollout_inference_stats.total_inference_seconds
                    ),
                    "ppo/seconds": ppo_seconds,
                    "ppo/policy_loss": metrics.policy_loss,
                    "ppo/value_loss": metrics.value_loss,
                    "ppo/entropy": metrics.entropy,
                    "ppo/total_loss": metrics.total_loss,
                    "ppo/approx_kl": metrics.approx_kl,
                    "ppo/max_approx_kl": (metrics.max_approx_kl),
                    "ppo/clip_fraction": (metrics.clip_fraction),
                    "ppo/early_stop": int(metrics.early_stop),
                    "ppo/mean_value": metrics.mean_value,
                    "ppo/mean_return": metrics.mean_return,
                    "optimizer/learning_rate": (
                        optimizer.param_groups[0]["lr"]
                    ),
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

                if iteration == target_iteration:
                    final_train_log = log_data
                elif wandb_run is not None:
                    wandb_run.log(log_data, step=iteration)

            # One evaluation per invocation, after this 10-step block.
            phase_id += 1

            eval_choices = make_opponent_schedule(
                battles=training_config.eval_battles,
                active_member_index=args.model_index,
                phase_id=phase_id,
                team_seed=training_config.team_seed,
                historical_pool=historical_pool,
            )

            inference_broker.reset_stats()
            evaluation_start = perf_counter()

            (
                eval_wins,
                eval_losses,
                eval_ties,
                eval_by_group,
                eval_by_model,
            ) = await evaluate_multiprocess(
                pool=pool,
                url=running_config.url,
                fmt=running_config.format,
                team_seed=training_config.team_seed,
                worker_count=running_config.workers,
                battle_lanes=running_config.battle_lanes,
                phase_id=phase_id,
                tensorizer=tensorizer,
                opponent_choices=eval_choices,
                active_slot_count=active_slot_count,
            )

            evaluation_seconds = perf_counter() - evaluation_start

            eval_inference_stats = inference_broker.snapshot_stats()

            eval_battle_count = eval_wins + eval_losses + eval_ties
            win_rate = eval_wins / eval_battle_count

            print()
            print(
                f"EVAL model={args.model_index:02d} "
                f"iteration={target_iteration} "
                f"wins={eval_wins} "
                f"losses={eval_losses} "
                f"ties={eval_ties} "
                f"win_rate={win_rate:.1%}"
            )

            for group, counts in sorted(eval_by_group.items()):
                group_battles = sum(counts)
                print(
                    f"  group={group} "
                    f"battles={group_battles} "
                    f"win_rate="
                    f"{counts[0] / group_battles:.1%}"
                )

            for opponent_name, counts in sorted(eval_by_model.items()):
                opponent_battles = sum(counts)
                print(
                    f"  opponent={opponent_name} "
                    f"battles={opponent_battles} "
                    f"win_rate="
                    f"{counts[0] / opponent_battles:.1%}"
                )

            if wandb_run is not None:
                eval_log: dict[str, float | int] = {
                    "iteration/index": target_iteration,
                    "population/model_index": (args.model_index),
                    "eval/wins": eval_wins,
                    "eval/losses": eval_losses,
                    "eval/ties": eval_ties,
                    "eval/win_rate": win_rate,
                    "eval/seconds": evaluation_seconds,
                    "eval/battles_per_second": (
                        eval_battle_count / evaluation_seconds
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

                add_eval_breakdown_logs(
                    log_data=eval_log,
                    prefix="eval",
                    by_group=eval_by_group,
                    by_model=eval_by_model,
                )

                if final_train_log is None:
                    raise RuntimeError(
                        "Missing final training log for evaluation step"
                    )

                final_train_log.update(eval_log)

                wandb_run.log(final_train_log, step=target_iteration)

            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            save_checkpoint(
                path=(checkpoint_dir / f"iteration_{target_iteration:05d}.pt"),
                model=model,
                optimizer=optimizer,
                iteration=target_iteration,
            )

            save_checkpoint(
                path=checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                iteration=target_iteration,
            )

            print(f"Saved {args.wandb_name} at iteration {target_iteration}")

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

    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--model-index", required=True, type=int)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", default=None)

    parser.add_argument("--set-training", nargs="*", default=[])
    parser.add_argument("--set-ppo", nargs="*", default=[])
    parser.add_argument("--set-reward", nargs="*", default=[])
    parser.add_argument("--set-running", nargs="*", default=[])
    parser.add_argument("--set-model", nargs="*", default=[])

    return parser.parse_args()


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

    await train_member(
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
