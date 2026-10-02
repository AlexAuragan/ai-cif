"""
This one is not trained from scratch but from crystal-x-00200 and against the SimpleHeuristic handler
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import queue
import random
import tempfile
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import asdict, dataclass, field
from functools import partial
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from statistics import median
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
from showdown_sdk.models.sdk.team_generators.team_generator import (
    BaseTeamGenerator,
)

import wandb
from ai_cif.inference.combat_handler import AsyncNeuralCombatHandler
from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel, TransformerBattleModel
from ai_cif.training.combat_handler import AsyncTrainingCombatHandler
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
    HISTORY_KIND_VOCAB_SIZE,
    HISTORY_REF_VOCAB_SIZE,
    STATUS_VOCAB_SIZE,
    WEATHER_VOCAB_SIZE,
    BattleTensorizer,
    BattleTensors,
)
from scripts.utils.battles import outcome_for, run_battle, team_generators
from scripts.utils.config import (
    MODEL_TYPES,
    PPO_TYPES,
    REWARD_TYPES,
    RUNNING_TYPES,
    TRAINING_TYPES,
    apply_overrides,
)
from scripts.utils.gpu import (
    GpuInferenceStats,
    SharedBattleBuffer,
    WorkerInferenceStats,
)
from scripts.utils.model import create_model
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
    own_hp_weight=0.4,
    enemy_hp_weight=0.4,
    speed_weight=0.0,
    speed_scale=40.0,
)

PPO_CONFIG = PPOConfig(
    learning_rate=1e-4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    epochs=6,
    minibatch_size=512,
    kl_target=0.02,
    kl_ratio_threshold=2,
    gamma=1,
    gae_lambda=0.98,
)


TRAINING_CONFIG = TrainingConfig(
    iterations=1000,
    rollout_battles=1000,
    # Kept because TrainingConfig requires it. Population evaluation below uses
    # EVAL_BATTLES_PER_OPPONENT instead of a single total battle count.
    eval_battles=1000,
    eval_interval=20,
    team_seed=42,
)


RUNNING_CONFIG = RunningConfig(
    url=DEFAULT_WEBSOCKET_URL,
    format="gen4randombattle",
    workers=20,
    threads=2,
    checkpoint_dir=Path("checkpoints"),
    wandb_project="ai-cif",
    wandb_entity=None,
    battle_lanes=10,
    gpu_batch_size=256,
    gpu_batch_wait_ms=2,
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
# All ten members start from the SAME supervised checkpoint. Each population
# round freezes the population at step 10*k and trains every member from
# 10*k -> 10*(k+1) against that frozen population. This keeps opponent strength
# fair even though members are trained sequentially on one desktop GPU.
POPULATION_SIZE = 10
STEPS_PER_ROUND = 10

# Change this one path to your supervised-learning checkpoint.
SUPERVISED_INITIAL_WEIGHTS = Path("data/models/platinium/platinium_00300.pt")
POPULATION_INITIAL_WEIGHTS = {
    index: SUPERVISED_INITIAL_WEIGHTS for index in range(1, POPULATION_SIZE + 1)
}

# Training distribution only. Evaluation has a separate fixed-style benchmark.
TRAIN_OPPONENT_WEIGHTS = {
    "simple_heuristics": 0.0,
    "current": 0.5,
    "historical": 0.5,
}

HISTORICAL_GROUP = "historical"
OWN_HISTORY_GROUP = "own_history"
COMPETITOR_HISTORY_GROUP = "competitor_history"
BASELINE_GROUP = "supervised_baseline"
SIMPLE_HEURISTICS_GROUP = "simple_heuristics"
CURRENT_GROUP = "current"
TRAINING_MODEL_KEY = "training"

# Only multiples of 100 become eligible historical strategies. The training
# historical pool stays bounded: baseline + broad coverage over old snapshots.
HISTORICAL_SNAPSHOT_INTERVAL = 100
HISTORICAL_POOL_SIZE = 10

# Benchmark/evaluation. Each listed opponent gets this many battles. Current
# population members are deliberately NOT part of this pool.
EVAL_BATTLES_PER_OPPONENT = 1000
EVAL_COMPETITOR_HISTORY_PER_SNAPSHOT = 1

# Recovery is intentionally conservative. A member must BOTH regress on the
# common benchmark opponents versus its previous eval and sit below the current
# population median before extra PPO updates are considered.
BENCHMARK_DROP_TOLERANCE = 0.02
RECOVERY_REQUIRE_BELOW_POPULATION_MEDIAN = True
RECOVERY_POPULATION_MARGIN = 0.0
RECOVERY_UPDATES_PER_BLOCK = 10
MAX_RECOVERY_BLOCKS = 2

INITIAL_STEP = 0
LAST_STEP = 1000
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"
POPULATION_STATE_FILENAME = "population_state.json"
EVAL_RESULTS_DIRNAME = "eval_results"


def historical_model_key(member_index: int, iteration: int) -> str:
    return f"historical/model_{member_index:02d}/iteration_{iteration:05d}"


def historical_snapshot_iterations(current_iteration: int) -> list[int]:
    return list(
        range(
            HISTORICAL_SNAPSHOT_INTERVAL,
            current_iteration,
            HISTORICAL_SNAPSHOT_INTERVAL,
        )
    )


def make_historical_pool(
    *, current_iteration: int, team_seed: int
) -> list[tuple[int, int]]:
    # All members share the same supervised checkpoint at iteration 0, so one
    # reference is enough; loading ten identical copies only wastes GPU memory.
    pool: list[tuple[int, int]] = [(1, INITIAL_STEP)]
    snapshots = historical_snapshot_iterations(current_iteration)

    if not snapshots or HISTORICAL_POOL_SIZE <= 1:
        return pool[:HISTORICAL_POOL_SIZE]

    rng = random.Random(team_seed + current_iteration * 1_000_003)

    # First take one model from as many snapshot generations as possible. This
    # avoids accidentally sampling a pool made entirely of near-identical ages.
    snapshot_order = list(snapshots)
    rng.shuffle(snapshot_order)

    for iteration in snapshot_order:
        members = list(range(1, POPULATION_SIZE + 1))
        rng.shuffle(members)
        pool.append((members[0], iteration))

        if len(pool) >= HISTORICAL_POOL_SIZE:
            return pool

    # If the run is still young, fill remaining slots with other members from
    # the snapshot generations that do exist.
    remaining = [
        (member_index, iteration)
        for iteration in snapshots
        for member_index in range(1, POPULATION_SIZE + 1)
        if (member_index, iteration) not in pool
    ]
    rng.shuffle(remaining)
    pool.extend(remaining[: HISTORICAL_POOL_SIZE - len(pool)])
    return pool


def load_historical_opponents(
    *,
    device: torch.device,
    model_config: ModelConfig,
    running_config: RunningConfig,
    population_prefix: str,
    historical_pool: list[tuple[int, int]],
) -> dict[str, BattleModel | TransformerBattleModel]:
    models: dict[str, BattleModel | TransformerBattleModel] = {}

    for member_index, iteration in historical_pool:
        weights = member_weights_at_iteration(
            running_config=running_config,
            population_prefix=population_prefix,
            member_index=member_index,
            iteration=iteration,
        )

        model = create_model(device, model_config, "transfomer", weights)
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


@dataclass(frozen=True)
class OpponentChoice:
    group: str
    model_key: str | None
    model_name: str


def evaluation_history_choices(
    *, active_member_index: int, population_step: int, team_seed: int
) -> tuple[list[OpponentChoice], list[tuple[int, int]]]:
    choices: list[OpponentChoice] = []
    model_refs: list[tuple[int, int]] = []

    # One shared supervised baseline. All ten iteration-0 members are identical.
    baseline_ref = (1, INITIAL_STEP)
    model_refs.append(baseline_ref)
    choices.extend(
        OpponentChoice(
            group=BASELINE_GROUP,
            model_key=historical_model_key(*baseline_ref),
            model_name=BASELINE_GROUP,
        )
        for _ in range(EVAL_BATTLES_PER_OPPONENT)
    )

    for iteration in historical_snapshot_iterations(population_step):
        own_ref = (active_member_index, iteration)
        model_refs.append(own_ref)
        choices.extend(
            OpponentChoice(
                group=OWN_HISTORY_GROUP,
                model_key=historical_model_key(*own_ref),
                model_name=(
                    f"own/model_{active_member_index:02d}/"
                    f"iteration_{iteration:05d}"
                ),
            )
            for _ in range(EVAL_BATTLES_PER_OPPONENT)
        )

        competitors = [
            member_index
            for member_index in range(1, POPULATION_SIZE + 1)
            if member_index != active_member_index
        ]
        rng = random.Random(
            team_seed + active_member_index * 10_007 + iteration * 1_000_003
        )
        rng.shuffle(competitors)

        for competitor_index in competitors[
            :EVAL_COMPETITOR_HISTORY_PER_SNAPSHOT
        ]:
            competitor_ref = (competitor_index, iteration)
            model_refs.append(competitor_ref)
            choices.extend(
                OpponentChoice(
                    group=COMPETITOR_HISTORY_GROUP,
                    model_key=historical_model_key(*competitor_ref),
                    model_name=(
                        f"competitor/model_{competitor_index:02d}/"
                        f"iteration_{iteration:05d}"
                    ),
                )
                for _ in range(EVAL_BATTLES_PER_OPPONENT)
            )

    # Preserve order while removing duplicate model loads.
    unique_refs = list(dict.fromkeys(model_refs))
    return choices, unique_refs


def make_evaluation_opponent_schedule(
    *,
    active_member_index: int,
    population_step: int,
    phase_id: int,
    team_seed: int,
) -> tuple[list[OpponentChoice], list[tuple[int, int]]]:
    history_choices, history_refs = evaluation_history_choices(
        active_member_index=active_member_index,
        population_step=population_step,
        team_seed=team_seed,
    )

    choices = [
        OpponentChoice(
            group=SIMPLE_HEURISTICS_GROUP,
            model_key=None,
            model_name=SIMPLE_HEURISTICS_GROUP,
        )
        for _ in range(EVAL_BATTLES_PER_OPPONENT)
    ]
    choices.extend(history_choices)

    # Use the same shuffle seed for every member at a given evaluation step so
    # SimpleHeuristics/baseline see the same team-seed positions across models.
    rng = random.Random(team_seed + phase_id * 1_000_003)
    rng.shuffle(choices)
    return choices, history_refs


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
        models: dict[str, BattleModel | TransformerBattleModel],
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

    if STEPS_PER_ROUND <= 0:
        raise ValueError("STEPS_PER_ROUND must be positive")

    if HISTORICAL_SNAPSHOT_INTERVAL <= 0:
        raise ValueError("HISTORICAL_SNAPSHOT_INTERVAL must be positive")

    if HISTORICAL_POOL_SIZE <= 0:
        raise ValueError("HISTORICAL_POOL_SIZE must be positive")

    if EVAL_BATTLES_PER_OPPONENT <= 0:
        raise ValueError("EVAL_BATTLES_PER_OPPONENT must be positive")

    if EVAL_COMPETITOR_HISTORY_PER_SNAPSHOT < 0:
        raise ValueError(
            "EVAL_COMPETITOR_HISTORY_PER_SNAPSHOT must be non-negative"
        )

    if EVAL_COMPETITOR_HISTORY_PER_SNAPSHOT >= POPULATION_SIZE:
        raise ValueError(
            "EVAL_COMPETITOR_HISTORY_PER_SNAPSHOT must be smaller than "
            "POPULATION_SIZE"
        )

    if LAST_STEP < INITIAL_STEP:
        raise ValueError("LAST_STEP must be >= INITIAL_STEP")

    if BENCHMARK_DROP_TOLERANCE < 0.0:
        raise ValueError("BENCHMARK_DROP_TOLERANCE must be non-negative")

    if RECOVERY_UPDATES_PER_BLOCK <= 0:
        raise ValueError("RECOVERY_UPDATES_PER_BLOCK must be positive")

    if MAX_RECOVERY_BLOCKS < 0:
        raise ValueError("MAX_RECOVERY_BLOCKS must be non-negative")

    total_weight = sum(TRAIN_OPPONENT_WEIGHTS.values())

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            f"TRAIN_OPPONENT_WEIGHTS must sum to 1.0, got {total_weight}"
        )

    if any(weight < 0.0 for weight in TRAIN_OPPONENT_WEIGHTS.values()):
        raise ValueError("Training opponent weights must be non-negative")

    supported_groups = {
        SIMPLE_HEURISTICS_GROUP,
        CURRENT_GROUP,
        HISTORICAL_GROUP,
    }
    unknown_groups = set(TRAIN_OPPONENT_WEIGHTS) - supported_groups

    if unknown_groups:
        raise ValueError(
            "Unknown training opponent groups: "
            + ", ".join(sorted(unknown_groups))
        )

    if not SUPERVISED_INITIAL_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Missing supervised initial weights: {SUPERVISED_INITIAL_WEIGHTS}"
        )


def _weighted_group_counts(battles: int) -> dict[str, int]:
    exact = {
        group: battles * weight
        for group, weight in TRAIN_OPPONENT_WEIGHTS.items()
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
    discarded_battles = 0
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

        while True:
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
                    "Discarding failed eval battle and retrying: "
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

    if discarded_battles > 0:
        print(
            f"Discarded {discarded_battles} failed eval battles while "
            f"collecting {len(opponent_choices)} completed battles"
        )

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

    team_generator_1, team_generator_2 = team_generators(
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

    team_generator_1, team_generator_2 = team_generators(
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


def member_run_name(*, population_prefix: str, member_index: int) -> str:
    return f"{population_prefix}-{member_index}"


def population_checkpoint_dir(
    *, running_config: RunningConfig, population_prefix: str
) -> Path:
    return running_config.checkpoint_dir / population_prefix


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
        if not SUPERVISED_INITIAL_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"Missing supervised initial weights: {SUPERVISED_INITIAL_WEIGHTS}"
            )
        return SUPERVISED_INITIAL_WEIGHTS

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
            f"Missing frozen opponent checkpoint for member {member_index} "
            f"at population step {iteration}: {checkpoint}"
        )
    return checkpoint


@dataclass(frozen=True)
class MemberProgress:
    iteration: int
    ppo_updates_total: int
    recovery_updates_total: int


@dataclass
class PopulationState:
    round_start: int = INITIAL_STEP
    current_member: int = 1
    phase: str = "train"
    eval_member: int = 1
    recovery_queue: list[int] = field(default_factory=list)
    recovery_index: int = 0
    recovery_blocks_done: dict[str, int] = field(default_factory=dict)
    recovery_target_ppo_updates: int | None = None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_member_checkpoint(
    *,
    path: Path,
    model: BattleModel | TransformerBattleModel,
    optimizer: torch.optim.Optimizer,
    progress: MemberProgress,
) -> None:
    _atomic_torch_save(
        path,
        {
            "iteration": progress.iteration,
            "ppo_updates_total": progress.ppo_updates_total,
            "recovery_updates_total": progress.recovery_updates_total,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
    )


def read_member_progress_from_checkpoint(path: Path) -> MemberProgress:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} must contain a dict")

    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError(
            f"Checkpoint {path} has invalid iteration {iteration!r}"
        )

    ppo_updates_total = checkpoint.get("ppo_updates_total", iteration)
    recovery_updates_total = checkpoint.get("recovery_updates_total", 0)

    if isinstance(ppo_updates_total, bool) or not isinstance(
        ppo_updates_total, int
    ):
        raise TypeError(
            f"Checkpoint {path} has invalid ppo_updates_total "
            f"{ppo_updates_total!r}"
        )

    if isinstance(recovery_updates_total, bool) or not isinstance(
        recovery_updates_total, int
    ):
        raise TypeError(
            f"Checkpoint {path} has invalid recovery_updates_total "
            f"{recovery_updates_total!r}"
        )

    return MemberProgress(
        iteration=iteration,
        ppo_updates_total=ppo_updates_total,
        recovery_updates_total=recovery_updates_total,
    )


def member_progress_on_disk(
    *, running_config: RunningConfig, population_prefix: str, member_index: int
) -> MemberProgress:
    latest = (
        member_checkpoint_dir(
            running_config=running_config,
            population_prefix=population_prefix,
            member_index=member_index,
        )
        / "latest.pt"
    )

    if not latest.is_file():
        return MemberProgress(
            iteration=INITIAL_STEP,
            ppo_updates_total=0,
            recovery_updates_total=0,
        )
    return read_member_progress_from_checkpoint(latest)


def load_active_member(
    *,
    device: torch.device,
    model_config: ModelConfig,
    ppo_config: PPOConfig,
    checkpoint_dir: Path,
    member_index: int,
) -> tuple[
    BattleModel | TransformerBattleModel, torch.optim.Optimizer, MemberProgress
]:
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

        progress = read_member_progress_from_checkpoint(latest)

        model = create_model(device, model_config, "transfomer")
        model.load_state_dict(model_state)
        model.eval()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=ppo_config.learning_rate
        )
        optimizer.load_state_dict(optimizer_state)

        for param_group in optimizer.param_groups:
            param_group["lr"] = ppo_config.learning_rate

        print(
            f"Resumed model {member_index:02d} from {latest} at "
            f"population_step={progress.iteration} "
            f"ppo_updates={progress.ppo_updates_total} "
            f"recovery_updates={progress.recovery_updates_total}"
        )
        return model, optimizer, progress

    model = create_model(
        device, model_config, "transfomer", SUPERVISED_INITIAL_WEIGHTS
    )
    model.eval()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo_config.learning_rate
    )
    progress = MemberProgress(
        iteration=INITIAL_STEP, ppo_updates_total=0, recovery_updates_total=0
    )

    print(
        f"Starting model {member_index:02d} from shared supervised weights "
        f"{SUPERVISED_INITIAL_WEIGHTS} at population step {INITIAL_STEP}"
    )
    return model, optimizer, progress


def load_frozen_opponents(
    *,
    device: torch.device,
    model_config: ModelConfig,
    running_config: RunningConfig,
    population_prefix: str,
    active_member_index: int,
    opponent_iteration: int,
) -> dict[str, BattleModel | TransformerBattleModel]:
    models: dict[str, BattleModel | TransformerBattleModel] = {}

    # At step 0 every population member is bit-for-bit identical. Load one GPU
    # model and alias the nine opponent keys to it instead of wasting memory on
    # nine copies of the supervised checkpoint.
    shared_initial_model: BattleModel | TransformerBattleModel | None = None
    if opponent_iteration == INITIAL_STEP:
        shared_initial_model = create_model(
            device, model_config, "transfomer", SUPERVISED_INITIAL_WEIGHTS
        )
        shared_initial_model.eval()
        shared_initial_model.requires_grad_(False)

    for member_index in range(1, POPULATION_SIZE + 1):
        if member_index == active_member_index:
            continue

        weights = member_weights_at_iteration(
            running_config=running_config,
            population_prefix=population_prefix,
            member_index=member_index,
            iteration=opponent_iteration,
        )
        if shared_initial_model is not None:
            model = shared_initial_model
        else:
            model = create_model(device, model_config, "transfomer", weights)
            model.eval()
            model.requires_grad_(False)

        models[population_model_key(member_index)] = model
        print(
            f"Frozen current opponent model={member_index:02d} "
            f"population_step={opponent_iteration} weights={weights}"
        )

    return models


def population_state_path(
    *, running_config: RunningConfig, population_prefix: str
) -> Path:
    return (
        population_checkpoint_dir(
            running_config=running_config, population_prefix=population_prefix
        )
        / POPULATION_STATE_FILENAME
    )


def save_population_state(
    *,
    state: PopulationState,
    running_config: RunningConfig,
    population_prefix: str,
) -> None:
    _atomic_write_json(
        population_state_path(
            running_config=running_config, population_prefix=population_prefix
        ),
        asdict(state),
    )


def evaluation_result_path(
    *,
    running_config: RunningConfig,
    population_prefix: str,
    population_step: int,
) -> Path:
    return (
        population_checkpoint_dir(
            running_config=running_config, population_prefix=population_prefix
        )
        / EVAL_RESULTS_DIRNAME
        / f"iteration_{population_step:05d}.json"
    )


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def latest_completed_evaluation_before(
    *,
    running_config: RunningConfig,
    population_prefix: str,
    population_step: int,
) -> dict | None:
    directory = (
        population_checkpoint_dir(
            running_config=running_config, population_prefix=population_prefix
        )
        / EVAL_RESULTS_DIRNAME
    )

    if not directory.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("iteration_*.json"):
        stem = path.stem
        try:
            step = int(stem.removeprefix("iteration_"))
        except ValueError:
            continue
        if step < population_step:
            candidates.append((step, path))

    for _, path in sorted(candidates, reverse=True):
        payload = load_json(path)
        if payload.get("complete") is True:
            return payload
    return None


def infer_population_state(
    *, running_config: RunningConfig, population_prefix: str, eval_interval: int
) -> PopulationState:
    progresses = [
        member_progress_on_disk(
            running_config=running_config,
            population_prefix=population_prefix,
            member_index=member_index,
        )
        for member_index in range(1, POPULATION_SIZE + 1)
    ]
    iterations = [progress.iteration for progress in progresses]
    minimum = min(iterations)

    # If every member is synchronized exactly on an evaluation boundary, first
    # check whether that evaluation finished. This matters if the coordinator
    # state file was lost after training completed but before/during evaluation.
    if len(set(iterations)) == 1 and minimum > INITIAL_STEP:
        synchronized_eval_due = (
            minimum % eval_interval == 0 or minimum == LAST_STEP
        )
        synchronized_eval_path = evaluation_result_path(
            running_config=running_config,
            population_prefix=population_prefix,
            population_step=minimum,
        )
        if synchronized_eval_due and (
            not synchronized_eval_path.is_file()
            or load_json(synchronized_eval_path).get("complete") is not True
        ):
            print(
                "Reconstructed synchronized population state; resuming "
                f"evaluation at step {minimum}."
            )
            return PopulationState(
                round_start=max(INITIAL_STEP, minimum - STEPS_PER_ROUND),
                current_member=POPULATION_SIZE + 1,
                phase="eval",
                eval_member=1,
            )

    round_start = (minimum // STEPS_PER_ROUND) * STEPS_PER_ROUND
    target = min(round_start + STEPS_PER_ROUND, LAST_STEP)

    for member_index, progress in enumerate(progresses, start=1):
        if progress.iteration < target:
            print(
                "Reconstructed population state from checkpoints: "
                f"round_start={round_start}, resume_member={member_index}"
            )
            return PopulationState(
                round_start=round_start,
                current_member=member_index,
                phase="train",
            )

    # All members completed the inferred round. If this is an evaluation step
    # and there is no completed eval file, resume evaluation instead of silently
    # advancing to the next training round.
    completed_step = target
    eval_path = evaluation_result_path(
        running_config=running_config,
        population_prefix=population_prefix,
        population_step=completed_step,
    )
    eval_due = (
        completed_step % eval_interval == 0 or completed_step == LAST_STEP
    )

    if eval_due and (
        not eval_path.is_file()
        or load_json(eval_path).get("complete") is not True
    ):
        print(
            "Reconstructed population state after a completed training round; "
            f"resuming evaluation at step {completed_step}."
        )
        return PopulationState(
            round_start=max(INITIAL_STEP, completed_step - STEPS_PER_ROUND),
            current_member=POPULATION_SIZE + 1,
            phase="eval",
            eval_member=1,
        )

    return PopulationState(
        round_start=completed_step, current_member=1, phase="train"
    )


def load_population_state(
    *, running_config: RunningConfig, population_prefix: str, eval_interval: int
) -> PopulationState:
    path = population_state_path(
        running_config=running_config, population_prefix=population_prefix
    )

    if not path.is_file():
        state = infer_population_state(
            running_config=running_config,
            population_prefix=population_prefix,
            eval_interval=eval_interval,
        )
        save_population_state(
            state=state,
            running_config=running_config,
            population_prefix=population_prefix,
        )
        return state

    payload = load_json(path)
    state = PopulationState(
        round_start=int(payload["round_start"]),
        current_member=int(payload.get("current_member", 1)),
        phase=str(payload.get("phase", "train")),
        eval_member=int(payload.get("eval_member", 1)),
        recovery_queue=[
            int(value) for value in payload.get("recovery_queue", [])
        ],
        recovery_index=int(payload.get("recovery_index", 0)),
        recovery_blocks_done={
            str(key): int(value)
            for key, value in payload.get("recovery_blocks_done", {}).items()
        },
        recovery_target_ppo_updates=(
            None
            if payload.get("recovery_target_ppo_updates") is None
            else int(payload["recovery_target_ppo_updates"])
        ),
    )
    print(
        f"Loaded population state: phase={state.phase} "
        f"round_start={state.round_start} "
        f"current_member={state.current_member}"
    )
    return state


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
    progress: MemberProgress,
    parameter_count: int,
):
    if args.no_wandb:
        return None

    run_name = member_run_name(
        population_prefix=population_prefix, member_index=model_index
    )
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
        print(f"Resumed W&B run {run_id} for {run_name}")
    else:
        running_dict = asdict(running_config)
        running_dict["checkpoint_dir"] = str(running_dict["checkpoint_dir"])
        run = wandb.init(
            project=running_config.wandb_project,
            entity=running_config.wandb_entity,
            group=args.wandb_group,
            name=run_name,
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
                "steps_per_round": STEPS_PER_ROUND,
                "historical_snapshot_interval": HISTORICAL_SNAPSHOT_INTERVAL,
                "historical_pool_size": HISTORICAL_POOL_SIZE,
                "training_opponent_weights": TRAIN_OPPONENT_WEIGHTS,
                "eval_battles_per_opponent": EVAL_BATTLES_PER_OPPONENT,
                "eval_competitor_history_per_snapshot": (
                    EVAL_COMPETITOR_HISTORY_PER_SNAPSHOT
                ),
                "benchmark_drop_tolerance": BENCHMARK_DROP_TOLERANCE,
                "recovery_updates_per_block": RECOVERY_UPDATES_PER_BLOCK,
                "max_recovery_blocks": MAX_RECOVERY_BLOCKS,
                "initial_weights": str(SUPERVISED_INITIAL_WEIGHTS),
                "parameter_count": parameter_count,
                "resumed_from_population_step": progress.iteration,
                "resumed_from_ppo_updates": progress.ppo_updates_total,
            },
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run.id)
        print(f"Created W&B run {run.id} for {run_name}")

    # W&B's internal event counter may advance during recovery, but charts use
    # population/step. Extra recovery PPO updates therefore do not pretend to be
    # later comparable population iterations.
    wandb.define_metric("population/step")
    wandb.define_metric("*", step_metric="population/step")
    return run


def training_phase_id(
    *, population_step: int, member_index: int, ppo_updates_total: int
) -> int:
    return (
        population_step * 10_000
        + member_index * 100
        + (ppo_updates_total % 100)
    )


def evaluation_phase_id(*, population_step: int, member_index: int) -> int:
    # Evaluations run sequentially, so usernames may safely reuse the same phase
    # id. Keeping it member-independent gives each model identical team seeds.
    _ = member_index
    return 50_000_000 + population_step

def _training_group_log_data(
    *,
    prefix: str,
    trajectories: PackedRollout,
    opponent_choices: list[OpponentChoice],
) -> dict[str, float | int]:
    if trajectories.battle_count != len(opponent_choices):
        raise RuntimeError(
            "Training rollout/opponent mismatch: "
            f"{trajectories.battle_count} trajectories for "
            f"{len(opponent_choices)} opponent choices"
        )

    outcomes = trajectories.outcomes.tolist()
    rewards = trajectories.rewards.tolist()

    groups: dict[str, dict[str, float | int]] = {}

    for choice, outcome, reward in zip(
        opponent_choices,
        outcomes,
        rewards,
        strict=True,
    ):
        if choice.group not in groups:
            groups[choice.group] = {
                "battles": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "reward_sum": 0.0,
            }

        stats = groups[choice.group]

        stats["battles"] += 1
        stats["reward_sum"] += float(reward)

        if outcome > 0.0:
            stats["wins"] += 1
        elif outcome < 0.0:
            stats["losses"] += 1
        else:
            stats["ties"] += 1

    log_data: dict[str, float | int] = {}

    for group, stats in groups.items():
        battles = int(stats["battles"])
        wins = int(stats["wins"])
        losses = int(stats["losses"])
        ties = int(stats["ties"])
        reward_sum = float(stats["reward_sum"])

        group_prefix = f"{prefix}/{group}"

        log_data[f"{group_prefix}/battles"] = battles
        log_data[f"{group_prefix}/wins"] = wins
        log_data[f"{group_prefix}/losses"] = losses
        log_data[f"{group_prefix}/ties"] = ties
        log_data[f"{group_prefix}/win_rate"] = wins / battles
        log_data[f"{group_prefix}/mean_reward"] = reward_sum / battles

    return log_data

def _training_log_data(
    *,
    member_index: int,
    population_step: int,
    progress: MemberProgress,
    trajectories: PackedRollout,
    opponent_choices: list[OpponentChoice],
    rollout_seconds: float,
    ppo_seconds: float,
    rollout_inference_stats: GpuInferenceStats,
    metrics,
    recovery: bool,
) -> dict[str, float | int]:
    wins, losses, ties, decisions = summarize_trajectories(trajectories)
    battle_count = trajectories.battle_count
    mean_reward = mean_trajectory_reward(trajectories)
    reward_breakdowns = trajectories.reward_breakdowns

    prefix = "recovery" if recovery else "train"

    log_data: dict[str, float | int] = {
        "population/step": population_step,
        "population/model_index": member_index,
        "population/ppo_updates_total": progress.ppo_updates_total,
        "population/recovery_updates_total": progress.recovery_updates_total,

        f"{prefix}/battles": battle_count,
        f"{prefix}/decisions": decisions,
        f"{prefix}/wins": wins,
        f"{prefix}/losses": losses,
        f"{prefix}/ties": ties,
        f"{prefix}/win_rate": wins / battle_count,
        f"{prefix}/mean_reward": mean_reward,
        f"{prefix}/mean_decisions_per_battle": decisions / battle_count,

        f"{prefix}/rollout_seconds": rollout_seconds,
        f"{prefix}/rollout_battles_per_second": (
            battle_count / rollout_seconds
        ),

        f"{prefix}/ppo_seconds": ppo_seconds,
        f"{prefix}/policy_loss": metrics.policy_loss,
        f"{prefix}/value_loss": metrics.value_loss,
        f"{prefix}/entropy": metrics.entropy,
        f"{prefix}/total_loss": metrics.total_loss,
        f"{prefix}/approx_kl": metrics.approx_kl,
        f"{prefix}/max_approx_kl": metrics.max_approx_kl,
        f"{prefix}/clip_fraction": metrics.clip_fraction,
        f"{prefix}/early_stop": int(metrics.early_stop),
        f"{prefix}/mean_value": metrics.mean_value,
        f"{prefix}/mean_return": metrics.mean_return,

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
    }

    log_data.update(
        _training_group_log_data(
            prefix=prefix,
            trajectories=trajectories,
            opponent_choices=opponent_choices,
        )
    )

    return log_data


async def train_member_block(
    *,
    args: argparse.Namespace,
    member_index: int,
    round_start: int,
    target_step: int,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    reward_config: RewardConfig,
    running_config: RunningConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
    population_prefix: str,
) -> None:
    device = torch.device("cuda")
    checkpoint_dir = member_checkpoint_dir(
        running_config=running_config,
        population_prefix=population_prefix,
        member_index=member_index,
    )
    model, optimizer, progress = load_active_member(
        device=device,
        model_config=model_config,
        ppo_config=ppo_config,
        checkpoint_dir=checkpoint_dir,
        member_index=member_index,
    )

    if not round_start <= progress.iteration <= target_step:
        raise RuntimeError(
            f"Model {member_index:02d} is at population step "
            f"{progress.iteration}, expected {round_start}..{target_step} "
            f"for this round"
        )

    historical_pool = make_historical_pool(
        current_iteration=round_start, team_seed=training_config.team_seed
    )
    opponent_models = load_frozen_opponents(
        device=device,
        model_config=model_config,
        running_config=running_config,
        population_prefix=population_prefix,
        active_member_index=member_index,
        opponent_iteration=round_start,
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
        model_index=member_index,
        population_prefix=population_prefix,
        progress=progress,
        parameter_count=parameter_count,
    )

    context = multiprocessing.get_context("spawn")
    active_slot_count = running_config.workers * running_config.battle_lanes
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
    inference_broker = PopulationGpuInferenceBroker(
        models={
            TRAINING_MODEL_KEY: model,
            **opponent_models,
            **historical_models,
        },
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        shared_buffer=shared_buffer,
        max_batch_size=running_config.gpu_batch_size,
        batch_wait_ms=running_config.gpu_batch_wait_ms,
    )

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False
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
            prefix=f"ai-cif-population-{member_index:02d}-"
        ) as temporary_directory_string:
            temporary_directory = Path(temporary_directory_string)

            while progress.iteration < target_step:
                iteration = progress.iteration + 1
                next_total_updates = progress.ppo_updates_total + 1
                phase_id = training_phase_id(
                    population_step=iteration,
                    member_index=member_index,
                    ppo_updates_total=next_total_updates,
                )
                opponent_choices = make_opponent_schedule(
                    battles=training_config.rollout_battles,
                    active_member_index=member_index,
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

                ppo_start = perf_counter()
                metrics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    trajectories=trajectories,
                    config=ppo_config,
                    device=device,
                )
                ppo_seconds = perf_counter() - ppo_start

                progress = MemberProgress(
                    iteration=iteration,
                    ppo_updates_total=next_total_updates,
                    recovery_updates_total=progress.recovery_updates_total,
                )

                # Save BEFORE W&B logging. If the machine dies after this point,
                # training resumes from this PPO update instead of replaying it.
                save_member_checkpoint(
                    path=checkpoint_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    progress=progress,
                )

                log_data = _training_log_data(
                    member_index=member_index,
                    population_step=iteration,
                    progress=progress,
                    trajectories=trajectories,
                    rollout_seconds=rollout_seconds,
                    ppo_seconds=ppo_seconds,
                    rollout_inference_stats=rollout_inference_stats,
                    opponent_choices=opponent_choices,
                    metrics=metrics,
                    recovery=False,
                )
                if wandb_run is not None:
                    wandb_run.log(log_data)

                print(
                    f"model={member_index:02d} "
                    f"population_step={iteration} "
                    f"ppo_updates={progress.ppo_updates_total} "
                    f"round_opponents={round_start}"
                )

            # Round-boundary checkpoint is the immutable opponent snapshot used
            # by every other member during the NEXT population round. Recovery
            # may intentionally overwrite this same logical-step checkpoint.
            save_member_checkpoint(
                path=checkpoint_dir / f"iteration_{target_step:05d}.pt",
                model=model,
                optimizer=optimizer,
                progress=progress,
            )

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


async def train_member_recovery_updates(
    *,
    args: argparse.Namespace,
    member_index: int,
    round_start: int,
    population_step: int,
    target_ppo_updates: int,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    reward_config: RewardConfig,
    running_config: RunningConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
    population_prefix: str,
) -> None:
    device = torch.device("cuda")
    checkpoint_dir = member_checkpoint_dir(
        running_config=running_config,
        population_prefix=population_prefix,
        member_index=member_index,
    )
    model, optimizer, progress = load_active_member(
        device=device,
        model_config=model_config,
        ppo_config=ppo_config,
        checkpoint_dir=checkpoint_dir,
        member_index=member_index,
    )

    if progress.iteration != population_step:
        raise RuntimeError(
            f"Recovery for model {member_index:02d} expected population step "
            f"{population_step}, found {progress.iteration}"
        )

    if progress.ppo_updates_total >= target_ppo_updates:
        return

    historical_pool = make_historical_pool(
        current_iteration=round_start, team_seed=training_config.team_seed
    )
    opponent_models = load_frozen_opponents(
        device=device,
        model_config=model_config,
        running_config=running_config,
        population_prefix=population_prefix,
        active_member_index=member_index,
        opponent_iteration=round_start,
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
        model_index=member_index,
        population_prefix=population_prefix,
        progress=progress,
        parameter_count=parameter_count,
    )

    context = multiprocessing.get_context("spawn")
    active_slot_count = running_config.workers * running_config.battle_lanes
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
    inference_broker = PopulationGpuInferenceBroker(
        models={
            TRAINING_MODEL_KEY: model,
            **opponent_models,
            **historical_models,
        },
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        shared_buffer=shared_buffer,
        max_batch_size=running_config.gpu_batch_size,
        batch_wait_ms=running_config.gpu_batch_wait_ms,
    )

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False
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
            prefix=f"ai-cif-recovery-{member_index:02d}-"
        ) as temporary_directory_string:
            temporary_directory = Path(temporary_directory_string)

            while progress.ppo_updates_total < target_ppo_updates:
                next_total_updates = progress.ppo_updates_total + 1
                phase_id = training_phase_id(
                    population_step=population_step,
                    member_index=member_index,
                    ppo_updates_total=next_total_updates,
                )
                opponent_choices = make_opponent_schedule(
                    battles=training_config.rollout_battles,
                    active_member_index=member_index,
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

                ppo_start = perf_counter()
                metrics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    trajectories=trajectories,
                    config=ppo_config,
                    device=device,
                )
                ppo_seconds = perf_counter() - ppo_start

                progress = MemberProgress(
                    iteration=population_step,
                    ppo_updates_total=next_total_updates,
                    recovery_updates_total=(
                        progress.recovery_updates_total + 1
                    ),
                )
                save_member_checkpoint(
                    path=checkpoint_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    progress=progress,
                )

                log_data = _training_log_data(
                    member_index=member_index,
                    population_step=population_step,
                    progress=progress,
                    trajectories=trajectories,
                    rollout_seconds=rollout_seconds,
                    ppo_seconds=ppo_seconds,
                    opponent_choices=opponent_choices,
                    rollout_inference_stats=rollout_inference_stats,
                    metrics=metrics,
                    recovery=True,
                )
                if wandb_run is not None:
                    wandb_run.log(log_data)

                print(
                    f"RECOVERY model={member_index:02d} "
                    f"population_step={population_step} "
                    f"ppo_updates={progress.ppo_updates_total} "
                    f"recovery_updates={progress.recovery_updates_total}"
                )

            save_member_checkpoint(
                path=checkpoint_dir / f"iteration_{population_step:05d}.pt",
                model=model,
                optimizer=optimizer,
                progress=progress,
            )

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


def evaluation_to_dict(
    *,
    wins: int,
    losses: int,
    ties: int,
    by_group: dict[str, tuple[int, int, int]],
    by_model: dict[str, tuple[int, int, int]],
    evaluation_seconds: float,
) -> dict:
    battles = wins + losses + ties
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "battles": battles,
        "win_rate": wins / battles,
        "seconds": evaluation_seconds,
        "by_group": {key: list(value) for key, value in by_group.items()},
        "by_model": {key: list(value) for key, value in by_model.items()},
    }


async def evaluate_member_at_step(
    *,
    args: argparse.Namespace,
    member_index: int,
    population_step: int,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    reward_config: RewardConfig,
    running_config: RunningConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
    population_prefix: str,
    log_prefix: str,
) -> dict:
    device = torch.device("cuda")
    checkpoint_dir = member_checkpoint_dir(
        running_config=running_config,
        population_prefix=population_prefix,
        member_index=member_index,
    )
    model, _, progress = load_active_member(
        device=device,
        model_config=model_config,
        ppo_config=ppo_config,
        checkpoint_dir=checkpoint_dir,
        member_index=member_index,
    )
    if progress.iteration != population_step:
        raise RuntimeError(
            f"Evaluation expected model {member_index:02d} at population step "
            f"{population_step}, found {progress.iteration}"
        )

    phase_id = evaluation_phase_id(
        population_step=population_step, member_index=member_index
    )
    eval_choices, historical_refs = make_evaluation_opponent_schedule(
        active_member_index=member_index,
        population_step=population_step,
        phase_id=phase_id,
        team_seed=training_config.team_seed,
    )
    historical_models = load_historical_opponents(
        device=device,
        model_config=model_config,
        running_config=running_config,
        population_prefix=population_prefix,
        historical_pool=historical_refs,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    wandb_run = init_wandb_run(
        args=args,
        checkpoint_dir=checkpoint_dir,
        running_config=running_config,
        training_config=training_config,
        ppo_config=ppo_config,
        reward_config=reward_config,
        model_index=member_index,
        population_prefix=population_prefix,
        progress=progress,
        parameter_count=parameter_count,
    )

    context = multiprocessing.get_context("spawn")
    active_slot_count = running_config.workers * running_config.battle_lanes
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
    inference_broker = PopulationGpuInferenceBroker(
        models={TRAINING_MODEL_KEY: model, **historical_models},
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        shared_buffer=shared_buffer,
        max_batch_size=running_config.gpu_batch_size,
        batch_wait_ms=running_config.gpu_batch_wait_ms,
    )

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False
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

        result = evaluation_to_dict(
            wins=eval_wins,
            losses=eval_losses,
            ties=eval_ties,
            by_group=eval_by_group,
            by_model=eval_by_model,
            evaluation_seconds=evaluation_seconds,
        )

        print(
            f"EVAL model={member_index:02d} population_step={population_step} "
            f"battles={result['battles']} win_rate={result['win_rate']:.1%}"
        )
        for group, counts in sorted(eval_by_group.items()):
            group_battles = sum(counts)
            print(
                f"  group={group} battles={group_battles} "
                f"win_rate={counts[0] / group_battles:.1%}"
            )

        if wandb_run is not None:
            log_data: dict[str, float | int] = {
                "population/step": population_step,
                "population/model_index": member_index,
                "population/ppo_updates_total": progress.ppo_updates_total,
                "population/recovery_updates_total": (
                    progress.recovery_updates_total
                ),
                f"{log_prefix}/wins": eval_wins,
                f"{log_prefix}/losses": eval_losses,
                f"{log_prefix}/ties": eval_ties,
                f"{log_prefix}/battles": result["battles"],
                f"{log_prefix}/win_rate": result["win_rate"],
                f"{log_prefix}/seconds": evaluation_seconds,
                f"{log_prefix}/gpu_requests": eval_inference_stats.requests,
                f"{log_prefix}/gpu_batches": eval_inference_stats.batches,
            }
            add_eval_breakdown_logs(
                log_data=log_data,
                prefix=log_prefix,
                by_group=eval_by_group,
                by_model=eval_by_model,
            )
            wandb_run.log(log_data)

        return result

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


def _aggregate_named_models(
    result: dict, names: set[str]
) -> tuple[int, int, int]:
    wins = losses = ties = 0
    by_model = result["by_model"]
    for name in names:
        counts = by_model[name]
        wins += int(counts[0])
        losses += int(counts[1])
        ties += int(counts[2])
    return wins, losses, ties


def benchmark_comparison(current: dict, previous: dict | None) -> dict:
    if previous is None:
        return {
            "has_previous": False,
            "common_opponents": 0,
            "current_common_win_rate": None,
            "previous_common_win_rate": None,
            "drop": None,
        }

    common_names = set(current["by_model"]) & set(previous["by_model"])
    if not common_names:
        return {
            "has_previous": False,
            "common_opponents": 0,
            "current_common_win_rate": None,
            "previous_common_win_rate": None,
            "drop": None,
        }

    current_counts = _aggregate_named_models(current, common_names)
    previous_counts = _aggregate_named_models(previous, common_names)
    current_battles = sum(current_counts)
    previous_battles = sum(previous_counts)
    current_rate = current_counts[0] / current_battles
    previous_rate = previous_counts[0] / previous_battles
    return {
        "has_previous": True,
        "common_opponents": len(common_names),
        "current_common_win_rate": current_rate,
        "previous_common_win_rate": previous_rate,
        "drop": previous_rate - current_rate,
    }


def previous_member_final_result(
    previous_eval: dict | None, member_index: int
) -> dict | None:
    if previous_eval is None:
        return None
    member = previous_eval.get("members", {}).get(str(member_index))
    if not isinstance(member, dict):
        return None
    final = member.get("final")
    return final if isinstance(final, dict) else None


def prepare_recovery_decisions(
    *, evaluation: dict, previous_eval: dict | None
) -> list[int]:
    members = evaluation["members"]
    initial_rates = [
        float(members[str(member_index)]["initial"]["win_rate"])
        for member_index in range(1, POPULATION_SIZE + 1)
    ]
    population_median = median(initial_rates)
    evaluation["population_median_initial_win_rate"] = population_median
    queue: list[int] = []

    for member_index in range(1, POPULATION_SIZE + 1):
        member_entry = members[str(member_index)]
        current = member_entry["initial"]
        previous = previous_member_final_result(previous_eval, member_index)
        comparison = benchmark_comparison(current, previous)
        below_population = (
            float(current["win_rate"])
            < population_median - RECOVERY_POPULATION_MARGIN
        )
        dropped = (
            comparison["has_previous"]
            and comparison["drop"] is not None
            and float(comparison["drop"]) > BENCHMARK_DROP_TOLERANCE
        )
        eligible = dropped and (
            below_population or not RECOVERY_REQUIRE_BELOW_POPULATION_MEDIAN
        )
        comparison["below_population_median"] = below_population
        comparison["recovery_eligible"] = eligible
        member_entry["comparison"] = comparison
        member_entry["final"] = current
        member_entry.setdefault("recovery", {"blocks": 0, "attempts": []})
        if eligible:
            queue.append(member_index)

    return queue


def recovery_still_needed(
    *, current: dict, previous: dict | None
) -> tuple[bool, dict]:
    comparison = benchmark_comparison(current, previous)
    needed = (
        comparison["has_previous"]
        and comparison["drop"] is not None
        and float(comparison["drop"]) > BENCHMARK_DROP_TOLERANCE
    )
    comparison["recovery_still_needed"] = needed
    return needed, comparison


async def run_population(
    *,
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    reward_config: RewardConfig,
    running_config: RunningConfig,
    model_config: ModelConfig,
    tensorizer: BattleTensorizer,
) -> None:
    validate_population_config()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Population training requires a CUDA/ROCm PyTorch device"
        )

    population_prefix = args.wandb_name
    state = load_population_state(
        running_config=running_config,
        population_prefix=population_prefix,
        eval_interval=training_config.eval_interval,
    )

    while state.round_start < LAST_STEP:
        target_step = min(state.round_start + STEPS_PER_ROUND, LAST_STEP)

        if state.phase == "train":
            while state.current_member <= POPULATION_SIZE:
                member_index = state.current_member
                print()
                print(
                    f"=== ROUND {state.round_start}->{target_step} "
                    f"MODEL {member_index:02d}/{POPULATION_SIZE:02d} ==="
                )
                await train_member_block(
                    args=args,
                    member_index=member_index,
                    round_start=state.round_start,
                    target_step=target_step,
                    ppo_config=ppo_config,
                    training_config=training_config,
                    reward_config=reward_config,
                    running_config=running_config,
                    model_config=model_config,
                    tensorizer=tensorizer,
                    population_prefix=population_prefix,
                )

                # Persist the NEXT member only after this member's target-step
                # checkpoint exists. A crash during model i therefore restarts i,
                # never model 1 and never skips unfinished work.
                state.current_member += 1
                save_population_state(
                    state=state,
                    running_config=running_config,
                    population_prefix=population_prefix,
                )

            eval_due = (
                target_step % training_config.eval_interval == 0
                or target_step == LAST_STEP
            )
            if eval_due:
                state.phase = "eval"
                state.eval_member = 1
                save_population_state(
                    state=state,
                    running_config=running_config,
                    population_prefix=population_prefix,
                )
            else:
                state.round_start = target_step
                state.current_member = 1
                save_population_state(
                    state=state,
                    running_config=running_config,
                    population_prefix=population_prefix,
                )
                continue

        if state.phase == "eval":
            eval_path = evaluation_result_path(
                running_config=running_config,
                population_prefix=population_prefix,
                population_step=target_step,
            )
            if eval_path.is_file():
                evaluation = load_json(eval_path)
            else:
                evaluation = {
                    "population_step": target_step,
                    "complete": False,
                    "members": {},
                }

            while state.eval_member <= POPULATION_SIZE:
                member_index = state.eval_member
                member_key = str(member_index)
                if (
                    member_key not in evaluation["members"]
                    or "initial" not in evaluation["members"][member_key]
                ):
                    initial = await evaluate_member_at_step(
                        args=args,
                        member_index=member_index,
                        population_step=target_step,
                        ppo_config=ppo_config,
                        training_config=training_config,
                        reward_config=reward_config,
                        running_config=running_config,
                        model_config=model_config,
                        tensorizer=tensorizer,
                        population_prefix=population_prefix,
                        log_prefix="eval_initial",
                    )
                    evaluation["members"].setdefault(member_key, {})[
                        "initial"
                    ] = initial
                    _atomic_write_json(eval_path, evaluation)

                state.eval_member += 1
                save_population_state(
                    state=state,
                    running_config=running_config,
                    population_prefix=population_prefix,
                )

            previous_eval = latest_completed_evaluation_before(
                running_config=running_config,
                population_prefix=population_prefix,
                population_step=target_step,
            )
            recovery_queue = prepare_recovery_decisions(
                evaluation=evaluation, previous_eval=previous_eval
            )
            _atomic_write_json(eval_path, evaluation)

            print(
                f"Evaluation step {target_step}: recovery candidates = "
                f"{recovery_queue or 'none'}"
            )
            state.phase = "recovery"
            state.recovery_queue = recovery_queue
            state.recovery_index = 0
            state.recovery_blocks_done = {
                str(member_index): int(
                    evaluation["members"][str(member_index)]
                    .get("recovery", {})
                    .get("blocks", 0)
                )
                for member_index in recovery_queue
            }
            state.recovery_target_ppo_updates = None
            save_population_state(
                state=state,
                running_config=running_config,
                population_prefix=population_prefix,
            )

        if state.phase == "recovery":
            eval_path = evaluation_result_path(
                running_config=running_config,
                population_prefix=population_prefix,
                population_step=target_step,
            )
            evaluation = load_json(eval_path)
            previous_eval = latest_completed_evaluation_before(
                running_config=running_config,
                population_prefix=population_prefix,
                population_step=target_step,
            )

            while state.recovery_index < len(state.recovery_queue):
                member_index = state.recovery_queue[state.recovery_index]
                member_key = str(member_index)
                blocks_done = state.recovery_blocks_done.get(member_key, 0)

                if blocks_done >= MAX_RECOVERY_BLOCKS:
                    state.recovery_index += 1
                    state.recovery_target_ppo_updates = None
                    save_population_state(
                        state=state,
                        running_config=running_config,
                        population_prefix=population_prefix,
                    )
                    continue

                if state.recovery_target_ppo_updates is None:
                    progress = member_progress_on_disk(
                        running_config=running_config,
                        population_prefix=population_prefix,
                        member_index=member_index,
                    )
                    state.recovery_target_ppo_updates = (
                        progress.ppo_updates_total + RECOVERY_UPDATES_PER_BLOCK
                    )
                    save_population_state(
                        state=state,
                        running_config=running_config,
                        population_prefix=population_prefix,
                    )

                await train_member_recovery_updates(
                    args=args,
                    member_index=member_index,
                    round_start=state.round_start,
                    population_step=target_step,
                    target_ppo_updates=state.recovery_target_ppo_updates,
                    ppo_config=ppo_config,
                    training_config=training_config,
                    reward_config=reward_config,
                    running_config=running_config,
                    model_config=model_config,
                    tensorizer=tensorizer,
                    population_prefix=population_prefix,
                )

                post_recovery = await evaluate_member_at_step(
                    args=args,
                    member_index=member_index,
                    population_step=target_step,
                    ppo_config=ppo_config,
                    training_config=training_config,
                    reward_config=reward_config,
                    running_config=running_config,
                    model_config=model_config,
                    tensorizer=tensorizer,
                    population_prefix=population_prefix,
                    log_prefix="eval_recovery",
                )
                previous = previous_member_final_result(
                    previous_eval, member_index
                )
                still_needed, comparison = recovery_still_needed(
                    current=post_recovery, previous=previous
                )

                blocks_done += 1
                state.recovery_blocks_done[member_key] = blocks_done
                member_entry = evaluation["members"][member_key]
                member_entry["final"] = post_recovery
                member_entry["comparison_after_recovery"] = comparison
                recovery_entry = member_entry.setdefault(
                    "recovery", {"blocks": 0, "attempts": []}
                )
                recovery_entry["blocks"] = blocks_done
                recovery_entry.setdefault("attempts", []).append(
                    {
                        "block": blocks_done,
                        "target_ppo_updates": state.recovery_target_ppo_updates,
                        "result": post_recovery,
                        "comparison": comparison,
                    }
                )
                _atomic_write_json(eval_path, evaluation)

                state.recovery_target_ppo_updates = None
                if not still_needed or blocks_done >= MAX_RECOVERY_BLOCKS:
                    state.recovery_index += 1
                save_population_state(
                    state=state,
                    running_config=running_config,
                    population_prefix=population_prefix,
                )

            evaluation["complete"] = True
            _atomic_write_json(eval_path, evaluation)
            state.round_start = target_step
            state.current_member = 1
            state.phase = "train"
            state.eval_member = 1
            state.recovery_queue = []
            state.recovery_index = 0
            state.recovery_blocks_done = {}
            state.recovery_target_ppo_updates = None
            save_population_state(
                state=state,
                running_config=running_config,
                population_prefix=population_prefix,
            )

    print(
        f"Population {population_prefix!r} complete at population step "
        f"{LAST_STEP}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-name",
        required=True,
        help=(
            "Population prefix. Member runs are named <prefix>-1 ... <prefix>-10."
        ),
    )
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

    await run_population(
        args=args,
        ppo_config=ppo_config,
        training_config=training_config,
        reward_config=reward_config,
        running_config=running_config,
        model_config=model_config,
        tensorizer=tensorizer,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    asyncio.run(main())
    print(f"took {perf_counter() - t0}")
