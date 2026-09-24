import argparse
import asyncio
import csv
import math
import multiprocessing
import os
import queue
import threading
import traceback
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from itertools import combinations_with_replacement
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from time import perf_counter
from typing import override

import torch
from dotenv import load_dotenv
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import (
    MaxBasePowerCombatHandler,
    RandomMoveCombatHandler,
    SimpleHeuristicsCombatHandler,
)
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
from tqdm import tqdm

from ai_cif.model.model import BattleModel, create_battle_model
from ai_cif.vectorization.tensorizer import (
    ACTION_COUNT,
    BattleTensorizer,
    BattleTensors,
)
from scripts.utils.battles import outcome_for, run_battle
from scripts.utils.gpu import SharedBattleBuffer
from scripts.utils.multithreading import split_battles, worker_initializer

load_dotenv()

DEFAULT_WEBSOCKET_URL = (
    os.environ.get("DEFAULT_WEBSOCKET_URL")
    or "ws://127.0.0.1:8000/showdown/websocket"
)

CSV_COLUMNS = [
    "format",
    "model_1",
    "model_2",
    "winrate",
    "variance",
    "nb_battles",
    "wins",
    "losses",
    "ties",
]

HEURISTIC_NAMES = {"random", "MaxBasePower", "SimpleHeuristics"}


@dataclass(frozen=True)
class Participant:
    index: int
    name: str
    checkpoint: Path | str

    @property
    def is_neural(self) -> bool:
        return isinstance(self.checkpoint, Path)


@dataclass(frozen=True)
class PairResult:
    battles: int
    wins: int
    losses: int
    ties: int
    score_sum: float
    score_squared_sum: float

    @property
    def winrate(self) -> float:
        if self.battles == 0:
            return 0.0
        return self.score_sum / self.battles

    @property
    def score_variance(self) -> float:
        """Unbiased sample variance of scores in {0, 0.5, 1}."""
        if self.battles <= 1:
            return 0.0

        mean = self.winrate
        numerator = self.score_squared_sum - self.battles * mean * mean
        return max(0.0, numerator / (self.battles - 1))

    @property
    def winrate_variance(self) -> float:
        """Estimated variance of the reported mean winrate."""
        if self.battles == 0:
            return 0.0
        return self.score_variance / self.battles

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.winrate_variance)


@dataclass
class WorkerInferenceStats:
    requests: int = 0
    shared_write_seconds: float = 0.0
    queue_put_seconds: float = 0.0
    response_wait_seconds: float = 0.0


@dataclass(frozen=True)
class EvalInferenceRequest:
    worker_index: int
    slot_index: int
    request_id: int
    model_index: int


@dataclass(frozen=True)
class EvalInferenceResponse:
    slot_index: int
    request_id: int
    logits: tuple[float, ...] | None
    error: str | None = None


type AsyncEvalInferenceFn = Callable[[BattleTensors], Awaitable[torch.Tensor]]
type PendingInference = dict[tuple[int, int], asyncio.Future[torch.Tensor]]


_EVAL_REQUEST_QUEUE: ProcessQueue | None = None
_EVAL_RESPONSE_QUEUES: list[ProcessQueue] | None = None
_EVAL_SHARED_BUFFER: SharedBattleBuffer | None = None


def discover_participants(models_dir: Path) -> list[Participant]:
    checkpoints = sorted(models_dir.glob("*/*.pt"))

    raw: list[tuple[str, Path | str]] = [
        ("random", "random"),
        ("MaxBasePower", "MaxBasePower"),
        ("SimpleHeuristics", "SimpleHeuristics"),
    ]

    for checkpoint in checkpoints:
        if checkpoint.stem in HEURISTIC_NAMES:
            raise ValueError(
                f"{checkpoint} uses reserved model name '{checkpoint.stem}'"
            )
        raw.append((checkpoint.stem, checkpoint))

    return [
        Participant(index=index, name=name, checkpoint=checkpoint)
        for index, (name, checkpoint) in enumerate(raw)
    ]


def load_neural_models(
    *,
    participants: list[Participant],
    required_indices: set[int],
    device: torch.device,
) -> tuple[dict[int, BattleModel], BattleTensorizer]:
    tensorizer = BattleTensorizer(max_history=32, vocab_gen=4)
    models: dict[int, BattleModel] = {}

    for participant in participants:
        if participant.index not in required_indices:
            continue
        if not participant.is_neural:
            raise RuntimeError(
                f"Participant {participant.name!r} was marked for GPU loading "
                "but is not neural"
            )

        model, _ = create_battle_model(
            device=device, max_history=tensorizer.max_history, vocab_gen=4
        )

        checkpoint_path = participant.checkpoint
        if not isinstance(checkpoint_path, Path):
            raise TypeError("Neural participant checkpoint must be a Path")

        # Keep optimizer state and other checkpoint payloads off the GPU.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint {checkpoint_path} must contain a dict")

        if "model" in checkpoint:
            model_state = checkpoint["model"]
        else:
            model_state = checkpoint

        if not isinstance(model_state, dict):
            raise TypeError(
                f"Checkpoint {checkpoint_path} has invalid model state"
            )

        model.load_state_dict(model_state)
        model.eval()
        models[participant.index] = model

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        print(
            f"Loaded {participant.name}: "
            f"model_index={participant.index} parameters={parameter_count:,}"
        )

    return models, tensorizer


class AsyncEvalNeuralCombatHandler(AsyncBaseCombatHandler):
    def __init__(
        self, *, tensorizer: BattleTensorizer, infer: AsyncEvalInferenceFn
    ) -> None:
        self.tensorizer = tensorizer
        self.infer = infer

    @override
    async def async_select_top_actions(
        self, battle_state: BattleState
    ) -> list[Action]:
        features = battle_to_features(battle_state)
        tensors = self.tensorizer.tensorize(features)
        logits = await self.infer(tensors)

        if logits.shape != (ACTION_COUNT,):
            raise RuntimeError(
                f"Expected policy logits shape ({ACTION_COUNT},), "
                f"got {tuple(logits.shape)}"
            )

        legal_indices = torch.where(tensors.action_mask)[0]
        if legal_indices.numel() == 0:
            raise RuntimeError("Model received a state with no legal actions")

        scores = logits[legal_indices]
        ranking = torch.argsort(scores, descending=True)
        ranked_indices = legal_indices[ranking]

        return [self._decode_action(int(index)) for index in ranked_indices]

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


def eval_gpu_worker_initializer(
    torch_threads: int,
    request_queue: ProcessQueue,
    response_queues: list[ProcessQueue],
    shared_buffer: SharedBattleBuffer,
) -> None:
    worker_initializer(torch_threads)

    global _EVAL_REQUEST_QUEUE
    global _EVAL_RESPONSE_QUEUES
    global _EVAL_SHARED_BUFFER

    _EVAL_REQUEST_QUEUE = request_queue
    _EVAL_RESPONSE_QUEUES = response_queues
    _EVAL_SHARED_BUFFER = shared_buffer


def _worker_transport(
    worker_index: int,
) -> tuple[ProcessQueue, ProcessQueue, SharedBattleBuffer]:
    request_queue = _EVAL_REQUEST_QUEUE
    response_queues = _EVAL_RESPONSE_QUEUES
    shared_buffer = _EVAL_SHARED_BUFFER

    if (
        request_queue is None
        or response_queues is None
        or shared_buffer is None
    ):
        raise RuntimeError("Evaluation GPU transport was not initialized")

    if not 0 <= worker_index < len(response_queues):
        raise IndexError(f"No response queue for worker {worker_index}")

    return request_queue, response_queues[worker_index], shared_buffer


async def response_pump(
    *, worker_index: int, pending: PendingInference
) -> None:
    """Known-good pacing: one blocking queue read via asyncio.to_thread."""
    _, response_queue, _ = _worker_transport(worker_index)

    while True:
        response = await asyncio.to_thread(response_queue.get)

        if response is None:
            return

        if not isinstance(response, EvalInferenceResponse):
            raise TypeError(
                "GPU inference broker returned invalid response "
                f"of type {type(response).__name__}"
            )

        key = (response.slot_index, response.request_id)
        future = pending.pop(key, None)

        if future is None:
            raise RuntimeError(
                "Received evaluation inference response with no pending request: "
                f"slot={response.slot_index} request={response.request_id}"
            )

        if response.error is not None:
            future.set_exception(
                RuntimeError(
                    f"GPU evaluation inference failed:\n{response.error}"
                )
            )
            continue

        if response.logits is None:
            future.set_exception(
                RuntimeError("GPU evaluation inference returned empty logits")
            )
            continue

        if len(response.logits) != ACTION_COUNT:
            future.set_exception(
                RuntimeError(
                    f"Expected {ACTION_COUNT} policy logits, "
                    f"got {len(response.logits)}"
                )
            )
            continue

        future.set_result(torch.tensor(response.logits, dtype=torch.float32))


def stop_response_pump(worker_index: int) -> None:
    _, response_queue, _ = _worker_transport(worker_index)
    response_queue.put(None)


def make_remote_infer(
    *,
    worker_index: int,
    slot_index: int,
    model_index: int,
    pending: PendingInference,
    stats: WorkerInferenceStats,
    timeout_seconds: float = 120.0,
) -> AsyncEvalInferenceFn:
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be > 0")

    request_queue, _, shared_buffer = _worker_transport(worker_index)

    if not 0 <= slot_index < shared_buffer.slot_count:
        raise IndexError(f"No shared inference slot {slot_index}")

    next_request_id = 0

    async def infer(observation: BattleTensors) -> torch.Tensor:
        nonlocal next_request_id

        request_id = next_request_id
        next_request_id += 1

        key = (slot_index, request_id)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[torch.Tensor] = loop.create_future()

        if key in pending:
            raise RuntimeError(
                "Duplicate pending evaluation inference request: "
                f"slot={slot_index} request={request_id}"
            )

        pending[key] = future

        shared_buffer.write(slot_index, observation)

        request_queue.put(
            EvalInferenceRequest(
                worker_index=worker_index,
                slot_index=slot_index,
                request_id=request_id,
                model_index=model_index,
            )
        )

        stats.requests += 1

        try:
            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=timeout_seconds
            )
            return result

        except BaseException:
            pending.pop(key, None)

            if not future.done():
                future.cancel()

            raise

    return infer


class MultiModelGpuInferenceBroker:
    """Collect requests globally, then run one forward per requested model."""

    def __init__(
        self,
        *,
        models: dict[int, BattleModel],
        device: torch.device,
        request_queue: ProcessQueue,
        response_queues: list[ProcessQueue],
        shared_buffer: SharedBattleBuffer,
        max_batch_size: int,
        batch_wait_ms: float,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("MultiModelGpuInferenceBroker requires CUDA/ROCm")
        if not models:
            raise ValueError("MultiModelGpuInferenceBroker requires models")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if batch_wait_ms < 0.0:
            raise ValueError("batch_wait_ms must be >= 0")

        self.models = models
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.shared_buffer = shared_buffer
        self.max_batch_size = max_batch_size
        # Only one matchup runs at a time, so at most two model ids are active.
        # Collect up to two model-batches before partitioning by model.
        self.collection_limit = max_batch_size * 2
        self.batch_wait_seconds = batch_wait_ms / 1000.0

        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Evaluation inference broker is already running")

        self._thread = threading.Thread(
            target=self._run, name="eval-gpu-inference", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return

        self.request_queue.put(None)
        thread.join(timeout=30.0)

        if thread.is_alive():
            raise RuntimeError(
                "Evaluation inference broker did not stop cleanly"
            )

        self._thread = None

    def _run(self) -> None:
        stop_after_batch = False

        while True:
            item = self.request_queue.get()

            if item is None:
                return

            if not isinstance(item, EvalInferenceRequest):
                raise TypeError(
                    "Evaluation inference queue received invalid object "
                    f"of type {type(item).__name__}"
                )

            requests = [item]

            if self.batch_wait_seconds > 0.0:
                deadline = perf_counter() + self.batch_wait_seconds

                while len(requests) < self.collection_limit:
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

                    if not isinstance(next_item, EvalInferenceRequest):
                        raise TypeError(
                            "Evaluation inference queue received invalid object "
                            f"of type {type(next_item).__name__}"
                        )

                    requests.append(next_item)
            else:
                while len(requests) < self.collection_limit:
                    try:
                        next_item = self.request_queue.get_nowait()
                    except queue.Empty:
                        break

                    if next_item is None:
                        stop_after_batch = True
                        break

                    if not isinstance(next_item, EvalInferenceRequest):
                        raise TypeError(
                            "Evaluation inference queue received invalid object "
                            f"of type {type(next_item).__name__}"
                        )

                    requests.append(next_item)

            groups: dict[int, list[EvalInferenceRequest]] = {}
            for request in requests:
                groups.setdefault(request.model_index, []).append(request)

            for model_index, group in groups.items():
                for start in range(0, len(group), self.max_batch_size):
                    self._process_model_group(
                        model_index, group[start : start + self.max_batch_size]
                    )

            if stop_after_batch:
                return

    def _process_model_group(
        self, model_index: int, requests: list[EvalInferenceRequest]
    ) -> None:
        try:
            model = self.models.get(model_index)
            if model is None:
                raise KeyError(
                    f"No GPU model registered for index {model_index}"
                )

            cpu_batch = self.shared_buffer.batch(
                [request.slot_index for request in requests]
            )

            gpu_batch = cpu_batch.to(self.device)

            with torch.inference_mode():
                logits, _ = model(gpu_batch)

            logits_cpu = logits.detach().cpu()

            if logits_cpu.shape != (len(requests), ACTION_COUNT):
                raise RuntimeError(
                    "Unexpected evaluation logits shape: "
                    f"{tuple(logits_cpu.shape)}"
                )

            for index, request in enumerate(requests):
                self.response_queues[request.worker_index].put(
                    EvalInferenceResponse(
                        slot_index=request.slot_index,
                        request_id=request.request_id,
                        logits=tuple(
                            float(value) for value in logits_cpu[index].tolist()
                        ),
                    )
                )

        except BaseException:
            for request in requests:
                self.response_queues[request.worker_index].put(
                    EvalInferenceResponse(
                        slot_index=request.slot_index,
                        request_id=request.request_id,
                        logits=None,
                        error=traceback.format_exc(),
                    )
                )
            raise


def create_handler(
    *,
    participant: Participant,
    worker_index: int,
    slot_index: int,
    pending: PendingInference,
    inference_stats: WorkerInferenceStats,
    tensorizer: BattleTensorizer,
):
    checkpoint = participant.checkpoint

    match checkpoint:
        case "random":
            return RandomMoveCombatHandler()
        case "MaxBasePower":
            return MaxBasePowerCombatHandler()
        case "SimpleHeuristics":
            return SimpleHeuristicsCombatHandler()

    if not isinstance(checkpoint, Path):
        raise TypeError(
            f"Unsupported participant checkpoint type for {participant.name}"
        )

    infer = make_remote_infer(
        worker_index=worker_index,
        slot_index=slot_index,
        model_index=participant.index,
        pending=pending,
        stats=inference_stats,
    )

    return AsyncEvalNeuralCombatHandler(tensorizer=tensorizer, infer=infer)


def _team_generators(
    *,
    fmt: str,
    team_seed: int,
    phase_id: int,
    battle_offset: int,
    lane_global_index: int,
) -> tuple[BaseTeamGenerator | None, BaseTeamGenerator | None]:
    if "randombattle" in fmt:
        return None, None

    seed = (
        team_seed
        + phase_id * 10_000_000
        + battle_offset * 1_000
        + lane_global_index * 2
    )
    return SampleTeamGenerator(seed), SampleTeamGenerator(seed + 1)


async def _pair_lane(
    *,
    model_1: Participant,
    model_2: Participant,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    battle_offset: int,
    worker_index: int,
    lane_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
    pending: PendingInference,
    inference_stats: WorkerInferenceStats,
) -> PairResult:
    lane_global_index = worker_index * battle_lanes + lane_index

    # Two distinct slots are required because both sides may request neural
    # inference concurrently on the same turn.
    slot_1 = lane_global_index * 2
    slot_2 = slot_1 + 1

    handler_1 = create_handler(
        participant=model_1,
        worker_index=worker_index,
        slot_index=slot_1,
        pending=pending,
        inference_stats=inference_stats,
        tensorizer=tensorizer,
    )
    handler_2 = create_handler(
        participant=model_2,
        worker_index=worker_index,
        slot_index=slot_2,
        pending=pending,
        inference_stats=inference_stats,
        tensorizer=tensorizer,
    )

    client_1 = Client(url, combat_handler=handler_1)
    client_2 = Client(url, combat_handler=handler_2)

    client_1.log_manager.disable()
    client_2.log_manager.disable()

    team_generator_1, team_generator_2 = _team_generators(
        fmt=fmt,
        team_seed=team_seed,
        phase_id=phase_id,
        battle_offset=battle_offset,
        lane_global_index=lane_global_index,
    )

    client_1_name = f"E{phase_id}A{lane_global_index}"
    client_2_name = f"E{phase_id}B{lane_global_index}"

    wins = 0
    losses = 0
    ties = 0
    score_sum = 0.0
    score_squared_sum = 0.0
    completed = 0

    try:
        await asyncio.gather(client_1.connect(), client_2.connect())
        await asyncio.gather(
            client_1.login(client_1_name), client_2.login(client_2_name)
        )

        while completed < battles:
            global_battle_index = battle_offset + completed
            model_1_on_client_1 = global_battle_index % 2 == 0

            if model_1_on_client_1:
                client_1.combat_handler = handler_1
                client_2.combat_handler = handler_2
                model_1_client = client_1
            else:
                client_1.combat_handler = handler_2
                client_2.combat_handler = handler_1
                model_1_client = client_2

            try:
                result_1, result_2 = await run_battle(
                    client_1,
                    client_2,
                    fmt=fmt,
                    team_generator_1=team_generator_1,
                    team_generator_2=team_generator_2,
                )
            except (
                BattleLifecycleError,
                BattleReproductionError,
                SDKTimeoutError,
            ) as error:
                print(
                    "Discarding failed evaluation battle and retrying: "
                    f"worker={worker_index} lane={lane_index} "
                    f"{type(error).__name__}: {error}"
                )

                await asyncio.gather(
                    client_1.close(), client_2.close(), return_exceptions=True
                )
                await asyncio.gather(
                    client_1.ensure_connected(), client_2.ensure_connected()
                )
                continue

            result = result_1 if model_1_client is client_1 else result_2

            if model_1_client.username is None:
                raise RuntimeError("Evaluation client has no username")

            outcome = outcome_for(result, model_1_client.username)

            if outcome > 0:
                wins += 1
                score = 1.0
            elif outcome < 0:
                losses += 1
                score = 0.0
            else:
                ties += 1
                score = 0.5

            score_sum += score
            score_squared_sum += score * score
            completed += 1

        return PairResult(
            battles=completed,
            wins=wins,
            losses=losses,
            ties=ties,
            score_sum=score_sum,
            score_squared_sum=score_squared_sum,
        )

    finally:
        await asyncio.gather(
            client_1.close(), client_2.close(), return_exceptions=True
        )


async def _pair_worker_async(
    *,
    model_1: Participant,
    model_2: Participant,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    battle_offset: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
) -> tuple[PairResult, dict[str, int | float]]:
    lane_counts = split_battles(battles, battle_lanes)

    lane_offsets: list[int] = []
    next_offset = battle_offset
    for count in lane_counts:
        lane_offsets.append(next_offset)
        next_offset += count

    pending: PendingInference = {}
    inference_stats = WorkerInferenceStats()

    has_neural = model_1.is_neural or model_2.is_neural
    pump_task: asyncio.Task[None] | None = None

    if has_neural:
        pump_task = asyncio.create_task(
            response_pump(worker_index=worker_index, pending=pending)
        )

    try:
        tasks = [
            asyncio.create_task(
                _pair_lane(
                    model_1=model_1,
                    model_2=model_2,
                    url=url,
                    fmt=fmt,
                    team_seed=team_seed,
                    battles=count,
                    battle_offset=lane_offsets[lane_index],
                    worker_index=worker_index,
                    lane_index=lane_index,
                    battle_lanes=battle_lanes,
                    phase_id=phase_id,
                    tensorizer=tensorizer,
                    pending=pending,
                    inference_stats=inference_stats,
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
                f"Worker {worker_index} finished with "
                f"{len(pending)} pending inference requests"
            )

        return merge_pair_results(results), asdict(inference_stats)

    finally:
        if pump_task is not None:
            stop_response_pump(worker_index)
            await pump_task


def _pair_worker(
    model_1: Participant,
    model_2: Participant,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    battle_offset: int,
    worker_index: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
) -> tuple[PairResult, dict[str, int | float]]:
    return asyncio.run(
        _pair_worker_async(
            model_1=model_1,
            model_2=model_2,
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            battles=battles,
            battle_offset=battle_offset,
            worker_index=worker_index,
            battle_lanes=battle_lanes,
            phase_id=phase_id,
            tensorizer=tensorizer,
        )
    )


def merge_pair_results(results: list[PairResult]) -> PairResult:
    return PairResult(
        battles=sum(result.battles for result in results),
        wins=sum(result.wins for result in results),
        losses=sum(result.losses for result in results),
        ties=sum(result.ties for result in results),
        score_sum=sum(result.score_sum for result in results),
        score_squared_sum=sum(result.score_squared_sum for result in results),
    )


async def evaluate_pair_multiprocess(
    *,
    pool: ProcessPoolExecutor,
    model_1: Participant,
    model_2: Participant,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    battle_offset: int,
    worker_count: int,
    battle_lanes: int,
    phase_id: int,
    tensorizer: BattleTensorizer,
) -> PairResult:
    counts = split_battles(battles, worker_count)

    worker_offsets: list[int] = []
    next_offset = battle_offset
    for count in counts:
        worker_offsets.append(next_offset)
        next_offset += count

    loop = asyncio.get_running_loop()
    tasks = []

    for worker_index, count in enumerate(counts):
        if count <= 0:
            continue

        tasks.append(
            loop.run_in_executor(
                pool,
                partial(
                    _pair_worker,
                    model_1,
                    model_2,
                    url,
                    fmt,
                    team_seed,
                    count,
                    worker_offsets[worker_index],
                    worker_index,
                    battle_lanes,
                    phase_id,
                    tensorizer,
                ),
            )
        )

    worker_results = await asyncio.gather(*tasks)

    results = [result for result, _ in worker_results]

    return merge_pair_results(results)


def read_scores(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return {}

        missing = [
            column for column in CSV_COLUMNS if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{path} is missing CSV columns: {missing}")

        rows: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in reader:
            key = (row["format"], row["model_1"], row["model_2"])
            rows[key] = row
        return rows


def write_scores(
    path: Path, rows: dict[tuple[str, str, str], dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


def row_to_result(row: dict[str, str]) -> PairResult:
    battles = int(row["nb_battles"])
    wins = int(row["wins"])
    losses = int(row["losses"])
    ties = int(row["ties"])

    if wins + losses + ties != battles:
        raise ValueError(
            "CSV result is inconsistent: "
            f"battles={battles} W/L/T={wins}/{losses}/{ties}"
        )

    return PairResult(
        battles=battles,
        wins=wins,
        losses=losses,
        ties=ties,
        score_sum=wins + 0.5 * ties,
        score_squared_sum=wins + 0.25 * ties,
    )


def result_to_row(
    *, fmt: str, model_1: Participant, model_2: Participant, result: PairResult
) -> dict[str, str]:
    return {
        "format": fmt,
        "model_1": model_1.name,
        "model_2": model_2.name,
        "winrate": f"{result.winrate:.8f}",
        "variance": f"{result.winrate_variance:.12f}",
        "nb_battles": str(result.battles),
        "wins": str(result.wins),
        "losses": str(result.losses),
        "ties": str(result.ties),
    }


async def evaluate_all(args: argparse.Namespace) -> None:
    models_dir = Path(args.models_dir)
    scores_path = Path(args.scores)
    participants = discover_participants(models_dir)
    pairs = list(combinations_with_replacement(participants, 2))
    rows = read_scores(scores_path)

    pending_pairs: list[
        tuple[int, Participant, Participant, PairResult | None, int]
    ] = []
    required_model_indices: set[int] = set()

    for pair_index, (model_1, model_2) in enumerate(pairs, start=1):
        key = (args.fmt, model_1.name, model_2.name)
        existing_row = rows.get(key)
        existing_result = (
            row_to_result(existing_row) if existing_row is not None else None
        )
        existing_battles = (
            existing_result.battles if existing_result is not None else 0
        )
        remaining = max(0, args.battles - existing_battles)

        if remaining <= 0:
            continue

        pending_pairs.append(
            (pair_index, model_1, model_2, existing_result, remaining)
        )

        if model_1.is_neural:
            required_model_indices.add(model_1.index)
        if model_2.is_neural:
            required_model_indices.add(model_2.index)

    max_concurrency = args.workers * args.battle_lanes

    print(f"Format: {args.fmt}")
    print(f"Models directory: {models_dir}")
    print(f"Scores CSV: {scores_path}")
    print(
        "Participants: "
        + ", ".join(participant.name for participant in participants)
    )
    print(f"Pairs including self-play: {len(pairs)}")
    print(f"Pairs requiring work: {len(pending_pairs)}")
    print(f"Target battles per pair: {args.battles}")
    print(f"Workers: {args.workers}")
    print(f"Battle lanes per worker: {args.battle_lanes}")
    print(f"Max battles in flight: {max_concurrency}")
    print(f"PyTorch threads per worker: {args.worker_torch_threads}")
    print(f"GPU inference max batch: {args.gpu_batch_size}")
    print(f"GPU inference batch wait: {args.gpu_batch_wait_ms:.3f} ms")

    if not pending_pairs:
        print("All matchups already satisfy the requested battle count.")
        return

    if required_model_indices and not torch.cuda.is_available():
        raise RuntimeError(
            "eval_gpu.py requires CUDA/ROCm for neural participants"
        )

    device = (
        torch.device("cuda") if required_model_indices else torch.device("cpu")
    )
    models, tensorizer = load_neural_models(
        participants=participants,
        required_indices=required_model_indices,
        device=device,
    )

    context = multiprocessing.get_context("spawn")

    # Every lane reserves two observation slots because both sides may be neural.
    slot_count = args.workers * args.battle_lanes * 2

    request_queue = context.Queue(
        maxsize=max(slot_count * 2, args.gpu_batch_size * 2)
    )
    response_queues = [
        context.Queue(maxsize=max(args.battle_lanes * 4, 8))
        for _ in range(args.workers)
    ]
    shared_buffer = SharedBattleBuffer.create(
        slot_count=slot_count, max_history=tensorizer.max_history
    )

    broker: MultiModelGpuInferenceBroker | None = None
    if models:
        broker = MultiModelGpuInferenceBroker(
            models=models,
            device=device,
            request_queue=request_queue,
            response_queues=response_queues,
            shared_buffer=shared_buffer,
            max_batch_size=args.gpu_batch_size,
            batch_wait_ms=args.gpu_batch_wait_ms,
        )
        broker.start()

    pool: ProcessPoolExecutor | None = None
    pool_terminated = False

    try:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=eval_gpu_worker_initializer,
            initargs=(
                args.worker_torch_threads,
                request_queue,
                response_queues,
                shared_buffer,
            ),
        )

        work_by_pair_index = {
            pair_index: (model_1, model_2, existing_result, remaining)
            for pair_index, model_1, model_2, existing_result, remaining in pending_pairs
        }

        for pair_index, (model_1, model_2) in enumerate(
            tqdm(pairs, desc="Evaluating matchups", unit="pair"), start=1
        ):
            work = work_by_pair_index.get(pair_index)

            if work is None:
                key = (args.fmt, model_1.name, model_2.name)
                existing = rows.get(key)
                existing_battles = (
                    int(existing["nb_battles"]) if existing else 0
                )
                tqdm.write(
                    f"[{pair_index}/{len(pairs)}] "
                    f"{model_1.name} vs {model_2.name}: "
                    f"already has {existing_battles} battles, skipping"
                )
                continue

            _, _, existing_result, remaining = work
            existing_battles = existing_result.battles if existing_result else 0

            tqdm.write(
                f"[{pair_index}/{len(pairs)}] "
                f"{model_1.name} vs {model_2.name}: "
                f"running {remaining} battle(s) "
                f"from offset {existing_battles}"
            )

            new_result = await evaluate_pair_multiprocess(
                pool=pool,
                model_1=model_1,
                model_2=model_2,
                url=args.url,
                fmt=args.fmt,
                team_seed=args.team_seed,
                battles=remaining,
                battle_offset=existing_battles,
                worker_count=args.workers,
                battle_lanes=args.battle_lanes,
                phase_id=pair_index,
                tensorizer=tensorizer,
            )

            if existing_result is None:
                result = new_result
            else:
                result = merge_pair_results([existing_result, new_result])

            key = (args.fmt, model_1.name, model_2.name)
            rows[key] = result_to_row(
                fmt=args.fmt, model_1=model_1, model_2=model_2, result=result
            )
            write_scores(scores_path, rows)

    except BaseException:
        if pool is not None:
            pool_terminated = True
            pool.terminate_workers()
        raise

    finally:
        if pool is not None and not pool_terminated:
            pool.shutdown(wait=True, cancel_futures=True)

        if broker is not None:
            broker.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", default=DEFAULT_WEBSOCKET_URL)
    parser.add_argument("--format", dest="fmt", default="gen1randombattle")
    parser.add_argument(
        "--battles",
        type=int,
        default=1000,
        help="Target total number of battles for every pair",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--battle-lanes", type=int, default=5)
    parser.add_argument("--worker-torch-threads", type=int, default=2)
    parser.add_argument("--gpu-batch-size", type=int, default=32)
    parser.add_argument("--gpu-batch-wait-ms", type=float, default=0.5)
    parser.add_argument("--team-seed", type=int, default=42)
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--scores", default="experiments/scores.csv")

    args = parser.parse_args()

    if args.battles < 1:
        parser.error("--battles must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.battle_lanes < 1:
        parser.error("--battle-lanes must be at least 1")
    if args.worker_torch_threads < 1:
        parser.error("--worker-torch-threads must be at least 1")
    if args.gpu_batch_size < 1:
        parser.error("--gpu-batch-size must be at least 1")
    if args.gpu_batch_wait_ms < 0.0:
        parser.error("--gpu-batch-wait-ms must be >= 0")

    return args


async def main() -> None:
    args = parse_args()
    await evaluate_all(args)


if __name__ == "__main__":
    asyncio.run(main())
