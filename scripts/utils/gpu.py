import asyncio
import queue
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from multiprocessing.queues import Queue as ProcessQueue
from time import perf_counter

import torch

from ai_cif.model.model import BattleModel
from ai_cif.vectorization.tensorizer import (
    ACTION_COUNT,
    BASE_STATS_DIM,
    FIELD_NUMERIC_DIM,
    HISTORY_NUMERIC_DIM,
    MOVE_CATEGORY_COUNT,
    MOVE_NUMERIC_DIM,
    MOVES_PER_POKEMON,
    POKEMON_NUMERIC_DIM,
    POKEMON_SLOTS,
    TYPE_COUNT,
    BattleBatch,
    BattleTensors,
)
from scripts.utils.multithreading import worker_initializer

type AsyncInferenceFn = Callable[
    [BattleTensors], Awaitable[tuple[torch.Tensor, float]]
]

type PendingInference = dict[
    tuple[int, int], asyncio.Future[tuple[torch.Tensor, float]]
]


@dataclass(frozen=True)
class InferenceRequest:
    worker_index: int
    slot_index: int
    request_id: int


@dataclass(frozen=True)
class InferenceResponse:
    slot_index: int
    request_id: int
    logits: tuple[float, ...] | None
    value: float | None
    error: str | None = None


@dataclass(frozen=True)
class GpuInferenceStats:
    requests: int
    batches: int
    max_batch_size: int

    total_batch_wait_seconds: float
    total_gather_seconds: float
    total_inference_seconds: float
    total_dispatch_seconds: float

    @property
    def mean_batch_size(self) -> float:
        if self.batches == 0:
            return 0.0
        return self.requests / self.batches

    @property
    def requests_per_inference_second(self) -> float:
        if self.total_inference_seconds <= 0.0:
            return 0.0
        return self.requests / self.total_inference_seconds


@dataclass(frozen=True)
class SharedBattleBuffer:
    """Preallocated shared-memory observation slots."""

    base_species_ids: torch.Tensor
    species_ids: torch.Tensor
    form_ids: torch.Tensor

    pokemon_types: torch.Tensor
    pokemon_base_stats: torch.Tensor

    move_ids: torch.Tensor
    move_types: torch.Tensor
    move_categories: torch.Tensor
    move_numeric: torch.Tensor

    item_ids: torch.Tensor
    ability_ids: torch.Tensor
    status_ids: torch.Tensor
    pokemon_numeric: torch.Tensor
    pokemon_mask: torch.Tensor

    weather_id: torch.Tensor
    field_numeric: torch.Tensor

    history_kind: torch.Tensor
    history_move: torch.Tensor
    history_species: torch.Tensor
    history_form: torch.Tensor
    history_actor: torch.Tensor
    history_target: torch.Tensor
    history_reason: torch.Tensor
    history_numeric: torch.Tensor
    history_mask: torch.Tensor
    history_length: torch.Tensor

    action_mask: torch.Tensor

    @classmethod
    def create(cls, *, slot_count: int, max_history: int) -> SharedBattleBuffer:
        if slot_count <= 0:
            raise ValueError("slot_count must be > 0")

        if max_history <= 0:
            raise ValueError("max_history must be > 0")

        def shared_zeros(
            shape: tuple[int, ...], *, dtype: torch.dtype
        ) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype).share_memory_()

        slots = slot_count
        history = max_history

        return cls(
            base_species_ids=shared_zeros(
                (slots, POKEMON_SLOTS), dtype=torch.long
            ),
            species_ids=shared_zeros((slots, POKEMON_SLOTS), dtype=torch.long),
            form_ids=shared_zeros((slots, POKEMON_SLOTS), dtype=torch.long),
            pokemon_types=shared_zeros(
                (slots, POKEMON_SLOTS, TYPE_COUNT), dtype=torch.float32
            ),
            pokemon_base_stats=shared_zeros(
                (slots, POKEMON_SLOTS, BASE_STATS_DIM), dtype=torch.float32
            ),
            move_ids=shared_zeros(
                (slots, POKEMON_SLOTS, MOVES_PER_POKEMON), dtype=torch.long
            ),
            move_types=shared_zeros(
                (slots, POKEMON_SLOTS, MOVES_PER_POKEMON, TYPE_COUNT),
                dtype=torch.float32,
            ),
            move_categories=shared_zeros(
                (slots, POKEMON_SLOTS, MOVES_PER_POKEMON, MOVE_CATEGORY_COUNT),
                dtype=torch.float32,
            ),
            move_numeric=shared_zeros(
                (slots, POKEMON_SLOTS, MOVES_PER_POKEMON, MOVE_NUMERIC_DIM),
                dtype=torch.float32,
            ),
            item_ids=shared_zeros((slots, POKEMON_SLOTS), dtype=torch.long),
            ability_ids=shared_zeros((slots, POKEMON_SLOTS), dtype=torch.long),
            status_ids=shared_zeros((slots, POKEMON_SLOTS), dtype=torch.long),
            pokemon_numeric=shared_zeros(
                (slots, POKEMON_SLOTS, POKEMON_NUMERIC_DIM), dtype=torch.float32
            ),
            pokemon_mask=shared_zeros((slots, POKEMON_SLOTS), dtype=torch.bool),
            weather_id=shared_zeros((slots,), dtype=torch.long),
            field_numeric=shared_zeros(
                (slots, FIELD_NUMERIC_DIM), dtype=torch.float32
            ),
            history_kind=shared_zeros((slots, history), dtype=torch.long),
            history_move=shared_zeros((slots, history), dtype=torch.long),
            history_species=shared_zeros((slots, history), dtype=torch.long),
            history_form=shared_zeros((slots, history), dtype=torch.long),
            history_actor=shared_zeros((slots, history), dtype=torch.long),
            history_target=shared_zeros((slots, history), dtype=torch.long),
            history_reason=shared_zeros((slots, history), dtype=torch.long),
            history_numeric=shared_zeros(
                (slots, history, HISTORY_NUMERIC_DIM), dtype=torch.float32
            ),
            history_mask=shared_zeros((slots, history), dtype=torch.bool),
            history_length=shared_zeros((slots,), dtype=torch.long),
            action_mask=shared_zeros((slots, ACTION_COUNT), dtype=torch.bool),
        )

    @property
    def slot_count(self) -> int:
        return self.base_species_ids.shape[0]

    def write(self, slot_index: int, observation: BattleTensors) -> None:
        if not 0 <= slot_index < self.slot_count:
            raise IndexError(
                f"slot_index {slot_index} is outside shared buffer"
            )

        self.base_species_ids[slot_index].copy_(observation.base_species_ids)
        self.species_ids[slot_index].copy_(observation.species_ids)
        self.form_ids[slot_index].copy_(observation.form_ids)

        self.pokemon_types[slot_index].copy_(observation.pokemon_types)
        self.pokemon_base_stats[slot_index].copy_(
            observation.pokemon_base_stats
        )

        self.move_ids[slot_index].copy_(observation.move_ids)
        self.move_types[slot_index].copy_(observation.move_types)
        self.move_categories[slot_index].copy_(observation.move_categories)
        self.move_numeric[slot_index].copy_(observation.move_numeric)

        self.item_ids[slot_index].copy_(observation.item_ids)
        self.ability_ids[slot_index].copy_(observation.ability_ids)
        self.status_ids[slot_index].copy_(observation.status_ids)
        self.pokemon_numeric[slot_index].copy_(observation.pokemon_numeric)
        self.pokemon_mask[slot_index].copy_(observation.pokemon_mask)

        self.weather_id[slot_index].copy_(observation.weather_id)
        self.field_numeric[slot_index].copy_(observation.field_numeric)

        self.history_kind[slot_index].copy_(observation.history_kind)
        self.history_move[slot_index].copy_(observation.history_move)
        self.history_species[slot_index].copy_(observation.history_species)
        self.history_form[slot_index].copy_(observation.history_form)
        self.history_actor[slot_index].copy_(observation.history_actor)
        self.history_target[slot_index].copy_(observation.history_target)
        self.history_reason[slot_index].copy_(observation.history_reason)
        self.history_numeric[slot_index].copy_(observation.history_numeric)
        self.history_mask[slot_index].copy_(observation.history_mask)
        self.history_length[slot_index].copy_(observation.history_length)

        self.action_mask[slot_index].copy_(observation.action_mask)

    def batch(self, slot_indices: list[int]) -> BattleBatch:
        if not slot_indices:
            raise ValueError("Cannot create an empty shared-memory batch")

        indices = torch.tensor(slot_indices, dtype=torch.long)

        return BattleBatch(
            base_species_ids=self.base_species_ids.index_select(0, indices),
            species_ids=self.species_ids.index_select(0, indices),
            form_ids=self.form_ids.index_select(0, indices),
            pokemon_types=self.pokemon_types.index_select(0, indices),
            pokemon_base_stats=self.pokemon_base_stats.index_select(0, indices),
            move_ids=self.move_ids.index_select(0, indices),
            move_types=self.move_types.index_select(0, indices),
            move_categories=self.move_categories.index_select(0, indices),
            move_numeric=self.move_numeric.index_select(0, indices),
            item_ids=self.item_ids.index_select(0, indices),
            ability_ids=self.ability_ids.index_select(0, indices),
            status_ids=self.status_ids.index_select(0, indices),
            pokemon_numeric=self.pokemon_numeric.index_select(0, indices),
            pokemon_mask=self.pokemon_mask.index_select(0, indices),
            weather_id=self.weather_id.index_select(0, indices),
            field_numeric=self.field_numeric.index_select(0, indices),
            history_kind=self.history_kind.index_select(0, indices),
            history_move=self.history_move.index_select(0, indices),
            history_species=self.history_species.index_select(0, indices),
            history_form=self.history_form.index_select(0, indices),
            history_actor=self.history_actor.index_select(0, indices),
            history_target=self.history_target.index_select(0, indices),
            history_reason=self.history_reason.index_select(0, indices),
            history_numeric=self.history_numeric.index_select(0, indices),
            history_mask=self.history_mask.index_select(0, indices),
            history_length=self.history_length.index_select(0, indices),
            action_mask=self.action_mask.index_select(0, indices),
        )


_REQUEST_QUEUE: ProcessQueue | None = None
_RESPONSE_QUEUES: list[ProcessQueue] | None = None
_SHARED_BUFFER: SharedBattleBuffer | None = None


def gpu_worker_initializer(
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
        raise RuntimeError(
            "GPU inference transport was not initialized in this worker"
        )

    if not 0 <= worker_index < len(response_queues):
        raise IndexError(f"No response queue for worker {worker_index}")

    return request_queue, response_queues[worker_index], shared_buffer


async def response_pump(
    *, worker_index: int, pending: PendingInference
) -> None:
    """Route one worker response queue into per-request asyncio Futures."""

    _, response_queue, _ = _worker_transport(worker_index)

    while True:
        response = await asyncio.to_thread(response_queue.get)

        if response is None:
            return

        if not isinstance(response, InferenceResponse):
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
            future.set_exception(
                RuntimeError(f"GPU inference broker failed:\n{response.error}")
            )
            continue

        if response.logits is None or response.value is None:
            future.set_exception(
                RuntimeError("GPU inference broker returned an empty result")
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

        future.set_result(
            (torch.tensor(response.logits, dtype=torch.float32), response.value)
        )


def stop_response_pump(worker_index: int) -> None:
    _, response_queue, _ = _worker_transport(worker_index)
    response_queue.put(None)


@dataclass
class WorkerInferenceStats:
    requests: int = 0
    shared_write_seconds: float = 0.0
    queue_put_seconds: float = 0.0
    response_wait_seconds: float = 0.0


def make_remote_infer(
    *,
    worker_index: int,
    slot_index: int,
    pending: PendingInference,
    stats: WorkerInferenceStats | None = None,
    timeout_seconds: float = 120.0,
) -> AsyncInferenceFn:
    """Create the async inference callable injected into one battle lane."""

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

        request = InferenceRequest(
            worker_index=worker_index,
            slot_index=slot_index,
            request_id=request_id,
        )

        try:
            write_start = perf_counter()
            shared_buffer.write(slot_index, observation)
            write_end = perf_counter()

            try:
                request_queue.put_nowait(request)
            except queue.Full:
                # Do not block every battle lane in this worker if the
                # broker temporarily falls behind.
                await asyncio.to_thread(request_queue.put, request)

            put_end = perf_counter()

            if stats is not None:
                stats.requests += 1
                stats.shared_write_seconds += write_end - write_start
                stats.queue_put_seconds += put_end - write_end

            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=timeout_seconds
            )

            if stats is not None:
                stats.response_wait_seconds += perf_counter() - put_end

            return result

        except BaseException:
            pending.pop(key, None)

            if not future.done():
                future.cancel()

            raise

    return infer


class BatchedGpuInferenceBroker:
    """Batch async rollout requests onto the main process GPU model."""

    def __init__(
        self,
        *,
        model: BattleModel,
        device: torch.device,
        request_queue: ProcessQueue,
        response_queues: list[ProcessQueue],
        shared_buffer: SharedBattleBuffer,
        max_batch_size: int,
        batch_wait_ms: float,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                f"BatchedGpuInferenceBroker requires CUDA/ROCm, got {device}"
            )
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if batch_wait_ms < 0.0:
            raise ValueError("batch_wait_ms must be >= 0")

        self.model = model
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
        self._total_inference_seconds = 0.0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU inference broker is already running")

        self._thread = threading.Thread(
            target=self._run, name="batched-gpu-inference", daemon=True
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
            self._total_inference_seconds = 0.0
            self._total_batch_wait_seconds = 0.0
            self._total_gather_seconds = 0.0
            self._total_dispatch_seconds = 0.0

    def snapshot_stats(self) -> GpuInferenceStats:
        with self._stats_lock:
            return GpuInferenceStats(
                requests=self._requests,
                batches=self._batches,
                max_batch_size=self._max_observed_batch_size,
                total_inference_seconds=self._total_inference_seconds,
                total_batch_wait_seconds=self._total_batch_wait_seconds,
                total_dispatch_seconds=self._total_dispatch_seconds,
                total_gather_seconds=self._total_gather_seconds,
            )

    def _run(self) -> None:
        stop_after_batch = False

        while True:
            item = self.request_queue.get()

            if item is None:
                return

            if not isinstance(item, InferenceRequest):
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

                    if not isinstance(next_item, InferenceRequest):
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

                    if not isinstance(next_item, InferenceRequest):
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

    def _process_batch(self, requests: list[InferenceRequest]) -> None:
        gather_start = perf_counter()

        cpu_batch = self.shared_buffer.batch(
            [request.slot_index for request in requests]
        )

        gather_seconds = perf_counter() - gather_start

        inference_start = perf_counter()

        gpu_batch = cpu_batch.to(self.device)

        with torch.inference_mode():
            logits, values = self.model(gpu_batch)

        logits_cpu = logits.detach().cpu()
        values_cpu = values.detach().cpu()

        inference_seconds = perf_counter() - inference_start

        expected_batch = len(requests)

        # existing shape checks...

        dispatch_start = perf_counter()

        for index, request in enumerate(requests):
            self.response_queues[request.worker_index].put(
                InferenceResponse(
                    slot_index=request.slot_index,
                    request_id=request.request_id,
                    logits=tuple(
                        float(value) for value in logits_cpu[index].tolist()
                    ),
                    value=float(values_cpu[index].item()),
                )
            )

        dispatch_seconds = perf_counter() - dispatch_start

        with self._stats_lock:
            self._requests += expected_batch
            self._batches += 1
            self._max_observed_batch_size = max(
                self._max_observed_batch_size, expected_batch
            )
            self._total_gather_seconds += gather_seconds
            self._total_inference_seconds += inference_seconds
            self._total_dispatch_seconds += dispatch_seconds

    def _send_errors(
        self, requests: list[InferenceRequest], error_text: str
    ) -> None:
        for request in requests:
            self.response_queues[request.worker_index].put(
                InferenceResponse(
                    slot_index=request.slot_index,
                    request_id=request.request_id,
                    logits=None,
                    value=None,
                    error=error_text,
                )
            )
