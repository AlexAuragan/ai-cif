"""Sweep GPU rollout throughput across the knobs we care about.

Configure the sweep by editing the plain Python objects below -- there is no
command line interface. Every candidate configuration is a ``SweepConfig``;
fields left as ``None`` fall back to the current ``scripts/train.py`` default,
so ``SweepConfig()`` is exactly the baseline we ship today.

For each configuration this script calls ``scripts.train.train`` in-process
and reports:

* rollout time at iteration 10 and iteration 20,
* steady-state battles/second,
* number of crashed (discarded) battles.

Because the rollout workers run in separate processes, their "Discarded N
failed battles" output is captured at the file-descriptor level so it can be
counted while still being echoed live to the terminal.

Run it from the repository root, in the same environment as training::

    uv run scripts/sweep_gpu.py

The training server (``DEFAULT_WEBSOCKET_URL``) must be up, exactly as for
``scripts/train.py`` itself. Results are written as ``results.csv`` /
``results.json`` plus one raw log per run under ``experiments/sweeps``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import sys
import threading
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from scripts.train import (
    MODEL_CONFIG,
    POOL_CONFIG,
    PPO_CONFIG,
    REWARD_CONFIG,
    RUNNING_CONFIG,
    TENSORIZER,
    TRAINING_CONFIG,
    train,
)
from scripts.utils.config import RUNNING_TYPES, TRAINING_TYPES

# --------------------------------------------------------------------------
# Sweep definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """One point in the sweep. ``None`` means "use the train.py default"."""

    gpu_batch_wait_ms: float | None = None
    battle_lanes: int | None = None
    threads: int | None = None
    workers: int | None = None


# Candidate values per knob, written as offsets from the current scripts/train.py
# default (0 == the default), so the sweep keeps tracking train.py. Keep these
# small: a full combination costs one ~20 minute run.
GPU_BATCH_WAIT_MS_OFFSETS = (0.0,)
BATTLE_LANES_OFFSETS = (-1, 0, 1)
WORKERS_OFFSETS = (-4, 0, 4)
THREADS_VALUES = (1,)

# With COMBINE on, every combination is tried, so a config can add a lane while
# removing workers (or the other way around) instead of only moving one knob.
COMBINE = True


def _offset_floats(
    base: float, offsets: tuple[float, ...], minimum: float
) -> tuple[float, ...]:
    return tuple(sorted({max(minimum, base + offset) for offset in offsets}))


def _offset_ints(
    base: int, offsets: tuple[int, ...], minimum: int
) -> tuple[int, ...]:
    return tuple(sorted({max(minimum, base + offset) for offset in offsets}))


def _candidate_configs() -> list[SweepConfig]:
    wait_values = _offset_floats(
        RUNNING_CONFIG.gpu_batch_wait_ms, GPU_BATCH_WAIT_MS_OFFSETS, 0.0
    )
    lane_values = _offset_ints(
        RUNNING_CONFIG.battle_lanes, BATTLE_LANES_OFFSETS, 1
    )
    worker_values = _offset_ints(RUNNING_CONFIG.workers, WORKERS_OFFSETS, 1)
    thread_values = tuple(sorted({*THREADS_VALUES, RUNNING_CONFIG.threads}))

    if COMBINE:
        return [
            SweepConfig(
                gpu_batch_wait_ms=wait,
                battle_lanes=lanes,
                threads=threads,
                workers=workers,
            )
            for wait in wait_values
            for lanes in lane_values
            for threads in thread_values
            for workers in worker_values
        ]

    configs = [SweepConfig()]

    for wait in wait_values:
        if wait != RUNNING_CONFIG.gpu_batch_wait_ms:
            configs.append(SweepConfig(gpu_batch_wait_ms=wait))

    for lanes in lane_values:
        if lanes != RUNNING_CONFIG.battle_lanes:
            configs.append(SweepConfig(battle_lanes=lanes))

    for workers in worker_values:
        if workers != RUNNING_CONFIG.workers:
            configs.append(SweepConfig(workers=workers))

    for threads in thread_values:
        if threads != RUNNING_CONFIG.threads:
            configs.append(SweepConfig(threads=threads))

    return configs


# Configurations to run. Add explicit entries by hand for anything the offsets
# above do not cover, e.g. ``SweepConfig(battle_lanes=11, threads=1)``.
CONFIGS: list[SweepConfig] = _candidate_configs()

# Only run the first MAX_CONFIGS configurations (0 = all of them). Results are
# written after every run, so stopping early still leaves usable data.
MAX_CONFIGS = 0

# Rough minutes per full run, only used to print a time estimate up front.
ESTIMATED_MINUTES_PER_RUN = 20.0

# Training knobs shared by every run. Iteration 10 and 20 must exist, so keep
# ITERATIONS at 20 or higher. Set EVAL_INTERVAL very large to skip evaluation
# and measure pure rollout throughput.
ITERATIONS = TRAINING_CONFIG.iterations
ROLLOUT_BATTLES = TRAINING_CONFIG.rollout_battles
EVAL_BATTLES = TRAINING_CONFIG.eval_battles
EVAL_INTERVAL = TRAINING_CONFIG.eval_interval

# Repeat every configuration this many times to smooth out noise.
REPEATS = 1

# Iterations below this are excluded from the steady-state mean.
WARMUP_ITERATIONS = 5

OUTPUT_DIR = Path("experiments/sweeps")
KEEP_CHECKPOINTS = False
STOP_ON_ERROR = False
DRY_RUN = False

ITER_RE = re.compile(r"^iteration=(?P<iter>\d+) battles=\d+ decisions=\d+$")
ROLLOUT_RE = re.compile(
    r"^rollout_time=(?P<seconds>[\d.]+)s "
    r"battles/s=(?P<bps>[\d.]+) "
    r"decisions/s=(?P<dps>[\d.]+)$"
)
DISCARD_RE = re.compile(
    r"^Discarded (?P<crashed>\d+) failed battles "
    r"while collecting \d+ trajectories$"
)
RETRY_RE = re.compile(r"^Discarding failed battle and retrying: ")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    run_id: str
    workers: int
    battle_lanes: int
    threads: int
    gpu_batch_wait_ms: float
    ok: bool
    wall_seconds: float
    iterations_seen: int
    rollout_seconds: dict[int, float]
    battles_per_second: dict[int, float]
    decisions_per_second: dict[int, float]
    crashed_battles: int
    crash_retries: int
    error: str | None = None


class _OutputParser:
    """Pulls the metrics we need out of one run's combined output."""

    def __init__(self) -> None:
        self.rollout_seconds: dict[int, float] = {}
        self.battles_per_second: dict[int, float] = {}
        self.decisions_per_second: dict[int, float] = {}
        self.crashed_battles = 0
        self.crash_retries = 0
        self._current_iteration: int | None = None

    def feed(self, line: str) -> None:
        iteration = ITER_RE.match(line)
        if iteration:
            self._current_iteration = int(iteration.group("iter"))
            return

        rollout = ROLLOUT_RE.match(line)
        if rollout and self._current_iteration is not None:
            current = self._current_iteration
            self.rollout_seconds[current] = float(rollout.group("seconds"))
            self.battles_per_second[current] = float(rollout.group("bps"))
            self.decisions_per_second[current] = float(rollout.group("dps"))
            return

        discarded = DISCARD_RE.match(line)
        if discarded:
            self.crashed_battles += int(discarded.group("crashed"))
            return

        if RETRY_RE.match(line):
            self.crash_retries += 1


# --------------------------------------------------------------------------
# Output capture
# --------------------------------------------------------------------------


def run_captured(action: Callable[[], None]) -> tuple[str, str | None]:
    """Run ``action`` while teeing its stdout, including child processes.

    The rollout workers are spawned processes that inherit file descriptor 1,
    so redirecting it to a pipe captures their output too. A pump thread
    simultaneously echoes everything to the real terminal.

    Returns the captured output and, if ``action`` raised, a short error
    description. KeyboardInterrupt is deliberately not swallowed.
    """

    sys.stdout.flush()

    real_stdout_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.close(write_fd)

    collected: list[bytes] = []

    def pump() -> None:
        while True:
            chunk = os.read(read_fd, 65536)

            if not chunk:
                break

            collected.append(chunk)
            os.write(real_stdout_fd, chunk)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()

    error_text: str | None = None

    try:
        action()
    except Exception as error:  # noqa: BLE001 - keep the sweep going
        error_text = f"{type(error).__name__}: {error}"
    finally:
        sys.stdout.flush()
        # Restoring fd 1 closes the pipe's write end, so the pump sees EOF.
        os.dup2(real_stdout_fd, 1)
        pump_thread.join(timeout=10.0)
        os.close(read_fd)
        pump_thread.join(timeout=10.0)
        os.close(real_stdout_fd)

    return b"".join(collected).decode("utf-8", errors="replace"), error_text


# --------------------------------------------------------------------------
# Sweep machinery
# --------------------------------------------------------------------------


def _assert_supported_fields() -> None:
    required_running = (
        "workers",
        "battle_lanes",
        "threads",
        "gpu_batch_wait_ms",
    )
    required_training = (
        "iterations",
        "rollout_battles",
        "eval_battles",
        "eval_interval",
    )

    missing = [key for key in required_running if key not in RUNNING_TYPES]
    missing += [key for key in required_training if key not in TRAINING_TYPES]

    if missing:
        raise SystemExit(
            "train.py override maps are missing fields: " + ", ".join(missing)
        )


def _overrides_for(config: SweepConfig) -> dict[str, int | float]:
    overrides: dict[str, int | float] = {}

    for field in fields(config):
        value = getattr(config, field.name)

        if value is None:
            continue

        if field.name not in RUNNING_TYPES:
            raise ValueError(f"Unknown running field: {field.name}")

        overrides[field.name] = value

    return overrides


def _run_id(running) -> str:
    return (
        f"w{running.workers}_l{running.battle_lanes}"
        f"_t{running.threads}_g{running.gpu_batch_wait_ms:g}"
    )


def _effective_run_id(config: SweepConfig) -> str:
    running = copy(RUNNING_CONFIG)

    for key, value in _overrides_for(config).items():
        setattr(running, key, value)

    return _run_id(running)


def _train_once(training, running) -> None:
    # ``train`` only reads no_wandb/wandb_name/wandb_group from its args, but it
    # is typed as argparse.Namespace, so hand it one directly.
    args = argparse.Namespace(no_wandb=True, wandb_name=None, wandb_group=None)

    asyncio.run(
        train(
            args,
            copy(PPO_CONFIG),
            training,
            copy(REWARD_CONFIG),
            running,
            copy(MODEL_CONFIG),
            copy(POOL_CONFIG),
            copy(TENSORIZER),
        )
    )


def run_config(
    config: SweepConfig, *, stamp_dir: Path, repeat_index: int = 0
) -> RunResult:
    running = copy(RUNNING_CONFIG)
    training = copy(TRAINING_CONFIG)

    for key, value in _overrides_for(config).items():
        setattr(running, key, value)

    training.iterations = ITERATIONS
    training.rollout_battles = ROLLOUT_BATTLES
    training.eval_battles = EVAL_BATTLES
    training.eval_interval = EVAL_INTERVAL

    identifier = _run_id(running)
    if REPEATS > 1:
        identifier = f"{identifier}-r{repeat_index + 1}"

    running.checkpoint_dir = stamp_dir / "checkpoints" / identifier
    log_path = stamp_dir / f"{identifier}.log"

    print()
    print("=" * 78)
    print(f"run {identifier}")
    print(
        f"  workers={running.workers} battle_lanes={running.battle_lanes} "
        f"threads={running.threads} "
        f"gpu_batch_wait_ms={running.gpu_batch_wait_ms:g} "
        f"gpu_batch_size={running.gpu_batch_size}"
    )
    print("=" * 78)
    sys.stdout.flush()

    parser = _OutputParser()
    start = perf_counter()
    output, error_text = run_captured(partial(_train_once, training, running))
    wall_seconds = perf_counter() - start

    log_path.write_text(output, encoding="utf-8")

    for line in output.splitlines():
        parser.feed(line)

    if not KEEP_CHECKPOINTS:
        shutil.rmtree(running.checkpoint_dir, ignore_errors=True)

    torch.cuda.empty_cache()

    if error_text is not None:
        print(f"  [{identifier}] FAILED: {error_text}")
        sys.stdout.flush()

    result = RunResult(
        run_id=identifier,
        workers=running.workers,
        battle_lanes=running.battle_lanes,
        threads=running.threads,
        gpu_batch_wait_ms=running.gpu_batch_wait_ms,
        ok=error_text is None and bool(parser.rollout_seconds),
        wall_seconds=wall_seconds,
        iterations_seen=len(parser.rollout_seconds),
        rollout_seconds=parser.rollout_seconds,
        battles_per_second=parser.battles_per_second,
        decisions_per_second=parser.decisions_per_second,
        crashed_battles=parser.crashed_battles,
        crash_retries=parser.crash_retries,
        error=error_text,
    )

    print(
        f"  [{identifier}] done wall={wall_seconds:.1f}s "
        f"crashed={result.crashed_battles} "
        f"t@10={_mark(result.rollout_seconds, 10)} "
        f"t@20={_mark(result.rollout_seconds, 20)}"
    )
    sys.stdout.flush()

    return result


def _mark(mapping: dict[int, float], mark: int) -> str:
    value = mapping.get(mark)
    return "-" if value is None else f"{value:.2f}"


def steady_battles_per_second(result: RunResult) -> float | None:
    values = [
        value
        for iteration, value in result.battles_per_second.items()
        if iteration >= WARMUP_ITERATIONS
    ]

    if not values:
        values = list(result.battles_per_second.values())

    if not values:
        return None

    return sum(values) / len(values)


def _result_to_row(result: RunResult) -> dict[str, object]:
    steady = steady_battles_per_second(result)

    return {
        "run": result.run_id,
        "workers": result.workers,
        "battle_lanes": result.battle_lanes,
        "threads": result.threads,
        "gpu_batch_wait_ms": result.gpu_batch_wait_ms,
        "ok": result.ok,
        "wall_seconds": round(result.wall_seconds, 3),
        "iterations_seen": result.iterations_seen,
        "t_iter_10": result.rollout_seconds.get(10),
        "t_iter_20": result.rollout_seconds.get(20),
        "bps_iter_10": result.battles_per_second.get(10),
        "bps_iter_20": result.battles_per_second.get(20),
        "bps_steady": None if steady is None else round(steady, 3),
        "dps_iter_10": result.decisions_per_second.get(10),
        "dps_iter_20": result.decisions_per_second.get(20),
        "crashed_battles": result.crashed_battles,
        "crash_retries": result.crash_retries,
        "error": result.error,
    }


def write_results(results: list[RunResult], stamp_dir: Path) -> None:
    rows = [_result_to_row(result) for result in results]

    if rows:
        with (stamp_dir / "results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    payload = []

    for result, row in zip(results, rows, strict=True):
        entry = dict(row)
        entry["rollout_seconds"] = {
            str(key): value for key, value in result.rollout_seconds.items()
        }
        entry["battles_per_second"] = {
            str(key): value for key, value in result.battles_per_second.items()
        }
        entry["decisions_per_second"] = {
            str(key): value
            for key, value in result.decisions_per_second.items()
        }
        payload.append(entry)

    with (stamp_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def print_summary(results: list[RunResult]) -> None:
    print()
    print("=" * 78)
    print("summary")
    print("=" * 78)

    header = (
        f"{'run':<20}{'w':>4}{'lane':>6}{'thr':>5}{'wait':>7}"
        f"{'t@10':>8}{'t@20':>8}{'bps@10':>9}{'bps@20':>9}"
        f"{'bps_ss':>9}{'crash':>7}{'wall':>9}"
    )
    print(header)
    print("-" * len(header))

    def sort_key(result: RunResult) -> tuple[int, float]:
        steady = steady_battles_per_second(result)

        if not result.ok or steady is None:
            return (1, 0.0)

        return (0, -steady)

    ranked = sorted(results, key=sort_key)

    for result in ranked:
        steady = steady_battles_per_second(result)
        steady_text = "-" if steady is None else f"{steady:.2f}"

        print(
            f"{result.run_id:<20}"
            f"{result.workers:>4}"
            f"{result.battle_lanes:>6}"
            f"{result.threads:>5}"
            f"{result.gpu_batch_wait_ms:>7g}"
            f"{_mark(result.rollout_seconds, 10):>8}"
            f"{_mark(result.rollout_seconds, 20):>8}"
            f"{_mark(result.battles_per_second, 10):>9}"
            f"{_mark(result.battles_per_second, 20):>9}"
            f"{steady_text:>9}"
            f"{result.crashed_battles:>7}"
            f"{result.wall_seconds:>8.1f}s"
            f"{'' if result.ok else '  FAILED'}"
        )

    best = next(
        (
            result
            for result in ranked
            if result.ok and steady_battles_per_second(result) is not None
        ),
        None,
    )

    if best is not None:
        steady = steady_battles_per_second(best)
        print()
        print(
            f"best steady-state: {best.run_id} "
            f"bps_ss={steady:.2f} "
            f"t@10={_mark(best.rollout_seconds, 10)} "
            f"t@20={_mark(best.rollout_seconds, 20)} "
            f"crashed={best.crashed_battles}"
        )

    failed = [result for result in results if not result.ok]

    if failed:
        print()
        print(f"failed runs: {', '.join(result.run_id for result in failed)}")


def main() -> None:
    _assert_supported_fields()

    os.environ["PYTHONUNBUFFERED"] = "1"

    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_DIR

    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir

    stamp_dir = output_dir / stamp

    configs = CONFIGS[:MAX_CONFIGS] if MAX_CONFIGS > 0 else CONFIGS
    estimated_minutes = len(configs) * REPEATS * ESTIMATED_MINUTES_PER_RUN

    print(f"configurations: {len(configs)} x {REPEATS} repeat(s)")
    print(f"iterations: {ITERATIONS} rollout_battles: {ROLLOUT_BATTLES}")
    print(f"estimated time: {estimated_minutes / 60:.1f}h")

    for config in configs:
        print(f"  {_effective_run_id(config)}: {config}")

    if DRY_RUN:
        print("\ndry run: nothing executed")
        return

    stamp_dir.mkdir(parents=True, exist_ok=True)

    print(f"output: {stamp_dir}")
    print("make sure the training server is reachable before continuing")

    results: list[RunResult] = []

    try:
        for config in configs:
            for repeat_index in range(REPEATS):
                result = run_config(
                    config, stamp_dir=stamp_dir, repeat_index=repeat_index
                )
                results.append(result)
                write_results(results, stamp_dir)

                if not result.ok and STOP_ON_ERROR:
                    print(f"stopping after failed run {result.run_id}")
                    write_results(results, stamp_dir)
                    print_summary(results)
                    return
    except KeyboardInterrupt:
        print("\nsweep interrupted; writing partial results")
    finally:
        write_results(results, stamp_dir)
        print_summary(results)
        print(f"\nresults: {stamp_dir / 'results.csv'}")
        print(f"json:    {stamp_dir / 'results.json'}")


if __name__ == "__main__":
    main()
