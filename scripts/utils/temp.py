from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class BattleTiming:
    start_ns: int
    end_ns: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000_000


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower

    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rollout_tail(timings: list[BattleTiming]) -> str:
    if not timings:
        return "rollout_tail battles=0"

    durations = [timing.duration_seconds for timing in timings]

    first_start_ns = min(timing.start_ns for timing in timings)
    completion_offsets = sorted(
        (timing.end_ns - first_start_ns) / 1_000_000_000 for timing in timings
    )

    def completion_time(fraction: float) -> float:
        index = ceil(len(completion_offsets) * fraction) - 1
        index = max(0, min(index, len(completion_offsets) - 1))
        return completion_offsets[index]

    p50 = _percentile(durations, 0.50)
    p90 = _percentile(durations, 0.90)
    p95 = _percentile(durations, 0.95)
    p99 = _percentile(durations, 0.99)
    maximum = max(durations)

    finish_90 = completion_time(0.90)
    finish_95 = completion_time(0.95)
    finish_99 = completion_time(0.99)
    finish_100 = completion_time(1.00)

    tail_90_100 = finish_100 - finish_90
    tail_95_100 = finish_100 - finish_95
    tail_99_100 = finish_100 - finish_99

    return (
        "rollout_tail "
        f"battles={len(timings)} "
        f"duration_p50={p50:.3f}s "
        f"p90={p90:.3f}s "
        f"p95={p95:.3f}s "
        f"p99={p99:.3f}s "
        f"max={maximum:.3f}s "
        f"finish90={finish_90:.3f}s "
        f"finish95={finish_95:.3f}s "
        f"finish99={finish_99:.3f}s "
        f"finish100={finish_100:.3f}s "
        f"tail90_100={tail_90_100:.3f}s "
        f"tail95_100={tail_95_100:.3f}s "
        f"tail99_100={tail_99_100:.3f}s"
    )
