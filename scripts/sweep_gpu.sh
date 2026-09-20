
#!/usr/bin/env bash
set -uo pipefail

# Overnight benchmark for scripts/train_gpu.py
#
# Sweeps:
#   - rollout worker processes
#   - PyTorch threads per rollout process
#   - GPU max batch size
#
# battle-lanes is derived automatically so that all rollout battles can be
# in flight at once:
#   battle_lanes = ceil(ROLLOUT_BATTLES / workers)
#
# Raw logs, per-run metrics, and a ranked summary are written under:
#   benchmarks/train_gpu_<timestamp>/
#
# Optional environment overrides:
#   REPEATS=3 ITERATIONS=15 WARMUP=3 ./benchmark_train_gpu.sh
#   GPU_BATCH_WAIT_MS=0.5 RUN_TIMEOUT=20m ./benchmark_train_gpu.sh
#
# Edit the arrays below if you want a smaller/larger search.

WORKERS=(18 20 22)
THREADS=(1 2 3)
BATCH_SIZES=(16 24 32 40)
GPU_BATCH_WAIT_MS=(0.0 0.25 0.5 1)

REPEATS="${REPEATS:-2}"
ITERATIONS="${ITERATIONS:-15}"
WARMUP="${WARMUP:-3}"

ROLLOUT_BATTLES="${ROLLOUT_BATTLES:-100}"
EVAL_BATTLES="${EVAL_BATTLES:-50}"
# Keep periodic evaluation out of the benchmark iterations. The mandatory
# initial evaluation still warms the model/GPU before timed rollouts.
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"

RUN_TIMEOUT="${RUN_TIMEOUT:-20m}"
RANDOMIZE="${RANDOMIZE:-1}"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Run this script from inside the ai-cif git repository." >&2
    exit 1
}
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmarks/train_gpu_${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
CSV="${OUT_DIR}/runs.csv"
SUMMARY="${OUT_DIR}/summary.csv"

mkdir -p "$LOG_DIR"

cat > "$CSV" <<'CSV_HEADER'
repeat,workers,threads,batch_size,battle_lanes,status,iterations_used,mean_rollout_s,median_rollout_s,mean_battles_s,median_battles_s,mean_decisions_s,median_decisions_s,mean_batch,mean_gpu_requests_s,log
CSV_HEADER

CONFIG_FILE="${OUT_DIR}/configs.txt"
: > "$CONFIG_FILE"

for workers in "${WORKERS[@]}"; do
    for threads in "${THREADS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for gpu_batch_wait_ms in "${GPU_BATCH_WAIT_MS}"; do
                printf '%s %s %s %s\n' "$workers" "$threads" "$batch_size" "$gpu_batch_wait_ms">> "$CONFIG_FILE"
            done
        done
    done
done

if [[ "$RANDOMIZE" == "1" ]]; then
    shuf "$CONFIG_FILE" -o "$CONFIG_FILE"
fi

TOTAL_CONFIGS="$(wc -l < "$CONFIG_FILE")"
TOTAL_RUNS=$((TOTAL_CONFIGS * REPEATS))
RUN_INDEX=0

echo "Output:       $OUT_DIR"
echo "Configurations: $TOTAL_CONFIGS"
echo "Repeats:      $REPEATS"
echo "Total runs:   $TOTAL_RUNS"
echo "Iterations:   $ITERATIONS per run"
echo "Warmup skip:  first $WARMUP rollout iterations"
echo "Rollout battles: $ROLLOUT_BATTLES"
echo

append_metrics() {
    local log_file="$1"
    local repeat="$2"
    local workers="$3"
    local threads="$4"
    local batch_size="$5"
    local battle_lanes="$6"
    local status="$7"

    uv run python - \
        "$log_file" "$CSV" "$repeat" "$workers" "$threads" \
        "$batch_size" "$battle_lanes" "$status" "$WARMUP" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

(
    log_path,
    csv_path,
    repeat,
    workers,
    threads,
    batch_size,
    battle_lanes,
    status,
    warmup,
) = sys.argv[1:]

warmup = int(warmup)

rollout_re = re.compile(
    r"rollout_time=(?P<seconds>[0-9.]+)s "
    r"battles/s=(?P<bps>[0-9.]+) "
    r"decisions/s=(?P<dps>[0-9.]+)"
)
gpu_re = re.compile(
    r"^gpu_inference "
    r"requests=(?P<requests>\d+) "
    r"batches=(?P<batches>\d+) "
    r"mean_batch=(?P<mean_batch>[0-9.]+) "
    r"max_batch=(?P<max_batch>\d+) "
    r"inference_time=(?P<inference_seconds>[0-9.]+)s "
    r"requests/inference_s=(?P<gpu_rps>[0-9.]+)"
)

rows = []
last_rollout = None

for line in Path(log_path).read_text(errors="replace").splitlines():
    rollout_match = rollout_re.search(line)
    if rollout_match:
        last_rollout = {
            "seconds": float(rollout_match["seconds"]),
            "bps": float(rollout_match["bps"]),
            "dps": float(rollout_match["dps"]),
            "mean_batch": None,
            "gpu_rps": None,
        }
        rows.append(last_rollout)
        continue

    if last_rollout is not None and last_rollout["mean_batch"] is None:
        gpu_match = gpu_re.search(line)
        if gpu_match:
            last_rollout["mean_batch"] = float(gpu_match["mean_batch"])
            last_rollout["gpu_rps"] = float(gpu_match["gpu_rps"])

used = rows[warmup:]

def mean(key):
    values = [row[key] for row in used if row[key] is not None]
    return statistics.fmean(values) if values else float("nan")

def median(key):
    values = [row[key] for row in used if row[key] is not None]
    return statistics.median(values) if values else float("nan")

fields = [
    repeat,
    workers,
    threads,
    batch_size,
    battle_lanes,
    status,
    len(used),
    f"{mean('seconds'):.6f}",
    f"{median('seconds'):.6f}",
    f"{mean('bps'):.6f}",
    f"{median('bps'):.6f}",
    f"{mean('dps'):.6f}",
    f"{median('dps'):.6f}",
    f"{mean('mean_batch'):.6f}",
    f"{mean('gpu_rps'):.6f}",
    log_path,
]

with open(csv_path, "a", newline="") as handle:
    csv.writer(handle).writerow(fields)
PY
}

for repeat in $(seq 1 "$REPEATS"); do
    while read -r workers threads batch_size gpu_batch_wait_ms; do
        RUN_INDEX=$((RUN_INDEX + 1))

        # Enough lanes that every rollout battle can be active immediately.
        battle_lanes=$(((ROLLOUT_BATTLES + workers - 1) / workers))

        tag="r${repeat}_w${workers}_t${threads}_b${batch_size}_l${battle_lanes}"
        log_file="${LOG_DIR}/${tag}.log"

        echo "================================================================"
        echo "[$RUN_INDEX/$TOTAL_RUNS] $tag"
        echo "workers=$workers threads=$threads batch=$batch_size lanes=$battle_lanes"
        echo "================================================================"

        cmd=(
            uv run scripts/train_gpu.py
            --no-wandb
            --wandb-name "$tag"
            --set-running "workers=${workers}" "threads=${threads}"
            --battle-lanes "$battle_lanes"
            --gpu-batch-size "$batch_size"
            --gpu-batch-wait-ms "$gpu_batch_wait_ms"
            --set-training
                "iterations=${ITERATIONS}"
                "rollout_battles=${ROLLOUT_BATTLES}"
                "eval_battles=${EVAL_BATTLES}"
                "eval_interval=${EVAL_INTERVAL}"
        )

        set +e
        timeout --signal=INT --kill-after=30s "$RUN_TIMEOUT" \
            "${cmd[@]}" 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}
        set -e

        if [[ "$exit_code" -eq 0 ]]; then
            status="ok"
        elif [[ "$exit_code" -eq 124 ]]; then
            status="timeout"
        else
            status="exit_${exit_code}"
        fi

        append_metrics \
            "$log_file" "$repeat" "$workers" "$threads" \
            "$batch_size" "$gpu_batch_wait_ms" "$battle_lanes" "$status"

        echo
        echo "Finished $tag: $status"
        echo
    done < "$CONFIG_FILE"
done

uv run python - "$CSV" "$SUMMARY" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

runs_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

groups = defaultdict(list)

with runs_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        if row["status"] != "ok":
            continue
        if int(row["iterations_used"]) <= 0:
            continue

        key = (
            int(row["workers"]),
            int(row["threads"]),
            int(row["batch_size"]),
            int(row["battle_lanes"]),
        )
        groups[key].append(row)

summary = []

for (workers, threads, batch_size, battle_lanes), rows in groups.items():
    def avg(field):
        vals = [float(row[field]) for row in rows]
        vals = [v for v in vals if math.isfinite(v)]
        return statistics.fmean(vals) if vals else float("nan")

    summary.append(
        {
            "workers": workers,
            "threads": threads,
            "batch_size": batch_size,
            "battle_lanes": battle_lanes,
            "successful_repeats": len(rows),
            "mean_rollout_s": avg("mean_rollout_s"),
            "median_rollout_s": avg("median_rollout_s"),
            "mean_battles_s": avg("mean_battles_s"),
            "median_battles_s": avg("median_battles_s"),
            "mean_decisions_s": avg("mean_decisions_s"),
            "median_decisions_s": avg("median_decisions_s"),
            "mean_batch": avg("mean_batch"),
            "mean_gpu_requests_s": avg("mean_gpu_requests_s"),
        }
    )

summary.sort(key=lambda row: row["mean_battles_s"], reverse=True)

fieldnames = [
    "rank",
    "workers",
    "threads",
    "batch_size",
    "battle_lanes",
    "successful_repeats",
    "mean_rollout_s",
    "median_rollout_s",
    "mean_battles_s",
    "median_battles_s",
    "mean_decisions_s",
    "median_decisions_s",
    "mean_batch",
    "mean_gpu_requests_s",
]

with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for rank, row in enumerate(summary, start=1):
        formatted = {"rank": rank, **row}
        for key, value in list(formatted.items()):
            if isinstance(value, float):
                formatted[key] = f"{value:.4f}"
        writer.writerow(formatted)

print()
print("Top configurations by mean rollout battles/s")
print("=============================================")
print(
    f"{'rank':>4} {'workers':>7} {'thr':>4} {'batch':>5} {'lanes':>5} "
    f"{'battles/s':>10} {'decisions/s':>12} {'rollout_s':>10} {'mean_batch':>10}"
)

for rank, row in enumerate(summary[:15], start=1):
    print(
        f"{rank:>4} "
        f"{row['workers']:>7} "
        f"{row['threads']:>4} "
        f"{row['batch_size']:>5} "
        f"{row['battle_lanes']:>5} "
        f"{row['mean_battles_s']:>10.2f} "
        f"{row['mean_decisions_s']:>12.1f} "
        f"{row['mean_rollout_s']:>10.2f} "
        f"{row['mean_batch']:>10.2f}"
    )

print()
print(f"Full summary: {summary_path}")
print(f"Per-run data: {runs_path}")
PY

echo
echo "Benchmark complete."
echo "Raw logs: $LOG_DIR"
echo "Runs CSV:  $CSV"
echo "Summary:   $SUMMARY"
