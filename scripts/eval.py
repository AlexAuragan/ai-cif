import argparse
import asyncio
import csv
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import combinations_with_replacement
from pathlib import Path
from time import perf_counter

import torch
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import (
    MaxBasePowerCombatHandler,
    RandomMoveCombatHandler,
    SimpleHeuristicsCombatHandler,
)
from showdown_sdk.exceptions import BattleLifecycleError
from showdown_sdk.models.sdk import SampleTeamGenerator
from showdown_sdk.models.sdk.team_generators.team_generator import BaseTeamGenerator
from tqdm import tqdm

from ai_cif.inference.combat_handler import NeuralCombatHandler
from ai_cif.model.model import create_battle_model
from scripts.utils.battles import outcome_for, run_battle
from scripts.utils.multithreading import split_battles, worker_initializer

DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/showdown/websocket"

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


@dataclass(frozen=True)
class Participant:
    name: str
    checkpoint: Path | str


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


def discover_participants(models_dir: Path) -> list[Participant]:
    checkpoints = sorted(models_dir.glob("*/*.pt"))

    participants = [
        Participant(name="random", checkpoint="random"),
        Participant(name="MaxBasePower", checkpoint="MaxBasePower"),
        Participant(name="SimpleHeuristics", checkpoint="SimpleHeuristics"),
    ]

    for checkpoint in checkpoints:
        if checkpoint.stem in ("random", "MaxBasePower", "SimpleHeuristics"):
            raise ValueError(
                f"{checkpoint} uses reserved model name '{checkpoint.stem}'"
            )

        participants.append(Participant(name=checkpoint.stem, checkpoint=checkpoint))

    return participants


def load_neural_handler(checkpoint_path: str) -> NeuralCombatHandler:
    device = torch.device("cpu")

    model, tensorizer = create_battle_model(device=device, max_history=32, vocab_gen=4)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path!r} must contain a dict")

    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    if not isinstance(model_state, dict):
        raise TypeError(f"Checkpoint {checkpoint_path!r} has invalid model state")

    model.load_state_dict(model_state)
    model.eval()

    return NeuralCombatHandler(model=model, tensorizer=tensorizer, device=device)


def create_handler(checkpoint_path: str):
    match checkpoint_path:
        case "random":
            return RandomMoveCombatHandler()
        case "MaxBasePower":
            return MaxBasePowerCombatHandler()
        case "SimpleHeuristics":
            return SimpleHeuristicsCombatHandler()

    return load_neural_handler(checkpoint_path)


async def run_pair_worker_async(
    *,
    model_1_path: str,
    model_2_path: str,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    phase_id: int,
    side_offset: int,
) -> PairResult:
    handler_1 = create_handler(model_1_path)
    handler_2 = create_handler(model_2_path)

    client_1 = Client(url, combat_handler=handler_1)
    client_2 = Client(url, combat_handler=handler_2)

    client_1.log_manager.disable()
    client_2.log_manager.disable()

    team_generator_1: BaseTeamGenerator | None = None
    team_generator_2: BaseTeamGenerator | None = None

    if "randombattle" not in fmt:
        team_generator_1 = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index * 2
        )
        team_generator_2 = SampleTeamGenerator(
            team_seed + phase_id * 10_000 + worker_index * 2 + 1
        )

    client_1_name = f"E{phase_id}A{worker_index}"
    client_2_name = f"E{phase_id}B{worker_index}"

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
            # Alternate which Showdown side model_1 occupies so the result is
            # not coupled to always being challenger/client 1.
            model_1_on_client_1 = (side_offset + completed) % 2 == 0

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
            except BattleLifecycleError as error:
                print(f"Discarding failed evaluation battle and retrying: {error!r}")

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
        await asyncio.gather(client_1.close(), client_2.close(), return_exceptions=True)


def run_pair_worker(
    model_1_path: str,
    model_2_path: str,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    worker_index: int,
    phase_id: int,
    side_offset: int,
) -> PairResult:
    return asyncio.run(
        run_pair_worker_async(
            model_1_path=model_1_path,
            model_2_path=model_2_path,
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            battles=battles,
            worker_index=worker_index,
            phase_id=phase_id,
            side_offset=side_offset,
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
    worker_count: int,
    phase_id: int,
) -> PairResult:
    counts = split_battles(battles, worker_count)
    loop = asyncio.get_running_loop()

    tasks = []
    side_offset = 0

    model_1_path = str(model_1.checkpoint)
    model_2_path = str(model_2.checkpoint)

    for worker_index, count in enumerate(counts):
        if count <= 0:
            continue

        tasks.append(
            loop.run_in_executor(
                pool,
                partial(
                    run_pair_worker,
                    model_1_path,
                    model_2_path,
                    url,
                    fmt,
                    team_seed,
                    count,
                    worker_index,
                    phase_id,
                    side_offset,
                ),
            )
        )

        side_offset += count

    results = await asyncio.gather(*tasks)
    return merge_pair_results(results)


def read_scores(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return {}

        missing = [column for column in CSV_COLUMNS if column not in reader.fieldnames]

        if missing:
            raise ValueError(f"{path} is missing CSV columns: {missing}")

        rows: dict[tuple[str, str, str], dict[str, str]] = {}

        for row in reader:
            key = (row["format"], row["model_1"], row["model_2"])
            rows[key] = row

        return rows


def write_scores(path: Path, rows: dict[tuple[str, str, str], dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for key in sorted(rows):
            writer.writerow(rows[key])


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
    models_dir = Path("data/models")
    scores_path = Path("experiments/scores.csv")
    participants = discover_participants(models_dir)
    pairs = list(combinations_with_replacement(participants, 2))

    print(f"Format: {args.fmt}")
    print(f"Models directory: {models_dir}")
    print(f"Scores CSV: {scores_path}")
    print(
        "Participants: " + ", ".join(participant.name for participant in participants)
    )
    print(f"Pairs including self-play: {len(pairs)}")
    print(f"Target battles per pair: {args.battles}")
    print(f"Workers: {args.workers}")
    print(f"PyTorch threads per worker: {args.worker_torch_threads}")

    rows = read_scores(scores_path)

    context = multiprocessing.get_context("spawn")
    pool: ProcessPoolExecutor | None = None
    pool_terminated = False

    try:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=worker_initializer,
            initargs=(args.worker_torch_threads,),
        )

        for pair_index, (model_1, model_2) in enumerate(
            tqdm(pairs, desc="Evaluating matchups", unit="pair"), start=1
        ):
            key = (args.fmt, model_1.name, model_2.name)

            existing = rows.get(key)

            if existing is not None:
                existing_battles = int(existing["nb_battles"])

                if existing_battles >= args.battles:
                    tqdm.write(
                        f"[{pair_index}/{len(pairs)}] "
                        f"{model_1.name} vs {model_2.name}: "
                        f"already has {existing_battles} battles, skipping"
                    )
                    continue

                tqdm.write(
                    f"[{pair_index}/{len(pairs)}] "
                    f"{model_1.name} vs {model_2.name}: "
                    f"existing row has only {existing_battles} battles; "
                    f"rerunning the full {args.battles}"
                )
            else:
                tqdm.write(
                    f"[{pair_index}/{len(pairs)}] {model_1.name} vs {model_2.name}"
                )

            start = perf_counter()

            result = await evaluate_pair_multiprocess(
                pool=pool,
                model_1=model_1,
                model_2=model_2,
                url=args.url,
                fmt=args.fmt,
                team_seed=args.team_seed,
                battles=args.battles,
                worker_count=args.workers,
                phase_id=pair_index,
            )

            elapsed = perf_counter() - start

            rows[key] = result_to_row(
                fmt=args.fmt, model_1=model_1, model_2=model_2, result=result
            )

            # Persist after every completed matchup. If the process is
            # interrupted, all previously completed pairs are retained.
            write_scores(scores_path, rows)

            tqdm.write(
                f"  score={result.winrate:.2%} "
                f"W/L/T={result.wins}/{result.losses}/{result.ties} "
                f"variance={result.winrate_variance:.8f} "
                f"std_error={result.standard_error:.2%} "
                f"time={elapsed:.2f}s "
                f"battles/s={result.battles / elapsed:.2f}"
            )

    except BaseException:
        if pool is not None:
            pool_terminated = True
            pool.terminate_workers()
        raise

    finally:
        if pool is not None and not pool_terminated:
            pool.shutdown(wait=True, cancel_futures=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", default=DEFAULT_WEBSOCKET_URL)

    parser.add_argument("--format", dest="fmt", default="gen1randombattle")

    parser.add_argument(
        "--battles",
        type=int,
        default=1000,
        help="Target number of battles for every pair",
    )

    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--worker-torch-threads", type=int, default=4)

    parser.add_argument("--team-seed", type=int, default=42)

    args = parser.parse_args()

    if args.battles < 1:
        parser.error("--battles must be at least 1")

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.worker_torch_threads < 1:
        parser.error("--worker-torch-threads must be at least 1")

    return args


async def main() -> None:
    args = parse_args()
    await evaluate_all(args)


if __name__ == "__main__":
    asyncio.run(main())
