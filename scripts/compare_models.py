import asyncio
import gzip
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import IO, Any

import anyio
import torch
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import SimpleHeuristicsCombatHandler
from showdown_sdk.exceptions import BattleLifecycleError
from showdown_sdk.features import FEATURE_SCHEMA_VERSION, battle_to_features
from showdown_sdk.models.sdk import BattleState, SampleTeamGenerator
from showdown_sdk.models.sdk.team_generators.team_generator import (
    BaseTeamGenerator,
)

from ai_cif.model.model import BattleModel, create_battle_model
from ai_cif.vectorization.tensorizer import (
    DEFAULT_MAX_HISTORY,
    DEFAULT_VOCAB_GEN,
    BattleTensorizer,
    BattleTensors,
    collate_battles,
)
from scripts.utils.battles import run_battle
from scripts.utils.multithreading import split_battles, worker_initializer

DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/showdown/websocket"
DATASET_SCHEMA = "ai-cif-policy-state-bank-v1"
ACTION_LABELS = tuple(
    [f"move_{index}" for index in range(1, 5)]
    + [f"switch_{index}" for index in range(1, 7)]
)


@dataclass(frozen=True)
class WorkerResult:
    path: str
    battles: int
    states: int


class RecordingSimpleHeuristicsCombatHandler(SimpleHeuristicsCombatHandler):
    """SimpleHeuristics policy that snapshots every decision observation."""

    def __init__(self, *, side: str) -> None:
        self.side = side
        self.battle_id = -1
        self.records: list[dict[str, object]] = []
        self.tensorizer = BattleTensorizer(
            max_history=DEFAULT_MAX_HISTORY, vocab_gen=DEFAULT_VOCAB_GEN
        )

    def start_battle(self, battle_id: int) -> None:
        self.battle_id = battle_id
        self.records = []

    def take_records(self) -> list[dict[str, object]]:
        records = self.records
        self.records = []
        return records

    def select_top_actions(self, battle_state: BattleState):
        features = battle_to_features(battle_state)
        observation = self.tensorizer.tensorize(features)

        actions = super().select_top_actions(battle_state)
        if not actions:
            raise RuntimeError("SimpleHeuristics returned no actions")

        self.records.append(
            {
                "battle_id": self.battle_id,
                "side": self.side,
                "decision_index": len(self.records),
                "turn": features.field.turn,
                "heuristic_action": _action_to_index(actions[0]),
                "observation": _tensors_to_json(observation),
            }
        )

        return actions


def _action_to_index(action: tuple[str, int]) -> int:
    kind, number = action

    if kind == "move" and 1 <= number <= 4:
        return number - 1
    if kind == "switch" and 1 <= number <= 6:
        return number + 3

    raise ValueError(f"Unsupported Showdown action: {action!r}")


def _tensors_to_json(tensors: BattleTensors) -> dict[str, object]:
    return {
        "base_species_ids": tensors.base_species_ids.tolist(),
        "species_ids": tensors.species_ids.tolist(),
        "form_ids": tensors.form_ids.tolist(),
        "move_ids": tensors.move_ids.tolist(),
        "item_ids": tensors.item_ids.tolist(),
        "ability_ids": tensors.ability_ids.tolist(),
        "status_ids": tensors.status_ids.tolist(),
        "pokemon_numeric": tensors.pokemon_numeric.tolist(),
        "pokemon_mask": tensors.pokemon_mask.tolist(),
        "weather_id": tensors.weather_id.item(),
        "field_numeric": tensors.field_numeric.tolist(),
        "history_kind": tensors.history_kind.tolist(),
        "history_move": tensors.history_move.tolist(),
        "history_species": tensors.history_species.tolist(),
        "history_form": tensors.history_form.tolist(),
        "history_actor": tensors.history_actor.tolist(),
        "history_target": tensors.history_target.tolist(),
        "history_reason": tensors.history_reason.tolist(),
        "history_numeric": tensors.history_numeric.tolist(),
        "history_mask": tensors.history_mask.tolist(),
        "history_length": tensors.history_length.item(),
        "action_mask": tensors.action_mask.tolist(),
    }


def _require_list(data: dict[str, object], key: str) -> list[Any]:
    value = data[key]
    if not isinstance(value, list):
        raise TypeError(f"observation[{key!r}] must be a list")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"observation[{key!r}] must be an int")
    return value


def _tensors_from_json(data: dict[str, object]) -> BattleTensors:
    return BattleTensors(
        base_species_ids=torch.tensor(
            _require_list(data, "base_species_ids"), dtype=torch.long
        ),
        species_ids=torch.tensor(
            _require_list(data, "species_ids"), dtype=torch.long
        ),
        form_ids=torch.tensor(
            _require_list(data, "form_ids"), dtype=torch.long
        ),
        move_ids=torch.tensor(
            _require_list(data, "move_ids"), dtype=torch.long
        ),
        item_ids=torch.tensor(
            _require_list(data, "item_ids"), dtype=torch.long
        ),
        ability_ids=torch.tensor(
            _require_list(data, "ability_ids"), dtype=torch.long
        ),
        status_ids=torch.tensor(
            _require_list(data, "status_ids"), dtype=torch.long
        ),
        pokemon_numeric=torch.tensor(
            _require_list(data, "pokemon_numeric"), dtype=torch.float32
        ),
        pokemon_mask=torch.tensor(
            _require_list(data, "pokemon_mask"), dtype=torch.bool
        ),
        weather_id=torch.tensor(
            _require_int(data, "weather_id"), dtype=torch.long
        ),
        field_numeric=torch.tensor(
            _require_list(data, "field_numeric"), dtype=torch.float32
        ),
        history_kind=torch.tensor(
            _require_list(data, "history_kind"), dtype=torch.long
        ),
        history_move=torch.tensor(
            _require_list(data, "history_move"), dtype=torch.long
        ),
        history_species=torch.tensor(
            _require_list(data, "history_species"), dtype=torch.long
        ),
        history_form=torch.tensor(
            _require_list(data, "history_form"), dtype=torch.long
        ),
        history_actor=torch.tensor(
            _require_list(data, "history_actor"), dtype=torch.long
        ),
        history_target=torch.tensor(
            _require_list(data, "history_target"), dtype=torch.long
        ),
        history_reason=torch.tensor(
            _require_list(data, "history_reason"), dtype=torch.long
        ),
        history_numeric=torch.tensor(
            _require_list(data, "history_numeric"), dtype=torch.float32
        ),
        history_mask=torch.tensor(
            _require_list(data, "history_mask"), dtype=torch.bool
        ),
        history_length=torch.tensor(
            _require_int(data, "history_length"), dtype=torch.long
        ),
        action_mask=torch.tensor(
            _require_list(data, "action_mask"), dtype=torch.bool
        ),
    )


def _open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")  # type:ignore
    return path.open(mode, encoding="utf-8")


async def _collect_worker_async(
    *,
    output_path: str,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    battle_offset: int,
    worker_index: int,
) -> WorkerResult:
    handler_1 = RecordingSimpleHeuristicsCombatHandler(side="a")
    handler_2 = RecordingSimpleHeuristicsCombatHandler(side="b")

    client_1 = Client(url, combat_handler=handler_1)
    client_2 = Client(url, combat_handler=handler_2)
    client_1.log_manager.disable()
    client_2.log_manager.disable()

    team_generator_1: BaseTeamGenerator | None = None
    team_generator_2: BaseTeamGenerator | None = None

    if "randombattle" not in fmt:
        team_generator_1 = SampleTeamGenerator(team_seed + worker_index * 2)
        team_generator_2 = SampleTeamGenerator(team_seed + worker_index * 2 + 1)

    pid_suffix = os.getpid() % 100_000
    name_1 = f"C{worker_index}A{pid_suffix}"
    name_2 = f"C{worker_index}B{pid_suffix}"

    completed = 0
    state_count = 0
    path = Path(output_path)

    try:
        await asyncio.gather(client_1.connect(), client_2.connect())
        await asyncio.gather(client_1.login(name_1), client_2.login(name_2))

        async with await anyio.open_file(path, "w", encoding="utf-8") as output:
            while completed < battles:
                battle_id = battle_offset + completed
                handler_1.start_battle(battle_id)
                handler_2.start_battle(battle_id)

                try:
                    await run_battle(
                        client_1,
                        client_2,
                        fmt=fmt,
                        team_generator_1=team_generator_1,
                        team_generator_2=team_generator_2,
                    )
                except BattleLifecycleError as error:
                    print(
                        f"Discarding failed comparison battle and retrying: {error!r}"
                    )
                    await asyncio.gather(
                        client_1.close(),
                        client_2.close(),
                        return_exceptions=True,
                    )
                    await asyncio.gather(
                        client_1.ensure_connected(), client_2.ensure_connected()
                    )
                    continue

                records = handler_1.take_records() + handler_2.take_records()
                records.sort(
                    key=lambda record: (
                        int(record["turn"]),  # type: ignore
                        str(record["side"]),
                        int(record["decision_index"]),  # type: ignore
                    )
                )

                for record in records:
                    await output.write(
                        json.dumps(record, separators=(",", ":"))
                    )
                    await output.write("\n")

                state_count += len(records)
                completed += 1

        return WorkerResult(
            path=str(path), battles=completed, states=state_count
        )
    finally:
        await asyncio.gather(
            client_1.close(), client_2.close(), return_exceptions=True
        )


def _collect_worker(
    output_path: str,
    url: str,
    fmt: str,
    team_seed: int,
    battles: int,
    battle_offset: int,
    worker_index: int,
) -> WorkerResult:
    return asyncio.run(
        _collect_worker_async(
            output_path=output_path,
            url=url,
            fmt=fmt,
            team_seed=team_seed,
            battles=battles,
            battle_offset=battle_offset,
            worker_index=worker_index,
        )
    )


def collect_battle_states(
    output_path: str | Path,
    *,
    battles: int = 200,
    url: str = DEFAULT_WEBSOCKET_URL,
    fmt: str = "gen1randombattle",
    workers: int = 4,
    threads: int = 1,
    team_seed: int = 42,
) -> dict[str, object]:
    """Run SimpleHeuristics mirrors and save their decision states to JSON."""
    if battles <= 0:
        raise ValueError("battles must be > 0")
    if workers <= 0:
        raise ValueError("workers must be > 0")
    if threads <= 0:
        raise ValueError("threads must be > 0")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    counts = split_battles(battles, workers)
    offsets: list[int] = []
    offset = 0
    for count in counts:
        offsets.append(offset)
        offset += count

    context = multiprocessing.get_context("spawn")

    with tempfile.TemporaryDirectory(prefix="ai-cif-compare-") as temp_dir:
        temp = Path(temp_dir)

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=worker_initializer,
            initargs=(threads,),
        ) as pool:
            futures = []
            for worker_index, count in enumerate(counts):
                if count <= 0:
                    continue

                worker_path = temp / f"worker_{worker_index:03d}.jsonl"
                futures.append(
                    pool.submit(
                        _collect_worker,
                        str(worker_path),
                        url,
                        fmt,
                        team_seed,
                        count,
                        offsets[worker_index],
                        worker_index,
                    )
                )

            results = [future.result() for future in futures]

        results.sort(key=lambda result: result.path)
        state_count = sum(result.states for result in results)
        completed_battles = sum(result.battles for result in results)

        metadata = {
            "schema": DATASET_SCHEMA,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "format": fmt,
            "battles": completed_battles,
            "state_count": state_count,
            "collector": "SimpleHeuristics_vs_SimpleHeuristics",
            "perspectives": ["a", "b"],
            "tensorizer": {
                "max_history": DEFAULT_MAX_HISTORY,
                "vocab_gen": DEFAULT_VOCAB_GEN,
            },
        }

        with _open_text(destination, "wt") as output:
            output.write("{")
            first_metadata = True
            for key, value in metadata.items():
                if not first_metadata:
                    output.write(",")
                output.write(json.dumps(key))
                output.write(":")
                output.write(json.dumps(value, separators=(",", ":")))
                first_metadata = False

            output.write(',"states":[')
            first_state = True

            for result in results:
                with Path(result.path).open(
                    "r", encoding="utf-8"
                ) as worker_file:
                    for line in worker_file:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if not first_state:
                            output.write(",")
                        output.write(stripped)
                        first_state = False

            output.write("]}")

    print(
        f"Saved {state_count:,} states from {completed_battles} battles "
        f"to {destination}"
    )
    return metadata


def _load_model(checkpoint_path: Path, device: torch.device) -> BattleModel:
    model, _ = create_battle_model(
        device=device,
        max_history=DEFAULT_MAX_HISTORY,
        vocab_gen=DEFAULT_VOCAB_GEN,
    )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a dict")

    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    if not isinstance(model_state, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} has invalid model state")

    model.load_state_dict(model_state)
    model.eval()
    return model


def _model_names(paths: list[Path]) -> list[str]:
    names = [f"{path.parent.name}/{path.stem}" for path in paths]
    if len(set(names)) != len(names):
        raise ValueError(
            "Model labels are not unique; put checkpoints in uniquely named "
            + "parent directories"
        )
    return names


def _load_state_bank(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[BattleTensors]]:
    with _open_text(path, "rt") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise TypeError("State-bank JSON root must be an object")
    if payload.get("schema") != DATASET_SCHEMA:
        raise ValueError(
            f"Unsupported state-bank schema: {payload.get('schema')!r}"
        )

    tensorizer_config = payload.get("tensorizer")
    if not isinstance(tensorizer_config, dict):
        raise TypeError("State bank is missing tensorizer metadata")

    if tensorizer_config.get("max_history") != DEFAULT_MAX_HISTORY:
        raise ValueError("State bank uses a different max_history")
    if tensorizer_config.get("vocab_gen") != DEFAULT_VOCAB_GEN:
        raise ValueError("State bank uses a different vocab_gen")

    raw_states = payload.get("states")
    if not isinstance(raw_states, list):
        raise TypeError("State bank must contain a states list")

    metadata_states: list[dict[str, object]] = []
    observations: list[BattleTensors] = []

    for index, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, dict):
            raise TypeError(f"states[{index}] must be an object")

        observation = raw_state.get("observation")
        if not isinstance(observation, dict):
            raise TypeError(f"states[{index}].observation must be an object")

        observations.append(_tensors_from_json(observation))
        metadata_states.append(raw_state)

    return payload, metadata_states, observations


def _infer_model(
    model: BattleModel,
    observations: list[BattleTensors],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probability_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    confidence_chunks: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            chunk = observations[start : start + batch_size]
            batch = collate_battles(chunk).to(device)
            logits, values = model(batch)
            probabilities = torch.softmax(logits, dim=-1)

            tiny = torch.finfo(probabilities.dtype).tiny
            entropy = -(
                probabilities * probabilities.clamp_min(tiny).log()
            ).sum(dim=-1)
            confidence = probabilities.max(dim=-1).values

            probability_chunks.append(probabilities.cpu())
            value_chunks.append(values.cpu())
            entropy_chunks.append(entropy.cpu())
            confidence_chunks.append(confidence.cpu())

    return (
        torch.cat(probability_chunks, dim=0),
        torch.cat(value_chunks, dim=0),
        torch.cat(entropy_chunks, dim=0),
        torch.cat(confidence_chunks, dim=0),
    )


def _js_divergence(
    probabilities_a: torch.Tensor, probabilities_b: torch.Tensor
) -> torch.Tensor:
    mixture = 0.5 * (probabilities_a + probabilities_b)
    tiny = torch.finfo(probabilities_a.dtype).tiny

    log_a = probabilities_a.clamp_min(tiny).log()
    log_b = probabilities_b.clamp_min(tiny).log()
    log_mixture = mixture.clamp_min(tiny).log()

    kl_a = (probabilities_a * (log_a - log_mixture)).sum(dim=-1)
    kl_b = (probabilities_b * (log_b - log_mixture)).sum(dim=-1)
    return 0.5 * (kl_a + kl_b)


def _action_label(index: int) -> str:
    if not 0 <= index < len(ACTION_LABELS):
        raise ValueError(f"Invalid action index: {index}")
    return ACTION_LABELS[index]


def compare_models(
    state_bank_path: str | Path,
    model_paths: list[str | Path],
    *,
    output_path: str | Path | None = None,
    batch_size: int = 512,
    device: str = "cpu",
    example_count: int = 10,
) -> dict[str, object]:
    """Compare neural checkpoints on exactly the same frozen observations."""
    if len(model_paths) < 2:
        raise ValueError("At least two model checkpoints are required")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if example_count < 0:
        raise ValueError("example_count must be >= 0")

    bank_path = Path(state_bank_path)
    paths = [Path(path) for path in model_paths]
    names = _model_names(paths)
    torch_device = torch.device(device)

    dataset, state_metadata, observations = _load_state_bank(bank_path)
    if not observations:
        raise ValueError("State bank contains no observations")

    heuristic_actions = torch.tensor(
        [int(state["heuristic_action"]) for state in state_metadata],  # type:ignore
        dtype=torch.long,
    )

    probabilities_by_model: dict[str, torch.Tensor] = {}
    values_by_model: dict[str, torch.Tensor] = {}
    actions_by_model: dict[str, torch.Tensor] = {}
    model_summaries: dict[str, object] = {}

    for name, path in zip(names, paths, strict=True):
        print(f"Evaluating {name} on {len(observations):,} states")
        model = _load_model(path, torch_device)
        probabilities, values, entropy, confidence = _infer_model(
            model, observations, device=torch_device, batch_size=batch_size
        )
        actions = probabilities.argmax(dim=-1)

        counts = torch.bincount(actions, minlength=len(ACTION_LABELS))
        action_counts = {
            label: int(counts[index].item())
            for index, label in enumerate(ACTION_LABELS)
        }
        action_rates = {
            label: action_counts[label] / len(observations)
            for label in ACTION_LABELS
        }

        probabilities_by_model[name] = probabilities
        values_by_model[name] = values
        actions_by_model[name] = actions

        model_summaries[name] = {
            "checkpoint": str(path),
            "mean_entropy": float(entropy.mean().item()),
            "mean_top1_probability": float(confidence.mean().item()),
            "mean_value": float(values.mean().item()),
            "value_std": float(values.std(unbiased=False).item()),
            "switch_rate": float((actions >= 4).float().mean().item()),
            "heuristic_disagreement_rate": float(
                (actions != heuristic_actions).float().mean().item()
            ),
            "action_counts": action_counts,
            "action_rates": action_rates,
        }

        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    pair_summaries: list[dict[str, object]] = []

    for name_a, name_b in combinations(names, 2):
        probabilities_a = probabilities_by_model[name_a]
        probabilities_b = probabilities_by_model[name_b]
        actions_a = actions_by_model[name_a]
        actions_b = actions_by_model[name_b]
        values_a = values_by_model[name_a]
        values_b = values_by_model[name_b]

        divergence = _js_divergence(probabilities_a, probabilities_b)
        disagreement = actions_a != actions_b

        examples: list[dict[str, object]] = []
        count = min(example_count, len(observations))
        if count > 0:
            top_indices = torch.topk(divergence, k=count).indices.tolist()
            for state_index in top_indices:
                state = state_metadata[state_index]
                examples.append(
                    {
                        "state_index": state_index,
                        "battle_id": int(state["battle_id"]),
                        "side": str(state["side"]),
                        "turn": int(state["turn"]),
                        "js_divergence": float(divergence[state_index].item()),
                        "model_a_action": _action_label(
                            int(actions_a[state_index].item())
                        ),
                        "model_b_action": _action_label(
                            int(actions_b[state_index].item())
                        ),
                        "heuristic_action": _action_label(
                            int(heuristic_actions[state_index].item())
                        ),
                    }
                )

        pair_summaries.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "top1_disagreement_rate": float(
                    disagreement.float().mean().item()
                ),
                "mean_js_divergence_nats": float(divergence.mean().item()),
                "p95_js_divergence_nats": float(
                    torch.quantile(divergence, 0.95).item()
                ),
                "mean_abs_value_difference": float(
                    (values_a - values_b).abs().mean().item()
                ),
                "examples": examples,
            }
        )

    report: dict[str, object] = {
        "schema": "ai-cif-policy-comparison-v1",
        "state_bank": str(bank_path),
        "format": dataset.get("format"),
        "battle_count": dataset.get("battles"),
        "state_count": len(observations),
        "models": model_summaries,
        "pairs": pair_summaries,
    }

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _open_text(destination, "wt") as output:
            json.dump(report, output, indent=2)
        print(f"Saved comparison report to {destination}")

    print()
    print("Pairwise comparison")
    for pair in report["pairs"]:  # type: ignore
        if not isinstance(pair, dict):
            raise TypeError("Invalid pair result")
        print(
            f"{pair['model_a']} vs {pair['model_b']}: "
            f"disagreement={float(pair['top1_disagreement_rate']):.1%}, "
            f"JS={float(pair['mean_js_divergence_nats']):.4f}"
        )
    return report


if __name__ == "__main__":
    # collect_battle_states(
    #     "data/ref_battle_states.json.gz",
    #     battles=200,
    #     url=DEFAULT_WEBSOCKET_URL,
    #     fmt="gen1randombattle",
    #     workers=1,
    #     threads=1,
    # )

    report = compare_models(
        "data/ref_battle_states.json.gz",
        [
            "data/models/blue/1.pt",
            "data/models/blue/2.pt",
            "data/models/blue/3.pt",
            "data/models/blue/4.pt",
            "data/models/blue/5.pt",
            "data/models/blue/6.pt",
            "data/models/blue/7.pt",
            "data/models/blue/8.pt",
            "data/models/blue/9.pt",
            "data/models/blue/10.pt",
        ],
        output_path="experiments/comparison_seed.json",
        device="cpu",
    )
