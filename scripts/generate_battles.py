import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import RandomMoveCombatHandler
from showdown_sdk.features import battle_to_features
from showdown_sdk.models.sdk import BattleState, SampleTeamGenerator, TeamSet
from showdown_sdk.vectorizer import VectorizerConfig, vectorize_battle_features
from tqdm import tqdm

from ai_cif.vectorization.simplified_history import (
    SimplifiedHistoryEntry,
    build_simplified_history,
    tail_simplified_history,
)

DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/showdown/websocket"


@dataclass
class DecisionRecord:
    """One model input captured immediately before choosing an action."""

    player: str
    turn: int

    global_decision_index: int
    battle_decision_index: int

    force_switch: bool

    raw_history_length: int
    tactical_history_length: int
    recent_tactical_length: int
    oldest_recent_tactical_turn: int | None
    tactical_turn_span: int

    vector: list[int | float]
    ranked_actions: list[tuple[str, int]]

    def to_json(self, *, battle: int, fmt: str) -> dict[str, Any]:
        return {
            "battle": battle,
            "format": fmt,
            "player": self.player,
            "turn": self.turn,
            "global_decision_index": self.global_decision_index,
            "battle_decision_index": self.battle_decision_index,
            "force_switch": self.force_switch,
            "raw_history_length": self.raw_history_length,
            "tactical_history_length": self.tactical_history_length,
            "recent_tactical_length": self.recent_tactical_length,
            "oldest_recent_tactical_turn": self.oldest_recent_tactical_turn,
            "tactical_turn_span": self.tactical_turn_span,
            "vector_length": len(self.vector),
            "vector": self.vector,
            "ranked_actions": [list(action) for action in self.ranked_actions],
        }


class RecordingRandomHandler(RandomMoveCombatHandler):
    """Random policy that records the state seen at each decision point."""

    def __init__(
        self,
        *,
        player: str,
        history_length: int,
        tactical_history_length: int,
        switch_chance: float = 0.1,
    ) -> None:
        super().__init__(switch_chance=switch_chance)

        self.player = player
        self.vectorizer_config = VectorizerConfig(history_length=history_length)
        self.tactical_history_length = tactical_history_length

        self.records: list[DecisionRecord] = []

        self._global_decision_index = 0
        self._battle_decision_index = 0

    def start_battle(self) -> int:
        """Reset battle-local numbering and return the current record mark."""

        self._battle_decision_index = 0
        return len(self.records)

    def records_since(self, mark: int) -> list[DecisionRecord]:
        return self.records[mark:]

    def select_top_actions(
        self,
        battle_state: BattleState,
    ) -> list[tuple[str, int]]:
        features = battle_to_features(battle_state)

        vector = vectorize_battle_features(
            features,
            config=self.vectorizer_config,
        )

        tactical = build_simplified_history(features.history)
        recent_tactical = tail_simplified_history(
            tactical,
            max_entries=self.tactical_history_length,
        )

        oldest_recent_turn = _oldest_turn(recent_tactical)

        if oldest_recent_turn is None:
            tactical_turn_span = 0
        else:
            tactical_turn_span = max(
                0,
                features.field.turn - oldest_recent_turn + 1,
            )

        ranked_actions = super().select_top_actions(battle_state)

        self._global_decision_index += 1
        self._battle_decision_index += 1

        self.records.append(
            DecisionRecord(
                player=self.player,
                turn=features.field.turn,
                global_decision_index=self._global_decision_index,
                battle_decision_index=self._battle_decision_index,
                force_switch=features.force_switch,
                raw_history_length=len(features.history),
                tactical_history_length=len(tactical),
                recent_tactical_length=len(recent_tactical),
                oldest_recent_tactical_turn=oldest_recent_turn,
                tactical_turn_span=tactical_turn_span,
                vector=vector,
                ranked_actions=list(ranked_actions),
            )
        )

        return ranked_actions


def _oldest_turn(
    history: tuple[SimplifiedHistoryEntry, ...],
) -> int | None:
    for entry in history:
        if entry.turn is not None:
            return entry.turn

    return None


async def run_battle(
    client_1: Client,
    client_2: Client,
    *,
    fmt: str,
    team_generator: SampleTeamGenerator | None,
) -> None:
    await asyncio.gather(
        client_1.ensure_connected(),
        client_2.ensure_connected(),
    )

    if client_1.username is None or client_2.username is None:
        raise RuntimeError("Both clients must be logged in before starting")

    team_1: TeamSet | None = None
    team_2: TeamSet | None = None

    if team_generator is not None:
        team_1 = await team_generator.generate(
            fmt,
            lambda team: client_1.validate_team(fmt, team),
        )
        team_2 = await team_generator.generate(
            fmt,
            lambda team: client_2.validate_team(fmt, team),
        )

    await client_1.challenge(
        client_2.username,
        fmt,
        timeout=60,
        team=team_1,
    )
    await client_2.accept_challenge(
        client_1.username,
        team=team_2,
    )

    await asyncio.gather(
        client_1.battle_manager.room_ready.wait(),
        client_2.battle_manager.room_ready.wait(),
    )

    await asyncio.gather(
        client_1.wait_for_battle_end(timeout=300),
        client_2.wait_for_battle_end(timeout=300),
    )


def append_records(
    output: Path,
    *,
    battle: int,
    fmt: str,
    records: list[DecisionRecord],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("a", encoding="utf-8") as file:
        for record in records:
            json.dump(
                record.to_json(battle=battle, fmt=fmt),
                file,
                separators=(",", ":"),
            )
            file.write("\n")


def _print_summary(records: list[DecisionRecord]) -> None:
    if not records:
        return

    mature = [
        record
        for record in records
        if record.recent_tactical_length > 0
        and record.turn >= 10
    ]

    sample = mature if mature else records

    raw_per_turn = [
        record.raw_history_length / max(record.turn, 1)
        for record in sample
    ]

    tactical_per_turn = [
        record.tactical_history_length / max(record.turn, 1)
        for record in sample
    ]

    spans = [
        record.tactical_turn_span
        for record in sample
        if record.tactical_turn_span > 0
    ]

    compression = [
        record.tactical_history_length / record.raw_history_length
        for record in sample
        if record.raw_history_length > 0
    ]

    print()
    print("History summary")
    print("---------------")
    print(
        f"Decision points: {len(records)}"
    )
    print(
        "Average raw events / current turn: "
        f"{sum(raw_per_turn) / len(raw_per_turn):.2f}"
    )
    print(
        "Average tactical entries / current turn: "
        f"{sum(tactical_per_turn) / len(tactical_per_turn):.2f}"
    )

    if compression:
        average_compression = sum(compression) / len(compression)
        print(
            "Average tactical/raw history ratio: "
            f"{average_compression:.3f}"
        )

    if spans:
        ordered = sorted(spans)
        print(
            "Last tactical window turn coverage: "
            f"avg={sum(spans) / len(spans):.2f}, "
            f"p50={_percentile(ordered, 0.50)}, "
            f"p90={_percentile(ordered, 0.90)}, "
            f"max={ordered[-1]}"
        )


def _percentile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("values must not be empty")

    index = round((len(values) - 1) * q)
    return values[index]


async def generate(
    *,
    websocket_url: str,
    fmt: str,
    battles: int,
    history_length: int,
    tactical_history_length: int,
    output: Path,
    switch_chance: float,
    team_seed: int,
) -> None:
    handler_1 = RecordingRandomHandler(
        player="BOT1",
        history_length=history_length,
        tactical_history_length=tactical_history_length,
        switch_chance=switch_chance,
    )
    handler_2 = RecordingRandomHandler(
        player="BOT2",
        history_length=history_length,
        tactical_history_length=tactical_history_length,
        switch_chance=switch_chance,
    )

    client_1 = Client(
        websocket_url,
        combat_handler=handler_1,
    )
    client_2 = Client(
        websocket_url,
        combat_handler=handler_2,
    )

    team_generator: SampleTeamGenerator | None = None
    if "randombattle" not in fmt:
        team_generator = SampleTeamGenerator(team_seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    all_records: list[DecisionRecord] = []

    try:
        await asyncio.gather(
            client_1.connect(),
            client_2.connect(),
        )
        await asyncio.gather(
            client_1.login("BOT1"),
            client_2.login("BOT2"),
        )

        print("BOT1 connected")
        print("BOT2 connected")

        first_vector_length: int | None = None

        progress = tqdm(
            range(1, battles + 1),
            desc=fmt,
            unit="battle",
            dynamic_ncols=True,
        )

        for battle_number in progress:
            mark_1 = handler_1.start_battle()
            mark_2 = handler_2.start_battle()

            await run_battle(
                client_1,
                client_2,
                fmt=fmt,
                team_generator=team_generator,
            )

            records = (
                handler_1.records_since(mark_1)
                + handler_2.records_since(mark_2)
            )

            append_records(
                output,
                battle=battle_number,
                fmt=fmt,
                records=records,
            )

            all_records.extend(records)

            if records and first_vector_length is None:
                first_vector_length = len(records[0].vector)

            latest_spans = [
                record.tactical_turn_span
                for record in records
                if record.tactical_turn_span > 0
            ]

            average_span = (
                sum(latest_spans) / len(latest_spans)
                if latest_spans
                else 0.0
            )

            progress.set_postfix(
                decisions=len(all_records),
                vector_dim=first_vector_length or "?",
                tactical_span=f"{average_span:.1f}t",
            )

        print()
        print(
            f"Wrote {len(all_records)} decision records to {output}"
        )

        if first_vector_length is not None:
            print(f"Vector dimension: {first_vector_length}")

        _print_summary(all_records)

    finally:
        await asyncio.gather(
            client_1.close(),
            client_2.close(),
            return_exceptions=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=DEFAULT_WEBSOCKET_URL,
        help="Pokémon Showdown websocket URL",
    )
    parser.add_argument(
        "--format",
        default="gen1randombattle",
        dest="fmt",
        help="Showdown battle format",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=100,
        help="Number of battles to generate",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=32,
        help="Raw semantic events kept by the SDK flat vectorizer",
    )
    parser.add_argument(
        "--tactical-history-length",
        type=int,
        default=32,
        help="Recent tactical entries used when measuring turn coverage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/turn_vectors.jsonl"),
        help="JSONL output path",
    )
    parser.add_argument(
        "--switch-chance",
        type=float,
        default=0.1,
        help="Random bot switch probability",
    )
    parser.add_argument(
        "--team-seed",
        type=int,
        default=42,
        help="Seed used by SampleTeamGenerator for non-random formats",
    )

    args = parser.parse_args()

    if args.battles <= 0:
        parser.error("--battles must be > 0")

    if args.history_length < 0:
        parser.error("--history-length must be >= 0")

    if args.tactical_history_length < 0:
        parser.error("--tactical-history-length must be >= 0")

    if not 0.0 <= args.switch_chance <= 1.0:
        parser.error("--switch-chance must be between 0 and 1")

    return args


async def async_main() -> None:
    args = parse_args()

    await generate(
        websocket_url=args.url,
        fmt=args.fmt,
        battles=args.battles,
        history_length=args.history_length,
        tactical_history_length=args.tactical_history_length,
        output=args.output,
        switch_chance=args.switch_chance,
        team_seed=args.team_seed,
    )


if __name__ == "__main__":
    asyncio.run(async_main())
