from dataclasses import dataclass, field
from typing import Literal

from showdown_sdk.features import EventFeatures, PokemonRefFeatures

type HpChangeKind = Literal["damage", "heal", "sethp"]


@dataclass(frozen=True)
class HpChange:
    """One immediate HP consequence attached to a move action.

    hp_delta is signed and expressed as a fraction of max HP:

        -0.35 -> lost 35% HP
        +0.20 -> recovered 20% HP
    """

    kind: HpChangeKind
    target: PokemonRefFeatures | None

    hp_before: float | None
    hp_after: float | None
    hp_delta: float | None

    source_type: str | None = None
    crit: bool | None = None
    effectiveness: float | None = None


@dataclass(frozen=True)
class MoveTacticalEntry:
    """One move attempt plus the immediate HP effects caused by that action."""

    kind: Literal["move"] = field(default="move", init=False)

    turn: int | None = None
    action_id: int | None = None

    actor: PokemonRefFeatures | None = None
    target: PokemonRefFeatures | None = None
    move: str | None = None

    success: bool | None = None
    does_hit: bool | None = None
    failure_reason: str | None = None
    hit_count: int | None = None

    hp_changes: tuple[HpChange, ...] = ()


@dataclass(frozen=True)
class SwitchTacticalEntry:
    """A Pokémon entering the field."""

    kind: Literal["switch"] = field(default="switch", init=False)

    turn: int | None = None
    pokemon: PokemonRefFeatures | None = None
    species: str | None = None
    hp_ratio: float | None = None
    baton_pass: bool = False


@dataclass(frozen=True)
class CantTacticalEntry:
    """A Pokémon was unable to take its intended action."""

    kind: Literal["cant"] = field(default="cant", init=False)

    turn: int | None = None
    pokemon: PokemonRefFeatures | None = None
    move: str | None = None
    reason: str | None = None


type SimplifiedHistoryEntry = (
    MoveTacticalEntry | SwitchTacticalEntry | CantTacticalEntry
)


@dataclass
class _MoveBuilder:
    turn: int | None
    action_id: int | None
    actor: PokemonRefFeatures | None
    target: PokemonRefFeatures | None
    move: str | None
    success: bool | None
    does_hit: bool | None
    failure_reason: str | None
    hit_count: int | None
    hp_changes: list[HpChange] = field(default_factory=list)

    def freeze(self) -> MoveTacticalEntry:
        return MoveTacticalEntry(
            turn=self.turn,
            action_id=self.action_id,
            actor=self.actor,
            target=self.target,
            move=self.move,
            success=self.success,
            does_hit=self.does_hit,
            failure_reason=self.failure_reason,
            hit_count=self.hit_count,
            hp_changes=tuple(self.hp_changes),
        )


type _InternalEntry = _MoveBuilder | SwitchTacticalEntry | CantTacticalEntry


def build_simplified_history(
    history: tuple[EventFeatures, ...],
) -> tuple[SimplifiedHistoryEntry, ...]:
    """Compress full semantic history into tactical history.

    Causal rule
    -----------
    HP changes are attached to a move only when:

    1. the HP event carries an action_id,
    2. a move with the same action_id exists, and
    3. both events happened on the same turn.

    The same-turn check matters because delayed effects can retain information
    about the move that originally created them.  Folding later residual damage
    back into an old move would destroy temporal ordering.

    Everything else is ignored unless it is needed to maintain HP observations.
    In particular, item/ability/weather/status/stat events are expected to be
    represented by the current battle state.
    """

    entries: list[_InternalEntry] = []
    moves_by_action_id: dict[int, _MoveBuilder] = {}
    hp_by_pokemon: dict[tuple[str, int | str], float] = {}

    for event in history:
        event_type = event.event_type

        if event_type == "move":
            builder = _MoveBuilder(
                turn=event.turn,
                action_id=event.action_id,
                actor=event.source,
                target=event.target,
                move=event.move,
                success=event.success,
                does_hit=event.does_hit,
                failure_reason=_payload_str(event, "failure_reason"),
                hit_count=event.hit_count,
            )
            entries.append(builder)

            if event.action_id is not None:
                moves_by_action_id[event.action_id] = builder

            continue

        if event_type == "pokemonswitch":
            if event.source is not None and event.hp_ratio is not None:
                hp_by_pokemon[_pokemon_key(event.source)] = event.hp_ratio

            entries.append(
                SwitchTacticalEntry(
                    turn=event.turn,
                    pokemon=event.source,
                    species=event.species,
                    hp_ratio=event.hp_ratio,
                    baton_pass=_payload_bool(event, "baton_pass"),
                )
            )
            continue

        if event_type == "cant":
            entries.append(
                CantTacticalEntry(
                    turn=event.turn,
                    pokemon=event.source,
                    move=event.move,
                    reason=_payload_str(event, "reason"),
                )
            )
            continue

        if event_type in {"damage", "heal", "sethp"}:
            _consume_hp_event(
                event=event,
                kind=event_type,
                hp_by_pokemon=hp_by_pokemon,
                moves_by_action_id=moves_by_action_id,
            )

    frozen: list[SimplifiedHistoryEntry] = []

    for entry in entries:
        if isinstance(entry, _MoveBuilder):
            frozen.append(entry.freeze())
        else:
            frozen.append(entry)

    return tuple(frozen)


def tail_simplified_history(
    history: tuple[SimplifiedHistoryEntry, ...], max_entries: int = 32
) -> tuple[SimplifiedHistoryEntry, ...]:
    """Return the most recent tactical entries without changing compression.

    Keeping truncation separate lets model experiments change sequence length
    without changing the meaning of the tactical-history schema.
    """

    if max_entries < 0:
        raise ValueError("max_entries must be >= 0")

    if max_entries == 0:
        return ()

    return history[-max_entries:]


def build_recent_simplified_history(
    history: tuple[EventFeatures, ...], max_entries: int = 32
) -> tuple[SimplifiedHistoryEntry, ...]:
    """Convenience wrapper for the common build-then-tail operation."""

    return tail_simplified_history(
        build_simplified_history(history), max_entries=max_entries
    )


def _consume_hp_event(
    *,
    event: EventFeatures,
    kind: str,
    hp_by_pokemon: dict[tuple[str, int | str], float],
    moves_by_action_id: dict[int, _MoveBuilder],
) -> None:
    target = event.target

    hp_before: float | None = None
    hp_after = event.hp_ratio

    if target is not None:
        key = _pokemon_key(target)
        hp_before = hp_by_pokemon.get(key)

        if hp_after is not None:
            hp_by_pokemon[key] = hp_after

    hp_delta: float | None = None
    if hp_before is not None and hp_after is not None:
        hp_delta = hp_after - hp_before

    action_id = event.action_id
    if action_id is None:
        return

    move = moves_by_action_id.get(action_id)
    if move is None:
        return

    if move.turn != event.turn:
        return

    if kind == "damage":
        hp_kind: HpChangeKind = "damage"
    elif kind == "heal":
        hp_kind = "heal"
    elif kind == "sethp":
        hp_kind = "sethp"
    else:
        raise ValueError(f"Unsupported HP event type: {kind!r}")

    move.hp_changes.append(
        HpChange(
            kind=hp_kind,
            target=target,
            hp_before=hp_before,
            hp_after=hp_after,
            hp_delta=hp_delta,
            source_type=event.effect_source_type,
            crit=event.crit if hp_kind == "damage" else None,
            effectiveness=(event.effectiveness if hp_kind == "damage" else None),
        )
    )


def _pokemon_key(ref: PokemonRefFeatures) -> tuple[str, int | str]:
    """Build the most stable available identity key without guessing."""

    if ref.slot is not None:
        return (ref.side, ref.slot)

    if ref.pokemon_id is not None:
        return (ref.side, ref.pokemon_id)

    if ref.species is not None:
        # Last-resort fallback.  This can collide when a team contains duplicate
        # species, but it is still preferable to fabricating an identity.
        return (ref.side, f"species:{ref.species}")

    return (ref.side, "unknown")


def _payload_str(event: EventFeatures, key: str) -> str | None:
    value = event.payload.get(key)
    return value if isinstance(value, str) else None


def _payload_bool(event: EventFeatures, key: str) -> bool:
    value = event.payload.get(key)
    return value if isinstance(value, bool) else False


__all__ = [
    "CantTacticalEntry",
    "HpChange",
    "MoveTacticalEntry",
    "SimplifiedHistoryEntry",
    "SwitchTacticalEntry",
    "build_recent_simplified_history",
    "build_simplified_history",
    "tail_simplified_history",
]
