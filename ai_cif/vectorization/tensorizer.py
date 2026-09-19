from dataclasses import dataclass
from typing import Final

import torch
from showdown_sdk.features import (
    BattleFeatures,
    Knowledge,
    PokemonRefFeatures,
    SideConditionFeatures,
)
from showdown_sdk.vectorizer.vocabulary import (
    ability_id,
    id_limits,
    item_id,
    move_id,
    pokemon_id,
)
from torch import Tensor

from ai_cif.vectorization.simplified_history import (
    CantTacticalEntry,
    MoveTacticalEntry,
    SimplifiedHistoryEntry,
    SwitchTacticalEntry,
    build_recent_simplified_history,
)

type DeviceLike = str | torch.device


## Public schema constants

TEAM_SIZE: Final = 6
POKEMON_SLOTS: Final = TEAM_SIZE * 2
MOVES_PER_POKEMON: Final = 4
ACTION_COUNT: Final = 10
DEFAULT_MAX_HISTORY: Final = 32
DEFAULT_VOCAB_GEN: Final = 4

# Categorical IDs always reserve:
#
#   0 = NONE / padding / known absence
#   1 = UNKNOWN
#   2+ = concrete values
#
NONE_ID: Final = 0
UNKNOWN_ID: Final = 1
REAL_ID_OFFSET: Final = 1  # raw SDK IDs start at 1; concrete token = raw + 1

HISTORY_KIND_PAD: Final = 0
HISTORY_KIND_MOVE: Final = 1
HISTORY_KIND_SWITCH: Final = 2
HISTORY_KIND_CANT: Final = 3
HISTORY_KIND_VOCAB_SIZE: Final = 4

# Stable Pokémon references used inside simplified history.
#
#   0      = none / padding
#   1..6   = own slots 0..5
#   7..12  = opponent slots 0..5
#   13     = own Pokémon, slot unresolved
#   14     = opponent Pokémon, slot unresolved
HISTORY_REF_NONE: Final = 0
HISTORY_REF_SELF_UNKNOWN: Final = 13
HISTORY_REF_OPPONENT_UNKNOWN: Final = 14
HISTORY_REF_VOCAB_SIZE: Final = 15

STATUS_NAMES: Final = ("brn", "frz", "par", "psn", "slp", "tox", "fnt")
STATUS_IDS: Final = {name: i + 2 for i, name in enumerate(STATUS_NAMES)}
STATUS_VOCAB_SIZE: Final = 2 + len(STATUS_NAMES)

WEATHER_NAMES: Final = (
    "raindance",
    "sunnyday",
    "sandstorm",
    "hail",
    "snow",
    "desolateland",
    "primordialsea",
    "deltastream",
)
WEATHER_IDS: Final = {name: i + 2 for i, name in enumerate(WEATHER_NAMES)}
WEATHER_VOCAB_SIZE: Final = 2 + len(WEATHER_NAMES)

# Conditions relevant to the intended Gen 1-4 scope and already represented by
# the SDK feature layer / baseline vectorizer.
SIDE_CONDITION_NAMES: Final = (
    "stealthrock",
    "spikes",
    "toxicspikes",
    "reflect",
    "lightscreen",
    "safeguard",
    "mist",
    "tailwind",
)
SIDE_CONDITION_INDEX: Final = {
    name: i for i, name in enumerate(SIDE_CONDITION_NAMES)
}

CANT_REASON_NAMES: Final = (
    "slp",
    "frz",
    "par",
    "flinch",
    "recharge",
    "trapped",
    "partiallytrapped",
    "disable",
    "taunt",
    "encore",
    "confusion",
)
CANT_REASON_IDS: Final = {
    name: i + 2 for i, name in enumerate(CANT_REASON_NAMES)
}
CANT_REASON_VOCAB_SIZE: Final = 2 + len(CANT_REASON_NAMES)


# Pokémon numeric schema:
#
#  0 present/revealed
#  1 active
#  2 fainted
#  3 HP ratio
#  4 HP known
#  5 level / 100
#  6 level known
#  7 transformed
#  8 has type override
#  9 atk stage / 6
# 10 def stage / 6
# 11 spa stage / 6
# 12 spd stage / 6
# 13 spe stage / 6
# 14 accuracy stage / 6
# 15 evasion stage / 6
# 16 perish count / 4
# 17 perish count known
# 18 must recharge
# 19 attack stat / 500
# 20 defense stat / 500
# 21 special attack stat / 500
# 22 special defense stat / 500
# 23 speed stat / 500
# 24 own exact stats known
POKEMON_NUMERIC_DIM: Final = 25

# Field numeric schema:
#
#  0 generation / 4
#  1 turn / 100
#  2 Gen 1 desync flag
#  3 forced switch flag
#  4 own active slot / 5
#  5 own active slot known
#  6 opponent active slot / 5
#  7 opponent active slot known
#  8..15  own side conditions
# 16..23  opponent side conditions
# 24..31  field conditions
FIELD_NUMERIC_DIM: Final = 8 + 3 * len(SIDE_CONDITION_NAMES)

# Simplified history numeric schema:
#
#  0 age in turns / max_history
#  1 success
#  2 success known
#  3 does_hit
#  4 does_hit known
#  5 hit_count / 5
#  6 hit_count known
#  7 direct target HP delta
#  8 direct target HP delta known
#  9 actor HP delta (recoil/drain/recovery)
# 10 actor HP delta known
# 11 crit
# 12 crit known
# 13 effectiveness / 4
# 14 effectiveness known
# 15 switch HP ratio
# 16 switch HP known
# 17 baton pass
HISTORY_NUMERIC_DIM: Final = 18


## Tensor containers


@dataclass(frozen=True)
class BattleTensors:
    """One unbatched model observation.

    Shapes:
        base_species_ids:    [12]
        species_ids:         [12]
        form_ids:            [12]
        move_ids:            [12, 4]
        item_ids:            [12]
        ability_ids:         [12]
        status_ids:          [12]
        pokemon_numeric:     [12, POKEMON_NUMERIC_DIM]
        pokemon_mask:        [12]

        weather_id:          []
        field_numeric:       [FIELD_NUMERIC_DIM]

        history_kind:        [H]
        history_move:        [H]
        history_species:     [H]
        history_form:        [H]
        history_actor:       [H]
        history_target:      [H]
        history_reason:      [H]
        history_numeric:     [H, HISTORY_NUMERIC_DIM]
        history_mask:        [H]
        history_length:      []

        action_mask:         [10]
    """

    base_species_ids: Tensor
    species_ids: Tensor
    form_ids: Tensor
    move_ids: Tensor
    item_ids: Tensor
    ability_ids: Tensor
    status_ids: Tensor
    pokemon_numeric: Tensor
    pokemon_mask: Tensor

    weather_id: Tensor
    field_numeric: Tensor

    history_kind: Tensor
    history_move: Tensor
    history_species: Tensor
    history_form: Tensor
    history_actor: Tensor
    history_target: Tensor
    history_reason: Tensor
    history_numeric: Tensor
    history_mask: Tensor
    history_length: Tensor

    action_mask: Tensor

    def batched(self) -> BattleBatch:
        """Add a batch dimension of size one."""

        return BattleBatch(
            base_species_ids=self.base_species_ids.unsqueeze(0),
            species_ids=self.species_ids.unsqueeze(0),
            form_ids=self.form_ids.unsqueeze(0),
            move_ids=self.move_ids.unsqueeze(0),
            item_ids=self.item_ids.unsqueeze(0),
            ability_ids=self.ability_ids.unsqueeze(0),
            status_ids=self.status_ids.unsqueeze(0),
            pokemon_numeric=self.pokemon_numeric.unsqueeze(0),
            pokemon_mask=self.pokemon_mask.unsqueeze(0),
            weather_id=self.weather_id.unsqueeze(0),
            field_numeric=self.field_numeric.unsqueeze(0),
            history_kind=self.history_kind.unsqueeze(0),
            history_move=self.history_move.unsqueeze(0),
            history_species=self.history_species.unsqueeze(0),
            history_form=self.history_form.unsqueeze(0),
            history_actor=self.history_actor.unsqueeze(0),
            history_target=self.history_target.unsqueeze(0),
            history_reason=self.history_reason.unsqueeze(0),
            history_numeric=self.history_numeric.unsqueeze(0),
            history_mask=self.history_mask.unsqueeze(0),
            history_length=self.history_length.unsqueeze(0),
            action_mask=self.action_mask.unsqueeze(0),
        )

    def to(self, device: DeviceLike) -> BattleTensors:
        return BattleTensors(
            base_species_ids=self.base_species_ids.to(device),
            species_ids=self.species_ids.to(device),
            form_ids=self.form_ids.to(device),
            move_ids=self.move_ids.to(device),
            item_ids=self.item_ids.to(device),
            ability_ids=self.ability_ids.to(device),
            status_ids=self.status_ids.to(device),
            pokemon_numeric=self.pokemon_numeric.to(device),
            pokemon_mask=self.pokemon_mask.to(device),
            weather_id=self.weather_id.to(device),
            field_numeric=self.field_numeric.to(device),
            history_kind=self.history_kind.to(device),
            history_move=self.history_move.to(device),
            history_species=self.history_species.to(device),
            history_form=self.history_form.to(device),
            history_actor=self.history_actor.to(device),
            history_target=self.history_target.to(device),
            history_reason=self.history_reason.to(device),
            history_numeric=self.history_numeric.to(device),
            history_mask=self.history_mask.to(device),
            history_length=self.history_length.to(device),
            action_mask=self.action_mask.to(device),
        )


@dataclass(frozen=True)
class BattleBatch:
    """Batched version of BattleTensors.

    Every field has a leading batch dimension B.
    """

    base_species_ids: Tensor
    species_ids: Tensor
    form_ids: Tensor
    move_ids: Tensor
    item_ids: Tensor
    ability_ids: Tensor
    status_ids: Tensor
    pokemon_numeric: Tensor
    pokemon_mask: Tensor

    weather_id: Tensor
    field_numeric: Tensor

    history_kind: Tensor
    history_move: Tensor
    history_species: Tensor
    history_form: Tensor
    history_actor: Tensor
    history_target: Tensor
    history_reason: Tensor
    history_numeric: Tensor
    history_mask: Tensor
    history_length: Tensor

    action_mask: Tensor

    def to(self, device: DeviceLike) -> BattleBatch:
        return BattleBatch(
            base_species_ids=self.base_species_ids.to(device),
            species_ids=self.species_ids.to(device),
            form_ids=self.form_ids.to(device),
            move_ids=self.move_ids.to(device),
            item_ids=self.item_ids.to(device),
            ability_ids=self.ability_ids.to(device),
            status_ids=self.status_ids.to(device),
            pokemon_numeric=self.pokemon_numeric.to(device),
            pokemon_mask=self.pokemon_mask.to(device),
            weather_id=self.weather_id.to(device),
            field_numeric=self.field_numeric.to(device),
            history_kind=self.history_kind.to(device),
            history_move=self.history_move.to(device),
            history_species=self.history_species.to(device),
            history_form=self.history_form.to(device),
            history_actor=self.history_actor.to(device),
            history_target=self.history_target.to(device),
            history_reason=self.history_reason.to(device),
            history_numeric=self.history_numeric.to(device),
            history_mask=self.history_mask.to(device),
            history_length=self.history_length.to(device),
            action_mask=self.action_mask.to(device),
        )

    def model_kwargs(self) -> dict[str, Tensor]:
        """Keyword arguments for a model whose forward() follows this schema."""

        return {
            "base_species_ids": self.base_species_ids,
            "species_ids": self.species_ids,
            "form_ids": self.form_ids,
            "move_ids": self.move_ids,
            "item_ids": self.item_ids,
            "ability_ids": self.ability_ids,
            "status_ids": self.status_ids,
            "pokemon_numeric": self.pokemon_numeric,
            "pokemon_mask": self.pokemon_mask,
            "weather_id": self.weather_id,
            "field_numeric": self.field_numeric,
            "history_kind": self.history_kind,
            "history_move": self.history_move,
            "history_species": self.history_species,
            "history_form": self.history_form,
            "history_actor": self.history_actor,
            "history_target": self.history_target,
            "history_reason": self.history_reason,
            "history_numeric": self.history_numeric,
            "history_mask": self.history_mask,
            "history_length": self.history_length,
            "action_mask": self.action_mask,
        }


## Tensorizer


class BattleTensorizer:
    """Deterministic BattleFeatures -> BattleTensors converter."""

    def __init__(
        self,
        *,
        max_history: int = DEFAULT_MAX_HISTORY,
        vocab_gen: int = DEFAULT_VOCAB_GEN,
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be > 0")

        if vocab_gen < 1:
            raise ValueError("vocab_gen must be >= 1")

        self.max_history = max_history
        self.vocab_gen = vocab_gen

        limits = id_limits(vocab_gen)

        # Concrete SDK IDs start at 1 and we add one offset because 0/1 are
        # reserved for NONE and UNKNOWN.
        self.species_vocab_size = limits.pokemon + 2
        self.form_vocab_size = limits.form + 2
        self.move_vocab_size = limits.move + 2
        self.ability_vocab_size = limits.ability + 2
        self.item_vocab_size = limits.item + 2

    def tensorize(
        self, features: BattleFeatures, *, device: DeviceLike | None = None
    ) -> BattleTensors:
        """Tensorize one battle observation."""

        gen = features.format.gen
        if gen is None:
            raise ValueError("BattleFeatures.format.gen must be known")

        if gen > self.vocab_gen:
            raise ValueError(
                f"Battle generation {gen} exceeds vocabulary generation "
                f"{self.vocab_gen}; increase vocab_gen before training on it"
            )

        if len(features.own_team) != TEAM_SIZE:
            raise ValueError(
                f"Expected {TEAM_SIZE} own Pokémon slots, got {len(features.own_team)}"
            )

        if len(features.enemy_team) != TEAM_SIZE:
            raise ValueError(
                f"Expected {TEAM_SIZE} enemy Pokémon slots, "
                f"got {len(features.enemy_team)}"
            )

        categorical = self._tensorize_pokemon_categoricals(features)
        pokemon_numeric, pokemon_mask = self._tensorize_pokemon_numeric(
            features
        )

        field_numeric = self._tensorize_field(features)
        weather_id = torch.tensor(
            self._weather_token(features.field.weather), dtype=torch.long
        )

        history = build_recent_simplified_history(
            features.history, max_entries=self.max_history
        )
        history_tensors = self._tensorize_history(
            history=history, current_turn=features.field.turn
        )

        action_mask = self._tensorize_action_mask(features)

        output = BattleTensors(
            base_species_ids=categorical["base_species_ids"],
            species_ids=categorical["species_ids"],
            form_ids=categorical["form_ids"],
            move_ids=categorical["move_ids"],
            item_ids=categorical["item_ids"],
            ability_ids=categorical["ability_ids"],
            status_ids=categorical["status_ids"],
            pokemon_numeric=pokemon_numeric,
            pokemon_mask=pokemon_mask,
            weather_id=weather_id,
            field_numeric=field_numeric,
            history_kind=history_tensors["history_kind"],
            history_move=history_tensors["history_move"],
            history_species=history_tensors["history_species"],
            history_form=history_tensors["history_form"],
            history_actor=history_tensors["history_actor"],
            history_target=history_tensors["history_target"],
            history_reason=history_tensors["history_reason"],
            history_numeric=history_tensors["history_numeric"],
            history_mask=history_tensors["history_mask"],
            history_length=history_tensors["history_length"],
            action_mask=action_mask,
        )

        self._validate_shapes(output)

        if device is not None:
            return output.to(device)

        return output

    ## Pokémon

    def _tensorize_pokemon_categoricals(
        self, features: BattleFeatures
    ) -> dict[str, Tensor]:
        base_species_ids = torch.zeros(POKEMON_SLOTS, dtype=torch.long)
        species_ids = torch.zeros(POKEMON_SLOTS, dtype=torch.long)
        form_ids = torch.zeros(POKEMON_SLOTS, dtype=torch.long)
        move_ids = torch.zeros(
            (POKEMON_SLOTS, MOVES_PER_POKEMON), dtype=torch.long
        )
        item_ids = torch.zeros(POKEMON_SLOTS, dtype=torch.long)
        ability_ids = torch.zeros(POKEMON_SLOTS, dtype=torch.long)
        status_ids = torch.zeros(POKEMON_SLOTS, dtype=torch.long)

        for row, pokemon in enumerate(features.own_team):
            if not pokemon.present:
                continue

            base_species, _ = self._species_token(pokemon.species)
            current_species, current_form = self._species_token(
                pokemon.current_species or pokemon.species
            )

            base_species_ids[row] = base_species
            species_ids[row] = current_species
            form_ids[row] = current_form

            for move_slot, move in enumerate(pokemon.moves[:MOVES_PER_POKEMON]):
                if move.present:
                    move_ids[row, move_slot] = self._move_token(move.name)

            item_ids[row] = self._item_token_known(
                pokemon.item, gen=features.format.gen
            )
            ability_ids[row] = self._ability_token_known(
                pokemon.current_ability
                if pokemon.current_ability is not None
                else pokemon.base_ability,
                gen=features.format.gen,
            )
            status_ids[row] = self._status_token(pokemon.status.major)

        for enemy_index, pokemon in enumerate(features.enemy_team):
            row = TEAM_SIZE + enemy_index

            if not pokemon.revealed:
                # The slot itself exists, but all identity information is
                # unrevealed rather than known-absent.
                base_species_ids[row] = UNKNOWN_ID
                species_ids[row] = UNKNOWN_ID
                form_ids[row] = UNKNOWN_ID
                move_ids[row].fill_(UNKNOWN_ID)

                item_ids[row] = self._enemy_item_token(
                    pokemon.item, gen=features.format.gen
                )
                ability_ids[row] = self._enemy_ability_token(
                    pokemon.current_ability, gen=features.format.gen
                )
                status_ids[row] = NONE_ID
                continue

            base_species, _ = self._species_token(pokemon.species)
            current_species, current_form = self._species_token(
                pokemon.current_species or pokemon.species
            )

            base_species_ids[row] = base_species
            species_ids[row] = current_species
            form_ids[row] = current_form

            for move_slot, move in enumerate(pokemon.moves[:MOVES_PER_POKEMON]):
                move_ids[row, move_slot] = self._knowledge_move_token(move)

            item_ids[row] = self._enemy_item_token(
                pokemon.item, gen=features.format.gen
            )
            ability_ids[row] = self._enemy_ability_token(
                pokemon.current_ability, gen=features.format.gen
            )
            status_ids[row] = self._status_token(pokemon.status.major)

        return {
            "base_species_ids": base_species_ids,
            "species_ids": species_ids,
            "form_ids": form_ids,
            "move_ids": move_ids,
            "item_ids": item_ids,
            "ability_ids": ability_ids,
            "status_ids": status_ids,
        }

    def _tensorize_pokemon_numeric(
        self, features: BattleFeatures
    ) -> tuple[Tensor, Tensor]:
        numeric = torch.zeros(
            (POKEMON_SLOTS, POKEMON_NUMERIC_DIM), dtype=torch.float32
        )
        mask = torch.zeros(POKEMON_SLOTS, dtype=torch.bool)

        for row, pokemon in enumerate(features.own_team):
            if not pokemon.present:
                continue

            mask[row] = True
            self._fill_common_pokemon_numeric(
                numeric[row],
                present=True,
                active=pokemon.active,
                fainted=pokemon.fainted,
                hp_ratio=pokemon.hp_ratio,
                level=pokemon.level,
                transformed=pokemon.transformed,
                has_type_override=pokemon.type_override is not None,
                attack_stage=pokemon.status.attack_stage,
                defense_stage=pokemon.status.defense_stage,
                special_attack_stage=pokemon.status.special_attack_stage,
                special_defense_stage=pokemon.status.special_defense_stage,
                speed_stage=pokemon.status.speed_stage,
                accuracy_stage=pokemon.status.accuracy_stage,
                evasion_stage=pokemon.status.evasion_stage,
                perish_count=pokemon.status.perish_count,
                must_recharge=pokemon.status.must_recharge,
            )

            if pokemon.stats is not None:
                stats = pokemon.stats
                numeric[row, 19] = _normalize_stat(stats.attack)
                numeric[row, 20] = _normalize_stat(stats.defense)
                numeric[row, 21] = _normalize_stat(stats.special_attack)
                numeric[row, 22] = _normalize_stat(stats.special_defense)
                numeric[row, 23] = _normalize_stat(stats.speed)
                numeric[row, 24] = 1.0

        for enemy_index, pokemon in enumerate(features.enemy_team):
            row = TEAM_SIZE + enemy_index

            if not pokemon.revealed:
                continue

            mask[row] = True
            self._fill_common_pokemon_numeric(
                numeric[row],
                present=True,
                active=pokemon.active,
                fainted=pokemon.fainted,
                hp_ratio=pokemon.hp_ratio,
                level=pokemon.level,
                transformed=pokemon.transformed,
                has_type_override=pokemon.type_override is not None,
                attack_stage=pokemon.status.attack_stage,
                defense_stage=pokemon.status.defense_stage,
                special_attack_stage=pokemon.status.special_attack_stage,
                special_defense_stage=pokemon.status.special_defense_stage,
                speed_stage=pokemon.status.speed_stage,
                accuracy_stage=pokemon.status.accuracy_stage,
                evasion_stage=pokemon.status.evasion_stage,
                perish_count=pokemon.status.perish_count,
                must_recharge=pokemon.status.must_recharge,
            )

        return numeric, mask

    @staticmethod
    def _fill_common_pokemon_numeric(
        row: Tensor,
        *,
        present: bool,
        active: bool,
        fainted: bool,
        hp_ratio: float | None,
        level: int | None,
        transformed: bool,
        has_type_override: bool,
        attack_stage: int,
        defense_stage: int,
        special_attack_stage: int,
        special_defense_stage: int,
        speed_stage: int,
        accuracy_stage: int,
        evasion_stage: int,
        perish_count: int | None,
        must_recharge: bool,
    ) -> None:
        row[0] = float(present)
        row[1] = float(active)
        row[2] = float(fainted)

        if hp_ratio is not None:
            row[3] = _clamp_float(hp_ratio, 0.0, 1.0)
            row[4] = 1.0

        if level is not None:
            row[5] = _clamp_float(level / 100.0, 0.0, 1.0)
            row[6] = 1.0

        row[7] = float(transformed)
        row[8] = float(has_type_override)

        row[9] = _normalize_stage(attack_stage)
        row[10] = _normalize_stage(defense_stage)
        row[11] = _normalize_stage(special_attack_stage)
        row[12] = _normalize_stage(special_defense_stage)
        row[13] = _normalize_stage(speed_stage)
        row[14] = _normalize_stage(accuracy_stage)
        row[15] = _normalize_stage(evasion_stage)

        if perish_count is not None:
            row[16] = _clamp_float(perish_count / 4.0, 0.0, 1.0)
            row[17] = 1.0

        row[18] = float(must_recharge)

    ## Field

    def _tensorize_field(self, features: BattleFeatures) -> Tensor:
        field = torch.zeros(FIELD_NUMERIC_DIM, dtype=torch.float32)

        gen = features.format.gen
        assert gen is not None

        field[0] = gen / float(self.vocab_gen)
        field[1] = min(features.field.turn, 100) / 100.0
        field[2] = float(features.field.gen_1_desync)
        field[3] = float(features.force_switch)

        if features.own_active_slot is not None:
            field[4] = features.own_active_slot / 5.0
            field[5] = 1.0

        if features.enemy_active_slot is not None:
            field[6] = features.enemy_active_slot / 5.0
            field[7] = 1.0

        own = self._side_condition_vector(features.field.own_side_conditions)
        enemy = self._side_condition_vector(
            features.field.enemy_side_conditions
        )
        global_conditions = self._side_condition_vector(
            features.field.field_conditions
        )

        offset = 8
        width = len(SIDE_CONDITION_NAMES)

        field[offset : offset + width] = own
        offset += width
        field[offset : offset + width] = enemy
        offset += width
        field[offset : offset + width] = global_conditions

        return field

    @staticmethod
    def _side_condition_vector(
        conditions: tuple[SideConditionFeatures, ...],
    ) -> Tensor:
        vector = torch.zeros(len(SIDE_CONDITION_NAMES), dtype=torch.float32)

        for condition in conditions:
            index = SIDE_CONDITION_INDEX.get(condition.name)
            if index is None:
                continue

            # Layered hazards need counts; binary conditions are naturally 1.
            vector[index] = _clamp_float(condition.value / 3.0, 0.0, 1.0)

        return vector

    ## Simplified history

    def _tensorize_history(
        self, *, history: tuple[SimplifiedHistoryEntry, ...], current_turn: int
    ) -> dict[str, Tensor]:
        h = self.max_history

        history_kind = torch.zeros(h, dtype=torch.long)
        history_move = torch.zeros(h, dtype=torch.long)
        history_species = torch.zeros(h, dtype=torch.long)
        history_form = torch.zeros(h, dtype=torch.long)
        history_actor = torch.zeros(h, dtype=torch.long)
        history_target = torch.zeros(h, dtype=torch.long)
        history_reason = torch.zeros(h, dtype=torch.long)
        history_numeric = torch.zeros(
            (h, HISTORY_NUMERIC_DIM), dtype=torch.float32
        )
        history_mask = torch.zeros(h, dtype=torch.bool)

        if len(history) > h:
            raise ValueError(
                f"Received {len(history)} tactical entries for max_history={h}"
            )

        for index, entry in enumerate(history):
            history_mask[index] = True

            turn = entry.turn
            if turn is not None:
                age = max(current_turn - turn, 0)
                history_numeric[index, 0] = min(age, h) / float(h)

            if isinstance(entry, MoveTacticalEntry):
                history_kind[index] = HISTORY_KIND_MOVE
                history_move[index] = self._move_token(entry.move)
                history_actor[index] = self._ref_token(entry.actor)
                history_target[index] = self._ref_token(entry.target)

                if entry.success is not None:
                    history_numeric[index, 1] = float(entry.success)
                    history_numeric[index, 2] = 1.0

                if entry.does_hit is not None:
                    history_numeric[index, 3] = float(entry.does_hit)
                    history_numeric[index, 4] = 1.0

                if entry.hit_count is not None:
                    history_numeric[index, 5] = min(entry.hit_count, 5) / 5.0
                    history_numeric[index, 6] = 1.0

                self._fill_move_hp_summary(
                    row=history_numeric[index], entry=entry
                )

            elif isinstance(entry, SwitchTacticalEntry):
                history_kind[index] = HISTORY_KIND_SWITCH
                history_actor[index] = self._ref_token(entry.pokemon)

                species, form = self._species_token(entry.species)
                history_species[index] = species
                history_form[index] = form

                if entry.hp_ratio is not None:
                    history_numeric[index, 15] = _clamp_float(
                        entry.hp_ratio, 0.0, 1.0
                    )
                    history_numeric[index, 16] = 1.0

                history_numeric[index, 17] = float(entry.baton_pass)

            elif isinstance(entry, CantTacticalEntry):
                history_kind[index] = HISTORY_KIND_CANT
                history_actor[index] = self._ref_token(entry.pokemon)
                history_move[index] = self._move_token(entry.move)
                history_reason[index] = self._cant_reason_token(entry.reason)

            else:
                raise TypeError(
                    f"Unsupported tactical history entry: {type(entry).__name__}"
                )

        return {
            "history_kind": history_kind,
            "history_move": history_move,
            "history_species": history_species,
            "history_form": history_form,
            "history_actor": history_actor,
            "history_target": history_target,
            "history_reason": history_reason,
            "history_numeric": history_numeric,
            "history_mask": history_mask,
            "history_length": torch.tensor(len(history), dtype=torch.long),
        }

    def _fill_move_hp_summary(
        self, *, row: Tensor, entry: MoveTacticalEntry
    ) -> None:
        target_delta = 0.0
        target_delta_known = False

        actor_delta = 0.0
        actor_delta_known = False

        crit: bool | None = None
        effectiveness: float | None = None

        for hp_change in entry.hp_changes:
            if hp_change.hp_delta is not None and _same_ref(
                hp_change.target, entry.target
            ):
                target_delta += hp_change.hp_delta
                target_delta_known = True

                if crit is None and hp_change.crit is not None:
                    crit = hp_change.crit

                if (
                    effectiveness is None
                    and hp_change.effectiveness is not None
                ):
                    effectiveness = hp_change.effectiveness

            if hp_change.hp_delta is not None and _same_ref(
                hp_change.target, entry.actor
            ):
                actor_delta += hp_change.hp_delta
                actor_delta_known = True

        if target_delta_known:
            row[7] = _clamp_float(target_delta, -1.0, 1.0)
            row[8] = 1.0

        if actor_delta_known:
            row[9] = _clamp_float(actor_delta, -1.0, 1.0)
            row[10] = 1.0

        if crit is not None:
            row[11] = float(crit)
            row[12] = 1.0

        if effectiveness is not None:
            row[13] = _clamp_float(effectiveness / 4.0, 0.0, 1.0)
            row[14] = 1.0

    ## Actions

    @staticmethod
    def _tensorize_action_mask(features: BattleFeatures) -> Tensor:
        mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)

        for action in features.available_actions:
            if action.kind == "move":
                move = action.move
                if move is None:
                    raise ValueError(
                        "ActionFeatures(kind='move') has no move payload"
                    )

                index = move.request_index
                if not 0 <= index < 4:
                    raise ValueError(
                        f"Move request_index must be in [0, 3], got {index}"
                    )

                mask[index] = True
                continue

            if action.kind == "switch":
                switch = action.switch
                if switch is None:
                    raise ValueError(
                        "ActionFeatures(kind='switch') has no switch payload"
                    )

                slot = switch.team_slot
                if not 0 <= slot < TEAM_SIZE:
                    raise ValueError(
                        f"Switch team_slot must be in [0, 5], got {slot}"
                    )

                mask[4 + slot] = True
                continue

            raise ValueError(f"Unknown action kind: {action.kind!r}")

        return mask

    ## Vocabulary helpers

    def _species_token(self, name: str | None) -> tuple[int, int]:
        if name is None or name == "":
            return NONE_ID, NONE_ID

        try:
            species, form = pokemon_id(name, self.vocab_gen)
        except AssertionError, KeyError, TypeError, ValueError:
            return UNKNOWN_ID, UNKNOWN_ID

        return species + REAL_ID_OFFSET, form + REAL_ID_OFFSET

    def _move_token(self, name: str | None) -> int:
        if name is None or name == "":
            return NONE_ID

        try:
            raw = move_id(name, self.vocab_gen)
        except AssertionError, KeyError, TypeError, ValueError:
            return UNKNOWN_ID

        return raw + REAL_ID_OFFSET

    def _item_token_known(self, name: str | None, *, gen: int | None) -> int:
        # Held items do not exist in Gen 1 battles.
        if gen == 1:
            return NONE_ID

        if name is None or name == "":
            return NONE_ID

        try:
            raw = item_id(name, self.vocab_gen)
        except AssertionError, KeyError, TypeError, ValueError:
            return UNKNOWN_ID

        return raw + REAL_ID_OFFSET

    def _ability_token_known(self, name: str | None, *, gen: int | None) -> int:
        # Abilities do not exist before Gen 3.
        if gen is not None and gen <= 2:
            return NONE_ID

        if name is None or name == "":
            return NONE_ID

        try:
            raw = ability_id(name, self.vocab_gen)
        except AssertionError, KeyError, TypeError, ValueError:
            return UNKNOWN_ID

        return raw + REAL_ID_OFFSET

    def _knowledge_move_token(self, value: Knowledge[str]) -> int:
        if not value.known:
            return UNKNOWN_ID
        return self._move_token(value.value)

    def _enemy_item_token(
        self, value: Knowledge[str], *, gen: int | None
    ) -> int:
        if gen == 1:
            return NONE_ID

        if not value.known:
            return UNKNOWN_ID

        return self._item_token_known(value.value, gen=gen)

    def _enemy_ability_token(
        self, value: Knowledge[str], *, gen: int | None
    ) -> int:
        if gen is not None and gen <= 2:
            return NONE_ID

        if not value.known:
            return UNKNOWN_ID

        return self._ability_token_known(value.value, gen=gen)

    @staticmethod
    def _status_token(name: str | None) -> int:
        if name is None or name == "":
            return NONE_ID
        return STATUS_IDS.get(name, UNKNOWN_ID)

    @staticmethod
    def _weather_token(name: str | None) -> int:
        if name is None or name == "":
            return NONE_ID
        return WEATHER_IDS.get(name, UNKNOWN_ID)

    @staticmethod
    def _cant_reason_token(reason: str | None) -> int:
        if reason is None or reason == "":
            return NONE_ID
        return CANT_REASON_IDS.get(reason, UNKNOWN_ID)

    @staticmethod
    def _ref_token(ref: PokemonRefFeatures | None) -> int:
        if ref is None:
            return HISTORY_REF_NONE

        if ref.side == "self":
            if ref.slot is None:
                return HISTORY_REF_SELF_UNKNOWN
            if not 0 <= ref.slot < TEAM_SIZE:
                return HISTORY_REF_SELF_UNKNOWN
            return 1 + ref.slot

        if ref.side == "opponent":
            if ref.slot is None:
                return HISTORY_REF_OPPONENT_UNKNOWN
            if not 0 <= ref.slot < TEAM_SIZE:
                return HISTORY_REF_OPPONENT_UNKNOWN
            return 7 + ref.slot

        return HISTORY_REF_NONE

    ## Validation

    def _validate_shapes(self, tensors: BattleTensors) -> None:
        expected = {
            "base_species_ids": (POKEMON_SLOTS,),
            "species_ids": (POKEMON_SLOTS,),
            "form_ids": (POKEMON_SLOTS,),
            "move_ids": (POKEMON_SLOTS, MOVES_PER_POKEMON),
            "item_ids": (POKEMON_SLOTS,),
            "ability_ids": (POKEMON_SLOTS,),
            "status_ids": (POKEMON_SLOTS,),
            "pokemon_numeric": (POKEMON_SLOTS, POKEMON_NUMERIC_DIM),
            "pokemon_mask": (POKEMON_SLOTS,),
            "weather_id": (),
            "field_numeric": (FIELD_NUMERIC_DIM,),
            "history_kind": (self.max_history,),
            "history_move": (self.max_history,),
            "history_species": (self.max_history,),
            "history_form": (self.max_history,),
            "history_actor": (self.max_history,),
            "history_target": (self.max_history,),
            "history_reason": (self.max_history,),
            "history_numeric": (self.max_history, HISTORY_NUMERIC_DIM),
            "history_mask": (self.max_history,),
            "history_length": (),
            "action_mask": (ACTION_COUNT,),
        }

        actual = {
            "base_species_ids": tuple(tensors.base_species_ids.shape),
            "species_ids": tuple(tensors.species_ids.shape),
            "form_ids": tuple(tensors.form_ids.shape),
            "move_ids": tuple(tensors.move_ids.shape),
            "item_ids": tuple(tensors.item_ids.shape),
            "ability_ids": tuple(tensors.ability_ids.shape),
            "status_ids": tuple(tensors.status_ids.shape),
            "pokemon_numeric": tuple(tensors.pokemon_numeric.shape),
            "pokemon_mask": tuple(tensors.pokemon_mask.shape),
            "weather_id": tuple(tensors.weather_id.shape),
            "field_numeric": tuple(tensors.field_numeric.shape),
            "history_kind": tuple(tensors.history_kind.shape),
            "history_move": tuple(tensors.history_move.shape),
            "history_species": tuple(tensors.history_species.shape),
            "history_form": tuple(tensors.history_form.shape),
            "history_actor": tuple(tensors.history_actor.shape),
            "history_target": tuple(tensors.history_target.shape),
            "history_reason": tuple(tensors.history_reason.shape),
            "history_numeric": tuple(tensors.history_numeric.shape),
            "history_mask": tuple(tensors.history_mask.shape),
            "history_length": tuple(tensors.history_length.shape),
            "action_mask": tuple(tensors.action_mask.shape),
        }

        for name, expected_shape in expected.items():
            if actual[name] != expected_shape:
                raise RuntimeError(
                    f"{name} has shape {actual[name]}, expected {expected_shape}"
                )


## Batching


def collate_battles(examples: list[BattleTensors]) -> BattleBatch:
    """Stack equally configured BattleTensors into a training batch."""

    if not examples:
        raise ValueError("Cannot collate an empty batch")

    history_size = examples[0].history_kind.shape[0]

    for example in examples[1:]:
        if example.history_kind.shape[0] != history_size:
            raise ValueError(
                "All examples in a batch must use the same max_history"
            )

    return BattleBatch(
        base_species_ids=torch.stack([x.base_species_ids for x in examples]),
        species_ids=torch.stack([x.species_ids for x in examples]),
        form_ids=torch.stack([x.form_ids for x in examples]),
        move_ids=torch.stack([x.move_ids for x in examples]),
        item_ids=torch.stack([x.item_ids for x in examples]),
        ability_ids=torch.stack([x.ability_ids for x in examples]),
        status_ids=torch.stack([x.status_ids for x in examples]),
        pokemon_numeric=torch.stack([x.pokemon_numeric for x in examples]),
        pokemon_mask=torch.stack([x.pokemon_mask for x in examples]),
        weather_id=torch.stack([x.weather_id for x in examples]),
        field_numeric=torch.stack([x.field_numeric for x in examples]),
        history_kind=torch.stack([x.history_kind for x in examples]),
        history_move=torch.stack([x.history_move for x in examples]),
        history_species=torch.stack([x.history_species for x in examples]),
        history_form=torch.stack([x.history_form for x in examples]),
        history_actor=torch.stack([x.history_actor for x in examples]),
        history_target=torch.stack([x.history_target for x in examples]),
        history_reason=torch.stack([x.history_reason for x in examples]),
        history_numeric=torch.stack([x.history_numeric for x in examples]),
        history_mask=torch.stack([x.history_mask for x in examples]),
        history_length=torch.stack([x.history_length for x in examples]),
        action_mask=torch.stack([x.action_mask for x in examples]),
    )


## Helpers


def _same_ref(
    left: PokemonRefFeatures | None, right: PokemonRefFeatures | None
) -> bool:
    if left is None or right is None:
        return False

    if left.side != right.side:
        return False

    if left.slot is not None and right.slot is not None:
        return left.slot == right.slot

    if left.pokemon_id is not None and right.pokemon_id is not None:
        return left.pokemon_id == right.pokemon_id

    if left.species is not None and right.species is not None:
        return left.species == right.species

    return False


def _normalize_stage(value: int) -> float:
    return _clamp_float(value / 6.0, -1.0, 1.0)


def _normalize_stat(value: int | None) -> float:
    if value is None:
        return 0.0
    return _clamp_float(value / 500.0, 0.0, 2.0)


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


__all__ = [
    "ACTION_COUNT",
    "CANT_REASON_VOCAB_SIZE",
    "DEFAULT_MAX_HISTORY",
    "DEFAULT_VOCAB_GEN",
    "FIELD_NUMERIC_DIM",
    "HISTORY_KIND_VOCAB_SIZE",
    "HISTORY_NUMERIC_DIM",
    "HISTORY_REF_VOCAB_SIZE",
    "MOVES_PER_POKEMON",
    "NONE_ID",
    "POKEMON_NUMERIC_DIM",
    "POKEMON_SLOTS",
    "STATUS_VOCAB_SIZE",
    "TEAM_SIZE",
    "UNKNOWN_ID",
    "WEATHER_VOCAB_SIZE",
    "BattleBatch",
    "BattleTensorizer",
    "BattleTensors",
    "collate_battles",
]
