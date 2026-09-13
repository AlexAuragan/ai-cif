import asyncio
import math

import torch
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler import RandomMoveCombatHandler
from showdown_sdk.classes.dt import BattleResult
from showdown_sdk.models.sdk import SampleTeamGenerator, TeamSet

from ai_cif.model.config import ModelConfig
from ai_cif.model.model import BattleModel
from ai_cif.training.combat_handler import TrainingCombatHandler
from ai_cif.training.ppo import PPOConfig, ppo_update
from ai_cif.training.rewards import breakdown_for
from ai_cif.training.trajectory import Trajectory
from ai_cif.vectorization.tensorizer import (
    CANT_REASON_VOCAB_SIZE,
    FIELD_NUMERIC_DIM,
    HISTORY_KIND_VOCAB_SIZE,
    HISTORY_NUMERIC_DIM,
    HISTORY_REF_VOCAB_SIZE,
    POKEMON_NUMERIC_DIM,
    STATUS_VOCAB_SIZE,
    WEATHER_VOCAB_SIZE,
    BattleTensorizer,
)

WEBSOCKET_URL = "ws://127.0.0.1:8000/showdown/websocket"
FORMAT = "gen1randombattle"
BATTLE_COUNT = 4


def create_model(device: torch.device) -> tuple[BattleModel, BattleTensorizer]:
    tensorizer = BattleTensorizer(max_history=32, vocab_gen=4)

    config = ModelConfig(
        species_count=tensorizer.species_vocab_size,
        form_count=tensorizer.form_vocab_size,
        move_count=tensorizer.move_vocab_size,
        item_count=tensorizer.item_vocab_size,
        ability_count=tensorizer.ability_vocab_size,
        status_count=STATUS_VOCAB_SIZE,
        weather_count=WEATHER_VOCAB_SIZE,
        tactical_event_type_count=HISTORY_KIND_VOCAB_SIZE,
        history_ref_count=HISTORY_REF_VOCAB_SIZE,
        history_reason_count=CANT_REASON_VOCAB_SIZE,
    )

    model = BattleModel(
        config=config,
        pokemon_numeric_feature_count=POKEMON_NUMERIC_DIM,
        field_numeric_feature_count=FIELD_NUMERIC_DIM,
        tactical_numeric_feature_count=HISTORY_NUMERIC_DIM,
    )

    model.to(device)

    return model, tensorizer


async def run_battle(
    client_1: Client,
    client_2: Client,
    *,
    fmt: str,
    team_generator: SampleTeamGenerator | None,
) -> tuple[BattleResult, BattleResult]:
    await asyncio.gather(
        client_1.ensure_connected(), client_2.ensure_connected()
    )

    if client_1.username is None or client_2.username is None:
        raise RuntimeError("Both clients must be logged in")

    team_1: TeamSet | None = None
    team_2: TeamSet | None = None

    if team_generator is not None:
        team_1 = await team_generator.generate(
            fmt, lambda team: client_1.validate_team(fmt, team)
        )

        team_2 = await team_generator.generate(
            fmt, lambda team: client_2.validate_team(fmt, team)
        )

    await client_1.challenge(client_2.username, fmt, timeout=60, team=team_1)

    await client_2.accept_challenge(client_1.username, team=team_2)

    await asyncio.gather(
        client_1.battle_manager.room_ready.wait(),
        client_2.battle_manager.room_ready.wait(),
    )

    result_1, result_2 = await asyncio.gather(
        client_1.wait_for_battle_end(timeout=300),
        client_2.wait_for_battle_end(timeout=300),
    )

    return result_1, result_2


def outcome_for(result: BattleResult, username: str) -> float:
    if result.winner is None:
        return 0.0

    if result.winner == username:
        return 1.0

    return -1.0


async def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model, tensorizer = create_model(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model parameters: {parameter_count:,}")

    training_handler = TrainingCombatHandler(
        model=model, tensorizer=tensorizer, device=device
    )

    random_handler = RandomMoveCombatHandler()

    neural_client = Client(WEBSOCKET_URL, combat_handler=training_handler)

    random_client = Client(WEBSOCKET_URL, combat_handler=random_handler)

    team_generator: SampleTeamGenerator | None = None

    if "randombattle" not in FORMAT:
        team_generator = SampleTeamGenerator(42)

    trajectories: list[Trajectory] = []

    try:
        await asyncio.gather(neural_client.connect(), random_client.connect())

        await asyncio.gather(
            neural_client.login("BOT1"), random_client.login("BOT2")
        )

        print()
        print("Collecting trajectories")
        print("-----------------------")

        for battle_index in range(BATTLE_COUNT):
            training_handler.start_battle()

            result, _ = await run_battle(
                neural_client,
                random_client,
                fmt=FORMAT,
                team_generator=team_generator,
            )

            if neural_client.username is None:
                raise RuntimeError("Neural client lost its username")

            outcome = outcome_for(result, neural_client.username)
            reward = breakdown_for(result, outcome)

            trajectory = training_handler.finish_battle(outcome, reward)

            if not trajectory.decisions:
                raise RuntimeError("Collected a trajectory with no decisions")

            trajectories.append(trajectory)

            print(
                f"battle={battle_index + 1} "
                f"decisions={len(trajectory.decisions)} "
                f"outcome={outcome:+.0f}"
            )

    finally:
        await asyncio.gather(
            neural_client.close(), random_client.close(), return_exceptions=True
        )

    print()
    print("Validating trajectories")
    print("-----------------------")

    # Make sure start_battle() produced independent objects.
    trajectory_ids = {id(trajectory) for trajectory in trajectories}

    assert len(trajectory_ids) == len(trajectories)

    total_decisions = 0

    for trajectory in trajectories:
        assert trajectory.outcome in {-1.0, 0.0, 1.0}

        for decision in trajectory.decisions:
            assert 0 <= decision.action < 10
            assert math.isfinite(decision.log_prob)
            assert math.isfinite(decision.value)
            assert -1.0 <= decision.value <= 1.0

            total_decisions += 1

    print(f"Trajectories: {len(trajectories)}")
    print(f"Decisions: {total_decisions}")

    # Snapshot model before PPO.
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    ppo_config = PPOConfig(
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        epochs=1,
        minibatch_size=256,
    )

    print()
    print("Running one PPO update")
    print("----------------------")

    metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        trajectories=trajectories,
        config=ppo_config,
        device=device,
    )

    assert math.isfinite(metrics.policy_loss)
    assert math.isfinite(metrics.value_loss)
    assert math.isfinite(metrics.entropy)
    assert math.isfinite(metrics.total_loss)

    changed_parameters: list[str] = []

    for name, parameter in model.named_parameters():
        previous = before[name]
        current = parameter.detach().cpu()

        if not torch.equal(previous, current):
            changed_parameters.append(name)

    if not changed_parameters:
        raise RuntimeError("PPO completed but no model parameters changed")

    print(f"policy_loss: {metrics.policy_loss:+.6f}")
    print(f"value_loss:  {metrics.value_loss:+.6f}")
    print(f"entropy:     {metrics.entropy:+.6f}")
    print(f"total_loss:  {metrics.total_loss:+.6f}")

    print()
    print(f"Changed parameter tensors: {len(changed_parameters)}")

    print()
    print("First changed parameters:")

    for name in changed_parameters[:10]:
        print(f"  {name}")

    print()
    print("PPO smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
