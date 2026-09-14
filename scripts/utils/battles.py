import asyncio

from showdown_sdk.classes.client import Client
from showdown_sdk.classes.dt import BattleResult
from showdown_sdk.models.sdk import TeamSet, print_reproduction_teams
from showdown_sdk.models.sdk.team_generators.team_generator import BaseTeamGenerator


def outcome_for(result: BattleResult, username: str) -> float:
    if result.winner is None:
        return 0.0

    if result.winner == username:
        return 1.0

    return -1.0


async def generate_teams(
    client_1: Client,
    client_2: Client,
    fmt: str,
    team_generator_1: BaseTeamGenerator | None = None,
    team_generator_2: BaseTeamGenerator | None = None,
):
    team_1: TeamSet | None = None
    team_2: TeamSet | None = None

    if team_generator_1 is not None:
        team_1 = await team_generator_1.generate(
            fmt, lambda team: client_1.validate_team(fmt, team)
        )

    if team_generator_2 is not None:
        team_2 = await team_generator_2.generate(
            fmt, lambda team: client_2.validate_team(fmt, team)
        )

    return team_1, team_2


async def run_battle(
    client_1: Client,
    client_2: Client,
    *,
    fmt: str,
    team_generator_1: BaseTeamGenerator | None,
    team_generator_2: BaseTeamGenerator | None,
) -> tuple[BattleResult, BattleResult]:
    await asyncio.gather(client_1.ensure_connected(), client_2.ensure_connected())

    if client_1.username is None:
        raise RuntimeError("Client 1 is not logged in")

    if client_2.username is None:
        raise RuntimeError("Client 2 is not logged in")

    team_1, team_2 = await generate_teams(
        client_1, client_2, fmt, team_generator_1, team_generator_2
    )

    await client_1.challenge(client_2.username, fmt, timeout=60, team=team_1)
    await client_2.accept_challenge(client_1.username, team=team_2)

    battle_waiter_1: asyncio.Task[BattleResult] | None = None
    battle_waiter_2: asyncio.Task[BattleResult] | None = None

    try:
        await asyncio.gather(
            client_1.battle_manager.room_ready.wait(),
            client_2.battle_manager.room_ready.wait(),
        )

        battle_waiter_1 = asyncio.create_task(client_1.wait_for_battle_end(timeout=300))
        battle_waiter_2 = asyncio.create_task(client_2.wait_for_battle_end(timeout=300))

        result_1, result_2 = await asyncio.gather(battle_waiter_1, battle_waiter_2)

        return result_1, result_2

    except BaseException as error:
        waiters = [
            waiter
            for waiter in (battle_waiter_1, battle_waiter_2)
            if waiter is not None
        ]

        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()

        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)

        client_1.battle_manager.abandon_battle(error)
        client_2.battle_manager.abandon_battle(error)

        if (team_generator_1 or team_generator_2) is not None:
            print_reproduction_teams(team_1, team_2)
        else:
            print("\n========== TEAM 1 ==========")
            for pokemon in client_1.battle_manager.battle_state.team:
                print(pokemon)

            print("\n========== TEAM 2 ==========")

            for pokemon in client_2.battle_manager.battle_state.team:
                print(pokemon)

            print("============================\n")

        raise
