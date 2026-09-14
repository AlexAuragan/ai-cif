from dataclasses import dataclass

import torch
from showdown_sdk.classes.client import Client
from showdown_sdk.classes.combat_handler.random_handler import RandomMoveCombatHandler
from showdown_sdk.models.sdk import SampleTeamGenerator

from ai_cif.inference.combat_handler import NeuralCombatHandler
from ai_cif.model.model import BattleModel
from ai_cif.training.combat_handler import TrainingCombatHandler
from ai_cif.vectorization.tensorizer import BattleTensorizer


@dataclass
class BattleWorker:
    neural_client: Client
    random_client: Client
    training_handler: TrainingCombatHandler
    evaluation_handler: NeuralCombatHandler
    team_generator: SampleTeamGenerator | None


def create_worker(
    *,
    index: int,
    url: str,
    fmt: str,
    team_seed: int,
    model: BattleModel,
    tensorizer: BattleTensorizer,
    device: torch.device,
) -> BattleWorker:
    training_handler = TrainingCombatHandler(
        model=model, tensorizer=tensorizer, device=device
    )

    evaluation_handler = NeuralCombatHandler(
        model=model, tensorizer=tensorizer, device=device
    )

    neural_client = Client(url, combat_handler=training_handler)

    random_client = Client(url, combat_handler=RandomMoveCombatHandler())

    neural_client.log_manager.disable()
    random_client.log_manager.disable()

    team_generator: SampleTeamGenerator | None = None

    if "randombattle" not in fmt:
        team_generator = SampleTeamGenerator(team_seed + index)

    return BattleWorker(
        neural_client=neural_client,
        random_client=random_client,
        training_handler=training_handler,
        evaluation_handler=evaluation_handler,
        team_generator=team_generator,
    )


def split_battles(battles: int, worker_count: int) -> list[int]:
    base = battles // worker_count
    remainder = battles % worker_count

    return [base + (1 if index < remainder else 0) for index in range(worker_count)]
