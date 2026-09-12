from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from ai_cif.model.model import BattleModel
from ai_cif.training.trajectory import Trajectory
from ai_cif.vectorization.tensorizer import collate_battles


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    epochs: int = 4
    minibatch_size: int = 256


@dataclass(frozen=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


def ppo_update(
    *,
    model: BattleModel,
    optimizer: torch.optim.Optimizer,
    trajectories: list[Trajectory],
    config: PPOConfig,
    device: torch.device,
) -> PPOMetrics:
    decisions = [
        decision
        for trajectory in trajectories
        for decision in trajectory.decisions
    ]

    if not decisions:
        raise ValueError("No decisions to train on")

    if any(trajectory.outcome is None for trajectory in trajectories):
        raise ValueError("All trajectories must have a terminal outcome")

    observations = [
        decision.observation
        for decision in decisions
    ]

    actions = torch.tensor(
        [decision.action for decision in decisions],
        dtype=torch.long,
    )

    old_log_probs = torch.tensor(
        [decision.log_prob for decision in decisions],
        dtype=torch.float32,
    )

    old_values = torch.tensor(
        [decision.value for decision in decisions],
        dtype=torch.float32,
    )

    returns = torch.cat(
        [
            torch.full(
                (len(trajectory.decisions),),
                float(trajectory.outcome or 0),
                dtype=torch.float32,
            )
            for trajectory in trajectories
        ]
    )

    advantages = returns - old_values

    if advantages.numel() > 1:
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

    count = len(decisions)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []

    model.train()

    for _ in range(config.epochs):
        permutation = torch.randperm(count)

        for start in range(0, count, config.minibatch_size):
            indices = permutation[
                start : start + config.minibatch_size
            ]

            examples = [
                observations[int(index)]
                for index in indices
            ]

            batch = collate_battles(examples).to(device)

            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_returns = returns[indices]
            batch_advantages = advantages[indices]

            logits, values = model(batch)

            distribution = Categorical(logits=logits)

            new_log_probs = distribution.log_prob(
                batch_actions
            )

            entropy = distribution.entropy().mean()

            ratio = torch.exp(
                new_log_probs - batch_old_log_probs
            )

            unclipped = ratio * batch_advantages

            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * batch_advantages

            policy_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()

            value_loss = torch.nn.functional.mse_loss(
                values,
                batch_returns,
            )

            total_loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )

            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            total_losses.append(float(total_loss.item()))

    model.eval()

    return PPOMetrics(
        policy_loss=sum(policy_losses) / len(policy_losses),
        value_loss=sum(value_losses) / len(value_losses),
        entropy=sum(entropies) / len(entropies),
        total_loss=sum(total_losses) / len(total_losses),
    )
