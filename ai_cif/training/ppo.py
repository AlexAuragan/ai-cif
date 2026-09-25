from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from ai_cif.model.model import BattleModel
from ai_cif.training.trajectory import PackedRollout, Trajectory


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    epochs: int = 4
    minibatch_size: int = 256

    # Stop the epochs if the kl > kl_target * kl_ratio_threshold
    kl_target: float = 0.2
    kl_ratio_threshold: float | None = None

    # GAE
    gamma: float = 1.0
    gae_lambda: float = 0.95


@dataclass(frozen=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float

    approx_kl: float
    max_approx_kl: float
    clip_fraction: float

    early_stop: bool

    mean_value: float
    mean_return: float


def compute_gae(
    rollout: PackedRollout, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.empty_like(rollout.old_values)
    returns = torch.empty_like(rollout.old_values)

    offset = 0

    for trajectory_index, trajectory_length_tensor in enumerate(
        rollout.trajectory_lengths
    ):
        trajectory_length = int(trajectory_length_tensor.item())
        end = offset + trajectory_length

        values = rollout.old_values[offset:end]
        terminal_reward = rollout.rewards[trajectory_index]

        next_value = torch.tensor(0.0, dtype=values.dtype, device=values.device)
        next_advantage = torch.tensor(
            0.0, dtype=values.dtype, device=values.device
        )

        for local_index in range(trajectory_length - 1, -1, -1):
            is_terminal = local_index == trajectory_length - 1

            reward = (
                terminal_reward
                if is_terminal
                else torch.tensor(0.0, dtype=values.dtype, device=values.device)
            )

            value = values[local_index]

            delta = reward + gamma * next_value - value

            advantage = delta + gamma * gae_lambda * next_advantage

            advantages[offset + local_index] = advantage
            returns[offset + local_index] = advantage + value

            next_value = value
            next_advantage = advantage

        offset = end

    return advantages, returns


def ppo_update(
    *,
    model: BattleModel,
    optimizer: torch.optim.Optimizer,
    trajectories: list[Trajectory] | PackedRollout,
    config: PPOConfig,
    device: torch.device,
) -> PPOMetrics:
    rollout = (
        trajectories
        if isinstance(trajectories, PackedRollout)
        else PackedRollout.from_trajectories(trajectories)
    )

    if rollout.decision_count == 0:
        raise ValueError("No decisions to train on")

    advantages_cpu, returns_cpu = compute_gae(
        rollout, gamma=config.gamma, gae_lambda=config.gae_lambda
    )

    if advantages_cpu.numel() > 1:
        advantages_cpu = (advantages_cpu - advantages_cpu.mean()) / (
            advantages_cpu.std(unbiased=False) + 1e-8
        )

    actions = rollout.actions.to(device)
    old_log_probs = rollout.old_log_probs.to(device)
    old_values = rollout.old_values.to(device)
    returns = returns_cpu.to(device)
    advantages = advantages_cpu.to(device)

    count = rollout.decision_count

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []

    model.train()

    early_stop = False
    for _ in range(config.epochs):
        permutation = torch.randperm(count)

        for start in range(0, count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]

            batch = rollout.observations.index_select(indices).to(device)

            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_returns = returns[indices]
            batch_advantages = advantages[indices]

            logits, values = model(batch)

            distribution = Categorical(logits=logits)

            new_log_probs = distribution.log_prob(batch_actions)

            entropy = distribution.entropy().mean()

            log_ratio = new_log_probs - batch_old_log_probs

            ratio = torch.exp(log_ratio)

            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            kl_value = float(approx_kl.item())

            if (
                config.kl_ratio_threshold is not None
                and kl_value > config.kl_target * config.kl_ratio_threshold
            ):
                early_stop = True
                break

            clip_fraction = (
                ((ratio - 1.0).abs() > config.clip_epsilon).float().mean()
            )

            unclipped = ratio * batch_advantages

            clipped = (
                torch.clamp(
                    ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                )
                * batch_advantages
            )

            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = torch.nn.functional.mse_loss(values, batch_returns)

            total_loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )

            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            total_losses.append(float(total_loss.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))
        if early_stop:
            break

    model.eval()

    return PPOMetrics(
        policy_loss=sum(policy_losses) / (len(policy_losses) or 1),
        value_loss=sum(value_losses) / (len(value_losses) or 1),
        entropy=sum(entropies) / (len(entropies) or 1),
        total_loss=sum(total_losses) / (len(total_losses) or 1),
        approx_kl=sum(approx_kls) / (len(approx_kls) or 1),
        max_approx_kl=max(approx_kls),
        early_stop=early_stop,
        clip_fraction=sum(clip_fractions) / (len(clip_fractions) or 1),
        mean_value=float(old_values.mean().item()),
        mean_return=float(returns.mean().item()),
    )
