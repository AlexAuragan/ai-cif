from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ai_cif.dtpo.config import DTPOConfig
from ai_cif.dtpo.features import tree_features_batch
from ai_cif.dtpo.policy import DecisionTreePolicy
from ai_cif.dtpo.value import BattleValueModel
from ai_cif.training.trajectory import PackedRollout, Trajectory


@dataclass(frozen=True)
class DTPOMetrics:
    policy_objective_before: float
    policy_objective_after: float
    value_loss: float
    entropy: float
    mean_advantage: float
    mean_return: float
    tree_depth: int
    leaf_count: int
    tree_updated: bool


def _discounted_terminal_returns(
    rollout: PackedRollout, gamma: float
) -> Tensor:
    returns: list[Tensor] = []

    for length_tensor, reward in zip(
        rollout.trajectory_lengths, rollout.rewards, strict=True
    ):
        length = int(length_tensor.item())
        powers = torch.arange(length - 1, -1, -1, dtype=torch.float32)
        returns.append(reward.float() * gamma**powers)

    return torch.cat(returns)


def _policy_objective(
    logits: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    clip_epsilon: float,
) -> Tensor:
    masked_logits = logits.masked_fill(~action_mask, -1e9)
    log_probabilities = torch.log_softmax(masked_logits, dim=1)
    row_indices = torch.arange(actions.shape[0], device=actions.device)
    new_log_probs = log_probabilities[row_indices, actions]

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    return surrogate.sum()


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    return float(entropy.mean())


def _train_value_model(
    *,
    value_model: BattleValueModel,
    optimizer: torch.optim.Optimizer,
    features: Tensor,
    targets: Tensor,
    config: DTPOConfig,
    device: torch.device,
) -> float:
    count = features.shape[0]
    losses: list[float] = []

    value_model.train()

    for _ in range(config.value_epochs):
        permutation = torch.randperm(count, device=device)

        for start in range(0, count, config.value_minibatch_size):
            indices = permutation[start : start + config.value_minibatch_size]

            predicted = value_model(features[indices])
            loss = torch.nn.functional.mse_loss(predicted, targets[indices])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

    value_model.eval()

    return sum(losses) / (len(losses) or 1)


def dtpo_update(
    *,
    policy: DecisionTreePolicy,
    value_model: BattleValueModel,
    value_optimizer: torch.optim.Optimizer,
    trajectories: list[Trajectory] | PackedRollout,
    config: DTPOConfig,
    device: torch.device,
) -> DTPOMetrics:
    rollout = (
        trajectories
        if isinstance(trajectories, PackedRollout)
        else PackedRollout.from_trajectories(trajectories)
    )

    if rollout.decision_count == 0:
        raise ValueError("No decisions to train on")

    features_np = tree_features_batch(rollout.observations)
    action_mask_np = (
        rollout.observations.action_mask.detach().cpu().numpy().astype(bool)
    )

    features = torch.from_numpy(features_np).to(device)
    action_mask = torch.from_numpy(action_mask_np).to(device)
    actions = rollout.actions.to(device)
    old_log_probs = rollout.old_log_probs.to(device)

    returns = _discounted_terminal_returns(rollout, config.gamma).to(device)

    old_values = rollout.old_values.to(device)
    advantages = returns - old_values

    if config.normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    old_raw_probabilities_np = policy.raw_probabilities_batch(features_np)

    old_raw_logits = torch.log(
        torch.from_numpy(old_raw_probabilities_np)
        .to(device=device, dtype=torch.float32)
        .clamp_min(1e-8)
    )

    objective_before = float(
        _policy_objective(
            old_raw_logits,
            actions,
            old_log_probs,
            advantages,
            action_mask,
            config.clip_epsilon,
        ).item()
    )

    best_tree = policy.tree
    best_objective = objective_before

    current_raw_probabilities_np = old_raw_probabilities_np

    for _ in range(config.policy_updates):
        logits = torch.log(
            torch.from_numpy(current_raw_probabilities_np)
            .to(device=device, dtype=torch.float32)
            .clamp_min(1e-8)
        )
        logits.requires_grad_(True)

        objective = _policy_objective(
            logits,
            actions,
            old_log_probs,
            advantages,
            action_mask,
            config.clip_epsilon,
        )

        (gradient,) = torch.autograd.grad(objective, logits)

        target_logits = logits.detach() + config.learning_rate * gradient

        # The regression tree learns the latent 10-action
        # policy. Legality is imposed separately at inference.
        target_probabilities = (
            torch.softmax(target_logits, dim=1).detach().cpu().numpy()
        )

        candidate = policy.make_candidate(features_np, target_probabilities)

        candidate_raw_probabilities_np = policy.raw_probabilities_batch(
            features_np, tree=candidate
        )

        candidate_logits = torch.log(
            torch.from_numpy(candidate_raw_probabilities_np)
            .to(device=device, dtype=torch.float32)
            .clamp_min(1e-8)
        )

        candidate_objective = float(
            _policy_objective(
                candidate_logits,
                actions,
                old_log_probs,
                advantages,
                action_mask,
                config.clip_epsilon,
            ).item()
        )

        print(
            "DTPO candidate "
            f"depth={candidate.get_depth()} "
            f"leaves={candidate.get_n_leaves()} "
            f"objective="
            f"{objective_before:.6f}"
            f"->{candidate_objective:.6f}"
        )

        if candidate_objective >= best_objective:
            best_objective = candidate_objective
            best_tree = candidate

        current_raw_probabilities_np = candidate_raw_probabilities_np

    tree_updated = best_tree is not policy.tree

    if tree_updated:
        if best_tree is None:
            raise RuntimeError("Missing candidate tree")
        policy.replace_tree(best_tree)

    value_loss = _train_value_model(
        value_model=value_model,
        optimizer=value_optimizer,
        features=features,
        targets=returns,
        config=config,
        device=device,
    )

    final_probabilities = policy.probabilities_batch(
        features_np, action_mask_np
    )

    return DTPOMetrics(
        policy_objective_before=objective_before,
        policy_objective_after=best_objective,
        value_loss=value_loss,
        entropy=_entropy(final_probabilities),
        mean_advantage=float(advantages.mean().item()),
        mean_return=float(returns.mean().item()),
        tree_depth=policy.depth,
        leaf_count=policy.leaf_count,
        tree_updated=tree_updated,
    )
