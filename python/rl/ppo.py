"""Proximal Policy Optimization.

The update itself is short; the diagnostics around it are not, deliberately. A
PPO run that is subtly wrong still trains, just badly, and the only way to tell
the difference is to watch approximate KL, clip fraction and explained variance
rather than the reward curve alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .policy import ActorCritic
from .rollout import RolloutBatch


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # Clipping the value function too is optional and off by default: it helps
    # when returns are large and unnormalised, and here they are already scaled.
    clip_range_value: float | None = None
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch_size: int = 256
    # Stops the epoch loop once the updated policy has moved too far from the
    # one that collected the data. Without it a bad batch can move the policy
    # somewhere it cannot recover from, which shows up as a run that collapses
    # from a good score in a single update.
    target_kl: float | None = 0.02
    normalize_advantages: bool = True
    anneal_learning_rate: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "PPOConfig":
        known = {f for f in PPOConfig.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown PPO settings: {sorted(unknown)}")
        return PPOConfig(**data)


@dataclass
class UpdateStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0
    epochs_run: int = 0
    stopped_early: bool = False
    learning_rate: float = 0.0


class PPO:
    def __init__(
        self,
        policy: ActorCritic,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.policy = policy
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def update(self, batch: RolloutBatch) -> UpdateStats:
        config = self.config
        stats = UpdateStats(learning_rate=self.optimizer.param_groups[0]["lr"])

        total = len(batch)
        indices = np.arange(total)
        minibatch_size = min(config.minibatch_size, total)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        grad_norms: list[float] = []

        for epoch in range(config.epochs):
            np.random.shuffle(indices)
            epoch_kls: list[float] = []

            for start in range(0, total, minibatch_size):
                slice_indices = torch.as_tensor(
                    indices[start : start + minibatch_size], device=self.device
                )
                observations = batch.observations[slice_indices]
                actions = batch.actions[slice_indices]
                old_log_probs = batch.log_probs[slice_indices]
                advantages = batch.advantages[slice_indices]
                returns = batch.returns[slice_indices]
                old_values = batch.values[slice_indices]

                if config.normalize_advantages and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_probs, entropy, values = self.policy.evaluate_actions(observations, actions)
                ratio = torch.exp(log_probs - old_log_probs)

                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                if config.clip_range_value is None:
                    value_loss = 0.5 * (values - returns).pow(2).mean()
                else:
                    clipped_values = old_values + torch.clamp(
                        values - old_values, -config.clip_range_value, config.clip_range_value
                    )
                    value_loss = 0.5 * torch.max(
                        (values - returns).pow(2), (clipped_values - returns).pow(2)
                    ).mean()

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator. The naive
                    # (old - new).mean() is unbiased but noisy enough that a
                    # target-KL early stop on it fires almost at random.
                    log_ratio = log_probs - old_log_probs
                    approx_kl = torch.mean(torch.exp(log_ratio) - 1 - log_ratio).item()
                    clip_fraction = (
                        (torch.abs(ratio - 1) > config.clip_range).float().mean().item()
                    )

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_mean.item())
                kls.append(approx_kl)
                epoch_kls.append(approx_kl)
                clip_fractions.append(clip_fraction)
                grad_norms.append(float(grad_norm))

            stats.epochs_run = epoch + 1
            if config.target_kl is not None and np.mean(epoch_kls) > config.target_kl:
                stats.stopped_early = True
                break

        stats.policy_loss = float(np.mean(policy_losses))
        stats.value_loss = float(np.mean(value_losses))
        stats.entropy = float(np.mean(entropies))
        stats.approx_kl = float(np.mean(kls))
        stats.clip_fraction = float(np.mean(clip_fractions))
        stats.grad_norm = float(np.mean(grad_norms))
        return stats


def ppo_clipped_objective(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_range: float
) -> torch.Tensor:
    """The clipped surrogate, exposed on its own so it can be tested directly.

    L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A), maximised - so the loss
    is its negation.
    """
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    return torch.min(unclipped, clipped)
