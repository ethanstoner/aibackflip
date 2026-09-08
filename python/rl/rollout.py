"""Rollout storage and generalised advantage estimation.

The only subtle part is the boundary handling, and it is subtle in a way that
trains anyway when it is wrong - which is why it is spelled out here and pinned
by tests.

Three distinct things happen at the end of a step:

* **terminated** - the episode failed. There is no future, so the bootstrap is
  zero.
* **truncated** - the episode hit the time limit. There *is* a future; it was
  simply cut off. The bootstrap is the critic's estimate at the observation the
  episode ended on, which is why the environment ships that observation.
  Treating this as termination teaches the critic that reaching the time limit
  is worth nothing, and the policy learns to avoid surviving.
* **neither** - bootstrap from the next step's value as usual.

Separately, the GAE recursion has to break at *any* episode end, because the
observation stored at the next index belongs to a different episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """A flattened rollout, ready for minibatching."""

    observations: torch.Tensor  # (T*N, obs_dim)
    actions: torch.Tensor  # (T*N, action_dim)
    log_probs: torch.Tensor  # (T*N,)
    advantages: torch.Tensor  # (T*N,)
    returns: torch.Tensor  # (T*N,)
    values: torch.Tensor  # (T*N,)

    def __len__(self) -> int:
        return self.observations.shape[0]


class RolloutBuffer:
    """Fixed (steps x environments) storage for one PPO iteration."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        shape = (num_steps, num_envs)

        self.observations = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.terminated = np.zeros(shape, dtype=bool)
        self.truncated = np.zeros(shape, dtype=bool)
        # Critic value at the observation a truncated episode ended on. Zero
        # everywhere else and never read there.
        self.bootstrap_values = np.zeros(shape, dtype=np.float32)

        self.step = 0

    def reset(self) -> None:
        self.step = 0

    @property
    def full(self) -> bool:
        return self.step >= self.num_steps

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        bootstrap_values: np.ndarray | None = None,
    ) -> None:
        if self.full:
            raise RuntimeError("rollout buffer is full; call compute_returns then reset")
        i = self.step
        self.observations[i] = observations
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.values[i] = values
        self.rewards[i] = rewards
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.bootstrap_values[i] = 0.0 if bootstrap_values is None else bootstrap_values
        self.step += 1

    def compute_advantages(
        self, last_values: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (advantages, returns), each shaped (steps, environments)."""
        advantages = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]

            # What the critic should look at for the state after this one.
            #   terminated -> nothing follows, so zero
            #   truncated  -> the state the episode was cut off in
            #   otherwise  -> the next stored value
            bootstrap = np.where(
                self.terminated[t],
                0.0,
                np.where(self.truncated[t], self.bootstrap_values[t], next_values),
            ).astype(np.float32)

            delta = self.rewards[t] + gamma * bootstrap - self.values[t]

            # The recursion breaks at *any* episode end, truncation included:
            # the advantage at index t+1 belongs to a different episode.
            episode_continues = ~(self.terminated[t] | self.truncated[t])
            last_gae = delta + gamma * gae_lambda * episode_continues * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns

    def to_batch(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        device: str | torch.device = "cpu",
    ) -> RolloutBatch:
        def flat(array: np.ndarray) -> torch.Tensor:
            reshaped = array.reshape(self.num_steps * self.num_envs, *array.shape[2:])
            return torch.as_tensor(reshaped, dtype=torch.float32, device=device)

        return RolloutBatch(
            observations=flat(self.observations),
            actions=flat(self.actions),
            log_probs=flat(self.log_probs),
            advantages=flat(advantages),
            returns=flat(returns),
            values=flat(self.values),
        )


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """1 - Var(target - prediction) / Var(target).

    The single most useful diagnostic for whether the critic is learning
    anything: 0 means it is no better than predicting the mean, and negative
    means it is actively worse. A policy cannot improve past a critic stuck near
    zero, so this is where a stalled run is diagnosed first.
    """
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    variance = np.var(targets)
    if variance < 1e-12:
        return float("nan")
    return float(1.0 - np.var(targets - predictions) / variance)
