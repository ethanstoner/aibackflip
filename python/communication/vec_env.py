"""Vectorised environment view over the UDP bridge.

Two things happen here that do not happen in the simulator:

* **Reward weighting.** The simulator sends raw, unweighted reward *terms* and
  this side turns them into a scalar. That keeps weights a config change rather
  than a rebuild, and keeps every component available for logging - which is the
  only practical way to spot reward hacking, since an exploit shows up as one
  term saturating while the others flatline.
* **Episode bookkeeping.** Returns and lengths are accumulated per environment
  so a finished episode can be reported once, complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .client import BridgeError, EnvClient
from .protocol import Spec, State


@dataclass
class RewardWeights:
    """Maps named reward terms onto a scalar.

    Terms are non-negative magnitudes; the sign lives here. Anything named
    ``*_cost`` is expected to carry a negative weight.
    """

    weights: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def standing() -> "RewardWeights":
        """Starting weights for the standing task.

        These are a starting point, not a tuned result. The per-component logs
        exist precisely so they can be argued with once training has run.
        """
        return RewardWeights(
            {
                "alive": 1.0,
                "pelvis_height": 1.0,
                "head_height": 0.5,
                "chest_upright": 1.0,
                "head_upright": 0.3,
                "com_over_support": 0.5,
                "foot_contact": 0.2,
                "horizontal_drift_cost": -0.3,
                "vertical_drift_cost": -0.1,
                "angular_drift_cost": -0.1,
                "action_cost": -0.2,
                "torque_cost": -0.1,
                "joint_limit_cost": -1.0,
            }
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "RewardWeights":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        weights = data.get("reward_weights", data)
        return cls({str(k): float(v) for k, v in weights.items()})

    def to_vector(self, names: list[str]) -> np.ndarray:
        """Aligns the weights to the server's term order.

        Unknown names are an error rather than a silent zero: a typo in a config
        would otherwise remove a reward term and be nearly impossible to spot in
        a training curve.
        """
        unknown = set(self.weights) - set(names)
        if unknown:
            raise ValueError(
                f"reward weights name terms the simulator does not provide: {sorted(unknown)}. "
                f"Available: {names}"
            )
        return np.array([self.weights.get(name, 0.0) for name in names], dtype=np.float32)


@dataclass
class EpisodeSummary:
    env_index: int
    ret: float
    length: int
    terminated: bool
    truncated: bool


class HumanoidVecEnv:
    """Batched environment with scalar rewards and episode accounting."""

    def __init__(
        self,
        client: EnvClient,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        if client.spec is None:
            raise BridgeError("client must be connected before wrapping it")
        self.client = client
        self.spec: Spec = client.spec
        self.reward_weights = reward_weights or RewardWeights.standing()
        self._weight_vector = self.reward_weights.to_vector(self.spec.reward_names)

        self.num_envs = self.spec.num_envs
        self.obs_dim = self.spec.obs_dim
        self.action_dim = self.spec.action_dim

        self._returns = np.zeros(self.num_envs, dtype=np.float64)
        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        # Running sum of each reward component over the current episode, so a
        # finished episode can report where its return came from.
        self._term_sums = np.zeros((self.num_envs, self.spec.reward_dim), dtype=np.float64)
        self.last_episode_terms: np.ndarray | None = None

    @property
    def reward_names(self) -> list[str]:
        return list(self.spec.reward_names)

    def reset(self, seed: int | None = None) -> np.ndarray:
        state = self.client.reset(seed)
        self._returns[:] = 0.0
        self._lengths[:] = 0
        self._term_sums[:] = 0.0
        return state.observations

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Applies one control step.

        Returns ``(observations, rewards, terminated, truncated, info)`` where
        ``observations`` is already the next episode's first observation for any
        environment that finished. ``info["final_observation"]`` holds the state
        those episodes ended in, which is what a truncated episode must
        bootstrap from.
        """
        state: State = self.client.step_batch(actions)

        rewards = (state.reward_terms @ self._weight_vector).astype(np.float32)
        self._returns += rewards
        self._lengths += 1
        self._term_sums += state.reward_terms

        episodes: list[EpisodeSummary] = []
        finished = np.flatnonzero(state.final_mask)
        for env_index in finished:
            episodes.append(
                EpisodeSummary(
                    env_index=int(env_index),
                    ret=float(self._returns[env_index]),
                    length=int(self._lengths[env_index]),
                    terminated=bool(state.terminated[env_index]),
                    truncated=bool(state.truncated[env_index]),
                )
            )
        if len(finished):
            self.last_episode_terms = self._term_sums[finished].mean(axis=0).copy()
            self._returns[finished] = 0.0
            self._lengths[finished] = 0
            self._term_sums[finished] = 0.0

        info = {
            "reward_terms": state.reward_terms,
            "episode_step": state.episode_step,
            "final_mask": state.final_mask,
            "final_observation": state.final_observations,
            "episodes": episodes,
            "step": state.step,
        }
        return state.observations, rewards, state.terminated, state.truncated, info

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "HumanoidVecEnv":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def connect(
    host: str = "127.0.0.1",
    port: int = 51234,
    num_envs: int = 25,
    seed: int = 0,
    reward_weights: RewardWeights | None = None,
    timeout: float = 1.0,
) -> HumanoidVecEnv:
    """Convenience: open a client, handshake, and wrap it."""
    client = EnvClient(host, port, timeout=timeout)
    client.connect(num_envs=num_envs, seed=seed)
    return HumanoidVecEnv(client, reward_weights)
