"""Running normalisation for observations and returns.

Observation normalisation is not optional here. The observation mixes
dimensionless ratios near 1 with angular velocities that reach tens of radians
per second; fed raw to a tanh network, the small channels are invisible and the
large ones saturate the first layer.

The statistics are part of the policy, not of the training loop: a checkpoint
loaded without them sees a completely different input distribution and behaves
like an untrained network. They are saved and restored together.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Streaming mean and variance using Chan's parallel algorithm.

    Batch-at-a-time rather than element-at-a-time, and numerically stable for the
    hundreds of millions of samples a long training run produces - the naive
    "sum of squares minus square of sum" loses all precision well before that.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        # Unit variance until real data arrives, so normalising before the first
        # update divides by one rather than by zero.
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0.0

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.shape[0] == 0:
            return
        self.update_from_moments(batch.mean(axis=0), batch.var(axis=0), batch.shape[0])

    def update_from_moments(self, mean: np.ndarray, var: np.ndarray, count: int) -> None:
        # The first batch defines the statistics outright rather than being
        # merged into a (mean 0, var 1) prior.
        #
        # A prior looks harmless and is not. Merging carries a
        # delta^2 * (n_a*n_b/total) term, and when the data sits far from zero
        # that delta is the full offset: a stream centred on 1e6 with a spread
        # of 1e-2 ends up reporting a variance of ~250 instead of ~1e-4, and
        # every observation is then scaled to nothing. The pseudo-count exists
        # only to avoid dividing by zero, which starting from an empty state
        # handles directly.
        if self.count == 0.0:
            self.mean = np.array(mean, dtype=np.float64, copy=True)
            self.var = np.array(var, dtype=np.float64, copy=True)
            self.count = float(count)
            return

        delta = mean - self.mean
        total = self.count + count

        self.mean = self.mean + delta * (count / total)
        m_a = self.var * self.count
        m_b = var * count
        m2 = m_a + m_b + np.square(delta) * (self.count * count / total)
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64)
        if mean.shape != self.mean.shape:
            raise ValueError(
                f"normaliser shape mismatch: checkpoint has {mean.shape}, "
                f"this policy expects {self.mean.shape}"
            )
        self.mean = mean
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ObservationNormalizer:
    """Whitens observations, with clipping and a freeze switch."""

    def __init__(self, obs_dim: int, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.rms = RunningMeanStd((obs_dim,))
        self.clip = clip
        self.epsilon = epsilon
        # Evaluation and playback must not keep updating the statistics, or a
        # long test run slowly changes the policy's inputs.
        self.frozen = False

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        return self.normalize(obs, update=update)

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if update and not self.frozen:
            self.rms.update(obs)
        normalized = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        # Clipping matters on the first few batches, when the variance estimate
        # is still poor and an outlier can produce a value in the hundreds.
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def freeze(self) -> None:
        self.frozen = True

    def state_dict(self) -> dict:
        return {"rms": self.rms.state_dict(), "clip": self.clip, "epsilon": self.epsilon}

    def load_state_dict(self, state: dict) -> None:
        self.rms.load_state_dict(state["rms"])
        self.clip = float(state.get("clip", self.clip))
        self.epsilon = float(state.get("epsilon", self.epsilon))


class ReturnNormalizer:
    """Scales rewards by the standard deviation of the discounted return.

    Not a mean shift: subtracting a constant from every reward changes the
    optimal policy whenever episode lengths vary, and here they vary a lot -
    surviving longer is the entire task. Only the scale is divided out, which
    leaves the ordering of policies untouched while keeping the value targets in
    a range the critic can fit.
    """

    def __init__(self, num_envs: int, gamma: float = 0.99, clip: float = 10.0,
                 epsilon: float = 1e-8) -> None:
        # Shape (1,) rather than a true scalar: every return from every
        # environment is a sample of the same distribution, so they are stacked
        # into one column and the resulting variance broadcasts over the batch.
        self.rms = RunningMeanStd((1,))
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.frozen = False

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float64)
        self.returns = self.returns * self.gamma + rewards
        if not self.frozen:
            self.rms.update(self.returns.reshape(-1, 1))
        scaled = rewards / np.sqrt(self.rms.var[0] + self.epsilon)
        # The accumulator has to reset where an episode ended, or returns from
        # separate episodes are chained into one runaway sum.
        self.returns[np.asarray(dones, dtype=bool)] = 0.0
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"rms": self.rms.state_dict(), "gamma": self.gamma, "clip": self.clip}

    def load_state_dict(self, state: dict) -> None:
        self.rms.load_state_dict(state["rms"])
        self.gamma = float(state.get("gamma", self.gamma))
        self.clip = float(state.get("clip", self.clip))
