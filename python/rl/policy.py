"""Actor-critic network and checkpointing.

Shape follows the plan: observation -> Linear -> Tanh -> Linear -> Tanh ->
Linear -> actions, with a matching critic. Two details are load-bearing and
worth stating rather than leaving implicit.

*Separate trunks.* The actor and critic do not share features. A shared trunk
saves parameters, but the value loss is typically an order of magnitude larger
than the policy loss and will dominate the shared layers' gradients - which
shows up as a policy that stops improving while the value loss keeps falling.

*A state-independent log standard deviation.* The spread of the action
distribution is a learned parameter rather than a network output. Letting the
network emit it invites collapsing the variance to nothing on whatever states
happen to look good early, and exploration never recovers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PolicyConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_init: float = -0.5
    # Bounds on the learned spread. The lower bound stops the distribution
    # collapsing to a point (after which the ratio in the PPO objective explodes
    # and nothing is explored); the upper bound stops a diverging run from
    # emitting nothing but saturated actions.
    log_std_min: float = -3.0
    log_std_max: float = 1.0


def _layer(in_features: int, out_features: int, gain: float) -> nn.Linear:
    """Orthogonal initialisation, which keeps activations from shrinking or
    exploding through a deep tanh stack far better than the default uniform."""
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(obs_dim: int, hidden_sizes: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = obs_dim
    for size in hidden_sizes:
        layers.append(_layer(last, size, gain=np.sqrt(2)))
        layers.append(nn.Tanh())
        last = size
    # A small gain on the output layer means the policy starts close to the rest
    # pose and the critic starts close to zero, rather than emitting large
    # arbitrary values that the first few updates have to undo.
    layers.append(_layer(last, out_dim, gain=out_gain))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = _mlp(config.obs_dim, config.hidden_sizes, config.action_dim, out_gain=0.01)
        self.critic = _mlp(config.obs_dim, config.hidden_sizes, 1, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), config.log_std_init))

    # ------------------------------------------------------------------ forward

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        # tanh keeps the mean inside the action range by construction, so the
        # only way a command can leave it is through exploration noise, which
        # the environment clamps.
        return torch.tanh(self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.action_mean(obs)
        log_std = self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        return Normal(mean, log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Samples an action. Returns (action, log_prob, value).

        The log probability is of the *unclipped* sample. The environment clamps
        to [-1, 1] when applying it; recomputing the probability after clipping
        would need a truncated-distribution correction for a difference that is
        negligible while the mean is inside the range.
        """
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, self.value(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Returns (log_prob, entropy, value) for actions taken earlier."""
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, self.value(obs)

    # ------------------------------------------------------------------ misc

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def current_std(self) -> np.ndarray:
        with torch.no_grad():
            return (
                self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)
                .exp()
                .cpu()
                .numpy()
            )


@dataclass
class CheckpointMeta:
    """Everything needed to reconstruct and audit a saved policy."""

    updates: int = 0
    env_steps: int = 0
    wall_seconds: float = 0.0
    best_return: float = float("-inf")
    reward_weights: dict = field(default_factory=dict)
    notes: str = ""


def save_checkpoint(
    path: str | Path,
    policy: ActorCritic,
    obs_normalizer,
    meta: CheckpointMeta,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict | None = None,
) -> None:
    """Writes a policy, its observation statistics and its provenance together.

    The normaliser travels with the weights because a checkpoint restored
    without it sees a different input distribution and behaves like an untrained
    network - a failure that looks like the training run was worthless.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "policy_config": asdict(policy.config),
        "policy": policy.state_dict(),
        "obs_normalizer": obs_normalizer.state_dict() if obs_normalizer is not None else None,
        "meta": asdict(meta),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    # Written to a temporary file and moved into place, so a run interrupted
    # mid-save leaves the previous checkpoint intact instead of a truncated one.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    expect_obs_dim: int | None = None,
    expect_action_dim: int | None = None,
):
    """Loads a checkpoint. Returns (policy, obs_normalizer_state, meta, payload).

    Dimensions are checked against the live environment when they are supplied.
    A silent mismatch would produce a policy reading the wrong fields, which
    looks like catastrophic forgetting rather than a loading error.
    """
    from .normalization import ObservationNormalizer

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = PolicyConfig(**payload["policy_config"])

    if expect_obs_dim is not None and config.obs_dim != expect_obs_dim:
        raise ValueError(
            f"checkpoint was trained on {config.obs_dim} observations, "
            f"the environment provides {expect_obs_dim}"
        )
    if expect_action_dim is not None and config.action_dim != expect_action_dim:
        raise ValueError(
            f"checkpoint has {config.action_dim} actions, "
            f"the environment expects {expect_action_dim}"
        )

    policy = ActorCritic(config).to(device)
    policy.load_state_dict(payload["policy"])

    normalizer = None
    if payload.get("obs_normalizer") is not None:
        normalizer = ObservationNormalizer(config.obs_dim)
        normalizer.load_state_dict(payload["obs_normalizer"])

    meta = CheckpointMeta(**payload.get("meta", {}))
    return policy, normalizer, meta, payload
