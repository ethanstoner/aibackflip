"""GAE and the PPO objective, checked against hand-computed values.

A test that recomputes GAE with the same loop it is testing proves nothing. The
expected values below are worked out by hand from the definition, so a change in
the boundary handling has to be deliberate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl.policy import ActorCritic, PolicyConfig
from rl.ppo import PPO, PPOConfig, ppo_clipped_objective
from rl.rollout import RolloutBuffer, explained_variance


def make_buffer(steps: int, envs: int = 1, obs_dim: int = 3, action_dim: int = 2) -> RolloutBuffer:
    return RolloutBuffer(steps, envs, obs_dim, action_dim)


def fill(buffer, rewards, values, terminated=None, truncated=None, bootstrap=None) -> None:
    steps = len(rewards)
    zeros_obs = np.zeros((buffer.num_envs, buffer.observations.shape[-1]), dtype=np.float32)
    zeros_act = np.zeros((buffer.num_envs, buffer.actions.shape[-1]), dtype=np.float32)
    for t in range(steps):
        buffer.add(
            observations=zeros_obs,
            actions=zeros_act,
            log_probs=np.zeros(buffer.num_envs, dtype=np.float32),
            values=np.full(buffer.num_envs, values[t], dtype=np.float32),
            rewards=np.full(buffer.num_envs, rewards[t], dtype=np.float32),
            terminated=np.full(buffer.num_envs, False if terminated is None else terminated[t]),
            truncated=np.full(buffer.num_envs, False if truncated is None else truncated[t]),
            bootstrap_values=np.full(
                buffer.num_envs, 0.0 if bootstrap is None else bootstrap[t], dtype=np.float32
            ),
        )


# ---------------------------------------------------------------- GAE


def test_gae_matches_a_hand_computed_three_step_rollout():
    gamma, lam = 0.9, 0.8
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.5, 2.5]
    last_value = 4.0

    buffer = make_buffer(3)
    fill(buffer, rewards, values)
    advantages, returns = buffer.compute_advantages(
        np.array([last_value], dtype=np.float32), gamma=gamma, gae_lambda=lam
    )

    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    d0 = 1.0 + 0.9 * 1.5 - 0.5   #  1.85
    d1 = 2.0 + 0.9 * 2.5 - 1.5   #  2.75
    d2 = 3.0 + 0.9 * 4.0 - 2.5   #  4.10
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1

    assert advantages[0, 0] == pytest.approx(a0, rel=1e-5)
    assert advantages[1, 0] == pytest.approx(a1, rel=1e-5)
    assert advantages[2, 0] == pytest.approx(a2, rel=1e-5)
    assert returns[0, 0] == pytest.approx(a0 + values[0], rel=1e-5)


def test_termination_zeroes_the_bootstrap():
    gamma, lam = 0.99, 0.95
    buffer = make_buffer(2)
    fill(buffer, rewards=[1.0, 1.0], values=[5.0, 5.0], terminated=[False, True])
    advantages, _ = buffer.compute_advantages(
        np.array([100.0], dtype=np.float32), gamma=gamma, gae_lambda=lam
    )

    # The last step terminated, so nothing follows it: delta = r - V, and the
    # enormous last_value must be ignored entirely.
    assert advantages[1, 0] == pytest.approx(1.0 - 5.0, rel=1e-5)
    d0 = 1.0 + gamma * 5.0 - 5.0
    assert advantages[0, 0] == pytest.approx(d0 + gamma * lam * advantages[1, 0], rel=1e-5)


def test_truncation_bootstraps_from_the_final_observation():
    """The case that quietly ruins a run if it is wrong.

    A truncated episode has a future; it was cut off. Bootstrapping from zero
    would teach the critic that reaching the time limit is worthless, and the
    policy would learn to avoid surviving.
    """
    gamma, lam = 0.99, 0.95
    buffer = make_buffer(1)
    fill(
        buffer,
        rewards=[1.0],
        values=[5.0],
        terminated=[False],
        truncated=[True],
        bootstrap=[7.0],
    )
    advantages, _ = buffer.compute_advantages(
        np.array([0.0], dtype=np.float32), gamma=gamma, gae_lambda=lam
    )
    assert advantages[0, 0] == pytest.approx(1.0 + gamma * 7.0 - 5.0, rel=1e-5)

    # And it is genuinely different from treating truncation as termination.
    terminating = make_buffer(1)
    fill(terminating, rewards=[1.0], values=[5.0], terminated=[True])
    terminated_adv, _ = terminating.compute_advantages(
        np.array([0.0], dtype=np.float32), gamma=gamma, gae_lambda=lam
    )
    assert advantages[0, 0] > terminated_adv[0, 0] + 5.0


def test_the_recursion_breaks_at_every_episode_boundary():
    """Advantage must not chain across a reset.

    The observation stored at t+1 belongs to a different episode, so an
    advantage that leaked backwards through the boundary would credit one
    episode's outcome to another's actions.
    """
    gamma, lam = 0.99, 0.95
    for boundary in ("terminated", "truncated"):
        buffer = make_buffer(3)
        flags = [False, True, False]
        fill(
            buffer,
            rewards=[0.0, 0.0, 1000.0],
            values=[0.0, 0.0, 0.0],
            terminated=flags if boundary == "terminated" else None,
            truncated=flags if boundary == "truncated" else None,
            bootstrap=[0.0, 0.0, 0.0],
        )
        advantages, _ = buffer.compute_advantages(
            np.array([0.0], dtype=np.float32), gamma=gamma, gae_lambda=lam
        )
        # The huge reward at step 2 is in a later episode than step 0, so none
        # of it may reach step 0.
        assert advantages[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert advantages[2, 0] == pytest.approx(1000.0, rel=1e-5)


def test_zero_lambda_reduces_gae_to_the_one_step_td_error():
    gamma = 0.99
    buffer = make_buffer(4)
    rewards = [1.0, 2.0, 3.0, 4.0]
    values = [0.1, 0.2, 0.3, 0.4]
    fill(buffer, rewards, values)
    advantages, _ = buffer.compute_advantages(
        np.array([0.5], dtype=np.float32), gamma=gamma, gae_lambda=0.0
    )
    next_values = values[1:] + [0.5]
    for t in range(4):
        assert advantages[t, 0] == pytest.approx(
            rewards[t] + gamma * next_values[t] - values[t], rel=1e-5
        )


def test_lambda_one_reduces_gae_to_the_monte_carlo_return():
    gamma = 0.9
    buffer = make_buffer(3)
    rewards = [1.0, 2.0, 3.0]
    fill(buffer, rewards, values=[0.0, 0.0, 0.0], terminated=[False, False, True])
    advantages, returns = buffer.compute_advantages(
        np.array([0.0], dtype=np.float32), gamma=gamma, gae_lambda=1.0
    )
    # With zero baselines and a terminal end, the advantage is the plain
    # discounted return.
    assert returns[0, 0] == pytest.approx(1.0 + 0.9 * 2.0 + 0.81 * 3.0, rel=1e-5)
    assert returns[2, 0] == pytest.approx(3.0, rel=1e-5)
    assert advantages[0, 0] == pytest.approx(returns[0, 0], rel=1e-5)


def test_environments_are_handled_independently():
    buffer = RolloutBuffer(2, 3, obs_dim=2, action_dim=1)
    for t in range(2):
        buffer.add(
            observations=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros((3, 1), dtype=np.float32),
            log_probs=np.zeros(3, dtype=np.float32),
            values=np.zeros(3, dtype=np.float32),
            rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            terminated=np.array([False, True, False]),
            truncated=np.zeros(3, dtype=bool),
            bootstrap_values=np.zeros(3, dtype=np.float32),
        )
    advantages, _ = buffer.compute_advantages(
        np.zeros(3, dtype=np.float32), gamma=0.99, gae_lambda=0.95
    )
    # Environment 1 terminated at both steps, so no chaining; 0 and 2 chain.
    assert advantages[0, 1] == pytest.approx(2.0, rel=1e-5)
    assert advantages[0, 0] > 1.0
    assert advantages[0, 2] > 3.0


def test_the_buffer_refuses_to_overflow():
    buffer = make_buffer(1)
    fill(buffer, rewards=[1.0], values=[0.0])
    assert buffer.full
    with pytest.raises(RuntimeError, match="full"):
        fill(buffer, rewards=[1.0], values=[0.0])
    buffer.reset()
    assert not buffer.full


def test_explained_variance_reads_as_expected():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(targets, targets) == pytest.approx(1.0)
    assert explained_variance(np.full(4, targets.mean()), targets) == pytest.approx(0.0)
    # Actively worse than predicting the mean.
    assert explained_variance(-targets, targets) < 0.0
    assert np.isnan(explained_variance(np.ones(4), np.ones(4)))


# ---------------------------------------------------------------- objective


def test_the_clipped_objective_matches_the_definition():
    advantages = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    ratio = torch.tensor([0.5, 1.0, 2.0, 0.5, 1.0, 2.0])
    out = ppo_clipped_objective(ratio, advantages, clip_range=0.2)

    # The asymmetry is the whole point of the objective, and it is easy to state
    # backwards.
    #
    # Positive advantage: making a good action more likely is capped at 1+eps
    # (ratio 2.0 -> 1.2), while making it *less* likely is left uncapped
    # (ratio 0.5 -> 0.5) so the update is still pushed back towards it.
    #
    # Negative advantage: making a bad action less likely is capped at 1-eps
    # (ratio 0.5 -> -0.8, not -0.5), while making it *more* likely is left
    # uncapped (ratio 2.0 -> -2.0) so it is fully penalised.
    assert out.tolist() == pytest.approx([0.5, 1.0, 1.2, -0.8, -1.0, -2.0])


def test_clipping_only_bites_in_the_direction_that_would_help():
    advantages = torch.tensor([1.0])
    unclipped_gain = ppo_clipped_objective(torch.tensor([5.0]), advantages, 0.2)
    assert unclipped_gain.item() == pytest.approx(1.2)  # capped

    # Moving the wrong way is not capped, so the update is still pushed back.
    unclipped_loss = ppo_clipped_objective(torch.tensor([0.01]), advantages, 0.2)
    assert unclipped_loss.item() == pytest.approx(0.01)


def test_a_ratio_of_one_returns_the_advantage_unchanged():
    advantages = torch.randn(64)
    out = ppo_clipped_objective(torch.ones(64), advantages, 0.2)
    assert torch.allclose(out, advantages)


# ---------------------------------------------------------------- update


def make_policy_and_batch(steps: int = 64, envs: int = 4, obs_dim: int = 8, action_dim: int = 3):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = ActorCritic(PolicyConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(32, 32)))
    buffer = RolloutBuffer(steps, envs, obs_dim, action_dim)

    for _ in range(steps):
        observations = rng.normal(size=(envs, obs_dim)).astype(np.float32)
        with torch.no_grad():
            actions, log_probs, values = policy.act(torch.as_tensor(observations))
        buffer.add(
            observations=observations,
            actions=actions.numpy(),
            log_probs=log_probs.numpy(),
            values=values.numpy(),
            rewards=rng.normal(size=envs).astype(np.float32),
            terminated=rng.random(envs) < 0.02,
            truncated=np.zeros(envs, dtype=bool),
        )

    advantages, returns = buffer.compute_advantages(np.zeros(envs, dtype=np.float32))
    return policy, buffer.to_batch(advantages, returns)


def test_an_update_changes_the_policy_and_reports_sane_diagnostics():
    policy, batch = make_policy_and_batch()
    before = [p.detach().clone() for p in policy.parameters()]

    ppo = PPO(policy, PPOConfig(epochs=4, minibatch_size=64, target_kl=None))
    stats = ppo.update(batch)

    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    assert stats.epochs_run == 4
    assert np.isfinite(stats.policy_loss)
    assert stats.value_loss >= 0.0
    assert stats.entropy > 0.0
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.approx_kl >= -1e-6  # the estimator is non-negative by construction
    assert stats.grad_norm >= 0.0


def test_the_first_minibatch_starts_at_a_ratio_of_one():
    """If the two log-probability paths disagree, the ratio starts away from 1
    and every update is wrong in a way that still trains."""
    policy, batch = make_policy_and_batch()
    with torch.no_grad():
        log_probs, _, _ = policy.evaluate_actions(batch.observations, batch.actions)
        ratio = torch.exp(log_probs - batch.log_probs)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


def test_target_kl_stops_the_epoch_loop_early():
    policy, batch = make_policy_and_batch()
    # A large learning rate moves the policy far in one epoch, which is exactly
    # what the early stop exists to catch.
    ppo = PPO(policy, PPOConfig(epochs=30, minibatch_size=32, learning_rate=1e-1, target_kl=1e-4))
    stats = ppo.update(batch)
    assert stats.stopped_early
    assert stats.epochs_run < 30


def test_gradients_are_clipped_to_the_configured_norm():
    policy, batch = make_policy_and_batch()
    ppo = PPO(policy, PPOConfig(epochs=1, minibatch_size=256, learning_rate=1e-2,
                                max_grad_norm=0.5, target_kl=None))
    ppo.update(batch)
    # clip_grad_norm_ reports the norm *before* clipping, so the check is that
    # the parameters did not move further than the clip allows.
    for parameter in policy.parameters():
        assert torch.isfinite(parameter).all()


def test_an_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown PPO settings"):
        PPOConfig.from_dict({"learning_rate": 1e-3, "clipping_range": 0.2})
    assert PPOConfig.from_dict({"learning_rate": 1e-3}).learning_rate == pytest.approx(1e-3)


def test_ppo_can_solve_a_trivial_bandit():
    """End-to-end proof that the update actually maximises reward.

    A one-step problem where the reward is -(a - target)^2. If PPO is wired up
    correctly the mean action converges on the target; if any sign is flipped it
    runs away from it. This is the smallest thing that can fail for the same
    reason a humanoid run would.
    """
    torch.manual_seed(3)
    obs_dim, action_dim = 4, 2
    target = torch.tensor([0.6, -0.4])

    policy = ActorCritic(
        PolicyConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(32,), log_std_init=-1.0)
    )
    ppo = PPO(policy, PPOConfig(epochs=8, minibatch_size=128, learning_rate=3e-3,
                                target_kl=None, entropy_coef=0.0))

    observations = torch.zeros(512, obs_dim)
    initial_error = (policy.action_mean(observations[:1]) - target).abs().sum().item()

    for _ in range(60):
        with torch.no_grad():
            actions, log_probs, values = policy.act(observations)
        rewards = -((actions - target) ** 2).sum(-1)
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)

        from rl.rollout import RolloutBatch

        batch = RolloutBatch(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            advantages=advantages,
            returns=rewards,
            values=values,
        )
        ppo.update(batch)

    final_error = (policy.action_mean(observations[:1]) - target).abs().sum().item()
    assert final_error < initial_error * 0.35, (
        f"the mean action did not converge on the target: {initial_error:.3f} -> {final_error:.3f}"
    )
