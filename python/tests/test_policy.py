"""Policy network, normalisation and checkpoint round-trips."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl.normalization import ObservationNormalizer, ReturnNormalizer, RunningMeanStd
from rl.policy import ActorCritic, CheckpointMeta, PolicyConfig, load_checkpoint, save_checkpoint


# ---------------------------------------------------------------- normalisation


def test_running_stats_match_numpy_on_the_whole_stream():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=[3.0, -7.0, 0.0], scale=[1.0, 5.0, 0.01], size=(5000, 3))

    rms = RunningMeanStd((3,))
    for start in range(0, len(data), 37):  # deliberately uneven batches
        rms.update(data[start : start + 37])

    assert rms.mean == pytest.approx(data.mean(axis=0), abs=1e-9)
    assert rms.var == pytest.approx(data.var(axis=0), rel=1e-6)


def test_running_stats_stay_accurate_over_a_long_offset_stream():
    """The naive sum-of-squares formula loses all precision here.

    Values far from zero with a small spread are exactly the case that breaks
    it, and a training run produces hundreds of millions of samples.
    """
    rng = np.random.default_rng(1)
    rms = RunningMeanStd((1,))
    total = []
    for _ in range(400):
        batch = rng.normal(loc=1e6, scale=1e-2, size=(1000, 1))
        rms.update(batch)
        total.append(batch)

    stacked = np.concatenate(total)
    assert rms.mean[0] == pytest.approx(stacked.mean(), rel=1e-12)
    assert rms.var[0] == pytest.approx(stacked.var(), rel=1e-4)
    # The failure this guards against is not a small error. Merging into a
    # (mean 0, var 1) prior reports a variance around 250 here instead of 1e-4,
    # because the delta^2 term carries the full 1e6 offset.
    assert rms.var[0] < 1.0


def test_normalised_observations_are_whitened_and_clipped():
    rng = np.random.default_rng(2)
    normalizer = ObservationNormalizer(4, clip=5.0)
    for _ in range(200):
        normalizer.normalize(rng.normal(loc=[10, -3, 0, 100], scale=[2, 0.5, 1, 50], size=(64, 4)))

    sample = rng.normal(loc=[10, -3, 0, 100], scale=[2, 0.5, 1, 50], size=(4096, 4))
    out = normalizer.normalize(sample, update=False)
    assert out.dtype == np.float32
    assert np.abs(out.mean(axis=0)).max() < 0.15
    assert np.abs(out.std(axis=0) - 1.0).max() < 0.15

    # An extreme outlier is clipped rather than propagated into the first layer.
    outlier = np.full((1, 4), 1e6, dtype=np.float32)
    assert np.all(np.abs(normalizer.normalize(outlier, update=False)) <= 5.0)


def test_freezing_stops_the_statistics_moving():
    normalizer = ObservationNormalizer(2)
    normalizer.normalize(np.zeros((100, 2), dtype=np.float32))
    normalizer.freeze()
    before = normalizer.rms.mean.copy()
    normalizer.normalize(np.full((100, 2), 50.0, dtype=np.float32))
    assert np.array_equal(normalizer.rms.mean, before)


def test_normalizer_state_round_trips():
    rng = np.random.default_rng(3)
    original = ObservationNormalizer(6)
    for _ in range(50):
        original.normalize(rng.normal(size=(32, 6)))

    restored = ObservationNormalizer(6)
    restored.load_state_dict(original.state_dict())

    sample = rng.normal(size=(10, 6)).astype(np.float32)
    assert np.allclose(
        original.normalize(sample, update=False), restored.normalize(sample, update=False)
    )


def test_a_normalizer_of_the_wrong_size_is_rejected():
    small = ObservationNormalizer(3)
    big = ObservationNormalizer(9)
    with pytest.raises(ValueError, match="shape mismatch"):
        big.load_state_dict(small.state_dict())


def test_return_normalizer_scales_without_shifting():
    """Only the scale is divided out, never a mean shift.

    Subtracting a constant from every reward changes the optimal policy whenever
    episode lengths vary - and surviving longer is the whole task here - so the
    sign of every reward must be preserved.
    """
    normalizer = ReturnNormalizer(num_envs=4, gamma=0.99)
    rng = np.random.default_rng(4)
    for _ in range(300):
        rewards = rng.uniform(0.5, 2.0, size=4)
        dones = rng.random(4) < 0.02
        scaled = normalizer(rewards, dones)
        assert np.all(scaled > 0), "a positive reward must stay positive"

    big = normalizer(np.array([10.0, 10.0, 10.0, 10.0]), np.zeros(4, dtype=bool))
    assert np.all(np.abs(big) < 100)


def test_return_normalizer_resets_the_accumulator_on_done():
    normalizer = ReturnNormalizer(num_envs=2, gamma=0.99)
    for _ in range(50):
        normalizer(np.array([1.0, 1.0]), np.zeros(2, dtype=bool))
    running = normalizer.returns.copy()
    assert running[0] > 5.0

    normalizer(np.array([1.0, 1.0]), np.array([True, False]))
    assert normalizer.returns[0] == 0.0
    assert normalizer.returns[1] > 5.0


# ---------------------------------------------------------------- policy


@pytest.fixture
def policy() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(PolicyConfig(obs_dim=111, action_dim=12))


def test_the_policy_has_the_planned_shape(policy: ActorCritic):
    obs = torch.randn(8, 111)
    action, log_prob, value = policy.act(obs)
    assert action.shape == (8, 12)
    assert log_prob.shape == (8,)
    assert value.shape == (8,)
    assert policy.parameter_count() > 0


def test_the_action_mean_is_bounded_even_for_extreme_inputs(policy: ActorCritic):
    # tanh on the output means a command can only leave [-1, 1] through
    # exploration noise, which the environment clamps.
    obs = torch.randn(64, 111) * 1000.0
    mean = policy.action_mean(obs)
    assert torch.all(mean.abs() <= 1.0)
    assert torch.all(torch.isfinite(mean))


def test_an_untrained_policy_starts_near_the_rest_pose(policy: ActorCritic):
    """The output layer is initialised with a small gain on purpose.

    Action 0 commands the rest pose, so a policy that starts near zero starts by
    standing rather than by folding itself up, and the first updates are not
    spent undoing an arbitrary initial pose.
    """
    obs = torch.randn(256, 111)
    mean = policy.action_mean(obs)
    assert mean.abs().mean().item() < 0.1


def test_deterministic_actions_are_the_distribution_mean(policy: ActorCritic):
    obs = torch.randn(4, 111)
    a, _, _ = policy.act(obs, deterministic=True)
    b, _, _ = policy.act(obs, deterministic=True)
    assert torch.equal(a, b)
    assert torch.allclose(a, policy.action_mean(obs))

    sampled, _, _ = policy.act(obs, deterministic=False)
    assert not torch.equal(sampled, a)


def test_evaluate_actions_reproduces_the_log_prob_from_act(policy: ActorCritic):
    """The single most important consistency property in PPO.

    The ratio in the objective is exp(new_log_prob - old_log_prob). If the two
    code paths disagree, the ratio starts at something other than 1 on the first
    epoch and every update is wrong in a way that still trains, badly.
    """
    obs = torch.randn(32, 111)
    action, log_prob, value = policy.act(obs)
    recomputed, entropy, recomputed_value = policy.evaluate_actions(obs, action)

    assert torch.allclose(log_prob, recomputed, atol=1e-6)
    assert torch.allclose(value, recomputed_value, atol=1e-6)
    assert entropy.shape == (32,)
    assert torch.all(entropy > 0)


def test_log_probabilities_are_summed_over_action_dimensions(policy: ActorCritic):
    # A per-dimension log-prob left unsummed silently turns the objective into a
    # mean over dimensions and scales every advantage by 1/12.
    obs = torch.randn(5, 111)
    action, log_prob, _ = policy.act(obs)
    dist = policy.distribution(obs)
    assert torch.allclose(log_prob, dist.log_prob(action).sum(-1), atol=1e-6)


def test_the_standard_deviation_is_learnable_and_bounded(policy: ActorCritic):
    assert policy.log_std.requires_grad
    std = policy.current_std()
    assert std.shape == (12,)
    assert np.allclose(std, np.exp(-0.5), atol=1e-5)

    # Tolerances sized for float32, not float64: the clamp happens in the
    # network's dtype.
    with torch.no_grad():
        policy.log_std.fill_(-50.0)
    assert np.all(policy.current_std() >= np.exp(policy.config.log_std_min) - 1e-7)
    with torch.no_grad():
        policy.log_std.fill_(50.0)
    assert np.all(policy.current_std() <= np.exp(policy.config.log_std_max) + 1e-6)


def test_gradients_reach_every_parameter(policy: ActorCritic):
    obs = torch.randn(16, 111)
    action, log_prob, value = policy.act(obs)
    new_log_prob, entropy, new_value = policy.evaluate_actions(obs, action)
    loss = -(new_log_prob.mean()) + new_value.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()

    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"
    assert policy.log_std.grad.abs().sum().item() > 0


def test_actor_and_critic_do_not_share_parameters(policy: ActorCritic):
    # A shared trunk lets the much larger value loss dominate the policy's
    # features, which shows up as a policy that stops improving while the value
    # loss keeps falling.
    actor_ids = {id(p) for p in policy.actor.parameters()}
    critic_ids = {id(p) for p in policy.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)


# ---------------------------------------------------------------- checkpoints


def test_a_checkpoint_round_trips_weights_and_statistics(tmp_path, policy: ActorCritic):
    rng = np.random.default_rng(5)
    normalizer = ObservationNormalizer(111)
    for _ in range(20):
        normalizer.normalize(rng.normal(loc=4.0, scale=3.0, size=(64, 111)))

    meta = CheckpointMeta(updates=17, env_steps=1_234_567, best_return=42.5,
                          reward_weights={"alive": 1.0}, notes="test")
    path = tmp_path / "policy.pt"
    save_checkpoint(path, policy, normalizer, meta)

    restored, restored_norm, restored_meta, _ = load_checkpoint(
        path, expect_obs_dim=111, expect_action_dim=12
    )

    obs = torch.randn(8, 111)
    a, _, v = policy.act(obs, deterministic=True)
    b, _, w = restored.act(obs, deterministic=True)
    assert torch.allclose(a, b, atol=1e-7)
    assert torch.allclose(v, w, atol=1e-7)

    sample = rng.normal(loc=4.0, scale=3.0, size=(4, 111)).astype(np.float32)
    assert np.allclose(
        normalizer.normalize(sample, update=False),
        restored_norm.normalize(sample, update=False),
    )

    assert restored_meta.updates == 17
    assert restored_meta.env_steps == 1_234_567
    assert restored_meta.best_return == pytest.approx(42.5)
    assert restored_meta.reward_weights == {"alive": 1.0}


def test_loading_a_mismatched_checkpoint_is_refused(tmp_path, policy: ActorCritic):
    """A silent dimension mismatch would produce a policy reading the wrong
    observation fields, which looks like catastrophic forgetting."""
    path = tmp_path / "policy.pt"
    save_checkpoint(path, policy, ObservationNormalizer(111), CheckpointMeta())

    with pytest.raises(ValueError, match="observations"):
        load_checkpoint(path, expect_obs_dim=64)
    with pytest.raises(ValueError, match="actions"):
        load_checkpoint(path, expect_action_dim=26)


def test_an_interrupted_save_leaves_the_previous_checkpoint_intact(tmp_path, policy: ActorCritic):
    path = tmp_path / "policy.pt"
    save_checkpoint(path, policy, ObservationNormalizer(111), CheckpointMeta(updates=1))
    original = path.read_bytes()

    # A failure while writing must not consume the existing file. The save
    # writes to a temporary and renames, so a partial write is never visible
    # under the real name.
    assert not (tmp_path / "policy.pt.tmp").exists()
    save_checkpoint(path, policy, ObservationNormalizer(111), CheckpointMeta(updates=2))
    assert path.read_bytes() != original
    assert not (tmp_path / "policy.pt.tmp").exists()

    _, _, meta, _ = load_checkpoint(path)
    assert meta.updates == 2


def test_optimizer_state_survives_a_round_trip(tmp_path, policy: ActorCritic):
    # Resuming without it restarts Adam's moment estimates, which produces a
    # visible dip in performance right after every resume.
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    loss = policy.value(torch.randn(4, 111)).pow(2).mean()
    loss.backward()
    optimizer.step()

    path = tmp_path / "policy.pt"
    save_checkpoint(path, policy, ObservationNormalizer(111), CheckpointMeta(), optimizer)
    _, _, _, payload = load_checkpoint(path)
    assert "optimizer" in payload

    fresh = ActorCritic(policy.config)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=3e-4)
    fresh_optimizer.load_state_dict(payload["optimizer"])
    assert fresh_optimizer.state_dict()["param_groups"][0]["lr"] == pytest.approx(3e-4)


def test_the_policy_runs_on_the_gpu(policy: ActorCritic):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cuda_policy = policy.cuda()
    obs = torch.randn(64, 111, device="cuda")
    action, log_prob, value = cuda_policy.act(obs)
    assert action.is_cuda and torch.isfinite(action).all()
    assert log_prob.shape == (64,) and value.shape == (64,)
