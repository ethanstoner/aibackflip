"""End-to-end tests against a real ``aibf_env`` process.

These are the ones that decide whether M3 works. Everything else tests a codec
or a data structure; this drives the actual simulator over an actual socket.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from communication.client import BridgeError, EnvClient
from communication.protocol import Header, MessageType, encode_action
from communication.vec_env import HumanoidVecEnv, RewardWeights

pytestmark = pytest.mark.needs_sim


def connect(server, num_envs: int | None = None, **kwargs) -> EnvClient:
    client = EnvClient("127.0.0.1", server.port, **kwargs)
    client.connect(num_envs=num_envs or server.num_envs)
    return client


# ---------------------------------------------------------------- handshake


def test_handshake_reports_a_self_consistent_spec(server):
    with connect(server) as client:
        spec = client.spec
        assert spec is not None
        assert spec.num_envs == server.num_envs
        assert spec.action_dim == 12
        assert spec.obs_dim > 50
        assert spec.reward_dim == len(spec.reward_names)
        assert spec.reward_dim >= 13
        assert spec.control_hz == pytest.approx(60.0)
        assert spec.physics_hz == pytest.approx(240.0)

        # The names are what make per-component logging possible, so they have
        # to be complete rather than merely present.
        assert len(spec.action_names) == spec.action_dim
        assert len(spec.reward_names) == spec.reward_dim
        assert len(spec.observation_names) == spec.obs_dim
        assert "hip_l" in spec.action_names
        assert "alive" in spec.reward_names
        assert "pelvis_height" in spec.observation_names

        # Joint limits arrive with the spec so nothing on this side hardcodes them.
        assert np.all(spec.action_low < spec.action_high)
        assert np.all(spec.action_low <= 0.0)
        assert np.all(spec.action_high >= 0.0)


def test_the_client_can_request_a_different_environment_count(server):
    with connect(server, num_envs=7) as client:
        assert client.spec is not None
        assert client.spec.num_envs == 7
        state = client.reset()
        assert state.observations.shape[0] == 7


# ---------------------------------------------------------------- stepping


def test_observations_and_rewards_arrive_with_the_right_shapes(server):
    with connect(server) as client:
        spec = client.spec
        state = client.reset()
        assert state.step == 0
        assert state.observations.shape == (spec.num_envs, spec.obs_dim)
        assert state.reward_terms.shape == (spec.num_envs, spec.reward_dim)
        assert np.all(np.isfinite(state.observations))
        assert not state.done.any()

        rng = np.random.default_rng(0)
        for expected_step in range(1, 21):
            actions = rng.uniform(-1, 1, size=(spec.num_envs, spec.action_dim)).astype(np.float32)
            state = client.step_batch(actions)
            assert state.step == expected_step
            assert state.observations.shape == (spec.num_envs, spec.obs_dim)
            assert np.all(np.isfinite(state.observations))
            assert np.all(np.isfinite(state.reward_terms))


def test_actions_actually_reach_the_simulator(server):
    """Two different action streams must produce two different trajectories.

    Without this, a bridge that silently dropped every action would still pass
    the shape and finiteness checks above.
    """

    def run(action_value: float) -> np.ndarray:
        with connect(server) as client:
            spec = client.spec
            client.reset(seed=1234)
            actions = np.full((spec.num_envs, spec.action_dim), action_value, dtype=np.float32)
            for _ in range(30):
                state = client.step_batch(actions)
            return state.observations.copy()

    held_neutral = run(0.0)
    driven_hard = run(0.9)
    assert not np.allclose(held_neutral, driven_hard, atol=1e-4)


def test_a_zero_action_keeps_the_figure_upright_far_longer_than_a_random_one(server):
    """A physical sanity check on the whole chain.

    Zero maps to the rest pose, so a zero policy should stand; a uniformly
    random policy should fall over. If the bridge were scrambling actions the
    two would be indistinguishable.
    """
    spec_envs = server.num_envs

    with connect(server) as client:
        spec = client.spec
        client.reset(seed=7)
        zeros = np.zeros((spec_envs, spec.action_dim), dtype=np.float32)
        zero_finished = 0
        for _ in range(200):
            state = client.step_batch(zeros)
            zero_finished += int(state.terminated.sum())

    with connect(server) as client:
        spec = client.spec
        client.reset(seed=7)
        rng = np.random.default_rng(3)
        random_finished = 0
        for _ in range(200):
            actions = rng.uniform(-1, 1, size=(spec_envs, spec.action_dim)).astype(np.float32)
            state = client.step_batch(actions)
            random_finished += int(state.terminated.sum())

    assert random_finished > zero_finished


def test_reset_returns_the_simulation_to_step_zero(server):
    with connect(server) as client:
        spec = client.spec
        client.reset()
        actions = np.zeros((spec.num_envs, spec.action_dim), dtype=np.float32)
        for _ in range(15):
            client.step_batch(actions)
        assert client.step == 15

        state = client.reset()
        assert state.step == 0
        assert client.step == 0
        assert np.all(state.episode_step == 0)


def test_seeded_resets_reproduce_exactly(server):
    with connect(server) as client:
        first = client.reset(seed=98765).observations.copy()
        client.step_batch(
            np.zeros((client.spec.num_envs, client.spec.action_dim), dtype=np.float32)
        )
        second = client.reset(seed=98765).observations.copy()
        assert np.array_equal(first, second)

        different = client.reset(seed=11111).observations.copy()
        assert not np.array_equal(first, different)


# ---------------------------------------------------------------- episodes


def test_finished_episodes_auto_reset_and_report_a_final_observation(server):
    with connect(server) as client:
        spec = client.spec
        client.reset(seed=5)
        rng = np.random.default_rng(11)

        saw_finish = False
        for _ in range(400):
            actions = rng.uniform(-1, 1, size=(spec.num_envs, spec.action_dim)).astype(np.float32)
            state = client.step_batch(actions)
            finished = int(state.final_mask.sum())
            if finished == 0:
                assert state.final_observations.shape == (0, spec.obs_dim)
                continue

            saw_finish = True
            # The dense block carries exactly one row per finished environment.
            assert state.final_observations.shape == (finished, spec.obs_dim)
            assert np.all(np.isfinite(state.final_observations))
            # Every finished environment reports a positive length and is
            # already restarted, so its next observation is a fresh one.
            assert np.all(state.episode_step[state.final_mask] > 0)
            assert np.array_equal(state.final_mask, state.done)

        assert saw_finish, "a random policy should have ended at least one episode"


def test_a_truncated_episode_ends_at_the_time_limit_still_standing(deterministic_server):
    """Truncation must be reported separately from failure.

    Run against a config with reset noise off, so a zero action really does hold
    the rest pose and the episodes end by running out of time rather than by
    falling. The final observation must then show a figure that is still upright
    - exactly the information a zeroed value bootstrap would throw away.
    """
    with connect(deterministic_server) as client:
        spec = client.spec
        assert spec.max_episode_steps == 60, "the config file should have been honoured"
        client.reset(seed=2)
        zeros = np.zeros((spec.num_envs, spec.action_dim), dtype=np.float32)

        saw_truncation = False
        for _ in range(spec.max_episode_steps + 5):
            state = client.step_batch(zeros)
            if state.truncated.any():
                saw_truncation = True
                assert not state.terminated[state.truncated].any()
                assert np.all(state.episode_step[state.truncated] == spec.max_episode_steps)
                rows = state.final_observations_by_env()[state.truncated]
                pelvis_height_index = spec.observation_names.index("pelvis_height")
                assert np.all(rows[:, pelvis_height_index] > 0.7)
                break

        assert saw_truncation


def test_reset_noise_is_on_by_default(server):
    """Episodes must not all start from the same state.

    The rest pose is a perfectly symmetric equilibrium, so a policy trained from
    an unperturbed start would score well while having learned nothing. This is
    a property of the shipped default config, not of a test fixture, so it is
    checked against the default server.
    """
    with connect(server) as client:
        first = client.reset(seed=1).observations.copy()
        second = client.reset(seed=2).observations.copy()
        assert not np.array_equal(first, second)

        # Environments within one batch differ from each other too.
        assert not np.allclose(first[0], first[1], atol=1e-6)


# ---------------------------------------------------------------- resilience


def test_a_lost_reply_costs_one_retry_and_not_a_double_step(server):
    """Resending an ACTION for a step already served must not advance twice.

    This is the property the whole reliability design rests on, so it is tested
    by actually replaying a datagram rather than by trusting the argument.
    """
    with connect(server) as client:
        spec = client.spec
        client.reset(seed=1)
        actions = np.zeros((spec.num_envs, spec.action_dim), dtype=np.float32)
        client.step_batch(actions)
        assert client.step == 1

        # Replay the request for step 0 -> 1, exactly as a retry would.
        replay = encode_action(
            Header(MessageType.ACTION, session=client.session, step=0), actions,
            np.zeros(spec.num_envs, dtype=np.uint8),
        )
        client._socket.sendto(replay, client.address)
        data, _ = client._socket.recvfrom(65535)

        from communication import protocol

        state = protocol.decode_state(data)
        assert state.step == 1, "a duplicate request must be answered, not re-simulated"

        # And the simulation is still where it was, so the next step is 2.
        state = client.step_batch(actions)
        assert state.step == 2
        assert np.all(state.episode_step == 2)


def test_stale_and_malformed_datagrams_are_ignored(server):
    with connect(server) as client:
        spec = client.spec
        client.reset()
        actions = np.zeros((spec.num_envs, spec.action_dim), dtype=np.float32)
        for _ in range(5):
            client.step_batch(actions)

        # Garbage, then a far-future step: neither should disturb the server.
        client._socket.sendto(b"not a packet at all", client.address)
        client._socket.sendto(
            encode_action(
                Header(MessageType.ACTION, session=client.session, step=9999),
                actions,
                np.zeros(spec.num_envs, dtype=np.uint8),
            ),
            client.address,
        )
        time.sleep(0.1)

        state = client.step_batch(actions)
        assert state.step == 6
        assert np.all(np.isfinite(state.observations))


def test_a_mismatched_action_shape_is_refused_not_misread(server):
    """A shape the server did not advertise must produce a refusal.

    Reading it as though it were the right shape would feed the policy
    misaligned observations, which looks like a learning problem rather than a
    protocol one.
    """
    from communication import protocol

    with connect(server) as client:
        spec = client.spec
        client.reset()
        wrong = encode_action(
            Header(MessageType.ACTION, session=client.session, step=client.step),
            np.zeros((spec.num_envs, spec.action_dim + 3), dtype=np.float32),
            np.zeros(spec.num_envs, dtype=np.uint8),
        )
        client._socket.sendto(wrong, client.address)
        data, _ = client._socket.recvfrom(65535)
        assert protocol.decode_header(data).type is MessageType.ERROR
        assert "shape" in protocol.decode_error(data)

        # And the server is still usable afterwards.
        state = client.step_batch(np.zeros((spec.num_envs, spec.action_dim), dtype=np.float32))
        assert state.step == 1


def test_an_absent_server_raises_promptly_instead_of_hanging(unused_port):
    client = EnvClient("127.0.0.1", unused_port, timeout=0.05, retries=3)
    started = time.monotonic()
    with pytest.raises(BridgeError, match="Is aibf_env running"):
        client.connect(num_envs=2)
    # A training run must not wedge because the simulator was not started.
    assert time.monotonic() - started < 5.0
    client.close()


def test_a_restarted_server_is_detected_rather_than_silently_followed(server_factory):
    """A new simulator process means a new session id.

    Continuing against it would look like training on the same environments
    while actually starting from scratch, so the client refuses.
    """
    first = server_factory(num_envs=3, seed=1)
    client = EnvClient("127.0.0.1", first.port, timeout=0.5, retries=5)
    client.connect(num_envs=3)
    client.reset()
    original_session = client.session
    first.stop()

    second = server_factory(num_envs=3, seed=1)
    # Point the same client at a freshly started server on a new port.
    client.address = ("127.0.0.1", second.port)
    actions = np.zeros((3, client.spec.action_dim), dtype=np.float32)
    with pytest.raises(BridgeError):
        for _ in range(3):
            client.step_batch(actions)
    client.close()
    assert original_session != 0


# ---------------------------------------------------------------- vec env


def test_the_vector_env_produces_weighted_scalar_rewards(server):
    with connect(server) as client:
        env = HumanoidVecEnv(client, RewardWeights.standing())
        obs = env.reset(seed=3)
        assert obs.shape == (server.num_envs, env.obs_dim)

        rng = np.random.default_rng(4)
        total_episodes = 0
        for _ in range(300):
            actions = rng.uniform(-1, 1, size=(env.num_envs, env.action_dim)).astype(np.float32)
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert rewards.shape == (env.num_envs,)
            assert np.all(np.isfinite(rewards))
            assert obs.shape == (env.num_envs, env.obs_dim)
            total_episodes += len(info["episodes"])
            for episode in info["episodes"]:
                assert episode.length > 0
                assert np.isfinite(episode.ret)
                assert episode.terminated != episode.truncated
        assert total_episodes > 0


def test_standing_still_earns_more_than_flailing(server):
    """The reward has to prefer the behaviour the task wants.

    If a random policy scored as well as a stationary one, there would be
    nothing for PPO to climb.
    """

    def mean_reward(action_fn) -> float:
        with connect(server) as client:
            env = HumanoidVecEnv(client, RewardWeights.standing())
            env.reset(seed=21)
            total = 0.0
            steps = 150
            for _ in range(steps):
                _, rewards, _, _, _ = env.step(action_fn(env))
                total += float(rewards.mean())
            return total / steps

    rng = np.random.default_rng(9)
    still = mean_reward(lambda env: np.zeros((env.num_envs, env.action_dim), dtype=np.float32))
    flailing = mean_reward(
        lambda env: rng.uniform(-1, 1, size=(env.num_envs, env.action_dim)).astype(np.float32)
    )
    assert still > flailing


def test_unknown_reward_weight_names_are_rejected(server):
    with connect(server) as client:
        with pytest.raises(ValueError, match="does not provide"):
            HumanoidVecEnv(client, RewardWeights({"not_a_real_term": 1.0}))


def test_reward_weights_align_to_the_servers_term_order(server):
    with connect(server) as client:
        # A weight vector that isolates one term reproduces that term exactly,
        # which is what proves the alignment is by name and not by position.
        weights = RewardWeights({"pelvis_height": 1.0})
        env = HumanoidVecEnv(client, weights)
        env.reset(seed=1)
        actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        _, rewards, _, _, info = env.step(actions)
        index = env.reward_names.index("pelvis_height")
        assert rewards == pytest.approx(info["reward_terms"][:, index], abs=1e-5)
