"""Drives the simulator with random actions.

This is the bridge's smoke test and its benchmark. It proves the loop end to
end - Python receiving observations, C++ receiving and acting on actions - and
reports the round-trip throughput that decides whether the transport or the
physics is the limiting factor for training.

    python python/random_agent.py --envs 32 --steps 2000
    python python/random_agent.py --policy zero --show-obs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from communication.client import BridgeError, EnvClient  # noqa: E402
from communication.vec_env import HumanoidVecEnv, RewardWeights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51234)
    parser.add_argument("--envs", type=int, default=25)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=["random", "zero", "sine"],
        default="random",
        help="random: uniform in [-1,1]; zero: hold the rest pose; sine: a slow sweep",
    )
    parser.add_argument("--show-obs", action="store_true", help="print a sample observation")
    parser.add_argument("--timeout", type=float, default=2.0)
    return parser.parse_args()


def make_policy(name: str, num_envs: int, action_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    if name == "zero":
        return lambda step: np.zeros((num_envs, action_dim), dtype=np.float32)
    if name == "sine":
        # Every joint sweeps at a slightly different rate, so the motion is
        # obviously non-random when watched with --render on the server.
        rates = np.linspace(0.4, 1.6, action_dim)
        return lambda step: np.tile(
            np.sin(step * 0.05 * rates).astype(np.float32), (num_envs, 1)
        )
    return lambda step: rng.uniform(-1.0, 1.0, size=(num_envs, action_dim)).astype(np.float32)


def main() -> int:
    args = parse_args()

    client = EnvClient(args.host, args.port, timeout=args.timeout)
    try:
        spec = client.connect(num_envs=args.envs, seed=args.seed)
    except BridgeError as exc:
        print(f"could not reach the simulator: {exc}", file=sys.stderr)
        print(f"start it with:  aibf_env --headless --port {args.port}", file=sys.stderr)
        return 1

    print(f"connected to {args.host}:{args.port}")
    print(
        f"  {spec.num_envs} environments, obs {spec.obs_dim}, actions {spec.action_dim}, "
        f"reward terms {spec.reward_dim}"
    )
    print(f"  control {spec.control_hz:.0f} Hz, physics {spec.physics_hz:.0f} Hz, "
          f"episode limit {spec.max_episode_steps}")

    env = HumanoidVecEnv(client, RewardWeights.standing())
    obs = env.reset(seed=args.seed)
    print(f"  first observation block: {obs.shape} {obs.dtype}")

    if args.show_obs:
        print("\nsample observation (environment 0), first 12 fields:")
        for name, value in list(zip(spec.observation_names, obs[0]))[:12]:
            print(f"    {name:<24} {value:+.4f}")

    policy = make_policy(args.policy, env.num_envs, env.action_dim, args.seed)

    returns: list[float] = []
    lengths: list[int] = []
    reward_sum = 0.0
    started = time.perf_counter()

    for step in range(args.steps):
        obs, rewards, terminated, truncated, info = env.step(policy(step))
        reward_sum += float(rewards.mean())
        for episode in info["episodes"]:
            returns.append(episode.ret)
            lengths.append(episode.length)

        if not np.all(np.isfinite(obs)):
            print("observations went non-finite; aborting", file=sys.stderr)
            return 1

    elapsed = time.perf_counter() - started
    control_rate = args.steps / elapsed
    env_rate = control_rate * env.num_envs

    print(f"\n{args.steps} control steps in {elapsed:.2f}s")
    print(f"  {control_rate:8.0f} round trips/s")
    print(f"  {env_rate:8.0f} env-steps/s  ({env_rate / spec.control_hz / env.num_envs:.0f}x real time)")
    print(f"  mean reward per step: {reward_sum / args.steps:+.3f}")
    if returns:
        print(f"  {len(returns)} episodes: mean return {np.mean(returns):+.2f}, "
              f"mean length {np.mean(lengths):.1f} steps "
              f"({np.mean(lengths) / spec.control_hz:.2f}s)")
    else:
        print("  no episodes finished")

    print(f"  client: {client.stats}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
