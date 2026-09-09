"""Runs a trained checkpoint without any learning.

    aibf_env --render --envs 1 --port 51234          # to watch
    python python/test.py --model checkpoints/stand_best.pt

Nothing here backpropagates, and the observation normaliser is frozen so a long
evaluation cannot drift the policy's inputs out from under it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from communication.client import BridgeError, EnvClient  # noqa: E402
from communication.vec_env import HumanoidVecEnv, RewardWeights  # noqa: E402
from rl.policy import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="checkpoint to load")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51234)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions instead of using the distribution mean",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client = EnvClient(args.host, args.port, timeout=5.0)
    try:
        spec = client.connect(num_envs=args.envs, seed=args.seed)
    except BridgeError as exc:
        print(f"could not reach the simulator: {exc}", file=sys.stderr)
        return 1

    policy, normalizer, meta, _ = load_checkpoint(
        args.model, device=args.device,
        expect_obs_dim=spec.obs_dim, expect_action_dim=spec.action_dim,
    )
    policy.eval()
    if normalizer is not None:
        # Evaluation must not keep updating the statistics, or a long run
        # gradually changes what the policy is being shown.
        normalizer.freeze()

    print(f"loaded {args.model}")
    print(f"  trained for {meta.env_steps:,} env steps over {meta.updates} updates "
          f"({meta.wall_seconds / 60:.1f} min)")
    print(f"  best training return {meta.best_return:+.2f}")
    print(f"  policy actions are {'sampled' if args.stochastic else 'deterministic'}")

    # Reuse the weights the checkpoint was trained with, so the reported return
    # is comparable to the training curve rather than to a different objective.
    weights = (
        RewardWeights(dict(meta.reward_weights))
        if meta.reward_weights
        else RewardWeights.standing()
    )
    env = HumanoidVecEnv(client, weights)
    raw_obs = env.reset(seed=args.seed)

    returns: list[float] = []
    lengths: list[int] = []
    reasons: Counter[str] = Counter()
    peak_pelvis: list[float] = []

    pelvis_index = spec.observation_names.index("pelvis_height")
    step = 0
    while len(returns) < args.episodes and step < args.max_steps:
        obs = normalizer.normalize(raw_obs, update=False) if normalizer else raw_obs
        with torch.no_grad():
            actions, _, _ = policy.act(
                torch.as_tensor(obs, device=args.device), deterministic=not args.stochastic
            )
        raw_obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
        peak_pelvis.append(float(raw_obs[:, pelvis_index].mean()))

        for episode in info["episodes"]:
            returns.append(episode.ret)
            lengths.append(episode.length)
            reasons["time limit" if episode.truncated else "fell"] += 1
        step += 1

    env.close()

    if not returns:
        print("\nno episode finished within the step budget")
        print(f"  ran {step} steps; mean pelvis height {np.mean(peak_pelvis):.3f} of rest")
        return 0

    print(f"\n{len(returns)} episodes over {step} control steps")
    print(f"  return   mean {np.mean(returns):+8.2f}   min {np.min(returns):+8.2f}   "
          f"max {np.max(returns):+8.2f}")
    print(f"  length   mean {np.mean(lengths):8.1f}   min {np.min(lengths):8d}   "
          f"max {np.max(lengths):8d}   ({np.mean(lengths) / spec.control_hz:.1f}s)")
    print(f"  survived to the time limit: {reasons['time limit']}/{len(returns)}")
    print(f"  mean pelvis height while running: {np.mean(peak_pelvis):.3f} of rest height")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
