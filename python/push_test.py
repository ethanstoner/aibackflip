"""Measures how hard a policy can be shoved before it falls over.

Runs the policy against a deterministic disturbance schedule - one impulse every
N control steps, alternating side - and sweeps the magnitude. The result is a
survival curve, which is a claim that can be checked, unlike "it looks robust".

    python python/push_test.py --model checkpoints/stand_v1_best.pt
    python python/push_test.py --model a.pt --compare b.pt --magnitudes 0 20 40 60 80

Each magnitude gets its own simulator process with a generated config, so the
schedule is identical across policies and the comparison is fair.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from communication.client import BridgeError, EnvClient  # noqa: E402
from communication.vec_env import HumanoidVecEnv, RewardWeights  # noqa: E402
from rl.policy import load_checkpoint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_binary() -> Path:
    exe = "aibf_env.exe" if os.name == "nt" else "aibf_env"
    for candidate in (
        REPO_ROOT / "build" / "bin" / "Release" / exe,
        REPO_ROOT / "build" / "bin" / "Debug" / exe,
        REPO_ROOT / "build" / "bin" / exe,
    ):
        if candidate.is_file():
            return candidate
    raise SystemExit("aibf_env is not built; run scripts/build.ps1")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def write_config(path: Path, magnitude: float, interval: int, episode_steps: int) -> None:
    """A config with a fixed-magnitude shove every `interval` control steps.

    Reset noise is kept at the training default rather than zeroed: the question
    is whether the policy survives shoves from the states it normally starts in.
    """
    config = {
        "max_episode_steps": episode_steps,
        "disturbance": {
            "interval_steps": interval,
            "probability_per_step": 0.0,
            "impulse_min": magnitude,
            "impulse_max": magnitude,
        },
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def evaluate(model: Path, magnitude: float, args: argparse.Namespace,
             config_path: Path) -> dict:
    port = free_port()
    write_config(config_path, magnitude, args.interval, args.episode_steps)

    server = subprocess.Popen(
        [
            str(env_binary()), "--headless", "--quiet",
            "--port", str(port), "--envs", str(args.envs),
            "--seed", str(args.seed), "--config", str(config_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(0.5)

    try:
        client = EnvClient("127.0.0.1", port, timeout=5.0)
        spec = client.connect(num_envs=args.envs, seed=args.seed)

        policy, normalizer, meta, _ = load_checkpoint(
            model, device=args.device,
            expect_obs_dim=spec.obs_dim, expect_action_dim=spec.action_dim,
        )
        policy.eval()
        if normalizer is not None:
            normalizer.freeze()

        weights = (
            RewardWeights(dict(meta.reward_weights))
            if meta.reward_weights else RewardWeights.standing()
        )
        env = HumanoidVecEnv(client, weights)
        raw_obs = env.reset(seed=args.seed)

        lengths: list[int] = []
        survived = 0
        step = 0
        while len(lengths) < args.episodes and step < args.max_steps:
            obs = normalizer.normalize(raw_obs, update=False) if normalizer else raw_obs
            with torch.no_grad():
                actions, _, _ = policy.act(
                    torch.as_tensor(obs, device=args.device), deterministic=True
                )
            raw_obs, _, _, _, info = env.step(actions.cpu().numpy())
            for episode in info["episodes"]:
                lengths.append(episode.length)
                if episode.truncated:
                    survived += 1
            step += 1
        env.close()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

    if not lengths:
        return {"magnitude": magnitude, "episodes": 0, "survival": float("nan"),
                "mean_length": float("nan")}

    # Shoves per episode is what the survival rate is really conditioned on.
    shoves = args.episode_steps // args.interval
    return {
        "magnitude": magnitude,
        "episodes": len(lengths),
        "survival": survived / len(lengths),
        "mean_length": float(np.mean(lengths)),
        "shoves_if_surviving": shoves,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--compare", default=None, help="a second checkpoint to run alongside")
    parser.add_argument(
        "--magnitudes", type=float, nargs="+",
        default=[0, 15, 30, 45, 60, 80, 100, 130, 160],
        help="impulse magnitudes in N s",
    )
    parser.add_argument("--interval", type=int, default=90, help="control steps between shoves")
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [Path(args.model)]
    if args.compare:
        models.append(Path(args.compare))
    for model in models:
        if not model.is_file():
            print(f"no such checkpoint: {model}", file=sys.stderr)
            return 1

    print(f"one shove every {args.interval} control steps ({args.interval / 60:.1f}s), "
          f"alternating side, {args.episode_steps}-step episodes")
    print(f"a surviving episode absorbs {args.episode_steps // args.interval} shoves\n")

    results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / "push.json"
        for model in models:
            label = model.stem
            print(f"--- {label} ---")
            print(f"{'impulse':>9}  {'survived':>9}  {'mean length':>12}")
            rows = []
            for magnitude in args.magnitudes:
                try:
                    row = evaluate(model, magnitude, args, config_path)
                except BridgeError as exc:
                    print(f"{magnitude:>7.0f} Ns  bridge error: {exc}", file=sys.stderr)
                    continue
                rows.append(row)
                print(f"{magnitude:>7.0f} Ns  {row['survival'] * 100:>8.0f}%  "
                      f"{row['mean_length']:>12.0f}")
            results[label] = rows
            print()

    if len(results) == 2:
        (a_label, a_rows), (b_label, b_rows) = results.items()
        print(f"{'impulse':>9}  {a_label:>18}  {b_label:>18}")
        for a, b in zip(a_rows, b_rows):
            print(f"{a['magnitude']:>7.0f} Ns  {a['survival'] * 100:>17.0f}%  "
                  f"{b['survival'] * 100:>17.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
