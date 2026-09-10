"""Measures whether an acrobatic policy still completes its motion when shoved.

`push_test.py` answers the standing question: how hard can the figure be pushed
before it falls over. This answers the airborne one: how hard can it be hit
mid-flip before it stops landing the flip. Those are different questions, and a
survival rate cannot answer the second - an episode that gets knocked flat still
"survives" until the reference motion runs out.

So the measured quantity here is **rotation completed**, integrated from the
pelvis orientation across the episode, plus whether the episode reached the end
of the clip rather than being terminated for losing it.

    python python/flight_test.py --model checkpoints/imit_backflip_best.pt
    python python/flight_test.py --model a.pt --compare b.pt --magnitudes 0 20 40

## The witness

Every row also reports the "spin kick": how much the figure's angular velocity
jumped across the single step the shove landed on. That column exists because a
previous version of this sweep returned an identical survival rate at every
magnitude from 0 to 180 N.s, which reads as extraordinary robustness and was in
fact a measurement that never varied its own independent variable.

It is deliberately not the peak angular velocity over the episode. A backflip
spins at several rad/s under its own power, so a peak reads about the same
whether or not a disturbance ever arrived - it would certify a broken sweep as a
working one. A disturbance sweep whose witness column is flat is broken, and the
script says so rather than printing a robustness result.
"""

from __future__ import annotations

import argparse
import copy
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

# Observation slots this script reads. Imported rather than hardcoded would be
# better, but the layout lives in C++; docs/OBSERVATIONS.md is the contract.
OBS_PELVIS_HEIGHT = 0  # in units of the rest-pose height, so 1.0 is standing
OBS_ORIENTATION_SIN = 1
OBS_ORIENTATION_COS = 2
OBS_ANGULAR_VELOCITY = 5
OBS_FOOT_CONTACTS = 102  # two flags, left and right


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


def strip_json_comments(text: str) -> str:
    """Removes `//` line comments, which the C++ config parser accepts.

    String-aware, because a config value like "motions/backflip.json" must
    survive and a naive split on "//" would still be wrong the day a path
    contains one.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            out.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            out.append(character)
            index += 1
            continue
        if character == "/" and text[index + 1 : index + 2] == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        out.append(character)
        index += 1
    return "".join(out)


def load_config(path: Path) -> dict:
    return json.loads(strip_json_comments(path.read_text(encoding="utf-8")))


def write_config(path: Path, base: dict, magnitude: float, offset: float,
                 at_step: int) -> None:
    """The base config plus a disturbance block.

    Merged onto the real motion config rather than written standalone: the
    imitation section, the motor gains and the termination thresholds all have
    to survive, or the sweep measures a differently configured figure than the
    one the policy was trained on.
    """
    config = copy.deepcopy(base)
    config["disturbance"] = {
        "at_step": at_step,
        "probability_per_step": 0.0,
        "impulse_min": magnitude,
        "impulse_max": magnitude,
        "offset_max": offset,
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def evaluate(model: Path, magnitude: float, args: argparse.Namespace,
             base: dict, config_path: Path) -> dict:
    port = free_port()
    write_config(config_path, base, magnitude, args.offset, args.at_step)

    server = subprocess.Popen(
        [
            str(env_binary()), "--headless", "--quiet",
            "--port", str(port), "--envs", str(args.envs),
            "--seed", str(args.seed), "--config", str(config_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(0.5)

    rotations: list[float] = []
    completed = 0
    peak_spin = 0.0
    biggest_step = 0.0

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

        # Rotation is integrated per environment because auto-reset teleports
        # the observation back to a standing pose; differencing across that
        # boundary would invent a whole spin that never happened.
        previous = np.arctan2(raw_obs[:, OBS_ORIENTATION_SIN], raw_obs[:, OBS_ORIENTATION_COS])
        total = np.zeros(args.envs)

        # The witness is the jump in angular velocity across the step the shove
        # lands on, not the peak over the episode. A backflip spins at several
        # rad/s under its own power, so a peak would be the same with or without
        # a disturbance and would certify a broken sweep as a working one.
        episode_step = np.zeros(args.envs, dtype=np.int64)
        previous_spin = raw_obs[:, OBS_ANGULAR_VELOCITY].copy()
        kicks: list[float] = []

        # Peak height and time off the ground, per episode. Averaging height
        # over an episode says almost nothing about a jump - the reference only
        # asks for height during a fraction of the clip - so the peak is the
        # quantity that distinguishes leaving the ground from rising onto the
        # toes, which is a distinction an earlier jump policy failed.
        peak_height = np.full(args.envs, -np.inf)
        airborne_steps = np.zeros(args.envs, dtype=np.int64)
        peaks: list[float] = []
        airborne: list[int] = []

        step = 0
        while len(rotations) < args.episodes and step < args.max_steps:
            obs = normalizer.normalize(raw_obs, update=False) if normalizer else raw_obs
            with torch.no_grad():
                actions, _, _ = policy.act(
                    torch.as_tensor(obs, device=args.device), deterministic=True
                )
            raw_obs, _, _, _, info = env.step(actions.cpu().numpy())

            episode_step += 1
            spin = raw_obs[:, OBS_ANGULAR_VELOCITY]
            # A window of two covers the shove whether the schedule fires before
            # or after the step counter advances.
            on_the_beat = (episode_step == args.at_step) | (episode_step == args.at_step + 1)
            if on_the_beat.any():
                kicks.append(float(np.abs(spin - previous_spin)[on_the_beat].max()))
            previous_spin = spin.copy()
            peak_spin = max(peak_spin, float(np.abs(spin).max()))

            finished = info["final_mask"]
            final = info["final_observation"]

            # The state this step actually ended in. For an environment that
            # finished, the live observation has already been replaced by the
            # next episode's reset, so reading height and contacts from it
            # measures a figure standing at rest instead of one in mid-air.
            # Height survives that (a reset pose is shorter than a flip apex,
            # and this is a running maximum) but the contact flags do not: the
            # reset pose has both feet down, so the last airborne step of every
            # episode was being counted as grounded.
            ended_in = raw_obs.copy()
            if finished.any() and final.size:
                ended_in[finished] = final

            peak_height = np.maximum(peak_height, ended_in[:, OBS_PELVIS_HEIGHT])
            contacts = ended_in[:, OBS_FOOT_CONTACTS:OBS_FOOT_CONTACTS + 2]
            airborne_steps += (contacts.max(axis=1) < 0.5).astype(np.int64)

            # Advance every environment to where it actually ended up: the final
            # observation for the ones that finished, the live one for the rest.
            current = np.arctan2(ended_in[:, OBS_ORIENTATION_SIN],
                                 ended_in[:, OBS_ORIENTATION_COS])
            # Orientation travels as sin and cos, so a step's rotation can only
            # be recovered up to a multiple of 2*pi and the shortest arc is
            # assumed. That assumption is sound while the figure turns less
            # than pi in one control step and silently loses a whole turn if it
            # ever does not, so the largest step is carried out as a witness
            # rather than trusted.
            delta = np.arctan2(np.sin(current - previous), np.cos(current - previous))
            biggest_step = max(biggest_step, float(np.abs(delta).max()))
            total += delta
            previous = current

            for episode in info["episodes"]:
                index = episode.env_index
                rotations.append(float(np.degrees(total[index])))
                if episode.truncated and abs(np.degrees(total[index])) >= args.flip_degrees:
                    completed += 1
                peaks.append(float(peak_height[index]))
                airborne.append(int(airborne_steps[index]))
                total[index] = 0.0
                peak_height[index] = -np.inf
                airborne_steps[index] = 0
            # Post-reset environments restart from the fresh observation.
            if finished.any():
                previous = np.where(
                    finished,
                    np.arctan2(raw_obs[:, OBS_ORIENTATION_SIN], raw_obs[:, OBS_ORIENTATION_COS]),
                    previous,
                )
                episode_step[finished] = 0
            step += 1
        env.close()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

    kick = float(np.mean(kicks)) if kicks else float("nan")
    if not rotations:
        return {"magnitude": magnitude, "episodes": 0, "completed": float("nan"),
                "mean_rotation": float("nan"), "peak_spin": peak_spin, "kick": kick,
                "peak_height": float("nan"), "airborne": float("nan"),
                "biggest_step": biggest_step}

    # Control steps to seconds; the simulator runs control at 60 Hz.
    return {
        "magnitude": magnitude,
        "episodes": len(rotations),
        "completed": completed / len(rotations),
        "mean_rotation": float(np.mean(rotations)),
        "peak_spin": peak_spin,
        "kick": kick,
        "biggest_step": biggest_step,
        "peak_height": float(np.mean(peaks)),
        "airborne": float(np.mean(airborne)) / 60.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--compare", default=None, help="a second checkpoint to run alongside")
    parser.add_argument("--config", default="configs/env2d_backflip_capture.json",
                        help="the motion config the disturbance is merged into")
    parser.add_argument("--magnitudes", type=float, nargs="+",
                        default=[0, 10, 20, 30, 40, 60, 80],
                        help="impulse magnitudes in N s")
    parser.add_argument("--offset", type=float, default=0.25,
                        help="metres from the pelvis centre; 0 is a pure linear shove")
    parser.add_argument("--at-step", type=int, default=35,
                        help="the single control step the shove lands on")
    parser.add_argument("--flip-degrees", type=float, default=300.0,
                        help="rotation that counts as having completed the flip")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def report(label: str, rows: list[dict], args: argparse.Namespace) -> None:
    print(f"--- {label} ---")
    print(f"{'impulse':>9}  {'flips':>7}  {'rotation':>10}  {'peak':>7}  "
          f"{'airborne':>9}  {'spin kick':>10}")
    for row in rows:
        print(f"{row['magnitude']:>7.0f} Ns  {row['completed'] * 100:>6.0f}%  "
              f"{row['mean_rotation']:>9.0f}d  {row['peak_height']:>6.2f}x  "
              f"{row['airborne']:>8.2f}s  {row['kick']:>10.3f}")

    # The disturbance has to be visible in the physics before any robustness
    # claim is made from these numbers.
    spins = [row["kick"] for row in rows]
    if len(spins) > 1 and max(spins) - min(spins) < 0.05 * max(max(spins), 1e-6):
        print("\n  WARNING: peak spin is flat across the sweep. The impulse is not reaching")
        print("  the figure, so these rows do not measure disturbance robustness.")

    # Rotation is integrated from an orientation that travels as sin and cos,
    # so each step's turn is only recoverable up to a multiple of 2*pi and the
    # shortest arc is taken. Past pi in one control step that assumption drops
    # a whole revolution and the rotation column reads *low* while looking
    # entirely normal, which is the worst way for a headline number to fail.
    biggest = max((row["biggest_step"] for row in rows), default=0.0)
    margin = np.pi / biggest if biggest > 0 else float("inf")
    print(f"\n  largest single-step rotation {biggest:.3f} rad, "
          f"{margin:.0f}x under the {np.pi:.3f} rad the wrapped integral allows")
    if margin < 3.0:
        print("  WARNING: that is close enough that the rotation column may be dropping")
        print("  whole turns. Raise the control rate or integrate angular velocity instead.")
    print()


def main() -> int:
    args = parse_args()
    config_path_in = REPO_ROOT / args.config
    if not config_path_in.is_file():
        print(f"no such config: {config_path_in}", file=sys.stderr)
        return 1
    base = load_config(config_path_in)

    models = [Path(args.model)]
    if args.compare:
        models.append(Path(args.compare))
    for model in models:
        if not model.is_file():
            print(f"no such checkpoint: {model}", file=sys.stderr)
            return 1

    print(f"one shove on control step {args.at_step}, "
          f"applied {args.offset:.2f} m off the pelvis centre")
    print(f"a flip counts as completed at {args.flip_degrees:.0f} degrees "
          f"with the clip run to its end\n")

    results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / "flight.json"
        for model in models:
            rows = []
            for magnitude in args.magnitudes:
                try:
                    rows.append(evaluate(model, magnitude, args, base, config_path))
                except BridgeError as exc:
                    print(f"{magnitude:>7.0f} Ns  bridge error: {exc}", file=sys.stderr)
            results[model.stem] = rows
            report(model.stem, rows, args)

    if len(results) == 2:
        (a_label, a_rows), (b_label, b_rows) = results.items()
        print(f"{'impulse':>9}  {a_label:>20}  {b_label:>20}")
        for a, b in zip(a_rows, b_rows):
            print(f"{a['magnitude']:>7.0f} Ns  {a['completed'] * 100:>19.0f}%  "
                  f"{b['completed'] * 100:>19.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
