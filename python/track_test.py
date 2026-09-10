"""Why an imitation policy tracks its reference badly, joint by joint.

`pose_match` is one number for twelve joints and a whole clip. It runs 0.86 for
the arm raise and 0.08 to 0.20 for the acrobatic motions, and an aggregate that
low says only that something is wrong. Three explanations were open, and the
reward curve cannot tell them apart:

* the policy has not trained long enough,
* the falloff rates are tuned so the term is saturated and has no gradient,
* the reference asks for something the figure physically cannot do.

Those have different fingerprints once the error is broken out. Undertraining
spreads the error across joints and phases. A saturated term reads near zero
everywhere while the *error* is unremarkable. An untrackable reference shows up
as specific joints, at specific phases, moving at rates the motors cannot
follow - and the reference's own demanded rate is a number this prints next to
what the figure achieved.

    python python/track_test.py --model checkpoints/imit_backflip_best.pt
    python python/track_test.py --model checkpoints/imit_arm_best.pt \
                                --config configs/env2d_arm_raise.json

## The self-check

Every per-joint number here is computed in Python, from joint angles read out
of the observation and reference angles read out of `aibf_motions --dump`. The
simulator computes `pose_match` from its own copy of both. So the two must
agree, and the script recomputes the term and prints the disagreement.

That is not decoration. If Python samples the reference at a phase one control
step away from the one the reward used, or reads the wrong observation slice,
every per-joint number below would still look completely plausible - a
believable table of believable errors, describing nothing. The agreement line
is what separates a measurement from a well-formatted guess, and above a
threshold the script says the numbers are not trustworthy instead of printing
them.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from communication.client import BridgeError, EnvClient  # noqa: E402
from communication.vec_env import HumanoidVecEnv, RewardWeights  # noqa: E402
from reference import REPO_ROOT, ReferenceClip, tool_binary  # noqa: E402
from rl.policy import load_checkpoint  # noqa: E402

# Observation slices this script reads; docs/OBSERVATIONS.md is the contract.
OBS_JOINT_SINCOS = 66  # 12 joints, sin then cos, interleaved per joint
OBS_JOINT_VELOCITY = 90  # 12 joints, rad/s
OBS_PHASE_SINCOS = 109  # sin, cos of phase * 2pi

JOINT_COUNT = 12

# Falloff rates the simulator uses when the config does not override them,
# from ImitationScales in cpp/humanoid/RewardTerms.h.
DEFAULT_SCALES = {
    "pose": 2.0,
    "joint_velocity": 0.1,
    "end_effector": 40.0,
    "root": 20.0,
    "com": 10.0,
}

# Above this the Python reconstruction and the simulator's own term disagree by
# more than rounding, which means the two are not looking at the same thing and
# nothing derived from the reconstruction can be trusted.
AGREEMENT_TOLERANCE = 0.02


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def strip_json_comments(text: str) -> str:
    """Removes `//` line comments, which the C++ config parser accepts.

    String-aware, so a config value like "motions/backflip.json" survives.
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


def wrap(angle: np.ndarray) -> np.ndarray:
    """To (-pi, pi], matching wrapAngle in the simulator.

    A joint sitting near +/-pi must not read as a 2pi error, which during a
    flip is not a hypothetical: several joints pass through it.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def collect(model: Path, config_path: Path, args: argparse.Namespace) -> dict:
    """Runs the policy and returns per-step tracking data.

    Steps where an episode ended are read from `final_observation` rather than
    from the live observation. After an auto-reset the live observation already
    belongs to the *next* episode while the reward terms still describe the step
    that just finished, and pairing those two would compare a fresh standing
    pose against a reference phase near the end of the clip. That mistake would
    add a large error to the last step of every episode and look exactly like a
    policy that cannot land.
    """
    port = free_port()
    server = subprocess.Popen(
        [
            str(tool_binary("aibf_env")), "--headless", "--quiet",
            "--port", str(port), "--envs", str(args.envs),
            "--seed", str(args.seed), "--config", str(config_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(0.5)

    phases: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    terms: list[np.ndarray] = []
    episodes = 0
    term_names: list[str] = []

    try:
        client = EnvClient("127.0.0.1", port, timeout=5.0)
        spec = client.connect(num_envs=args.envs, seed=args.seed)
        term_names = list(spec.reward_names)

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

        step = 0
        while episodes < args.episodes and step < args.max_steps:
            obs = normalizer.normalize(raw_obs, update=False) if normalizer else raw_obs
            with torch.no_grad():
                actions, _, _ = policy.act(
                    torch.as_tensor(obs, device=args.device),
                    deterministic=not args.stochastic,
                )
            raw_obs, _, _, _, info = env.step(actions.cpu().numpy())

            # The state each reward term was computed from: the live
            # observation, except where an episode ended and the live one has
            # already been replaced by the next episode's reset.
            state = raw_obs.copy()
            finished = info["final_mask"]
            final = info["final_observation"]
            if finished.any() and final.size:
                state[finished] = final

            sincos = state[:, OBS_JOINT_SINCOS : OBS_JOINT_SINCOS + 2 * JOINT_COUNT]
            angles.append(np.arctan2(sincos[:, 0::2], sincos[:, 1::2]))
            velocities.append(state[:, OBS_JOINT_VELOCITY : OBS_JOINT_VELOCITY + JOINT_COUNT])
            phase = np.arctan2(state[:, OBS_PHASE_SINCOS], state[:, OBS_PHASE_SINCOS + 1])
            phases.append(np.mod(phase / (2.0 * np.pi), 1.0))
            terms.append(info["reward_terms"].copy())

            episodes += len(info["episodes"])
            step += 1
        env.close()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

    return {
        "phase": np.concatenate(phases),
        "angles": np.concatenate(angles),
        "velocities": np.concatenate(velocities),
        "terms": np.concatenate(terms),
        "term_names": term_names,
        "episodes": episodes,
    }


def analyse(data: dict, clip: ReferenceClip, scales: dict) -> dict:
    reference_angles = clip.angles_at(data["phase"])
    reference_rates = clip.rates_at(data["phase"])

    angle_error = wrap(data["angles"] - reference_angles)
    rate_error = data["velocities"] - reference_rates

    # The same arithmetic the simulator does, on numbers this side derived
    # independently. Agreement is the licence to believe the breakdown.
    rebuilt_pose = np.exp(-scales["pose"] * (angle_error**2).sum(axis=1))
    rebuilt_rate = np.exp(-scales["joint_velocity"] * (rate_error**2).sum(axis=1))

    names = data["term_names"]
    reported_pose = data["terms"][:, names.index("pose_match")]
    reported_rate = data["terms"][:, names.index("joint_velocity_match")]

    return {
        "angle_error": angle_error,
        "rate_error": rate_error,
        "rebuilt_pose": rebuilt_pose,
        "rebuilt_rate": rebuilt_rate,
        "reported_pose": reported_pose,
        "reported_rate": reported_rate,
        "pose_agreement": float(np.abs(rebuilt_pose - reported_pose).max()),
        "rate_agreement": float(np.abs(rebuilt_rate - reported_rate).max()),
    }


def report(label: str, data: dict, clip: ReferenceClip, result: dict,
           scales: dict, deciles: int) -> bool:
    steps = result["angle_error"].shape[0]
    print(f"--- {label} vs {clip.name} ---")
    print(f"{steps} control steps over {data['episodes']} episodes, "
          f"pose falloff k = {scales['pose']:g}\n")

    print(f"  {'':22s} {'reported':>10s} {'rebuilt':>10s} {'worst gap':>11s}")
    print(f"  {'pose_match':22s} {result['reported_pose'].mean():>10.4f} "
          f"{result['rebuilt_pose'].mean():>10.4f} {result['pose_agreement']:>11.2e}")
    print(f"  {'joint_velocity_match':22s} {result['reported_rate'].mean():>10.4f} "
          f"{result['rebuilt_rate'].mean():>10.4f} {result['rate_agreement']:>11.2e}")

    if result["pose_agreement"] > AGREEMENT_TOLERANCE:
        print("\n  STOP: the rebuilt pose term does not match the simulator's own.")
        print("  Python and the reward are not looking at the same state, so the")
        print("  per-joint breakdown below would be fiction. Not printed.")
        return False
    print("\n  the rebuilt term matches the simulator's, so the breakdown is of the")
    print("  same error the reward saw.\n")

    error = result["angle_error"]
    rms = np.sqrt((error**2).mean(axis=0))
    share = (error**2).sum(axis=0) / max((error**2).sum(), 1e-12)
    achieved = np.abs(data["velocities"]).max(axis=0)
    demanded = clip.peak_rate()

    order = np.argsort(-share)
    print(f"  {'joint':12s} {'rms err':>9s} {'worst':>8s} {'share':>7s} "
          f"{'clip asks':>11s} {'reached':>9s} {'':>6s}")
    for j in order:
        follows = achieved[j] >= 0.7 * demanded[j] or demanded[j] < 1e-6
        print(f"  {clip.joint_names[j]:12s} {rms[j]:>9.3f} {np.abs(error[:, j]).max():>8.3f} "
              f"{share[j] * 100:>6.1f}% {demanded[j]:>8.1f} r/s {achieved[j]:>7.1f} "
              f"{'' if follows else '  <- lags':>6s}")

    print(f"\n  {'phase':>12s} {'pose_match':>11s} {'angle rms':>10s} {'steps':>7s}")
    edges = np.linspace(0.0, 1.0, deciles + 1)
    bucket = np.clip(np.digitize(data["phase"], edges[1:-1]), 0, deciles - 1)
    for d in range(deciles):
        mask = bucket == d
        if not mask.any():
            continue
        band_rms = float(np.sqrt((error[mask] ** 2).mean()))
        print(f"  {edges[d]:>5.2f}-{edges[d + 1]:<6.2f} "
              f"{result['reported_pose'][mask].mean():>11.4f} {band_rms:>10.3f} "
              f"{int(mask.sum()):>7d}")

    ceiling = clip.pose_match_ceiling(scales["pose"])
    mean_rms = float(np.sqrt((error**2).mean()))
    print(f"\n  mean joint error {mean_rms:.3f} rad ({np.degrees(mean_rms):.1f} deg) "
          f"over {clip.joint_count} joints")
    print(f"  the reference's own limit violations cap pose_match at {ceiling:.4f}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/env2d_backflip_capture.json")
    parser.add_argument("--motion", default=None,
                        help="override the clip; defaults to the config's imitation.motion")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of taking the mean")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = REPO_ROOT / args.config
    if not config_path.is_file():
        print(f"no such config: {config_path}", file=sys.stderr)
        return 1
    config = load_config(config_path)

    imitation = config.get("imitation", {})
    motion = args.motion or imitation.get("motion")
    if not motion:
        print(f"{args.config} has no imitation.motion; there is nothing to track against",
              file=sys.stderr)
        return 1

    scales = dict(DEFAULT_SCALES)
    scales.update({k: float(v) for k, v in imitation.get("scales", {}).items()
                   if k in scales})

    model = Path(args.model)
    if not model.is_file():
        print(f"no such checkpoint: {model}", file=sys.stderr)
        return 1

    clip = ReferenceClip.load(motion)
    try:
        data = collect(model, config_path, args)
    except BridgeError as exc:
        print(f"bridge error: {exc}", file=sys.stderr)
        return 1
    if data["episodes"] == 0:
        print("no episodes finished; nothing to report", file=sys.stderr)
        return 1

    result = analyse(data, clip, scales)
    ok = report(model.stem, data, clip, result, scales, args.deciles)
    print()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
