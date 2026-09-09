"""PPO training against the humanoid environment server.

    aibf_env --headless --envs 32 --port 51234      # in one terminal
    python python/train.py --config configs/ppo_stand.json

Everything that decides whether a run worked is logged: per-component reward
means (the only practical way to spot reward hacking), approximate KL, clip
fraction, explained variance, and termination reasons. A reward curve alone
cannot distinguish "learning to stand" from "found an exploit".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from communication.client import BridgeError, EnvClient  # noqa: E402
from communication.vec_env import HumanoidVecEnv, RewardWeights  # noqa: E402
from rl.normalization import ObservationNormalizer, ReturnNormalizer  # noqa: E402
from rl.policy import ActorCritic, CheckpointMeta, PolicyConfig, save_checkpoint, load_checkpoint  # noqa: E402
from rl.ppo import PPO, PPOConfig  # noqa: E402
from rl.rollout import RolloutBuffer, explained_variance  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    name: str = "stand"
    host: str = "127.0.0.1"
    port: int = 51234
    num_envs: int = 32
    seed: int = 1

    total_env_steps: int = 20_000_000
    rollout_steps: int = 64  # per environment, so the batch is this x num_envs

    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_init: float = -0.5

    normalize_observations: bool = True
    normalize_rewards: bool = True

    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward_weights: dict = field(default_factory=lambda: RewardWeights.standing().weights)

    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    save_every_updates: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def load(path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        ppo = PPOConfig.from_dict(data.pop("ppo", {}))
        hidden = data.pop("hidden_sizes", None)
        config = TrainConfig(**data, ppo=ppo)
        if hidden is not None:
            config.hidden_sizes = tuple(hidden)
        return config

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


def make_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover - tensorboard is in requirements
        print("tensorboard unavailable; metrics will only be printed", file=sys.stderr)
        return None
    return SummaryWriter(log_dir=str(log_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="total environment steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to continue from")
    parser.add_argument("--log-every", type=int, default=1, help="updates between console lines")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainConfig.load(args.config) if args.config else TrainConfig()
    if args.name:
        config.name = args.name
    if args.port:
        config.port = args.port
    if args.envs:
        config.num_envs = args.envs
    if args.steps:
        config.total_env_steps = args.steps
    if args.seed is not None:
        config.seed = args.seed
    if args.device:
        config.device = args.device

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    # ---- connect ----
    client = EnvClient(config.host, config.port, timeout=5.0)
    try:
        spec = client.connect(num_envs=config.num_envs, seed=config.seed)
    except BridgeError as exc:
        print(f"could not reach the simulator: {exc}", file=sys.stderr)
        print(f"start it with:  aibf_env --headless --envs {config.num_envs} "
              f"--port {config.port}", file=sys.stderr)
        return 1

    env = HumanoidVecEnv(client, RewardWeights(dict(config.reward_weights)))
    print(f"connected: {spec.num_envs} envs, obs {spec.obs_dim}, actions {spec.action_dim}")

    # ---- policy ----
    policy_config = PolicyConfig(
        obs_dim=spec.obs_dim,
        action_dim=spec.action_dim,
        hidden_sizes=tuple(config.hidden_sizes),
        log_std_init=config.log_std_init,
    )
    policy = ActorCritic(policy_config).to(device)
    obs_normalizer = ObservationNormalizer(spec.obs_dim) if config.normalize_observations else None
    return_normalizer = (
        ReturnNormalizer(config.num_envs, gamma=config.ppo.gamma)
        if config.normalize_rewards
        else None
    )
    ppo = PPO(policy, config.ppo, device=device)

    meta = CheckpointMeta(reward_weights=dict(config.reward_weights))
    if args.resume:
        policy, restored_norm, meta, payload = load_checkpoint(
            args.resume, device=device,
            expect_obs_dim=spec.obs_dim, expect_action_dim=spec.action_dim,
        )
        if restored_norm is not None:
            obs_normalizer = restored_norm
        ppo = PPO(policy, config.ppo, device=device)
        if "optimizer" in payload:
            ppo.optimizer.load_state_dict(payload["optimizer"])
        print(f"resumed from {args.resume} at {meta.env_steps:,} env steps")

    print(f"policy: {policy.parameter_count():,} parameters on {device}")

    # ---- logging ----
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{config.name}-{stamp}"
    log_dir = REPO_ROOT / config.log_dir / run_name
    checkpoint_dir = REPO_ROOT / config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = make_writer(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    print(f"logging to {log_dir}")

    # ---- rollout ----
    buffer = RolloutBuffer(config.rollout_steps, config.num_envs, spec.obs_dim, spec.action_dim)
    raw_obs = env.reset(seed=config.seed)
    obs = obs_normalizer.normalize(raw_obs) if obs_normalizer else raw_obs.astype(np.float32)

    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)
    steps_per_update = config.rollout_steps * config.num_envs
    total_updates = max(1, config.total_env_steps // steps_per_update)

    env_steps = meta.env_steps
    started = time.perf_counter()
    best_return = meta.best_return

    try:
        for update in range(1, total_updates + 1):
            if config.ppo.anneal_learning_rate:
                progress = 1.0 - (update - 1) / total_updates
                ppo.set_learning_rate(config.ppo.learning_rate * progress)

            buffer.reset()
            term_sums = np.zeros(spec.reward_dim, dtype=np.float64)
            raw_reward_sum = 0.0
            # Split the wall clock between simulating and learning. Without it,
            # "training is slow" is unactionable - the fix for a rollout-bound
            # run (more environments) is the opposite of the fix for an
            # update-bound one (bigger minibatches, fewer epochs).
            rollout_started = time.perf_counter()

            for _ in range(config.rollout_steps):
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, device=device)
                    actions, log_probs, values = policy.act(obs_tensor)

                action_array = actions.cpu().numpy()
                next_raw_obs, rewards, terminated, truncated, info = env.step(action_array)
                raw_reward_sum += float(rewards.mean())
                term_sums += info["reward_terms"].mean(axis=0)

                # A truncated episode has to bootstrap from the state it was cut
                # off in, which the environment ships alongside the reset one.
                bootstrap = np.zeros(config.num_envs, dtype=np.float32)
                if truncated.any():
                    final_by_env = info["final_observation"]
                    scattered = np.zeros((config.num_envs, spec.obs_dim), dtype=np.float32)
                    scattered[info["final_mask"]] = final_by_env
                    final_rows = scattered[truncated]
                    if obs_normalizer is not None:
                        # Not used to update the statistics: these are terminal
                        # states, over-represented relative to how often the
                        # policy actually visits them.
                        final_rows = obs_normalizer.normalize(final_rows, update=False)
                    with torch.no_grad():
                        bootstrap[truncated] = (
                            policy.value(torch.as_tensor(final_rows, device=device))
                            .cpu()
                            .numpy()
                        )

                stored_rewards = rewards
                if return_normalizer is not None:
                    stored_rewards = return_normalizer(rewards, terminated | truncated)

                buffer.add(
                    observations=obs,
                    actions=action_array,
                    log_probs=log_probs.cpu().numpy(),
                    values=values.cpu().numpy(),
                    rewards=stored_rewards,
                    terminated=terminated,
                    truncated=truncated,
                    bootstrap_values=bootstrap,
                )

                for episode in info["episodes"]:
                    recent_returns.append(episode.ret)
                    recent_lengths.append(episode.length)

                raw_obs = next_raw_obs
                obs = (
                    obs_normalizer.normalize(raw_obs)
                    if obs_normalizer
                    else raw_obs.astype(np.float32)
                )
                env_steps += config.num_envs

            rollout_seconds = time.perf_counter() - rollout_started

            update_started = time.perf_counter()
            with torch.no_grad():
                last_values = policy.value(torch.as_tensor(obs, device=device)).cpu().numpy()

            advantages, returns = buffer.compute_advantages(
                last_values, gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda
            )
            batch = buffer.to_batch(advantages, returns, device=device)
            stats = ppo.update(batch)
            update_seconds = time.perf_counter() - update_started

            variance_explained = explained_variance(buffer.values, returns)
            elapsed = time.perf_counter() - started
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            mean_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")

            if writer is not None:
                writer.add_scalar("rollout/episode_return", mean_return, env_steps)
                writer.add_scalar("rollout/episode_length", mean_length, env_steps)
                writer.add_scalar("rollout/reward_per_step", raw_reward_sum / config.rollout_steps, env_steps)
                writer.add_scalar("train/policy_loss", stats.policy_loss, env_steps)
                writer.add_scalar("train/value_loss", stats.value_loss, env_steps)
                writer.add_scalar("train/entropy", stats.entropy, env_steps)
                writer.add_scalar("train/approx_kl", stats.approx_kl, env_steps)
                writer.add_scalar("train/clip_fraction", stats.clip_fraction, env_steps)
                writer.add_scalar("train/grad_norm", stats.grad_norm, env_steps)
                writer.add_scalar("train/explained_variance", variance_explained, env_steps)
                writer.add_scalar("train/learning_rate", stats.learning_rate, env_steps)
                writer.add_scalar("train/epochs_run", stats.epochs_run, env_steps)
                writer.add_scalar("train/action_std", float(policy.current_std().mean()), env_steps)
                writer.add_scalar("perf/env_steps_per_second", env_steps / max(elapsed, 1e-9), env_steps)
                writer.add_scalar("perf/rollout_seconds", rollout_seconds, env_steps)
                writer.add_scalar("perf/update_seconds", update_seconds, env_steps)
                writer.add_scalar(
                    "perf/update_fraction",
                    update_seconds / max(rollout_seconds + update_seconds, 1e-9), env_steps
                )
                # Per-component reward means. A policy that has found an exploit
                # shows one of these saturating while the rest flatline, which is
                # invisible in the scalar reward.
                for name, value in zip(env.reward_names, term_sums / config.rollout_steps):
                    writer.add_scalar(f"reward_terms/{name}", float(value), env_steps)

            if update % args.log_every == 0:
                print(
                    f"update {update:5d}/{total_updates}  "
                    f"steps {env_steps:>11,}  "
                    f"return {mean_return:+8.2f}  "
                    f"len {mean_length:6.1f}  "
                    f"kl {stats.approx_kl:.4f}  "
                    f"clip {stats.clip_fraction:.3f}  "
                    f"ev {variance_explained:+.3f}  "
                    f"ent {stats.entropy:+.2f}  "
                    f"{env_steps / max(elapsed, 1e-9):7.0f} steps/s  "
                    f"(sim {rollout_seconds:.2f}s + learn {update_seconds:.2f}s, "
                    f"{stats.epochs_run}ep)"
                    + ("  [kl stop]" if stats.stopped_early else "")
                )

            meta.updates = update
            meta.env_steps = env_steps
            meta.wall_seconds = elapsed
            if recent_returns and mean_return > best_return:
                best_return = mean_return
                meta.best_return = best_return
                save_checkpoint(
                    checkpoint_dir / f"{config.name}_best.pt", policy, obs_normalizer, meta,
                    ppo.optimizer,
                )
            if update % config.save_every_updates == 0:
                save_checkpoint(
                    checkpoint_dir / f"{config.name}_latest.pt", policy, obs_normalizer, meta,
                    ppo.optimizer,
                )

    except KeyboardInterrupt:
        print("\ninterrupted; saving")
    finally:
        save_checkpoint(
            checkpoint_dir / f"{config.name}_latest.pt", policy, obs_normalizer, meta, ppo.optimizer
        )
        if writer is not None:
            writer.close()
        env.close()

    print(f"\nfinished: {meta.env_steps:,} env steps, best mean return {meta.best_return:+.2f}")
    print(f"checkpoints in {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
