"""Shared fixtures: locating the built C++ binaries and running a live server."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _binary(name: str) -> Path | None:
    """Finds a built executable, preferring Release over Debug."""
    exe = f"{name}.exe" if os.name == "nt" else name
    for candidate in (
        REPO_ROOT / "build" / "bin" / "Release" / exe,
        REPO_ROOT / "build" / "bin" / "Debug" / exe,
        REPO_ROOT / "build" / "bin" / exe,
        REPO_ROOT / "build" / exe,
    ):
        if candidate.is_file():
            return candidate
    return None


def free_udp_port() -> int:
    """Asks the OS for a free port.

    Tests must not hardcode 51234: a developer running the simulator by hand
    would otherwise make the suite fail in a way that looks like a bridge bug.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def unused_port() -> int:
    return free_udp_port()


@pytest.fixture(scope="session")
def env_binary() -> Path:
    path = _binary("aibf_env")
    if path is None:
        pytest.skip("aibf_env is not built; run scripts/build.ps1")
    return path


@pytest.fixture(scope="session")
def fixture_binary() -> Path:
    path = _binary("aibf_fixture")
    if path is None:
        pytest.skip("aibf_fixture is not built; run scripts/build.ps1")
    return path


class ServerProcess:
    """A running aibf_env, torn down at the end of the test."""

    def __init__(
        self, binary: Path, num_envs: int, seed: int = 1, config_path: Path | None = None
    ) -> None:
        self.port = free_udp_port()
        self.num_envs = num_envs
        command = [
            str(binary),
            "--headless",
            "--quiet",
            "--port",
            str(self.port),
            "--envs",
            str(num_envs),
            "--seed",
            str(seed),
        ]
        if config_path is not None:
            command += ["--config", str(config_path)]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # The client retries, so a short settle is enough; this only avoids
        # burning the first few retries on a process that has not bound yet.
        time.sleep(0.3)
        if self.process.poll() is not None:
            output = self.process.stdout.read() if self.process.stdout else ""
            raise RuntimeError(f"aibf_env exited immediately:\n{output}")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


@pytest.fixture
def server(env_binary: Path):
    """A four-environment server on a private port."""
    running = ServerProcess(env_binary, num_envs=4)
    try:
        yield running
    finally:
        running.stop()


@pytest.fixture
def server_factory(env_binary: Path):
    """For tests that need a specific environment count or to restart a server."""
    started: list[ServerProcess] = []

    def make(num_envs: int = 4, seed: int = 1, config_path: Path | None = None) -> ServerProcess:
        running = ServerProcess(
            env_binary, num_envs=num_envs, seed=seed, config_path=config_path
        )
        started.append(running)
        return running

    try:
        yield make
    finally:
        for running in started:
            running.stop()


@pytest.fixture
def deterministic_server(env_binary: Path, tmp_path: Path):
    """A server with reset noise disabled and a short episode limit.

    Reset noise is on by default, which is right for training and wrong for a
    test that needs the figure to survive to the time limit under a fixed
    policy. Writing a config also exercises the JSON loading path end to end.
    """
    config = {
        "max_episode_steps": 60,
        "reset_noise": {
            "root_angle": 0.0,
            "root_height": 0.0,
            "joint_angle": 0.0,
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
        },
    }
    path = tmp_path / "deterministic.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    running = ServerProcess(env_binary, num_envs=2, seed=1, config_path=path)
    try:
        yield running
    finally:
        running.stop()
