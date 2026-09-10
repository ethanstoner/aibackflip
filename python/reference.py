"""The reference motion, on the Python side, without a second sampler.

The imitation reward grades a policy against a clip that the C++ side samples
with cubic Hermite interpolation. Grading the same policy from Python needs the
same numbers, and there are only two ways to get them: re-implement Hermite
sampling here, or ask the implementation that the reward itself uses.

This does the second. ``aibf_motions --dump`` prints the reference at a dense
grid of phases and this module reads that table and interpolates within it. A
duplicated interpolator would be a place for the two sides to disagree quietly,
and a disagreement between the grader and the reward would look exactly like a
policy tracking badly - which is the thing being measured, so it must not also
be the thing that can be wrong.

Reading a table costs interpolation error instead. That error is bounded and
measurable rather than structural: ``max_grid_error`` refines the grid and
reports how much the answers move, and the default of 1024 samples puts it
around a microradian on every clip in ``motions/``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Dense enough that linear interpolation between neighbours is far below any
# tracking error worth reporting, and small enough to dump in milliseconds.
DEFAULT_SAMPLES = 1024


def tool_binary(name: str) -> Path:
    """Locates a built C++ tool, Release first."""
    exe = f"{name}.exe" if os.name == "nt" else name
    for candidate in (
        REPO_ROOT / "build" / "bin" / "Release" / exe,
        REPO_ROOT / "build" / "bin" / "Debug" / exe,
        REPO_ROOT / "build" / "bin" / exe,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} is not built; run scripts/build.ps1")


def _header_value(lines: list[str], key: str) -> str:
    """Pulls ``key=value`` out of the ``#`` comment block."""
    for line in lines:
        if not line.startswith("#"):
            break
        for field in line[1:].strip().split(" "):
            if field.startswith(f"{key}="):
                return field[len(key) + 1 :]
    raise ValueError(f"the dump header has no {key}")


@dataclass
class ReferenceClip:
    """A reference motion sampled onto a uniform phase grid."""

    name: str
    duration: float
    loop: bool
    joint_names: list[str]
    joint_limits: np.ndarray  # (J, 2), lower and upper, radians
    phase: np.ndarray  # (S,) strictly increasing from 0 to 1
    root_height: np.ndarray  # (S,) metres
    root_angle: np.ndarray  # (S,) radians, never wrapped
    angles: np.ndarray  # (S, J) radians
    rates: np.ndarray  # (S, J) radians per second

    # ---------------------------------------------------------------- loading

    @staticmethod
    def load(path: str | Path, samples: int = DEFAULT_SAMPLES) -> "ReferenceClip":
        clip = Path(path)
        if not clip.is_absolute():
            clip = REPO_ROOT / clip
        if not clip.is_file():
            raise FileNotFoundError(f"no such motion clip: {clip}")

        result = subprocess.run(
            [str(tool_binary("aibf_motions")), "--dump", str(clip), "--samples", str(samples)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"aibf_motions --dump failed: {result.stderr.strip()}")
        return ReferenceClip.parse(result.stdout)

    @staticmethod
    def parse(text: str) -> "ReferenceClip":
        lines = [line for line in text.splitlines() if line.strip()]
        comments = [line for line in lines if line.startswith("#")]
        body = [line for line in lines if not line.startswith("#")]
        if len(body) < 3:
            raise ValueError("the dump has no header row and at least two samples")

        name = _header_value(comments, "name")
        duration = float(_header_value(comments, "duration"))
        loop = _header_value(comments, "loop") == "1"
        joint_count = int(_header_value(comments, "joints"))
        joint_names = _header_value(comments, "joint_names").split(",")
        limits = np.array(
            [[float(part) for part in pair.split(":")]
             for pair in _header_value(comments, "joint_limits").split(",")],
            dtype=np.float64,
        )

        columns = body[0].split(",")
        expected = 5 + 2 * joint_count
        if len(columns) != expected:
            raise ValueError(f"the dump has {len(columns)} columns, expected {expected}")

        table = np.array([[float(v) for v in row.split(",")] for row in body[1:]],
                         dtype=np.float64)
        return ReferenceClip(
            name=name,
            duration=duration,
            loop=loop,
            joint_names=joint_names,
            joint_limits=limits,
            phase=table[:, 0],
            root_height=table[:, 3],
            root_angle=table[:, 4],
            angles=table[:, 5 : 5 + joint_count],
            rates=table[:, 5 + joint_count : 5 + 2 * joint_count],
        )

    # ---------------------------------------------------------------- queries

    @property
    def joint_count(self) -> int:
        return self.angles.shape[1]

    def _lookup(self, table: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Per-column linear interpolation of `table` at each phase.

        Phases outside [0, 1] clamp for a one-shot clip and wrap for a looping
        one, matching what the sampler does with time.
        """
        query = np.asarray(phase, dtype=np.float64).ravel()
        if self.loop:
            query = query - np.floor(query)
        out = np.empty((query.size, table.shape[1]), dtype=np.float64)
        for column in range(table.shape[1]):
            out[:, column] = np.interp(query, self.phase, table[:, column])
        return out

    def angles_at(self, phase: np.ndarray) -> np.ndarray:
        """(N, J) reference joint angles."""
        return self._lookup(self.angles, phase)

    def rates_at(self, phase: np.ndarray) -> np.ndarray:
        """(N, J) reference joint rates, radians per second."""
        return self._lookup(self.rates, phase)

    def root_at(self, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,) reference pelvis height and unwrapped pelvis angle."""
        stacked = np.column_stack((self.root_height, self.root_angle))
        values = self._lookup(stacked, phase)
        return values[:, 0], values[:, 1]

    # ------------------------------------------------------- what it demands

    def peak_rate(self) -> np.ndarray:
        """(J,) the fastest the clip ever asks each joint to move, rad/s.

        This is the number to compare against measured motor bandwidth. A joint
        asked to move faster than its motor can follow makes the reference
        untrackable no matter how long the policy trains, and that failure looks
        identical to a training failure from the reward curve alone.
        """
        return np.abs(self.rates).max(axis=0)

    def limit_headroom(self) -> np.ndarray:
        """(J,) the closest the clip ever comes to a joint limit, in radians.

        Negative means the clip asks for a pose outside the limits, which the
        solver will refuse: the reference is then unreachable by construction.
        Near zero means the joint is pinned against a constraint for part of the
        clip, where a motor has no authority to correct an error in one
        direction.
        """
        lower = self.angles - self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1] - self.angles
        return np.minimum(lower.min(axis=0), upper.min(axis=0))

    def limit_excess(self) -> np.ndarray:
        """(S, J) how far outside its limits the clip asks each joint to go.

        Zero everywhere the reference is legal. Non-zero here is not an
        authoring slip: `clampToLimits` clamps keyframes, and cubic Hermite
        overshoots between them, so a clip with twelve legal poses can still
        sweep past a limit on the way from one to the next.
        """
        lower = self.joint_limits[:, 0] - self.angles
        upper = self.angles - self.joint_limits[:, 1]
        return np.maximum(np.maximum(lower, upper), 0.0)

    def pose_match_ceiling(self, scale: float = 2.0) -> float:
        """The best `pose_match` the reference itself permits, in [0, 1].

        A joint held at its limit cannot reach a target beyond it, so wherever
        the clip asks for an illegal pose the tracking reward has a maximum
        below 1 that no amount of training can pass. Worth knowing as a number:
        an unreachable ceiling of 0.997 is a curiosity, and one of 0.4 would be
        the whole explanation for a policy that never tracks.
        """
        excess = self.limit_excess()
        return float(np.exp(-scale * (excess**2).sum(axis=1)).min())

    def max_grid_error(self, path: str | Path) -> float:
        """Worst joint-angle disagreement, in radians, against a 4x finer grid.

        The cost of reading a table instead of calling the sampler. Reported
        rather than assumed, because "dense enough" is a claim about the clip's
        curvature and not about the number 1024.
        """
        fine = ReferenceClip.load(path, samples=4 * (len(self.phase) - 1) + 1)
        return float(np.abs(self.angles_at(fine.phase) - fine.angles).max())
