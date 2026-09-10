"""The Python view of a reference clip agrees with the C++ sampler.

`reference.py` deliberately does not re-implement Hermite interpolation. It
reads a dense table from the same sampler the imitation reward uses and
interpolates within it, which trades a structural risk (two interpolators that
can disagree) for a numerical one (a grid that might be too coarse). These
tests measure the numerical risk instead of assuming it away, and check the
parse against the header the tool actually prints rather than against a
hardcoded column order.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import reference
from reference import REPO_ROOT, ReferenceClip

CLIPS = ["arm_raise", "squat", "jump", "forward_roll", "backflip"]


@pytest.fixture(scope="module")
def motions_binary() -> Path:
    try:
        return reference.tool_binary("aibf_motions")
    except FileNotFoundError:
        pytest.skip("aibf_motions is not built; run scripts/build.ps1")


@pytest.fixture(scope="module")
def backflip(motions_binary: Path) -> ReferenceClip:
    return ReferenceClip.load("motions/backflip.json")


def test_every_shipped_clip_loads(motions_binary: Path):
    for name in CLIPS:
        clip = ReferenceClip.load(f"motions/{name}.json", samples=64)
        assert clip.name == name
        assert clip.joint_count == 12
        assert clip.duration > 0
        assert np.isfinite(clip.angles).all()
        assert np.isfinite(clip.rates).all()


def test_the_phase_grid_spans_both_endpoints(backflip: ReferenceClip):
    """A one-shot clip's final pose has to be in the table, not one step short.

    Sampling i/samples instead of i/(samples-1) would stop just before the
    landing, which is the part of a backflip that matters most.
    """
    assert backflip.phase[0] == pytest.approx(0.0)
    assert backflip.phase[-1] == pytest.approx(1.0)
    assert np.all(np.diff(backflip.phase) > 0)


def test_joint_names_come_from_the_dump_not_from_here(backflip: ReferenceClip):
    assert backflip.joint_names[:3] == ["waist", "neck", "shoulder_l"]
    assert len(backflip.joint_names) == backflip.joint_count
    assert backflip.joint_limits.shape == (backflip.joint_count, 2)
    assert np.all(backflip.joint_limits[:, 0] < backflip.joint_limits[:, 1])


def test_lookup_reproduces_the_grid_it_was_built_from(backflip: ReferenceClip):
    """Interpolating at the sample phases must return the samples exactly."""
    assert backflip.angles_at(backflip.phase) == pytest.approx(backflip.angles)
    assert backflip.rates_at(backflip.phase) == pytest.approx(backflip.rates)


def test_the_grid_is_dense_enough_to_be_worth_reading(motions_binary: Path):
    """The cost of a table lookup instead of a call into the sampler.

    Measured against a 4x finer grid rather than asserted from the sample
    count: "dense enough" is a claim about how sharply the clip turns, and the
    backflip turns much harder than the arm raise.

    The rate bound is the loose one and it is the one that matters. Angles
    converge quadratically with the grid; rates converge only linearly, because
    the rate curve has a corner at every keyframe and no amount of refinement
    makes linear interpolation second-order across a corner. Measured worst
    case at 4096 samples: 8.5e-6 rad and 2.3e-2 rad/s, against joint rates that
    reach 50 rad/s during a flip.
    """
    worst_angle = worst_rate = 0.0
    for name in CLIPS:
        path = f"motions/{name}.json"
        clip = ReferenceClip.load(path)
        angle_error, rate_error = clip.max_grid_error(path)
        worst_angle = max(worst_angle, angle_error)
        worst_rate = max(worst_rate, rate_error)
        assert angle_error < 1e-4, f"{name} needs a denser grid: {angle_error:.2e} rad"
    assert worst_angle < 2e-5
    assert worst_rate < 5e-2


def test_a_looping_clip_wraps_and_a_one_shot_clip_clamps(motions_binary: Path):
    loop = ReferenceClip.load("motions/arm_raise.json")
    once = ReferenceClip.load("motions/backflip.json")
    if loop.loop:
        assert loop.angles_at([1.25]) == pytest.approx(loop.angles_at([0.25]), abs=1e-6)
    assert once.angles_at([1.5]) == pytest.approx(once.angles_at([1.0]))
    assert once.angles_at([-0.5]) == pytest.approx(once.angles_at([0.0]))


def test_peak_rate_finds_the_joint_the_clip_drives_hardest(backflip: ReferenceClip):
    """A tuck is a knee motion, so the knees must dominate.

    This is the quantity that decides whether a reference is trackable at all:
    it is compared against measured motor bandwidth, and a joint asked to move
    faster than its motor can follow makes the clip unreachable no matter how
    long a policy trains.
    """
    rates = backflip.peak_rate()
    assert rates.shape == (backflip.joint_count,)
    assert (rates > 0).all()
    knees = [backflip.joint_names.index("knee_l"), backflip.joint_names.index("knee_r")]
    assert max(rates[knees]) > 5.0


def test_limit_headroom_is_signed_and_finds_the_tightest_joint(backflip: ReferenceClip):
    headroom = backflip.limit_headroom()
    assert headroom.shape == (backflip.joint_count,)
    # The tightest joints in a tuck are the hips, and they are tight: the
    # authored pose sits within a couple of degrees of the limit.
    assert headroom.min() < 0.05
    assert backflip.joint_names[int(np.argmin(headroom))].startswith("hip")


def test_a_shipped_clip_may_leave_its_limits_but_not_by_enough_to_matter():
    """Clamping the keyframes does not make the sampled clip legal.

    `clampToLimits` clamps authored poses and Hermite overshoots between them,
    so forward_roll leaves its waist and neck limits by 0.027 rad over 17% of
    the clip and the backflip leaves its hip limits by 0.017 rad over 4%. That
    is a real defect and it is measured here rather than asserted away.

    The threshold is on the *consequence*, not on the excess: an illegal
    reference caps `pose_match` at a value no policy can pass, and what matters
    is whether that cap is anywhere near the 0.08 to 0.20 the acrobatic
    policies actually score. At 0.997 it is not, so this is a thing to know and
    not the thing to fix. If a clip is ever re-authored into a real violation,
    this fails and says which.
    """
    for name in CLIPS:
        clip = ReferenceClip.load(f"motions/{name}.json")
        ceiling = clip.pose_match_ceiling()
        assert ceiling > 0.99, (
            f"{name} asks for a pose outside its joint limits by enough to cap "
            f"pose_match at {ceiling:.4f}, which a policy can never beat"
        )


def test_a_broken_dump_is_rejected_rather_than_half_parsed():
    with pytest.raises(ValueError):
        ReferenceClip.parse("# name=x duration=1 loop=0 joints=2\nphase\n0\n")
    with pytest.raises(ValueError):
        # Header promises two joints, body delivers columns for one.
        ReferenceClip.parse(
            "# name=x duration=1 loop=0 joints=2\n"
            "# joint_names=a,b\n# joint_limits=-1:1,-1:1\n"
            "phase,time,root_x,root_y,root_angle,q0,dq0\n"
            "0,0,0,1,0,0,0\n1,1,0,1,0,0,0\n"
        )


def test_a_missing_clip_fails_loudly(motions_binary: Path):
    with pytest.raises(FileNotFoundError):
        ReferenceClip.load("motions/there_is_no_such_motion.json")


def test_the_tool_refuses_a_degenerate_sample_count(motions_binary: Path):
    result = subprocess.run(
        [str(motions_binary), "--dump", str(REPO_ROOT / "motions" / "backflip.json"),
         "--samples", "1"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "samples" in result.stderr
