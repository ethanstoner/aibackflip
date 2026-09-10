"""Phase reconstruction, which the whole tracking analysis rests on.

The observation carries phase as sin and cos of phase x 2pi, so phase 1.0 and
phase 0.0 encode to exactly the same pair. For a one-shot clip those are
different instants: the end of a backflip is not the start of one, and the
reference velocity there is not the same number. Reading the encoding alone put
the final step of every episode at the beginning of the clip, which showed up
as a rebuilt `joint_velocity_match` disagreeing with the simulator's by 0.35 on
33 steps out of 2184 - and as nothing at all in the per-joint table, which
looked entirely reasonable while being wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from communication.client import BridgeError
from track_test import decode_phase, field_indices, resolve_phase, wrap

PHASE = 109  # index of phase_sin in the observation


JOINTS = ["waist", "neck", "shoulder_l", "elbow_l", "shoulder_r", "elbow_r",
          "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]


def observation_names() -> list[str]:
    """A layout shaped like the simulator's, with the joint blocks in order."""
    names = ["pelvis_height", "pelvis_sin", "pelvis_cos", "pad_a", "pad_b", "pelvis_w"]
    names += [f"{joint}_{suffix}" for joint in JOINTS for suffix in ("sin", "cos")]
    names += [f"{joint}_vel" for joint in JOINTS]
    names += ["phase_sin", "phase_cos"]
    return names


def test_slots_are_resolved_by_name_not_by_position():
    slot = field_indices(observation_names(), JOINTS)
    names = observation_names()
    for index, joint in enumerate(JOINTS):
        assert names[slot["joint_sin"][index]] == f"{joint}_sin"
        assert names[slot["joint_cos"][index]] == f"{joint}_cos"
        assert names[slot["joint_vel"][index]] == f"{joint}_vel"
    assert names[slot["phase_sin"]] == "phase_sin"


def test_a_reordered_observation_still_reads_the_right_joint():
    """The point of resolving by name rather than tidiness.

    The clip's joint order and the observation's are maintained in two
    different places, and every comparison downstream pairs them index by
    index. Reversing one here must not silently grade the left knee against
    the right hip.
    """
    names = observation_names()
    reversed_joints = list(reversed(JOINTS))
    slot = field_indices(names, reversed_joints)
    for index, joint in enumerate(reversed_joints):
        assert names[slot["joint_vel"][index]] == f"{joint}_vel"


def test_a_layout_without_the_fields_stops_the_run():
    with pytest.raises(BridgeError):
        field_indices(["pelvis_height", "phase_sin", "phase_cos"], JOINTS)
    with pytest.raises(BridgeError):
        # Phase missing: a non-imitation observation, which has nothing to
        # track against and must not be reported as perfect tracking.
        field_indices([n for n in observation_names() if not n.startswith("phase")], JOINTS)


def observation_with_phase(values) -> np.ndarray:
    obs = np.zeros((len(values), PHASE + 2), dtype=np.float64)
    obs[:, PHASE] = np.sin(np.asarray(values) * 2.0 * np.pi)
    obs[:, PHASE + 1] = np.cos(np.asarray(values) * 2.0 * np.pi)
    return obs


def test_decode_round_trips_every_phase_except_the_endpoint():
    wanted = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999])
    assert decode_phase(observation_with_phase(wanted)) == pytest.approx(wanted, abs=1e-9)


def test_one_and_zero_are_indistinguishable_on_the_wire():
    """Not a defect to fix here - it is what the encoding is.

    Asserted on the encoded pair rather than on what decoding returns, because
    decoding cannot round-trip a difference that the two floats on the wire do
    not carry. `mod 1.0` sends the two of them to opposite ends of the range,
    which is precisely why the fold has to be resolved from context.
    """
    encoded = observation_with_phase([0.0, 1.0])
    assert encoded[0, PHASE] == pytest.approx(encoded[1, PHASE], abs=1e-12)
    assert encoded[0, PHASE + 1] == pytest.approx(encoded[1, PHASE + 1], abs=1e-12)


def test_a_backwards_step_in_a_one_shot_clip_is_the_fold():
    encoded = np.array([0.0, 0.0, 0.30])
    previous = np.array([0.98, 0.02, 0.28])
    resolved = resolve_phase(encoded, previous, loops=False)
    assert resolved[0] == 1.0  # ran off the end, clamped, folded onto zero
    assert resolved[1] == 0.0  # genuinely near the start, moving forward
    assert resolved[2] == 0.30  # ordinary step, untouched


def test_a_looping_clip_is_left_alone():
    """A loop really does come back to zero, so the same drop is not a fold."""
    encoded = np.array([0.01])
    previous = np.array([0.99])
    assert resolve_phase(encoded, previous, loops=True)[0] == pytest.approx(0.01)


def test_wrap_matches_the_simulators_convention():
    # A joint near +/-pi must not read as a 2pi error; several pass through it
    # during a flip.
    assert wrap(np.array([np.pi + 0.1]))[0] == pytest.approx(-np.pi + 0.1, abs=1e-9)
    # Several turns out, well clear of the +/-pi knife edge where the sign of a
    # rounding error decides which end you land on.
    assert wrap(np.array([3 * np.pi - 0.1]))[0] == pytest.approx(np.pi - 0.1, abs=1e-9)
    ordinary = np.array([-1.0, 0.0, 1.0])
    assert wrap(ordinary) == pytest.approx(ordinary, abs=1e-12)
