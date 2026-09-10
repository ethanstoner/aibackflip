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

from track_test import decode_phase, resolve_phase, wrap

PHASE = 109  # index of phase_sin in the observation


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
