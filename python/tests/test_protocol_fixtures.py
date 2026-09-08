"""Cross-language protocol agreement.

Round-tripping a packet through one language proves only that the codec is
self-consistent. These tests exercise the seam: C++ writes packets that Python
decodes, and Python writes a packet that C++ decodes. A byte-order or layout
disagreement fails here and nowhere else.

The values are deliberately awkward - negative zero, denormals, the maximum
uint32, floats whose bytes read differently under either endianness - so an
accidental agreement is not possible.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from communication import protocol
from communication.protocol import Header, MessageType, ProtocolError


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory, request) -> Path:
    binary = request.getfixturevalue("fixture_binary")
    directory = tmp_path_factory.mktemp("protocol")
    result = subprocess.run(
        [str(binary), "write", str(directory)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"aibf_fixture write failed:\n{result.stdout}{result.stderr}"
    return directory


def test_cpp_spec_decodes_in_python(fixtures: Path):
    spec = protocol.decode_spec((fixtures / "spec.bin").read_bytes())

    assert spec.num_envs == 3
    assert spec.obs_dim == 5
    assert spec.action_dim == 2
    assert spec.reward_dim == 2
    assert spec.control_hz == pytest.approx(60.0)
    assert spec.physics_hz == pytest.approx(240.0)
    assert spec.max_episode_steps == 1000
    assert spec.action_low.tolist() == pytest.approx([-2.5, -0.125])
    assert spec.action_high.tolist() == pytest.approx([0.75, 3.0])
    assert spec.action_names == ["knee_l", "hip_l"]
    assert spec.reward_names == ["alive", "torque_cost"]
    assert spec.observation_names == ["a", "b", "c", "d", "e"]


def test_cpp_state_decodes_in_python(fixtures: Path):
    data = (fixtures / "state.bin").read_bytes()
    header = protocol.decode_header(data)
    assert header.type is MessageType.STATE
    assert header.session == 0xABCD1234
    assert header.step == 42

    state = protocol.decode_state(data)
    assert state.observations.shape == (3, 5)
    assert state.reward_terms.shape == (3, 2)

    # Bit-exact, not approximate. A float that survives to seven digits but
    # differs in its last bit is a transport bug hiding as a precision issue.
    expected_first_row = np.array(
        [0.0, -0.0, 0.5, -0.25, 1.0 / 3.0], dtype=np.float32
    )
    assert state.observations[0].tobytes() == expected_first_row.tobytes()
    # Negative zero survived rather than being normalised to +0.
    assert np.signbit(state.observations[0][1])

    assert state.observations[1][2] == np.float32(12345.6789)
    assert state.observations[2].tolist() == pytest.approx([3, 4, 5, 6, 7])

    assert state.terminated.tolist() == [False, True, False]
    assert state.truncated.tolist() == [False, False, True]
    assert state.done.tolist() == [False, True, True]
    assert state.episode_step.tolist() == [7, 250, 4294967295]

    assert state.final_mask.tolist() == [False, True, True]
    assert state.final_observations.shape == (2, 5)
    assert state.final_observations[0].tolist() == pytest.approx([-1, -2, -3, -4, -5])
    assert state.final_observations[1].tolist() == pytest.approx([-6, -7, -8, -9, -10])


def test_final_observations_scatter_back_to_environment_rows(fixtures: Path):
    state = protocol.decode_state((fixtures / "state.bin").read_bytes())
    by_env = state.final_observations_by_env()
    assert by_env.shape == (3, 5)
    assert by_env[0].tolist() == [0, 0, 0, 0, 0]  # env 0 did not finish
    assert by_env[1].tolist() == pytest.approx([-1, -2, -3, -4, -5])
    assert by_env[2].tolist() == pytest.approx([-6, -7, -8, -9, -10])


def test_python_action_decodes_in_cpp(fixture_binary: Path, tmp_path: Path):
    actions = np.array([[-1.0, 0.0], [1.0, 0.5], [-0.5, 0.25]], dtype=np.float32)
    reset_mask = np.array([1, 0, 1], dtype=np.uint8)
    packet = protocol.encode_action(
        Header(MessageType.ACTION, session=0x0BADF00D, step=99), actions, reset_mask
    )

    path = tmp_path / "action.bin"
    path.write_bytes(packet)

    result = subprocess.run(
        [str(fixture_binary), "check", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert "OK" in result.stdout


def test_header_bytes_are_little_endian():
    # Pinned directly, because a round trip alone would pass even if both sides
    # were big-endian, and the C++ side asserts the same bytes.
    packet = protocol.encode_header(
        Header(MessageType.STATE, session=0x11223344, step=0xAABBCCDD)
    )
    assert len(packet) == protocol.HEADER_BYTES == 16
    assert packet[0:4] == b"AIBF"
    assert packet[4:6] == struct.pack("<H", protocol.VERSION)
    assert packet[6:8] == struct.pack("<H", int(MessageType.STATE))
    assert packet[8:12] == bytes([0x44, 0x33, 0x22, 0x11])
    assert packet[12:16] == bytes([0xDD, 0xCC, 0xBB, 0xAA])


def test_malformed_packets_raise_rather_than_crash(fixtures: Path):
    data = (fixtures / "state.bin").read_bytes()

    with pytest.raises(ProtocolError):
        protocol.decode_header(b"")
    with pytest.raises(ProtocolError):
        protocol.decode_header(b"XXXX" + data[4:])
    with pytest.raises(ProtocolError):
        protocol.decode_header(data[:4] + b"\x63\x00" + data[6:])  # version 99
    with pytest.raises(ProtocolError):
        protocol.decode_spec(data)  # a STATE decoded as a SPEC

    # Every truncation must be rejected, not just a convenient one.
    for cut in range(len(data)):
        with pytest.raises(ProtocolError):
            protocol.decode_state(data[:cut])


def test_a_lying_count_cannot_force_a_huge_allocation(fixtures: Path):
    data = bytearray((fixtures / "state.bin").read_bytes())
    data[protocol.HEADER_BYTES + 0 : protocol.HEADER_BYTES + 2] = struct.pack("<H", 65535)
    data[protocol.HEADER_BYTES + 2 : protocol.HEADER_BYTES + 4] = struct.pack("<H", 65535)
    with pytest.raises(ProtocolError):
        protocol.decode_state(bytes(data))


def test_encode_action_rejects_a_mismatched_reset_mask():
    actions = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ProtocolError):
        protocol.encode_action(
            Header(MessageType.ACTION), actions, np.zeros(2, dtype=np.uint8)
        )
    with pytest.raises(ProtocolError):
        protocol.encode_action(
            Header(MessageType.ACTION), np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.uint8)
        )


def test_action_encoding_is_float32_regardless_of_input_dtype():
    # numpy defaults to float64; sending that would double every packet and
    # desynchronise the layout.
    actions = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    packet = protocol.encode_action(
        Header(MessageType.ACTION), actions, np.zeros(2, dtype=np.uint8)
    )
    payload = packet[protocol.HEADER_BYTES + 4 + 2 :]
    assert len(payload) == 2 * 2 * 4
    assert np.frombuffer(payload, dtype="<f4").tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
