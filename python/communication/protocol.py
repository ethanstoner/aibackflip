"""Wire format for the C++/Python bridge.

This is the mirror of ``cpp/net/Protocol.cpp`` and has to agree with it byte for
byte. Everything is little-endian and packed explicitly with ``struct`` and
``numpy``; nothing relies on ``ctypes`` layout or dtype alignment, because C++
padding rules and numpy alignment are free to disagree and a mismatch there
shows up as a policy learning from garbage rather than as a crash.

The byte layout is documented in ``docs/PROTOCOL.md``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

MAGIC = 0x46424941  # b"AIBF" read as a little-endian uint32
VERSION = 1

# Largest datagram either side will send or accept. Matches kMaxPacketBytes.
MAX_PACKET_BYTES = 60000

_HEADER = struct.Struct("<IHHII")  # magic, version, type, session, step
HEADER_BYTES = _HEADER.size
assert HEADER_BYTES == 16

_U16 = struct.Struct("<H")
_U64 = struct.Struct("<Q")
_F32 = struct.Struct("<f")


class MessageType(IntEnum):
    HELLO = 1
    SPEC = 2
    ACTION = 3
    STATE = 4
    RESET = 5
    BYE = 6
    ERROR = 7


class ProtocolError(Exception):
    """Raised when a datagram cannot be decoded."""


@dataclass(frozen=True)
class Header:
    type: MessageType
    session: int = 0
    step: int = 0


@dataclass
class Spec:
    """Everything the trainer needs in order to talk to the simulator.

    Dimensions come from the server rather than being hardcoded here, so a
    config change on the C++ side cannot silently desynchronise the two.
    """

    num_envs: int
    obs_dim: int
    action_dim: int
    reward_dim: int
    control_hz: float
    physics_hz: float
    max_episode_steps: int
    action_low: np.ndarray
    action_high: np.ndarray
    action_names: list[str] = field(default_factory=list)
    reward_names: list[str] = field(default_factory=list)
    observation_names: list[str] = field(default_factory=list)


@dataclass
class State:
    """One control step's result for the whole batch.

    ``observations`` is post-reset for any environment that finished this step;
    everything else describes the episode that ended. The state such an episode
    ended in is in ``final_observations``, which a truncated episode needs in
    order to bootstrap its value estimate.
    """

    step: int
    session: int
    observations: np.ndarray  # (num_envs, obs_dim) float32
    reward_terms: np.ndarray  # (num_envs, reward_dim) float32
    terminated: np.ndarray  # (num_envs,) bool
    truncated: np.ndarray  # (num_envs,) bool
    episode_step: np.ndarray  # (num_envs,) uint32
    final_mask: np.ndarray  # (num_envs,) bool
    final_observations: np.ndarray  # (num_finished, obs_dim) float32, dense

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated

    def final_observations_by_env(self) -> np.ndarray:
        """The dense final-observation block scattered back to (num_envs, obs_dim).

        Rows for environments that did not finish are zero. Convenient when the
        caller would rather index by environment than track the packing.
        """
        num_envs, obs_dim = self.observations.shape
        out = np.zeros((num_envs, obs_dim), dtype=np.float32)
        if self.final_observations.size:
            out[self.final_mask] = self.final_observations
        return out


# ---------------------------------------------------------------- encoding


def encode_header(header: Header) -> bytes:
    return _HEADER.pack(MAGIC, VERSION, int(header.type), header.session, header.step)


def encode_hello(header: Header, num_envs: int, seed: int) -> bytes:
    return encode_header(header) + _U16.pack(num_envs) + _U64.pack(seed & 0xFFFFFFFFFFFFFFFF)


def encode_reset(header: Header, seed: int = 0, reseed: bool = False) -> bytes:
    return (
        encode_header(header)
        + _U64.pack(seed & 0xFFFFFFFFFFFFFFFF)
        + bytes([1 if reseed else 0])
    )


def encode_bye(header: Header) -> bytes:
    return encode_header(header)


def encode_action(header: Header, actions: np.ndarray, reset_mask: np.ndarray) -> bytes:
    """``actions`` is (num_envs, action_dim); ``reset_mask`` is (num_envs,)."""
    actions = np.ascontiguousarray(actions, dtype="<f4")
    if actions.ndim != 2:
        raise ProtocolError(f"actions must be 2-D, got shape {actions.shape}")
    num_envs, action_dim = actions.shape

    mask = np.ascontiguousarray(reset_mask, dtype=np.uint8).reshape(-1)
    if mask.size != num_envs:
        raise ProtocolError(f"reset mask has {mask.size} entries, expected {num_envs}")

    return (
        encode_header(header)
        + _U16.pack(num_envs)
        + _U16.pack(action_dim)
        + mask.tobytes()
        + actions.tobytes()
    )


# ---------------------------------------------------------------- decoding


class _Reader:
    """Bounds-checked cursor. Every read validates, because a UDP socket
    accepts datagrams from anywhere and a decoder that trusts its input is a
    denial-of-service waiting for a stray packet."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self._data = data
        self._pos = pos

    def _take(self, count: int) -> memoryview:
        end = self._pos + count
        if count < 0 or end > len(self._data):
            raise ProtocolError(
                f"packet truncated: wanted {count} bytes at offset {self._pos}, "
                f"have {len(self._data) - self._pos}"
            )
        chunk = memoryview(self._data)[self._pos : end]
        self._pos = end
        return chunk

    def u16(self) -> int:
        return _U16.unpack(self._take(2))[0]

    def u64(self) -> int:
        return _U64.unpack(self._take(8))[0]

    def f32(self) -> float:
        return _F32.unpack(self._take(4))[0]

    def u8(self) -> int:
        return self._take(1)[0]

    def string(self) -> str:
        length = self.u16()
        return bytes(self._take(length)).decode("utf-8", errors="replace")

    def f32_array(self, count: int) -> np.ndarray:
        # Copied rather than viewed: the caller's buffer is a transient recv
        # buffer, and a view into it would alias the next packet.
        return np.frombuffer(self._take(count * 4), dtype="<f4").astype(np.float32, copy=True)

    def u8_array(self, count: int) -> np.ndarray:
        return np.frombuffer(self._take(count), dtype=np.uint8).copy()

    def u32_array(self, count: int) -> np.ndarray:
        return np.frombuffer(self._take(count * 4), dtype="<u4").copy()

    def string_list(self) -> list[str]:
        return [self.string() for _ in range(self.u16())]


def decode_header(data: bytes) -> Header:
    if not data or len(data) < HEADER_BYTES:
        raise ProtocolError(f"packet too short for a header: {len(data)} bytes")
    if len(data) > MAX_PACKET_BYTES:
        raise ProtocolError(f"packet too large: {len(data)} bytes")

    magic, version, raw_type, session, step = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:08X}")
    if version != VERSION:
        raise ProtocolError(f"protocol version {version}, expected {VERSION}")
    try:
        message_type = MessageType(raw_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown message type {raw_type}") from exc
    return Header(type=message_type, session=session, step=step)


def _open(data: bytes, expected: MessageType) -> tuple[Header, _Reader]:
    header = decode_header(data)
    if header.type is not expected:
        raise ProtocolError(f"expected {expected.name}, got {header.type.name}")
    return header, _Reader(data, HEADER_BYTES)


def decode_spec(data: bytes) -> Spec:
    _, reader = _open(data, MessageType.SPEC)
    num_envs = reader.u16()
    obs_dim = reader.u16()
    action_dim = reader.u16()
    reward_dim = reader.u16()
    control_hz = reader.f32()
    physics_hz = reader.f32()
    max_episode_steps = reader.u16()
    action_low = reader.f32_array(action_dim)
    action_high = reader.f32_array(action_dim)
    return Spec(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        control_hz=control_hz,
        physics_hz=physics_hz,
        max_episode_steps=max_episode_steps,
        action_low=action_low,
        action_high=action_high,
        action_names=reader.string_list(),
        reward_names=reader.string_list(),
        observation_names=reader.string_list(),
    )


def decode_state(data: bytes) -> State:
    header, reader = _open(data, MessageType.STATE)
    num_envs = reader.u16()
    obs_dim = reader.u16()
    reward_dim = reader.u16()

    # Guard the arithmetic before allocating: a corrupt count in a 40-byte
    # datagram must not ask numpy for gigabytes.
    needed = num_envs * (obs_dim + reward_dim) * 4 + num_envs * 7 + 2
    if HEADER_BYTES + 6 + needed > len(data):
        raise ProtocolError(
            f"state packet claims {num_envs}x{obs_dim} observations "
            f"but is only {len(data)} bytes"
        )

    observations = reader.f32_array(num_envs * obs_dim).reshape(num_envs, obs_dim)
    reward_terms = reader.f32_array(num_envs * reward_dim).reshape(num_envs, reward_dim)
    terminated = reader.u8_array(num_envs).astype(bool)
    truncated = reader.u8_array(num_envs).astype(bool)
    episode_step = reader.u32_array(num_envs)
    final_mask = reader.u8_array(num_envs).astype(bool)

    final_rows = reader.u16()
    if final_rows > num_envs:
        # popcount of a mask over num_envs entries cannot exceed num_envs.
        raise ProtocolError(f"state packet claims {final_rows} final observations for {num_envs} envs")
    final_observations = reader.f32_array(final_rows * obs_dim).reshape(final_rows, obs_dim)

    if int(final_mask.sum()) != final_rows:
        raise ProtocolError(
            f"final mask selects {int(final_mask.sum())} environments "
            f"but {final_rows} observations were sent"
        )

    return State(
        step=header.step,
        session=header.session,
        observations=observations,
        reward_terms=reward_terms,
        terminated=terminated,
        truncated=truncated,
        episode_step=episode_step,
        final_mask=final_mask,
        final_observations=final_observations,
    )


def decode_error(data: bytes) -> str:
    _, reader = _open(data, MessageType.ERROR)
    return reader.string()
