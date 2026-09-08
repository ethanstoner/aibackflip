"""UDP client for the humanoid environment server.

Reliability is built on making the exchange idempotent rather than on invented
sequence numbers. Every ACTION names the step it is answering, and the server
either applies it once or - if it has already advanced - replies with the cached
STATE for that step. So a retry after a lost datagram is always safe: it can
never cause the simulation to step twice for one action.

Failure handling is explicit because a training run is long and unattended:

* a lost datagram in either direction costs one retry,
* a server that restarts is detected through the session id and re-handshaked,
* a server that stays silent raises rather than blocking forever.
"""

from __future__ import annotations

import socket
import time

import numpy as np

from . import protocol
from .protocol import Header, MessageType, ProtocolError, Spec, State


class BridgeError(RuntimeError):
    """The simulator could not be reached or spoke nonsense."""


class EnvClient:
    """Request/response client for one environment server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 51234,
        *,
        timeout: float = 1.0,
        retries: int = 40,
        receive_buffer_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.address = (host, port)
        self.timeout = timeout
        self.retries = retries

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(timeout)
        try:
            # The default receive buffer is small enough that a burst of 26 KB
            # datagrams can be dropped by the kernel before Python sees them.
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_bytes)
        except OSError:
            pass

        self.spec: Spec | None = None
        self.session = 0
        self.step = 0
        self.closed = False

        # Counters worth logging; a rising retry count usually means the
        # simulator is saturated rather than that anything is broken.
        self.stats = {
            "packets_sent": 0,
            "packets_received": 0,
            "timeouts": 0,
            "stale_dropped": 0,
            "malformed_dropped": 0,
            "reconnects": 0,
        }

    # ------------------------------------------------------------------ setup

    def connect(self, num_envs: int = 25, seed: int = 0) -> Spec:
        """Handshakes and returns the server's spec.

        Dimensions are read from the reply rather than assumed, so nothing on
        this side hardcodes an observation size that a config change can
        invalidate.
        """
        request = protocol.encode_hello(Header(MessageType.HELLO), num_envs, seed)
        reply = self._exchange(request, MessageType.SPEC, match_step=None)
        spec = protocol.decode_spec(reply)

        if spec.num_envs != num_envs:
            raise BridgeError(
                f"asked for {num_envs} environments, server built {spec.num_envs}"
            )
        if len(spec.action_low) != spec.action_dim:
            raise BridgeError("spec action limits do not match the action dimension")

        self.spec = spec
        self.session = protocol.decode_header(reply).session
        return spec

    def _require_spec(self) -> Spec:
        if self.spec is None:
            raise BridgeError("not connected; call connect() first")
        return self.spec

    # ------------------------------------------------------------------ rpc

    def reset(self, seed: int | None = None) -> State:
        self._require_spec()
        request = protocol.encode_reset(
            Header(MessageType.RESET, session=self.session),
            seed=seed or 0,
            reseed=seed is not None,
        )
        reply = self._exchange(request, MessageType.STATE, match_step=0)
        state = protocol.decode_state(reply)
        self.step = state.step
        self.session = state.session
        return state

    def step_batch(self, actions: np.ndarray, reset_mask: np.ndarray | None = None) -> State:
        """Applies one control step and returns the resulting state.

        ``reset_mask`` names environments to restart instead of stepping, which
        is how a finished episode is recycled. Resets are explicit so the caller
        gets to see the terminal observation first - PPO needs it to bootstrap
        the value function through a time-limit truncation.
        """
        spec = self._require_spec()
        actions = np.asarray(actions, dtype=np.float32).reshape(spec.num_envs, spec.action_dim)
        if reset_mask is None:
            reset_mask = np.zeros(spec.num_envs, dtype=np.uint8)

        request = protocol.encode_action(
            Header(MessageType.ACTION, session=self.session, step=self.step), actions, reset_mask
        )
        reply = self._exchange(request, MessageType.STATE, match_step=self.step + 1)
        state = protocol.decode_state(reply)
        self.step = state.step
        return state

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.spec is not None:
                self._socket.sendto(
                    protocol.encode_bye(Header(MessageType.BYE, session=self.session)),
                    self.address,
                )
        except OSError:
            pass
        finally:
            self._socket.close()

    def __enter__(self) -> "EnvClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport

    def _exchange(
        self, request: bytes, expected: MessageType, match_step: int | None
    ) -> bytes:
        """Sends `request` and waits for a matching reply, retrying on loss.

        Resending is safe by construction: the server treats a repeated request
        for a step it has already served as a request to resend, not to step
        again.
        """
        if self.closed:
            raise BridgeError("client is closed")

        last_problem = "no reply"
        for attempt in range(self.retries):
            try:
                self._socket.sendto(request, self.address)
                self.stats["packets_sent"] += 1
            except OSError as exc:
                raise BridgeError(f"could not send to {self.address}: {exc}") from exc

            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stats["timeouts"] += 1
                    last_problem = "timed out waiting for a reply"
                    break
                try:
                    self._socket.settimeout(remaining)
                    data, _ = self._socket.recvfrom(protocol.MAX_PACKET_BYTES)
                except socket.timeout:
                    self.stats["timeouts"] += 1
                    last_problem = "timed out waiting for a reply"
                    break
                except OSError as exc:
                    # On Windows a UDP send to a port with no listener makes the
                    # *next* recv fail with WSAECONNRESET. That means "not
                    # listening yet", so retry rather than give up.
                    last_problem = f"socket error: {exc}"
                    break

                self.stats["packets_received"] += 1
                try:
                    header = protocol.decode_header(data)
                except ProtocolError as exc:
                    self.stats["malformed_dropped"] += 1
                    last_problem = str(exc)
                    continue

                if header.type is MessageType.ERROR:
                    raise BridgeError(f"server refused the request: {protocol.decode_error(data)}")

                if header.type is not expected:
                    self.stats["stale_dropped"] += 1
                    continue

                if self.session and header.session and header.session != self.session:
                    # The simulator restarted. Nothing the caller holds is valid
                    # any more, so say so rather than silently continuing
                    # against a fresh set of environments.
                    raise BridgeError(
                        "simulator session changed (server restarted); reconnect required"
                    )

                if match_step is not None and header.step != match_step:
                    self.stats["stale_dropped"] += 1
                    continue

                return data

        raise BridgeError(
            f"no valid {expected.name} from {self.address[0]}:{self.address[1]} "
            f"after {self.retries} attempts ({last_problem}). Is aibf_env running?"
        )
