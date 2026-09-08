# Wire protocol

The simulator (`aibf_env`) and the trainer (Python) talk over UDP on localhost.
Implementations: `cpp/net/Protocol.cpp` and `python/communication/protocol.py`. They agree
byte for byte, and `python/tests/test_protocol_fixtures.py` proves it in both directions
against packets written by `aibf_fixture`.

## Ground rules

- **Little-endian, always.** Every field is written a byte at a time. Nothing depends on host
  byte order, and the header layout is pinned by a test on both sides rather than only
  round-tripped — a round trip alone would pass even if both implementations were big-endian.
- **No struct punning.** Nothing is memcpy'd across the boundary. C++ padding rules and numpy
  dtype alignment are free to disagree, and a mismatch there surfaces as a policy learning from
  garbage rather than as a crash.
- **Struct-of-arrays payloads.** All observations, then all reward terms, then the flags. Each
  block is one `numpy` view on the Python side instead of a per-environment unpack.
- **Floats are IEEE-754 binary32.** The only thing assumed rather than encoded. True on every
  platform either side runs on.
- **Decoders validate everything.** A UDP socket accepts datagrams from anywhere. Every length
  is checked against the remaining buffer before it is used to size anything.
- **Maximum datagram: 60000 bytes.** Inside the 65507-byte UDP payload limit with room to
  spare. At 32 environments and a 111-dimensional observation a STATE packet is about 15 KB.

## Header — 16 bytes, on every message

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | magic | `"AIBF"`, i.e. `0x46424941` as a little-endian u32 |
| 4 | 2 | version | currently 1; a mismatch is rejected, not tolerated |
| 6 | 2 | type | see below |
| 8 | 4 | session | identifies one run of the simulator |
| 12 | 4 | step | which control step this message concerns |

`session` is generated when the simulator starts. If the server restarts, the session changes
and the client raises instead of silently continuing against a fresh set of environments that
merely looks like the old one.

## Message types

| Value | Name | Direction |
|---|---|---|
| 1 | HELLO | Python → C++ |
| 2 | SPEC | C++ → Python |
| 3 | ACTION | Python → C++ |
| 4 | STATE | C++ → Python |
| 5 | RESET | Python → C++ |
| 6 | BYE | Python → C++ |
| 7 | ERROR | C++ → Python |

### HELLO → SPEC

```
HELLO:  u16 num_envs
        u64 seed

SPEC:   u16 num_envs, obs_dim, action_dim, reward_dim
        f32 control_hz, physics_hz
        u16 max_episode_steps
        f32[action_dim] action_lower      (joint limits)
        f32[action_dim] action_upper
        u16 count, then count x (u16 length, utf-8 bytes)   -- action names
        u16 count, then count x (u16 length, utf-8 bytes)   -- reward term names
        u16 count, then count x (u16 length, utf-8 bytes)   -- observation names
```

Dimensions and joint limits come *from the server*. Nothing on the Python side hardcodes an
observation size that a config change could invalidate. The names are what make per-component
reward logging possible, and they let the trainer refuse a config that references a reward term
the simulator does not provide instead of silently weighting it at zero.

A HELLO naming a different `num_envs` rebuilds the batch. Reconnecting with a different setting
is a normal thing to want, and the alternative is a silent mismatch.

### RESET → STATE

```
RESET:  u64 seed
        u8  reseed      (non-zero re-seeds the environments from `seed`)
```

Resets every environment and returns the batch to step 0.

### ACTION → STATE

```
ACTION: u16 num_envs, action_dim
        u8[num_envs]                 reset mask
        f32[num_envs * action_dim]   actions, normalized to [-1, 1]

STATE:  u16 num_envs, obs_dim, reward_dim
        f32[num_envs * obs_dim]      observations
        f32[num_envs * reward_dim]   reward terms (raw, unweighted)
        u8[num_envs]                 terminated
        u8[num_envs]                 truncated
        u32[num_envs]                episode step
        u8[num_envs]                 final mask
        u16                          final row count
        f32[rows * obs_dim]          final observations, dense
```

Actions outside `[-1, 1]` are clamped, not extrapolated. A Gaussian policy samples outside the
interval constantly, and extrapolating would hand the solver targets outside the joint's own
limits for the limit constraints to fight every step.

`reset mask` force-restarts an environment instead of stepping it, which is how a caller begins
a fresh rollout mid-stream. Ordinary episode ends do not need it — see below.

## Auto-reset and the final observation

Environments reset themselves when an episode ends. In the STATE that reports the end:

- `observations` is already the **new** episode's first observation,
- `reward terms`, `terminated`, `truncated` and `episode step` describe the episode that
  **ended**,
- `final mask` marks which environments ended, and `final observations` holds the state each of
  them ended in, packed densely in environment order.

Both halves are load-bearing. Auto-reset keeps every step a real transition, so a rollout is a
fixed (steps × environments) block with no holes to mask out. The final observation is still
required because a **truncated** episode has to bootstrap its value estimate from the state it
was cut off in — dropping it teaches the critic that reaching the time limit is worth zero,
which is the classic time-limit bootstrapping bug.

## Reliability

There are no retransmission timers and no sequence-number bookkeeping. The exchange is made
idempotent instead:

- An ACTION for the **current** step is applied once and answered with a fresh STATE.
- An ACTION for the step **just completed** means the client never received that reply, so the
  cached STATE is sent again. The simulation does not advance.
- Anything else is stale and dropped.

So a datagram lost in either direction costs exactly one retry, and a retry can never cause the
simulation to step twice for one action. `test_a_lost_reply_costs_one_retry_and_not_a_double_step`
replays a datagram to check this rather than trusting the argument.

The client retries with a timeout and gives up with a message naming the address, so a training
run started without the simulator fails in seconds instead of hanging.

A BYE makes the server forget its client but keep listening; a trainer that crashes and
restarts reconnects without the simulator being relaunched. `--exit-on-bye` opts into the
older shut-down-on-disconnect behaviour for scripted one-shot runs.

## Measured

Localhost, RTX 4090 workstation, one `aibf_env` process, `python/random_agent.py`:

| Environments | Round trips/s | Env-steps/s | vs real time |
|---|---|---|---|
| 1 | 6 831 | 6 831 | 114x |
| 8 | 2 918 | 23 346 | 49x |
| 25 | 1 328 | 33 198 | 22x |
| 32 | 1 073 | 34 352 | 18x |

Across 2002 round trips at 25 environments: zero timeouts, zero stale packets, zero malformed
packets.

The single-environment figure is the transport ceiling — about 6 800 round trips per second,
dominated by Python-side overhead rather than by the socket. At 32 environments the loop only
needs 1 073 of those, so there is roughly **6x headroom** and the bridge is not the limiting
factor; physics is. Headless physics alone runs at 37 700 env-steps/s for the same batch, so
the bridge costs about 9%.

That retires the plan's concern about UDP becoming the bottleneck. Shared memory stays unbuilt
until something measures slow.
