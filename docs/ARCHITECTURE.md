# Architecture

## Shape of the system

```
   C++ process                                    Python process
   ------------------------------                 ---------------------------
   EnvBatch (N humanoids)                         PPO trainer
     physics world  (240 Hz)                        actor-critic (PyTorch, CUDA)
     humanoid + joint motors                        observation normalization
     observation assembly            <-- UDP -->    rollout buffer + GAE
     reward *components*                            checkpointing
     termination                                    TensorBoard logging
     renderer (optional)
```

The simulator owns physics, contacts, and episode lifetime. The trainer owns the policy and
the optimizer. They meet at a single batched UDP round trip per control step.

## Why the pieces are the way they are

### Physics runs at 240 Hz, control at 60 Hz

Rendering frame rate never touches physics. One control step advances four fixed 1/240 s
physics substeps, so a policy trained headless at 20x real time behaves identically to one
being watched at 60 FPS. Substepping also lets the constraint solver stay stable at joint
torques high enough to throw a humanoid into the air, which a single 60 Hz step will not do.

### Own math library, no GLM

`cpp/core/Math.h` is dependency-free: Vec2/Vec3/Vec4, Mat2/Mat3/Mat4, and a quaternion type.
The plan originally called for GLM, but the physics needs quaternion integration,
skew-symmetric matrices, inertia-tensor inverses, and a 2x2 block solve — none of which GLM
provides for free — and the remainder (projection matrices) is about sixty lines. Owning the
whole thing means every operation the solver depends on has a test pinning its convention,
which matters more here than saving those sixty lines.

Two conventions are load-bearing and tested:

- Matrices are **column-major**, so `Mat3`/`Mat4` upload to OpenGL untransposed.
- 2D cross products follow Box2D: `cross(a,b)` is the scalar z-component, `cross(w, r)` is
  `omega x r` giving the velocity of a point at offset `r`, and `cross(r, w)` is its exact
  negation. If those two overloads ever disagree in sign, the solver pumps energy into the
  system instead of removing it, and the humanoid explodes. `Vec2.crossConventionsAgree`
  exists to catch precisely that.

### `Real` is a typedef

Everything is `float`. If an instability turns out to be precision-related rather than a
solver bug, `using Real = double;` in `Math.h` switches the entire codebase, and the wire
protocol converts explicitly rather than memcpy-ing, so nothing else breaks.

Fast-math is disabled in every build configuration (`/fp:precise`, `-ffp-contract=off`). NaN
and infinity checks in the solver are load-bearing; `/fp:fast` permits the compiler to assume
they cannot fire.

### Deterministic RNG

`cpp/core/Rng.h` is PCG32, not `<random>`. Standard-library distributions are not specified
to produce identical sequences across implementations, so a seed would not reproduce a
training run. PCG32 does, and it is faster than Mersenne Twister for this workload.

### No OpenGL loader dependency

`cpp/engine/GL.cpp` resolves the ~40 GL 3.3 core entry points actually used through
`glfwGetProcAddress`. Vendoring generated glad output would add a build-time code generator
for something the renderer can declare explicitly.

### One binary, two modes

`--headless` skips window creation entirely rather than selecting a different code path.
Training and playback step the identical simulation, so what you watch is what you trained.

### Batched UDP, explicit serialization

A single datagram carries every environment's observation. At 32 envs and a 186-dimensional
3D observation that is roughly 26 KB, comfortably inside the 65507-byte UDP payload limit,
and it costs one round trip per control step instead of 32.

Serialization is explicit little-endian `putF32`/`getF32` on both sides. Nothing memcpys a
struct across the boundary, so C++ padding rules and `numpy` dtype alignment cannot silently
disagree. A HELLO/SPEC handshake has the simulator report observation, action, and reward
dimensions plus joint limits, so the Python side never hardcodes a dimension that a config
change can invalidate.

Both sides use socket timeouts with bounded retries, and every packet carries a step number
so a stale or duplicated datagram is dropped rather than applied.

Full wire format: `docs/PROTOCOL.md` (written in M3).

### Reward components in C++, weights in Python

The simulator computes raw reward *terms* — height, uprightness, contact state, tracking
errors — because it owns the contacts and the centre of mass. It ships the vector of terms;
Python applies configurable weights and sums them.

This buys two things. Reward weights become a Python config change rather than a C++ rebuild,
and every component gets logged separately, which is the only practical way to catch reward
hacking: a policy that has found a degenerate exploit shows it as one component saturating
while the others flatline.

Termination stays in C++, next to the state it inspects.

### Reference State Initialization is not an optimization

For the imitation milestones, episodes start at a *random phase* of the reference motion
rather than always at frame zero. Without it, a policy has to master the takeoff before it
ever observes the landing, and acrobatic motions essentially never train. It is designed in
from the start, not retrofitted when training stalls.

## Directory map

| Path | Contents |
|---|---|
| `cpp/core/` | `Math.h`, `Rng.h`, `Test.h/.cpp`, JSON + config loading |
| `cpp/engine/` | window, GL loader, 2D and 3D renderers, camera, input, debug draw |
| `cpp/physics/` | rigid bodies, collision, joints, sequential-impulse solver, worlds |
| `cpp/humanoid/` | body construction, observation assembly, reward terms |
| `cpp/env/` | episodic environment wrapper and the parallel batch |
| `cpp/motion/` | keyframe clips, JSON serialization, phase sampling |
| `cpp/net/` | wire protocol and UDP server |
| `python/rl/` | policy, PPO, rollout buffer, reward weighting, normalization |
| `python/communication/` | protocol codec, UDP client, vectorized env adapter |

## Testing approach

The C++ suite (`cpp/tests/`) uses a small self-registering framework in `core/Test.h` — no
external dependency, so a fresh clone can run the physics tests before fetching anything.

Physics correctness is asserted numerically, not by looking at the window: impulse responses
against closed-form results, joint anchor drift under load, energy behaviour, and finite-
difference checks against the analytic formulas the solver uses. `Vec2.pointVelocityMatchesFiniteDifference`
is the template — it checks `omega x r` against the numerical derivative of an actual
rotation rather than restating the same formula twice.

The Python suite (`python/tests/`) covers the protocol codec, action mapping, reward
weighting, GAE, PPO clipping, and checkpoint round-trips, plus environment preconditions that
would otherwise surface hours into a training run.
