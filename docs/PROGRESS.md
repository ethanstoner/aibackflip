# Progress

Newest milestone last. Failed experiments are recorded alongside successful ones; a milestone
is only marked done when its exit criterion was actually observed, not when the code compiled.

---

## M0 — scaffold, build system, math core

**Status:** done (2026-09-08)

### Implemented

- CMake 3.20 project, C++20, MSVC and GCC warning sets. GLFW 3.4 pulled in via `FetchContent`
  (shallow clone); OpenGL located through `find_package`.
- `cpp/core/Math.h` — Vec2/Vec3/Vec4, Mat2/Mat3/Mat4, quaternions. Column-major matrices,
  Box2D 2D-cross conventions, first-order quaternion integration with renormalization, slerp
  with double-cover handling, projection and view matrices.
- `cpp/core/Rng.h` — PCG32 with uniform, bounded-integer (rejection sampled), Gaussian
  (Marsaglia polar), unit-circle and unit-sphere sampling.
- `cpp/core/Test.h/.cpp` — self-registering, dependency-free test framework with a name filter.
- `cpp/main.cpp` — argument parsing plus a `--check-gl` probe that opens a hidden GL 3.3 core
  context, so toolchain breakage surfaces immediately rather than in M1.
- Python package skeleton, `pytest.ini`, `conftest.py` that puts `python/` on `sys.path`.
- `scripts/build.ps1`, `scripts/test.ps1`.
- `README.md`, `docs/ARCHITECTURE.md`, this file.

### Build / run

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
.\build\bin\Release\aibackflip.exe --check-gl
```

### Tested

- **C++: 29 cases, all passing.** Cross-product convention agreement, `omega x r` against a
  finite-difference derivative, angle wrapping across the pi boundary, Mat2/Mat3 inverse
  round-trips, near-singular and exactly-singular inverse degradation, `skew(a)*b == cross(a,b)`,
  rotation orthonormality, quaternion/matrix agreement, Hamilton product ordering, slerp
  endpoints and short-path selection, projection and view matrix mappings, RNG reproducibility
  and moment checks.
- Quaternion integrator drift budget: spinning at 2 rad/s for one simulated second in 1/240 s
  steps lands within **1e-3 rad** of the analytic angle, and 20 000 steps of random angular
  velocity leaves `|q|` within 1e-4 of unit. That is the error budget the 3D engine inherits.
- **Python: 9 cases, all passing.** CUDA reachable with a real allocation (RTX 4090), default
  dtype float32, gradients flowing through the planned tanh-MLP shape, CPU/GPU forward-pass
  agreement, seeding reproducibility, `SummaryWriter` importable, float32 `tobytes` round-trip.
- `--check-gl` reports `GLFW 3.4.0 (Win32 WGL ...)` and successfully creates a GL 3.3 core
  context.

### Problems hit and how they were resolved

1. **`std::max` not found in `Rng.h`.** Missing `<algorithm>`; MSVC does not transitively
   include it from `<cmath>`. Added the include.
2. **MSVC C4723 "potential divide by 0" in `inverse(Mat2)`.** Two contributing causes. The
   test passed a compile-time-constant singular matrix, which is not representative of how
   the solver calls it, so that was replaced with a loop over progressively-more-parallel
   columns plus an exactly-singular case built from runtime RNG values. MSVC still folds
   `a*b - a*b` to a literal zero and warns about a division the guard above it already
   prevents, so C4723 is now suppressed for that one function with a comment explaining that
   anything reaching the division has `|det| >= 1e-12`. The guard itself was never wrong.
3. **RNG determinism test failed — and the test was the bug.** It compared `a.nextU32()` to
   `b.nextU32()` while also calling `a.uniform()` in the same loop, so `a` ran two draws ahead
   of `b` after the first iteration. Split into three focused cases: same seed reproduces,
   different seeds diverge, different streams diverge.

### Deviations from the plan

- **GLM dropped.** The plan called for fetching GLM for rendering math. The physics needs
  quaternion integration, `skew()`, inertia-tensor inverses and a 2x2 block solve regardless,
  and the rest is projection matrices. Owning all of it removes a dependency and puts every
  convention the solver relies on under test. Rationale in `docs/ARCHITECTURE.md`.
- **venv uses `--system-site-packages`.** Re-downloading ~2.5 GB of CUDA wheels into a project
  venv buys nothing when the system interpreter already has `torch 2.6.0+cu124` working.

### Known issues

- `main.cpp` does not simulate anything yet; that is M1.
- TensorBoard 2.21.0 installs cleanly but has not been exercised beyond an import.

### Next

M1 — 2D rigid-body physics, floor collision, the 13-body humanoid with 12 revolute joints,
and mouse interaction. Exit criterion: the humanoid collapses under gravity in a way that
looks and measures like rigid-body dynamics, joints hold their anchors under load, and
dragging a limb with the mouse moves the whole body through the constraint chain.

---

## M1 — 2D physics engine and humanoid

**Status:** done (2026-09-08)

### Implemented

**Physics** (`cpp/physics/`)

- `RigidBody2D` — capsule bodies (segment along local +Y swept by a radius; a disc is
  `halfLength = 0`), with an exact rectangle-plus-two-half-discs inertia derivation, impulse
  and force application, and `setCapsuleWithMass` which back-solves density from a named mass.
- `Collision2D` — segment/segment closest points, capsule vs half-space (up to two contact
  points, so a foot lies flat instead of pivoting on one corner), capsule vs capsule with a
  finite fallback normal for exactly coincident bodies.
- `Joint2D` — revolute joint stacking a motor, two one-sided limit constraints, and a 2-DOF
  point constraint, solved in that order so the anchor gets the final say.
- `World2D` — sequential-impulse solver with warm starting, speculative contacts, Coulomb
  friction, restitution with a resting threshold, a dedicated position pass, collision groups,
  a soft mouse spring, NaN detection, and diagnostic stats.

**Humanoid** (`cpp/humanoid/`) — 13 links, 12 revolute joints, 69 kg, 1.64 m. Authored as a
rest pose (world position and angle per link, plus each joint's world pivot); local anchors
and joint reference angles are derived from it, which is what makes "every joint reads zero
in the rest pose" true by construction rather than by luck. JSON load/save with per-name
partial overrides, plus a `validate()` that catches the authoring errors that would otherwise
appear as a figure exploding on step one.

**Rendering** (`cpp/engine/`) — hand-written GL 3.3 core loader (29 entry points), GLFW
window with a polled input snapshot, batched 2D renderer, and debug overlays for contacts,
joints, centre of mass, support polygon, joint targets, velocities and motion trails.

**Tooling** — `core/Png.h` writes PNGs with stored-deflate blocks, so the simulator can
screenshot itself with no image library; `--capture` runs a fixed number of simulated seconds
and writes frames, which is how rendered output gets checked rather than asserted.
`aibf_diag` reports measurements instead of assertions.

### Build / run

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
.\build\bin\Release\aibackflip.exe                    # interactive; see --help for controls
.\build\bin\Release\aibackflip.exe --motors
.\build\bin\Release\aibf_diag.exe                     # measurements
.\build\bin\Release\aibackflip.exe --dump-config configs\humanoid2d.json
```

### Tested

**121 C++ cases, all passing** (up from 29). Beyond the M0 math suite: collision manifolds
and normal directions, contact and friction response against closed-form expectations, joint
anchors under load, limits, motors, JSON round-trips, and the humanoid itself.

Measured numbers, from `aibf_diag`:

| Measurement | Result |
|---|---|
| Joint drift, 5-link chain yanked for 6 s — **position solver** | **1.8 mm** worst case |
| Same, Baumgarte velocity bias only | 21.4 mm worst case |
| Resting penetration, humanoid, 30 s | 1.3 mm, constant (no sinking) |
| Anchor error at rest, 30 s | 6e-8 m |
| Energy drift, undamped pendulum | -0.7% at 1 s, -5.6% at 10 s, -16% at 60 s |
| Headless throughput, 32 worlds | 37 700 env-steps/s (20x real time), single-threaded |
| Headless throughput, 1 world | 46 100 steps/s (768x real time) |

Two of those are worth reading carefully. **Energy drift is monotonically negative** at every
horizon — the solver only ever loses energy, never gains it. A constraint solver that gains
energy turns joints into motors and launches the figure, so the sign matters more than the
magnitude here, and 1 s of backflip does not care about a 60 s figure.

**Throughput is physics-bound, not transport-bound.** At 32 worlds the batched control rate is
1 178 steps/s, roughly an order of magnitude under what a localhost UDP round trip can sustain,
which retires the plan's concern about the bridge becoming the bottleneck. Threading across
environments is available later if it is ever needed; it is not needed yet.

Visual verification (`--capture`, frames inspected):

- Rest pose renders standing with both feet flat on the ground and every segment meeting at
  its joint.
- Unactuated, the figure collapses like a ragdoll — buckling at the knees, folding at the
  waist, limbs splaying, and settling on the floor.
- With motors holding the rest pose it stands still indefinitely.

### The motors hold a pose; they do not balance

Worth stating plainly, because the screenshot is misleading. With motors enabled the figure
stands at a pelvis height of 0.9989 m for 30 simulated seconds without drifting. That is not
balance — it is a perfectly symmetric equilibrium held by joint stiffness, with nothing
disturbing it. Shoving it says so:

| Sideways impulse on the pelvis | Outcome after 4 s |
|---|---|
| 0, 5, 15 N·s | still standing |
| 40 N·s | fallen (pelvis 0.12 m, chest inverted) |
| 90 N·s | fallen |

It survives small shoves through stiffness alone and has no recovery strategy whatsoever.
Learning to balance is M5, and being pushed and recovering is M6.

### Problems hit and how they were resolved

1. **Three joint tests failed with anchor errors 10-100x the threshold.** Cause: the test
   scaffolding pinned a rod to a static anchor body that physically overlapped it, so contact
   forces were fighting the joint. Real behaviour, wrong setup — connected bodies necessarily
   overlap at their pivots, which is exactly what collision groups are for. Fixing the tests
   to use a shared group dropped the pendulum's drift from >1e-4 m to under 1e-9 m. The
   humanoid does the same thing.
2. **A joint-limit test reported a 1.34 rad violation.** Not a solver failure: the test started
   the rod horizontal (1.57 rad) while imposing limits of ±0.2, so it began 1.37 rad outside
   its own range and the number being measured was how fast the solver dragged it back.
   Rewritten to start inside the limits; the real worst-case violation under repeated 60 rad/s
   slams is under 0.05 rad.
3. **Every duration in the test helpers was one step short.** `int(seconds / kDt)` with a
   non-representable `kDt` truncates 240.0 to 239. Caught by an exact free-fall assertion that
   was off by precisely one step's worth of travel. Now rounds.
4. **`Json::operator[](0)` was ambiguous.** A literal `0` is a null pointer constant, so it
   converts to `const char*` and therefore to `std::string`; against an auto-vivifying non-const
   `operator[](std::string)` the call ties (string overload wins on the object argument, index
   overload wins on the index). Resolved by making both subscripts const-only and moving
   mutation to an explicit `set()`.
5. **Arms and legs were indistinguishable on screen.** In a sagittal 2D figure every limb
   projects onto the same line, and shades of one blue made poses unreadable. Separate hues per
   limb group, right side darkened, and an explicit back-to-front draw order (far limbs, torso,
   near limbs) rather than index order, which had been burying the arms inside the chest.
6. **Configs dumped as `-0.800000012`.** Correct but unreadable. The JSON writer now emits the
   shortest representation that reads back to the same value, at float precision when the value
   came from a float.

### Known issues

- Capsule-vs-capsule produces a single contact point. Fine today: the humanoid disables its own
  self-collision and the only free bodies are discs. A second point will be needed if two
  elongated props ever have to rest against each other.
- Self-collision is off for the whole figure, so during a tuck the legs can pass through the
  chest. Standard for 2D sagittal humanoids and revisited if an imitation motion needs it.
- No on-screen text yet; live stats go to the window title. A font lands with the M11
  visualisation work.
- The solver is single-threaded. Fast enough (see above), so threading stays unbuilt until
  something measures slow.

### Next

M2 — the motor implementation landed early (it was cheaper to write while the solver was in
hand) and is already covered by six tests plus the perturbation table above. M2 therefore
narrows to: an interactive joint-posing mode for driving targets by keyboard, a torque/gain
sweep to check the defaults are sane per joint rather than plausible in aggregate, and
confirming limbs are driven rather than teleported for every joint, not just the ones tested.

---

## M2 — actuated joints

**Status:** done (2026-09-08)

### Implemented

The motor itself shipped with M1, so M2 is about proving it works on all twelve joints rather
than on the two the pendulum tests covered.

- **Interactive posing.** Up/down selects a joint, left/right drives its target, `0` returns
  every target to the rest pose. Driving a target auto-enables the motors — silently moving a
  value nothing reads is a confusing dead end. The window title shows the selected joint's
  commanded and actual angle side by side, so tracking error is visible while posing.
- **`--pin`** freezes the pelvis so joint ranges can be driven and inspected without the figure
  toppling. Purely a debugging aid.
- **`--pose <name>`** commands a built-in pose (`squat`, `reach`, `tuck`, `lunge`). These set
  joint *targets* only: the PD motors apply bounded torque, gravity and contacts push back, and
  the shape that appears is whatever the solver converges on. A pose the figure cannot
  physically hold simply does not appear.
- **`aibf_diag authority`** sweeps every joint and reports what it can actually do.

### The motor formulation, and why not the obvious one

The spec's `torque = kp*error - kd*rate` is implemented, but solved as an implicit soft
constraint rather than applied as an explicit external torque:

```
gamma (CFM) = 1 / (h*(kd + h*kp))
bias        = kp/(kd + h*kp) * C
softMass    = 1 / (invI_a + invI_b + gamma)
impulse     = -softMass * (Cdot + bias + gamma*accumulated),  clamped to +/- maxTorque*h
```

The explicit form is only stable while `kp*h^2` stays below the joint's inertia, which rules
out the stiffnesses an acrobatic motion needs. `JointMotor.staysStableAtStiffnessThatWouldBlowUpAnExplicitPD`
runs kp = 500 000 at 1/240 s — roughly 2.4x past the explicit stability limit for that link —
and tracks its target to 0.05 rad with zero velocity clamp events.

### Tested

**133 C++ cases, all passing** (up from 121). Eighteen cover the motors specifically.

Per-joint authority, pelvis pinned, commanded 70% of the way toward the further limit and held
2.5 s (`aibf_diag authority`):

| joint | target | reached | error | hold torque | % of ceiling |
|---|---|---|---|---|---|
| waist | -0.560 | -0.569 | 0.0089 | 35.5 N·m | 9% |
| neck | -0.420 | -0.422 | 0.0020 | 0.8 N·m | 1% |
| shoulder (each) | +2.100 | +2.090 | 0.0097 | 7.8 N·m | 6% |
| elbow (each) | +1.890 | +1.886 | 0.0039 | 2.0 N·m | 2% |
| hip (each) | +1.470 | +1.459 | 0.0111 | 44.5 N·m | 11% |
| knee (each) | -1.820 | -1.816 | 0.0038 | 11.5 N·m | 4% |
| ankle (each) | -0.630 | -0.630 | 0.0001 | 0.2 N·m | 0% |

Every joint reaches its commanded angle, worst error 0.0111 rad (0.64°), and none needs more
than 11% of its torque ceiling to hold. The residual error is not a defect: a proportional
motor holding a load settles at (load torque)/kp by definition.

Peak torque hits 100% of the ceiling on most joints, and that is meaningless — a step command
starts with an enormous `kp*error` and clamps instantly. Only the settled holding torque says
whether a joint is out of authority, which is why the diagnostic now reports both.

Robustness under conditions that match an untrained policy:

- 6 s of uniformly random normalized targets on all twelve joints, re-sampled at 60 Hz, pelvis
  pinned: stable, zero velocity clamp events, anchor error under 5e-3 m.
- The same for 10 s with the figure free to fall, so contacts and motors fight simultaneously:
  stable, anchor error under 1e-2 m, penetration under 0.02 m, every body finite.

Action mapping is pinned in both directions: -1 maps to the lower limit, +1 to the upper, 0 to
the midpoint, and out-of-range samples clamp rather than extrapolate. That last one matters —
a Gaussian policy samples outside [-1, 1] constantly, and extrapolating would hand the solver
targets outside the joint's own limits for the limit constraints to fight every step.

Visual verification (`--pose <name> --pin`, frames inspected):

- **squat** — torso leaning slightly forward, thighs forward, knees bent, shins back, foot flat,
  arms extended forward. Worst tracking error 0.0105 rad.
- **tuck** — knees drawn to the chest, arms folded, torso curled, head tucked. Recognisably the
  shape a backflip needs. Worst tracking error 0.0174 rad.
- **reach**, **lunge** — as commanded, worst errors 0.0040 and 0.0111 rad.

The squat confirmed every joint sign convention derived on paper: hip flexion positive, knee
flexion negative, ankle dorsiflexion positive, waist flexion negative, shoulder forward
positive. Had any of those been backwards the pose would have come out visibly wrong, which is
exactly why it was worth rendering rather than only asserting angles.

### Known issues

- **The torque ceilings are generous.** 400 N·m at the hip is roughly twice a human's peak
  (~2-3 N·m/kg for a 69 kg figure). That is deliberate headroom for now, but it is also an
  invitation for a policy to find superhuman solutions. If the learned motions in M5-M8 look
  physically implausible, the ceilings are the first thing to cut. Recorded here so that is a
  decision rather than an oversight.
- Damping ratios are conservative (roughly 5x critical at the knee). Safe, and it costs
  response speed; worth revisiting if imitation tracking turns out sluggish.
- Posing drives one joint at a time by keyboard. Fine for inspection; authoring whole reference
  motions is what the M7 animator is for.

### Next

M3 — the C++/Python UDP bridge: batched little-endian protocol with a HELLO/SPEC handshake,
the environment batch on the C++ side, the client and vectorised adapter on the Python side,
and a random-action loop proving both directions end to end with timeouts and recovery.

---

## M3 — the C++ to Python bridge

**Status:** done (2026-09-08)

M3 absorbed the observation and reward design that the plan had put in M4, because the protocol
cannot be defined without them. M4 narrows to normalisation, the policy network and
checkpointing.

### Implemented

- **`cpp/humanoid/Observation.cpp`** — a 111-value observation. Layout, rationale and the exact
  index map are in [`docs/OBSERVATIONS.md`](OBSERVATIONS.md).
- **`cpp/humanoid/RewardTerms.cpp`** — 13 raw, unweighted reward components. The simulator
  computes terms; Python applies weights and sums.
- **`cpp/env/Env2D.cpp`** — the episodic environment: control at 60 Hz over four 240 Hz physics
  substeps, reset noise, six named termination reasons, truncation kept distinct from
  termination, and a disturbance hook for M6.
- **`cpp/env/EnvBatch.cpp`** — N environments with auto-reset and final-observation capture.
- **`cpp/net/Protocol.cpp`**, **`UDPSocket.cpp`**, **`EnvServer.cpp`** — the wire format, a thin
  socket wrapper, and the request/response server. Documented in
  [`docs/PROTOCOL.md`](PROTOCOL.md).
- **`aibf_env`** — the environment host. `--headless` never touches OpenGL; `--render` shows one
  environment while serving the same batch; `--capture` screenshots what a policy is doing.
- **`aibf_fixture`** — writes reference packets for Python to decode and verifies packets Python
  encoded.
- **Python** — `communication/protocol.py`, `client.py`, `vec_env.py`, and `random_agent.py`.

### Build / run

```powershell
.\build\bin\Release\aibf_env.exe --headless --port 51234 --envs 25
python python\random_agent.py --envs 25 --steps 2000 --show-obs
.\build\bin\Release\aibf_env.exe --render --envs 8      # watch one while it trains
```

### Tested

**183 C++ cases and 37 Python cases, all passing** (up from 133 and 9).

The Python suite now includes 19 end-to-end tests that spawn a real `aibf_env` process on a
private port and drive it over a real socket, plus 9 cross-language protocol tests.

**Cross-language agreement is tested in both directions.** Round-tripping a packet inside one
language proves only that a codec is self-consistent. `aibf_fixture` writes SPEC and STATE
packets that the Python suite decodes and asserts on, and Python writes an ACTION packet that
the C++ side decodes and asserts on. The fixture values are deliberately awkward — negative
zero, denormals, `12345.6789f`, `UINT32_MAX` — and the comparison is bit-exact via `tobytes()`,
not `approx`. Negative zero surviving as negative zero is checked explicitly.

**Robustness is fuzzed rather than argued.** 3000 random buffers (a third with a valid preamble,
so the fuzz reaches the payload decoders rather than bouncing off the magic check) are fed to
every decoder; every truncation of a valid packet is rejected; a length field claiming 65535
environments in a 40-byte datagram is refused on the arithmetic before anything is allocated.

**Behavioural checks, not just plumbing.** Two different action streams must produce two
different trajectories, or a bridge that silently dropped every action would pass every shape
and finiteness check. A zero policy must end fewer episodes than a random one. A stationary
policy must score higher than a flailing one, or there would be nothing for PPO to climb.

### Measured

| Environments | Round trips/s | Env-steps/s | vs real time |
|---|---|---|---|
| 1 | 6 831 | 6 831 | 114x |
| 8 | 2 918 | 23 346 | 49x |
| 25 | 1 328 | 33 198 | 22x |
| 32 | 1 073 | 34 352 | 18x |

Zero timeouts, zero stale packets and zero malformed packets across 2002 round trips at 25
environments.

**The transport is not the bottleneck.** The single-environment figure is the round-trip
ceiling — about 6 800/s, dominated by Python-side overhead rather than the socket. At 32
environments the loop needs 1 073 of those, leaving roughly **6x headroom**. Against the
37 700 env-steps/s that headless physics alone achieves for the same batch, the bridge costs
about 9%. The plan's concern about UDP becoming the bottleneck is retired, and shared memory
stays unbuilt until something measures slow.

A random policy survives 60 control steps (1.0 s) on average before terminating, which is the
baseline M5 has to beat.

Visual verification: `aibf_env --render` was captured mid-episode while a Python sine policy
drove it, showing environment 0 buckling under actions arriving over the socket.

### Design decisions that changed during the milestone

Three things were designed one way, found wanting, and changed. All three would have been much
more expensive to discover during M5.

1. **Action 0 mapped to the middle of each joint range.** For the knee's `[-2.6, 0.05]` that is
   1.3 radians of flexion, so a freshly initialised policy — which outputs values near zero —
   would begin every rollout by folding the figure into a deep squat, and learning would have
   to climb out of that before it could start. The mapping is now piecewise-linear about the
   **rest pose**: 0 commands the rest angle, ±1 the limits. Joint limits are now validated to
   straddle zero so both branches are usable, and the knee's upper limit moved from exactly 0 to
   0.05 rad so half its action range is not dead.

2. **Explicit reset masks.** The original design had Python request a reset on the step after an
   episode ended. That wastes a control step per episode and, worse, punches a hole in the
   fixed (steps × environments) block PPO wants. Environments now auto-reset, and the STATE
   packet carries the observation each finished episode ended on. Both halves matter: auto-reset
   keeps every step a real transition, and the final observation is what a **truncated** episode
   bootstraps its value estimate from. Dropping it teaches the critic that reaching the time
   limit is worth zero — the classic time-limit bootstrapping bug.

3. **The server exited when a client disconnected.** A trainer that crashes and restarts should
   be able to reconnect without the simulator being relaunched too. BYE now makes the server
   forget its client and keep listening; `--exit-on-bye` opts into the old behaviour for
   scripted runs.

### Problems hit and how they were resolved

1. **Access violation in every environment test.** `EnvConfig`'s `Humanoid2DConfig` member was
   default-constructed, and *that* type's default has empty link and joint vectors — so the
   first index of `kPelvis` walked off the end of an empty vector. A struct whose default state
   is unusable is a trap; the default is now the real figure, and a test asserts a
   default-constructed `EnvConfig` validates.
2. **Reset noise was zero by default**, so every episode began from an identical state
   regardless of seed. Caught by a test asserting two different seeds produce different
   observations. Since the rest pose is a perfectly symmetric equilibrium, a policy trained from
   an unperturbed start would have scored well while learning nothing. Defaults are now small
   but non-zero, and a test pins that property on the shipped config rather than on a fixture.
3. **The encoder could emit undecodable packets.** A caller that left one array shorter than the
   count it declared produced a packet no decoder could read. The encoder now writes exactly the
   counts in the header, zero-filling a short vector.
4. **A test asserted translation invariance to 1e-5 and failed at 1e-5.** The cause was float
   precision at x = 37.5 m, not a leak of absolute position. The tolerance is now derived from
   float epsilon at the offset being tested, so it distinguishes rounding from a real leak
   instead of being loosened until it passes.

### Known issues

- The batch is single-threaded. At 34 000 env-steps/s with 6x transport headroom, threading
  would only help if the physics budget became the binding constraint — it has not.
- `--render` costs throughput because serving and drawing share a thread. Fine for watching;
  training runs headless.
- The reward weights in `RewardWeights.standing()` are a starting point, not a tuned result.
  Nothing has been trained yet.

### Next

M4 — observation normalisation with running mean/std saved alongside checkpoints, the
actor-critic network (tanh MLP, Gaussian policy with a state-independent log-std), and
checkpoint save/load. Then M5 puts PPO on top of it and finds out whether the humanoid can
learn to stand.
