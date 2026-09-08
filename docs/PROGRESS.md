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
