# aibackflip

A humanoid that learns to backflip, on a 2D rigid-body physics engine and a PPO trainer both
written from scratch. No physics library, no animation playback, no scripted trajectory: a
neural network outputs twelve motor targets at 60 Hz and a hand-written constraint solver does
the rest.

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue) ![tests](https://img.shields.io/badge/tests-313%20passing-brightgreen) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

![a learned backflip](docs/media/backflip.gif)

### Highlights

- **Lands 24 of 24 backflips**: +358° mean rotation, 0.71 s airborne, peak height 1.66x rest,
  measured by `flight_test.py` against a committed checkpoint.
- **Survives being hit mid-flip**: 83% of flips still land after a 200 N·s shove at the apex.
  Training against shoves lifts completion at 400 N·s from 12% to 44% over 96 episodes.
- **Physics verified numerically, not by eye**: 1.77 mm worst-case joint drift on a five-link
  chain yanked for 6 s, and energy that only ever decreases (-16% at 60 s, never a gain).
- **12,317 lines of C++ and 4,375 of Python, 313 tests**, two dependencies (GLFW for the
  window, PyTorch for autograd).

**C++20 · Python · PyTorch · OpenGL 3.3 · CMake · UDP**

---

## Overview

The point of the project is to own every layer between "apply a torque" and "land a backflip":
the rigid-body solver, the humanoid, the environment, the wire protocol, the RL algorithm and the
measurement tools. A 13-link, 69 kg, 1.64 m figure imitates hand-authored reference clips
DeepMimic-style, and PPO turns that into a policy that flips, rolls, jumps, stands and recovers
from shoves.

| | | | |
|---|---|---|---|
| ![forward roll](docs/media/roll.gif) | ![jump](docs/media/jump.gif) | ![push recovery](docs/media/push.gif) | ![shoved mid-flip](docs/media/shoved.gif) |
| forward roll, -360° | vertical jump | recovering from a 135 N·s shove | hit with 300 N·s at the apex |

---

## Architecture

```
    C++                                 UDP            Python
  +------------------------+                    +----------------------+
  | 64 environments        |   one batched      | PPO                  |
  |   rigid bodies         |   datagram per     |   actor-critic MLP   |
  |   contacts + friction  |   control step     |   GAE, clipped       |
  |   revolute joints      | <----------------> |   surrogate, KL stop |
  |   PD motors            |                    |   running obs norm   |
  |   111-value observation|                    |                      |
  |   18 raw reward terms  |                    |   reward *weights*   |
  +------------------------+                    +----------------------+
```

**Physics engine (C++).** A sequential-impulse constraint solver with warm starting, speculative
contacts, a Coulomb friction cone, a restitution threshold, and a dedicated position solver
rather than Baumgarte stabilisation. Capsule bodies, capsule-vs-halfspace and capsule-vs-capsule
narrow phase, collision groups, semi-implicit Euler.

**Humanoid.** Thirteen capsule links and twelve revolute joints. Each joint stacks an angular
motor, two one-sided limits and a 2-DOF point constraint, solved in that order so the point
constraint gets the final say on position. The figure is authored as a rest pose and the local
anchors are derived from it, so "every joint reads zero at rest" is true by construction.

**Learning (Python).** PPO with a clipped surrogate, GAE, advantage normalisation, entropy bonus,
gradient clipping, a KL-estimator early stop and learning-rate annealing. Separate actor and
critic trunks; a Gaussian policy with a tanh mean and clamped state-independent log-std.
Imitation rewards on pose, joint velocity, end effectors, root and centre of mass, with Reference
State Initialization and early termination.

**Bridge.** One UDP datagram carries all 64 environments per control step, with explicit
little-endian serialisation and no struct-padding assumptions. A HELLO/SPEC handshake has C++ tell
Python the observation, action and reward dimensions (111, 12, 18), so Python never hardcodes a
shape. A duplicate ACTION packet resends the cached STATE instead of stepping twice.

### Three design decisions that shaped everything else

**The simulator computes reward terms; Python owns the weights.** C++ is the only side that sees
contacts and the centre of mass, so it emits 18 raw non-negative terms and Python weights them
from JSON. Tuning needs no rebuild, and because every term is logged separately, reward hacking
shows up as one term climbing while the others stall. That is how the squat policy was caught
scoring 0.855 on pose matching while lying flat on its back. The term-by-term layout is in
[`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md).

**Joint motors are implicit PD, solved as constraints.** Explicit torque `kp*e - kd*v` is only
stable while `kp*h^2 < inertia`, which caps the gains far below what a backflip needs. Folding the
spring into the solver as a soft constraint (`gamma = 1/(h*(kd + h*kp))`) removes the ceiling.
Measured consequence: a knee follows a commanded 3 Hz swing at 47% amplitude with the standing
gains and 97% with the acrobatic ones, and the backflip is not trainable until that is fixed.

**Reference State Initialization is not a refinement.** Episodes start at a random phase of the
clip with that phase's velocities. Without it a policy has to master the takeoff before it ever
sees the landing, and a backflip takeoff is only worth anything once the landing works.

Design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Wire format:
[`docs/PROTOCOL.md`](docs/PROTOCOL.md).

---

## Engineering Highlights

- **Built a 2D rigid-body physics engine from scratch** (sequential impulses, friction cones,
  position solver, capsule collision) and verified it against closed-form impulse responses and
  finite-difference checks: 1.77 mm worst joint drift under load, energy monotonically
  non-increasing to 60 s.
- **Implemented PPO from scratch** in PyTorch (GAE, clipped surrogate, KL early stop) and trained
  a 12-DOF humanoid to backflip in about 17 minutes (12M environment steps at 11.4k to 12.5k
  env-steps/s, 64 environments).
- **Designed a batched UDP protocol** that moves 64 environments per datagram, with a
  self-describing handshake and idempotent requests, so neither side hardcodes a shape.
- **Replaced explicit PD torques with implicit PD constraints**, lifting knee tracking at 3 Hz
  from 47% to 97% amplitude and making the backflip trainable at all.
- **Built measurement tools that refuse to report a bad number.** `flight_test.py` carries a
  witness column for the angular-velocity jump across each shove and warns when it is flat, after
  a misconfigured sweep returned 23/24 at every magnitude. It also reports the largest per-step
  rotation against the π limit of its wrapped integral.
- **Found and fixed a reference-velocity sign flip at the loop point of every looping clip** that
  reward and observation agreed on perfectly, caught only by `track_test.py` recomputing the
  reward from an independently dumped table (2 steps out of 2,880).
- **Cut mean joint tracking error 28%** (0.475 to 0.340 rad) by retraining with a gentler pose
  falloff, with the flip unchanged at 100% and +356°.
- **Tested the stack with 313 cases** (216 C++, 97 Python) and re-ran every README number
  against its pinned command, which corrected three claims.

---

## Results

Every number here comes from a command in this repository run against a checkpoint committed
alongside it.

| behaviour | result | reproduce with |
|---|---|---|
| **Backflip** | +358° mean rotation, 24/24 complete, 0.71 s airborne, peak height 1.66x rest | `flight_test.py` |
| **Backflip, shoved mid-flight** | lands 100% up to 100 N·s, 83% at 200, 38% at 300 | `flight_test.py` |
| **Backflip, trained against shoves** | 44% at 400 N·s against 12% untrained, costing 6 points at zero disturbance (96 episodes) | `flight_test.py` |
| **Backflip, gentler pose falloff** | 0.340 rad mean joint error against 0.475, flip unchanged at 100% and +356° | `track_test.py` |
| **Forward roll** | -360°, 24/24 complete | `flight_test.py` |
| **Jump** | peak height 1.27x rest, 0.44 s with both feet clear | `flight_test.py` |
| **Standing** | 39, 40 and 39 of 40 episodes reach the 1000-step limit over seeds 0, 1, 2 | `test.py` |
| **Push recovery** | thirteen shoves per episode, 0.8 s apart: 100% survive at 100 N·s, 96% at 115, 46% at 130, 8% at 145 | `push_test.py --interval 45` |

```powershell
python python\flight_test.py --model checkpoints\imit_backflip_best.pt `
                             --magnitudes 0 100 200 300 400 --offset 0.4
python python\flight_test.py --model checkpoints\imit_backflip_robust_latest.pt `
                             --magnitudes 0 400 --offset 0.4 --episodes 96 --envs 8
python python\flight_test.py --model checkpoints\imit_roll_best.pt `
                             --config configs\env2d_forward_roll_capture.json --magnitudes 0
python python\flight_test.py --model checkpoints\imit_jump_v2_best.pt `
                             --config configs\env2d_jump_capture.json --magnitudes 0
python python\push_test.py   --model checkpoints\robust_v1_best.pt `
                             --interval 45 --magnitudes 100 115 130 145 160
python python\track_test.py  --model checkpoints\k025_gentle_best.pt `
                             --config configs\env2d_backflip_capture.json

# standing needs a running simulator; repeat with --seed 1 and --seed 2
.\build\bin\Release\aibf_env.exe --headless --quiet --exit-on-bye --envs 8 --port 51300 `
                                 --config configs\env2d.json
python python\test.py --model checkpoints\stand_v1_best.pt --port 51300 --envs 8 --episodes 40 --seed 0
```

`--envs` is part of the measurement: each environment is seeded separately, so the environment
count decides which episodes are drawn. The shove-trained row reads 43% at `--envs 12` and 44% at
the default 8.

**Robustness is bought, not found.** The shove-trained policy is a worse backflip: 94% at zero
disturbance against 100%, and 349° against 358°. Training against a gentler range gives up
nearly all of the robustness while still paying most of the cost. Both checkpoints are kept, and
the undisturbed one is the flip at the top because it is the better flip.

### How it learns

![training progression](docs/media/progression.gif)

Four checkpoints from one training run attempting the same reference motion: 120 updates, 720,
960 and 2,880. The stages were chosen by measurement rather than even spacing. Every snapshot
in the run, measured with `flight_test.py --magnitudes 0 --episodes 12` (the last batch of
environments can overshoot, so each row is 12 to 16 episodes):

| updates | flips completed | rotation | peak height | airborne |
|---|---|---|---|---|
| 120 | 0% | 89° | 1.12x | 0.33 s |
| 480 | 0% | 191° | 1.23x | 0.39 s |
| 840 | 8% | 260° | 1.20x | 0.50 s |
| 960 | 33% | 226° | 1.34x | 0.45 s |
| 1,080 | 94% | 342° | 1.65x | 0.70 s |
| **1,440** | **42%** | **290°** | **1.47x** | **0.53 s** |
| 1,920 | 94% | 353° | 1.57x | 0.70 s |
| 2,280 | 100% | 360° | 1.65x | 0.70 s |
| 2,880 | 100% | 358° | 1.61x | 0.74 s |

The curve is not monotonic. Update 1,080 already lands 94% and update 1,440 falls back to 42%
before recovering. That row is in bold rather than omitted: picking only the improving snapshots
would make a cleaner story and a false one. Rotation and airborne time move together because a
flip is won or lost at takeoff; the angular momentum is fixed the moment the feet leave the
ground.

### Limitations and work in progress

- **This is sagittal-plane 2D.** A cartwheel is a frontal-plane motion and cannot be represented
  in this figure at all. A 3D engine was built and trained to stand, then removed: the acrobatics
  did not land in the time available, and a half-finished second engine was worth less than a
  finished first one. That work and why it was cut is in `docs/PROGRESS.md` and the git history.
- **Tracking is loose on the acrobatic motions (M11, in progress).** It is not the motors: every
  joint reaches two to ten times the peak rate its reference asks for. It is the pose falloff,
  which saturates mid-clip, where `pose_match` reads 1e-4 to 3e-3 and the exponential has about
  2000x less gradient than a healthy one. Retraining the backflip at a falloff of 0.25 instead of
  2.0 cuts mean joint error from 0.475 rad to 0.340 (27° to 19°) at no measurable cost to the
  flip. That is still 19° per joint, the jump and roll have not been retrained, and the flagship
  checkpoint is still the original.
- **The backflip is fragile at takeoff.** Shoved at 200 N·s on control step 15, before the feet
  leave the ground, it completes 0% of flips against 83% for the same shove mid-flight
  (`flight_test.py ... --magnitudes 200 --offset 0.4 --at-step 15`). The robustness result is
  about the air, not the whole motion.
- Every "N/N" figure comes from a deterministic policy. `flight_test.py` does not measure
  stochastic sampling.

The full engineering log, including every experiment that failed, is in
[`docs/PROGRESS.md`](docs/PROGRESS.md).

---

## Getting Started

Requires Visual Studio 2022 BuildTools (MSVC 14.4x), CMake 3.20+ and PyTorch. GLFW is fetched at
configure time; nothing else is vendored. The venv inherits the system CUDA build of torch rather
than re-downloading it.

```powershell
python -m venv --system-site-packages venv
.\venv\Scripts\python.exe -m pip install -r python\requirements.txt
.\scripts\build.ps1                 # Release into build\bin\Release
```

Watch the trained backflip, or re-render the GIFs:

```powershell
.\scripts\evaluate.ps1 -Model checkpoints\imit_backflip_best.pt `
                       -EnvConfig configs\env2d_backflip_capture.json
.\scripts\showcase.ps1 -Only results
```

Train from scratch. 12M environment steps at 11.4k to 12.5k env-steps/s with 64 environments,
about 17 minutes, measured over two full runs. The bare-simulator benchmark
(`aibf_diag throughput`, 39k to 45k) is a different number: it excludes the bridge, observations,
reward terms, resets and the policy, which together are roughly three quarters of a training step.

```powershell
.\scripts\train.ps1 -Config configs\ppo_imitate_fast.json `
                    -EnvConfig configs\env2d_backflip.json -Name backflip -Envs 64
```

---

## Testing

```powershell
.\scripts\test.ps1                  # builds, then 216 C++ cases and 97 Python cases
```

The C++ suite asserts physics numerically: impulse responses against closed-form results, joint
anchor drift under load, energy behaviour, and finite-difference checks against the analytic
formulas the solver uses. The Python suite covers the wire protocol against C++-generated
fixtures, live bridge sessions against the simulator, the policy and running normalisation, PPO,
the reference-clip reader and `track_test.py`. Every acrobatic
result is confirmed twice: once by measurement, once by rendering frames and looking at them.

---

## Project Structure

```
cpp/core        math, JSON with comments, PNG writer, PCG32, test runner
cpp/physics     rigid bodies, capsule collision, revolute joints, sequential-impulse solver
cpp/humanoid    13-link figure, forward kinematics, observations, reward terms
cpp/env         episodic environment, batching, auto-reset with final-observation capture
cpp/motion      keyframe clips, Hermite interpolation, phase sampling
cpp/engine      OpenGL 3.3 renderer
cpp/tools       env server, diagnostics, protocol fixture, motion generator
python/rl       policy, PPO, GAE, running normalisation
python/         train.py, test.py (standing), push_test.py, flight_test.py (airborne),
                track_test.py (per-joint imitation error)
configs/        physics, body, task and hyperparameter configs
motions/        five hand-authored reference clips
scripts/        build, test, train, evaluate, showcase
```

---

## What I Learned

**A number belongs in the README only if a command regenerates it.** Applying that rule cost
three claims their original wording. Standing's "40/40" was one seed of three. The backflip's
airborne time and peak height were a maximum over episodes reported as a mean. Push recovery
read "absorbs thirteen 115 N·s shoves, falls at 145", but the printed command produced six
shoves, not thirteen; run properly, 115 is 96%, 145 is 8%, and 130 is a 46% coin flip the
sentence skipped. Every rounding went the flattering way, which is the reason to re-run rather
than re-read.

**A reward term that flags a problem changes nothing. Only a termination does.** The squat policy
lay on its back performing squat-shaped leg motions, because joint angles are root-relative. The
jump policy rose onto its toes and never left the ground. Raising a weight was the tempting fix
both times and adding a termination was the one that worked; for the jump a single threshold
moved peak height from 1.069 to 1.368. A related trap: an exponential tracking reward tuned to
the wrong scale has no gradient where it saturates, so a dead term still appears in the config,
the logs and the total while contributing nothing.

**Two sides that agree can both be wrong.** The reward and the observation read the reference
from one sampler, so they agreed perfectly on a velocity that flipped sign at every loop point.
Only an independent recomputation disagreed, and the same check then caught a second bug: phase
travels as sin and cos, so phase 1.0 and phase 0.0 are the same two floats, and the end of every
episode was being read as the start of the clip.

---

## License

MIT. See [`LICENSE`](LICENSE).
