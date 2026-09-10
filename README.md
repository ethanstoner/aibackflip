# aibackflip

**A humanoid that learns to backflip, on a physics engine written from scratch.**

No physics library. No animation playback, no inverse kinematics, no scripted trajectory. A
neural network reads joint angles and foot contacts at 60 Hz and outputs twelve motor targets,
and a hand-written rigid-body solver does the rest.

![a learned backflip](docs/media/backflip.gif)

| | | | |
|---|---|---|---|
| ![forward roll](docs/media/roll.gif) | ![jump](docs/media/jump.gif) | ![push recovery](docs/media/push.gif) | ![shoved mid-flip](docs/media/shoved.gif) |
| forward roll, -360 deg | vertical jump | recovering from a 135 N.s shove | hit with 300 N.s at the apex |

```
18,500 lines of C++      2D and 3D rigid-body physics, constraint solvers, humanoids, renderers
 3,500 lines of Python   PPO, GAE, policy, normalisation, evaluation harnesses
   371 tests             294 C++, 77 Python, all green
     2 dependencies      GLFW for the window, PyTorch for autograd. Nothing else is vendored.
```

---

## Results

Every number here is produced by a command in this repository, run against a checkpoint that is
committed alongside it. Nothing is claimed from watching the screen.

| behaviour | result | reproduce with |
|---|---|---|
| **Backflip** | +358 deg mean rotation, 24/24 episodes complete, 0.71 s airborne, peak height 1.66x rest | `flight_test.py` |
| **Backflip, shoved mid-flight** | lands 100% up to 100 N.s, 83% at 200 N.s, 38% at 300 N.s | `flight_test.py` |
| **Backflip, trained against shoves** | 44% at 400 N.s against 12% untrained, costing 6 points at zero disturbance (96 episodes) | `flight_test.py` |
| **Forward roll** | -360 deg, 24/24 episodes complete | `flight_test.py` |
| **Jump** | peak height 1.27x rest, 0.43 s with both feet clear of the ground | `flight_test.py` |
| **Standing** | 39 to 40 of 40 episodes reach the 1000-step limit across three seeds, surviving returns within 0.4% | `test.py` |
| **Push recovery** | absorbs thirteen 115 N.s shoves per episode, falls at 145 | `push_test.py` |

The complete engineering log, including every experiment that failed and what each one turned
out to be, is in [`docs/PROGRESS.md`](docs/PROGRESS.md).

---

## What was built

### The physics engine

A 2D rigid-body simulator in C++ with a sequential-impulse constraint solver: warm starting,
speculative contacts, a Coulomb friction cone, a restitution threshold, and a dedicated position
solver rather than Baumgarte stabilisation. Capsule bodies, capsule-vs-halfspace and
capsule-vs-capsule narrow phase, collision groups, semi-implicit Euler integration.

Verified numerically, not by eye. Joint anchor drift on a five-link chain yanked for six seconds
is **1.8 mm** worst case. Energy is monotonically non-increasing over a 240-second horizon: the
solver only ever loses energy, never gains it, which is the property that separates a stable
solver from one that quietly explodes.

### The humanoid

Thirteen capsule links, twelve revolute joints, 69 kg, 1.64 m. Each joint stacks an angular
motor, two one-sided limit constraints and a 2-DOF point constraint, solved in that order so the
point constraint gets the final say on position.

The figure is authored as a **rest pose** (a world position and angle per link, a world pivot per
joint) and the local anchors and reference angles are derived from it, so "every joint reads zero
at rest" is true by construction rather than by careful data entry.

### The learning

PPO written from scratch: clipped surrogate objective, GAE, advantage normalisation, entropy
bonus, gradient clipping, Schulman's low-variance KL estimator driving a target-KL early stop,
and learning-rate annealing. Actor and critic have separate trunks; the policy is Gaussian with a
tanh mean and a state-independent, clamped log-std.

Imitation is DeepMimic-style: exponential tracking rewards on pose, joint velocity, end-effector
position, root and centre of mass, with Reference State Initialization and early termination.

### The 3D engine

The same structure again in three dimensions, and the parts that are genuinely different rather
than the 2D code with a `z` added: real inertia tensors that rotate with the body, ball joints
with separate cone and twist limits, hinges that must actively forbid the two axes they do not
turn about, and a capsule renderer.

The orientation integrator is the piece worth naming. A tumbling body has no constant angular
velocity even with no torque on it, because the inertia tensor turns underneath omega, so
sampling omega once per step is first order however exactly the rotation is then applied. At this
engine's rate that costs a tumbling body **+2017% of its kinetic energy over five seconds**.
Angular momentum is conserved to machine precision the whole time, so the usual check passes the
broken version without complaint.

`EnvBatch` and `EnvServer` are templates over the environment rather than copies, and Python
needed no change at all to drive the 3D figure: the observation went from 111 values to 204 and
the action from 12 to 24, and the handshake carried it.

### The bridge

One UDP datagram carries all 64 environments per control step, with explicit little-endian
serialisation on both sides and no struct-padding assumptions anywhere. A HELLO/SPEC handshake
makes C++ tell Python the observation, action and reward dimensions, so Python never hardcodes a
shape. Requests are idempotent: a duplicate ACTION packet resends the cached STATE rather than
simulating the step twice.

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

Design rationale is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), the wire format in
[`docs/PROTOCOL.md`](docs/PROTOCOL.md), and the observation layouts in
[`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md) and
[`docs/OBSERVATIONS3D.md`](docs/OBSERVATIONS3D.md).

---

## Three design decisions that shaped everything else

**The simulator computes reward *terms*; Python owns the *weights*.** C++ is the only side that
can see contacts and the centre of mass, so it produces 18 raw non-negative quantities. Python
multiplies them by weights from a JSON file. Reward tuning then needs no rebuild, and because
every component is logged separately, reward hacking shows up as one term climbing while the
others stall. That is exactly how the squat policy was caught scoring 0.855 on pose matching
while lying flat on its back.

**Joint motors are implicit PD, solved as constraints.** An explicit torque `kp*e - kd*v` applied
per step is only stable while `kp*h^2 < inertia`, which caps the gains far below what a backflip
needs. Folding the spring into the solver as a soft constraint (`gamma = 1/(h*(kd + h*kp))`)
removes that ceiling. The measurable consequence: tracking bandwidth is `kp/kd`, and a knee
follows a commanded 3 Hz swing at 47% amplitude with the standing gains against 97% with the
acrobatic ones. The backflip is not trainable until that number is fixed.

**Reference State Initialization is not a refinement.** Episodes start at a random phase of the
reference clip, with the velocities belonging to that phase. Without it a policy has to master
the takeoff before it ever observes the landing, and for a backflip the takeoff is only worth
anything if the landing already works.

---

## Four things that were harder than expected

**A reward term that flags a problem changes nothing. Only a termination does.** This happened
three separate times. The squat policy lay on its back performing squat-shaped leg motions,
because joint angles are root-relative and a figure on the ground can hold the reference pose
perfectly. The jump policy rose onto its toes and never left the ground, hidden by episode
averaging on a reward the reference only demanded for 20 of 75 steps. In both cases raising the
weight was the tempting fix and adding a termination was the one that worked. Fixing the jump
moved peak height from 1.069 to 1.368 with a single threshold change and nothing else.

**Two reward terms were dead and nothing said so.** `joint_velocity_match` read exactly 0.000 for
a policy that was visibly performing the motion. An exponential tracking reward `exp(-k*e^2)` has
no gradient where it is saturated, so a term tuned to the wrong scale contributes nothing at all
while still appearing in the config, the logs and the total. Fixed by measuring what a working
policy actually achieves and putting the falloff in the middle of that range.

**A disturbance sweep returned an identical result at every magnitude.** 23/24 at 0 N.s and 23/24
at 180 N.s reads as extraordinary robustness. The generated config had been written standalone
instead of merged onto the motion config, so the simulator was running a different task, with no
motion clip and the wrong motor gains, and the "survival rate" was counting episodes that reached
a time limit doing nothing in particular. `flight_test.py` now carries a witness column measuring
the actual jump in angular velocity across the shove, and refuses to print a robustness result
when that column is flat.

**Robustness is bought, not found.** Fine-tuning the backflip against random off-centre shoves
more than triples completion at 400 N.s, from 12% to 44% over 96 episodes. It also makes the
policy a worse backflip: 94% at zero disturbance against 100%, and 349 degrees of rotation
against 357. Training against a gentler range gives up nearly all of the robustness while still
paying most of the cost, so there is no setting in between that avoids the trade. Both
checkpoints are kept, and the undisturbed one is the flip in the GIF at the top, because it is
the better flip.

---

## Verification

The rule this project runs on: **a number belongs in this README only if a command in this README
regenerates it.** Applying that rule cost two claims their original wording. Standing's "40/40
episodes" turned out to be one seed of three, the other two giving 39/40. The backflip's airborne
time and peak height were a maximum over episodes reported as though it were a mean. Both are
corrected above and the correction is recorded in `docs/PROGRESS.md`.

Physics correctness is asserted numerically: impulse responses against closed-form results, joint
anchor drift under load, energy behaviour, and finite-difference checks against the analytic
formulas the solver uses. Every acrobatic result is confirmed twice, once by measurement and once
by rendering the frames and looking at them.

```powershell
.\scripts\build.ps1                 # Release into build\bin\Release
.\scripts\test.ps1                  # 212 C++ cases and 77 Python cases
```

---

## Running it

Requires Visual Studio 2022 BuildTools (MSVC 14.4x), CMake 3.20+, and PyTorch. GLFW is fetched at
configure time; nothing else is vendored.

Watch a trained policy, or re-render the GIFs above:

```powershell
.\scripts\evaluate.ps1 -Model checkpoints\imit_backflip_best.pt `
                       -EnvConfig configs\env2d_backflip_capture.json
.\scripts\demo.ps1 -Only backflip
```

Reproduce the headline measurements:

```powershell
python python\flight_test.py --model checkpoints\imit_backflip_best.pt `
                             --magnitudes 0 100 200 300 400 --offset 0.4
python python\push_test.py   --model checkpoints\robust_v1_best.pt
```

Train the backflip from scratch (12M environment steps, about 45k steps/s with 64 environments):

```powershell
.\scripts\train.ps1 -Config configs\ppo_imitate_fast.json `
                    -EnvConfig configs\env2d_backflip.json -Name backflip -Envs 64
```

`venv\` is created with `--system-site-packages` so it inherits the installed CUDA build of torch
instead of re-downloading several GB:

```powershell
python -m venv --system-site-packages venv
.\venv\Scripts\python.exe -m pip install -r python\requirements.txt
```

---

## Layout

```
cpp/core        math, JSON with comments, PNG writer, PCG32, test runner
cpp/physics     rigid bodies, capsule collision, revolute joints, sequential-impulse solver
cpp/humanoid    13-link figure, forward kinematics, observations, reward terms
cpp/env         episodic environment, batching, auto-reset with final-observation capture
cpp/motion      keyframe clips, Hermite interpolation, phase sampling
cpp/engine      OpenGL 3.3 renderer with a hand-written 29-entry loader
cpp/tools       env server, diagnostics, protocol fixture, motion generator
python/rl       policy, PPO, GAE, running normalisation
python/         train.py, test.py, push_test.py (standing), flight_test.py (airborne)
configs/        physics, body, task and hyperparameter configs
motions/        five hand-authored reference clips
scripts/        build, test, train, evaluate, demo
```

---

## What is not done

- **The 3D figure stands but does not yet tumble.** The 3D engine is built and tested:
  quaternion orientation, real inertia tensors, ball joints with separate cone and twist limits,
  hinges with a two-axis angular lock, and a capsule renderer. The same PPO code trains it to
  stand. Acrobatic imitation in 3D is written and running but is not producing a flip yet, and
  the current numbers are in `docs/PROGRESS.md` rather than in this table.
- **The cartwheel exists as a reference clip, not as a learned behaviour.** It is the motion this
  whole exercise was pointed at, since a sagittal 2D figure has no frontal plane and cannot
  represent one at all. The clip is authored and renders correctly. No policy performs it yet.
- **Tracking is loose on the acrobatic motions.** `pose_match` runs 0.08 to 0.20 for the jump,
  roll and backflip against 0.86 for an arm raise. The policies match the root trajectory and
  improvise the limbs. Whether that needs more training, gentler falloffs or a more physically
  achievable reference has not been separated.
- **The backflip is fragile at takeoff.** Shoved at 200 N.s during launch it completes 0% of
  flips, against 83% for the same shove at mid-flight. The robustness result is about the air,
  not the whole motion.
- Every "N/N" figure comes from a deterministic policy. Stochastic sampling was swept only for
  the backflip, at 22/24.
