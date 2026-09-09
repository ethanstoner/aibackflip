# aibackflip

A humanoid that learns to backflip. The physics engine is custom C++ with no physics library,
the learning is PPO written from scratch in PyTorch, and the two talk over UDP.

![a learned backflip](docs/media/backflip.gif)

Every frame above is simulated. There is no animation playback, no inverse kinematics, no
scripted trajectory. A neural network reads joint angles and contacts at 60 Hz and outputs
twelve motor targets, and the rigid-body solver does the rest.

| | | |
|---|---|---|
| ![forward roll](docs/media/roll.gif) | ![jump](docs/media/jump.gif) | ![push recovery](docs/media/push.gif) |
| forward roll, -360 deg | vertical jump | recovering from a 135 N.s shove |

## What it does, measured

Every number below comes from a script in this repo, run against a checkpoint in
`checkpoints/`. Nothing is asserted from watching it.

| behaviour | result | measured by |
|---|---|---|
| Standing | 39-40 of 40 episodes reach the 1000-step limit (seeds 0, 1, 2); surviving returns spread 0.4% | `python/test.py` |
| Push recovery | absorbs 13 shoves an episode at **115 N.s**, falls at 145 (one every 45 steps) | `python/push_test.py` |
| Backflip | **+358 deg** mean rotation, 24/24 complete, 0.71 s airborne, peak height 1.66x rest | `python/flight_test.py` |
| Backflip under a mid-air shove | lands the flip **83% of the time at 200 N.s**, 100% up to 100 N.s | `python/flight_test.py` |
| Forward roll | **-360 deg**, 24/24 complete | `python/flight_test.py` |
| Jump | peak height 1.27x rest, 0.43 s with both feet clear of the ground | `python/flight_test.py` |

The full record, including the experiments that failed and what each one turned out to be, is in
[`docs/PROGRESS.md`](docs/PROGRESS.md).

## How it is put together

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

Three decisions that shaped everything else:

**The simulator computes reward *terms*; Python owns the *weights*.** C++ is the only side that
can see contacts and the centre of mass, so it produces 18 raw non-negative quantities. Python
multiplies them by weights from a JSON file. Reward tuning needs no rebuild, and because every
component is logged separately, reward hacking shows up as one term climbing while the others
stall. That is how the squat policy was caught scoring 0.855 on pose matching while lying flat
on its back.

**Joint motors are implicit PD, solved as constraints.** An explicit torque `kp*e - kd*v` applied
per step is only stable while `kp*h^2 < inertia`, which caps the gains far below what a backflip
needs. Folding the spring into the solver as a soft constraint (`gamma = 1/(h*(kd + h*kp))`)
removes that ceiling. The measurable consequence: tracking bandwidth is `kp/kd`, and a knee
follows a commanded 3 Hz swing at 47% amplitude with the standing gains against 97% with the
acrobatic ones.

**Reference State Initialization is not a refinement.** Episodes start at a random phase of the
reference clip, with the velocities belonging to that phase. Without it a policy has to master
the takeoff before it ever observes the landing, and for a backflip the takeoff is only worth
anything if the landing already works.

Design rationale in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the wire format in
[`docs/PROTOCOL.md`](docs/PROTOCOL.md); the observation layout in
[`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md).

## Three things that were harder than expected

**A reward term that flags a problem changes nothing. Only a termination does.** This happened
three separate times. The squat policy lay on its back doing squat-shaped leg motions, because
joint angles are root-relative and a figure on the ground can hold the reference pose perfectly.
The jump policy rose onto its toes and never left the ground, hidden by episode-averaging a
reward the reference only demanded for 20 of 75 steps. In both cases raising the weight was the
tempting fix and adding a termination was the one that worked.

**Two reward terms were dead and nobody noticed.** `joint_velocity_match` read exactly 0.000 for
a policy that was visibly performing the motion. An exponential tracking reward `exp(-k*e^2)` has
no gradient where it is saturated, so a term tuned to the wrong scale contributes nothing at all
while still appearing in the config, the logs and the total. Fixed by measuring what a working
policy actually achieves and putting the falloff in the middle of that range.

**A disturbance sweep returned an identical result at every magnitude.** 23/24 at 0 N.s and
23/24 at 180 N.s reads as extraordinary robustness. The generated config had been written
standalone instead of merged onto the motion config, so the simulator was running a different
task, with no motion clip and the wrong motor gains, and the "survival rate" was counting
episodes that ran to a time limit doing nothing. `flight_test.py` now carries a witness column
measuring the actual jump in angular velocity across the shove, and refuses to report a
robustness result when that column is flat.

## Build and run

Needs Visual Studio 2022 BuildTools (MSVC 14.4x), CMake 3.20+, and PyTorch. GLFW is fetched at
configure time; nothing else is vendored.

```powershell
.\scripts\build.ps1                 # Release into build\bin\Release
.\scripts\test.ps1                  # 212 C++ cases and 77 Python cases
```

Watch a trained policy:

```powershell
.\scripts\evaluate.ps1 -Model checkpoints\imit_backflip_best.pt `
                       -EnvConfig configs\env2d_backflip_capture.json
.\scripts\demo.ps1 -Only backflip   # re-render the GIF at the top of this file
```

Reproduce the headline measurements:

```powershell
python python\push_test.py   --model checkpoints\robust_v1_best.pt
python python\flight_test.py --model checkpoints\imit_backflip_best.pt `
                             --magnitudes 0 100 200 300 400 --offset 0.4
```

Train the backflip from scratch (12M environment steps at roughly 45k steps/s on one machine,
with 64 environments):

```powershell
.\scripts\train.ps1 -Config configs\ppo_imitate_fast.json `
                    -EnvConfig configs\env2d_backflip.json -Name backflip -Envs 64
```

`venv\` is created with `--system-site-packages` so it inherits the installed CUDA build of
torch instead of re-downloading several GB:

```powershell
python -m venv --system-site-packages venv
.\venv\Scripts\python.exe -m pip install -r python\requirements.txt
```

## Layout

```
cpp/core        math, JSON with comments, PNG writer, PCG32, test runner
cpp/physics     rigid bodies, capsule collision, revolute joints, sequential-impulse solver
cpp/humanoid    13-link figure, forward kinematics, observations, reward terms
cpp/env         episodic environment, batching, auto-reset with final-observation capture
cpp/motion      keyframe clips, Hermite interpolation, phase sampling
cpp/engine      OpenGL 3.3 renderer with a hand-written 29-entry loader
cpp/tools       env server, diagnostics, protocol fixture, motion generator
python/rl       policy, PPO, GAE, running normalization
python/         train.py, test.py, push_test.py (standing), flight_test.py (airborne)
configs/        physics, body, task and hyperparameter configs
motions/        five hand-authored reference clips
```

## What is not done

- **No 3D.** Everything here is sagittal-plane 2D. A cartwheel is a frontal-plane motion and
  cannot be represented in this figure at all, so it waits for the 3D engine.
- **Tracking is loose on the acrobatic motions.** `pose_match` runs 0.08 to 0.20 for the jump,
  roll and backflip against 0.86 for an arm raise. The policies match the root trajectory and
  improvise the limbs. Whether that needs more training, gentler falloffs or a more achievable
  reference has not been separated.
- **The backflip is fragile at takeoff.** Shoved at 200 N.s during launch it completes 0% of
  flips, against 83% for the same shove at mid-flight. The robustness result is about the air,
  not the whole motion.
- Every "N/N" figure comes from a deterministic policy; stochastic sampling was swept only for
  the backflip (22/24).
