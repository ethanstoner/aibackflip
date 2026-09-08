# aibackflip

A physics-based humanoid that learns to move by reinforcement learning, built from scratch.

The rigid-body simulation is custom C++ — no physics library. The learning is PPO in PyTorch,
talking to the simulator over UDP. The goal is a humanoid that learns to stand, resist being
shoved, and eventually perform a physically simulated backflip through DeepMimic-style
imitation of a hand-authored reference motion.

Nothing here is a scripted animation dressed up as AI. Where a behaviour has not been learned
yet, the docs say so.

## Status

See [`docs/PROGRESS.md`](docs/PROGRESS.md) for what actually works right now, including the
training experiments that failed and why.

## Build

Requires Visual Studio 2022 BuildTools (MSVC 14.4x), CMake 3.20+, and a GPU-capable PyTorch.
GLFW is fetched automatically at configure time.

```powershell
.\scripts\build.ps1                # Release into build\bin\Release
.\scripts\build.ps1 -Config Debug
.\scripts\build.ps1 -Clean         # wipe build\ and re-fetch dependencies
```

## Test

```powershell
.\scripts\test.ps1                 # builds, then runs the C++ and Python suites
.\scripts\test.ps1 -Filter Quat    # only C++ cases matching "Quat"
.\scripts\test.ps1 -PyOnly
```

## Python environment

`venv\` is created with `--system-site-packages` so it inherits the already-installed
`torch 2.6.0+cu124` instead of re-downloading several GB of CUDA wheels.

```powershell
python -m venv --system-site-packages venv
.\venv\Scripts\python.exe -m pip install -r python\requirements.txt
```

## Layout

```
cpp/        simulator, physics, renderer, UDP env server, motion authoring tool
python/     PPO training, policy, and the UDP client
configs/    timesteps, body dimensions, joint limits, reward weights, PPO hyperparameters
motions/    hand-authored reference motions for imitation learning
docs/       ARCHITECTURE.md, PROGRESS.md, PROTOCOL.md, OBSERVATIONS.md
scripts/    build.ps1, test.ps1
```

Design rationale lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
