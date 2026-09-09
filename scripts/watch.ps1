# Watch a trained policy in a window.
#
#   .\scripts\watch.ps1                                   the best standing policy
#   .\scripts\watch.ps1 -Model checkpoints\robust_v1_best.pt
#   .\scripts\watch.ps1 -Model checkpoints\imit_squat_best.pt -EnvConfig configs\env2d_squat.json
#
# While it runs, the simulator window takes:
#   x / z   shove the pelvis right / left   (hold shift for a hard shove)
#   b       throw a ball at it
#   mouse   left-drag a limb
#   t c j v trails, contacts, joint targets, velocities
#   wheel   zoom            esc  quit

param(
    [string]$Model = "checkpoints\stand_v1_best.pt",
    [string]$EnvConfig = "",
    [int]$Port = 51950,
    [int]$Episodes = 200,
    [int]$Seed = 0,
    [switch]$Stochastic
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvExe = Join-Path $Root "build\bin\Release\aibf_env.exe"
$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
if (-not (Test-Path $Model)) { $Model = Join-Path $Root $Model }
if (-not (Test-Path $Model)) { throw "no such checkpoint: $Model" }

# One environment, so the window shows the thing being evaluated rather than one
# of thirty-two.
$serverArgs = @("--render", "--quiet", "--port", "$Port", "--envs", "1", "--seed", "$Seed")
if ($EnvConfig -ne "") {
    $path = if (Test-Path $EnvConfig) { $EnvConfig } else { Join-Path $Root $EnvConfig }
    if (-not (Test-Path $path)) { throw "no such environment config: $EnvConfig" }
    $serverArgs += @("--config", $path)
}

Write-Host "watching $(Split-Path -Leaf $Model)" -ForegroundColor Cyan
Write-Host "  x/z shove   b ball   mouse drag   esc quit" -ForegroundColor DarkGray

$server = Start-Process -FilePath $EnvExe -ArgumentList $serverArgs -PassThru -WorkingDirectory $Root
Start-Sleep -Milliseconds 900
if ($server.HasExited) { throw "the simulator exited immediately" }

$arguments = @((Join-Path $Root "python\test.py"), "--model", $Model, "--port", "$Port",
               "--envs", "1", "--episodes", "$Episodes", "--seed", "$Seed")
if ($Stochastic) { $arguments += "--stochastic" }

try {
    & $Python @arguments
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
