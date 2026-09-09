# Starts the environment server and a training run together, and shuts the
# server down when training ends.
#
#   .\scripts\train.ps1
#   .\scripts\train.ps1 -Config configs\ppo_stand.json -Steps 5000000 -Name stand_v2
#   .\scripts\train.ps1 -Port 51500 -Envs 32

param(
    [string]$Config = "configs\ppo_stand.json",
    # Environment-side config (disturbances, reset noise, imitation). Passed to
    # aibf_env, not to the trainer - the simulator owns the task, Python owns
    # the learning.
    [string]$EnvConfig = "",
    [string]$Name = "",
    [int]$Port = 51234,
    [int]$Envs = 32,
    [long]$Steps = 0,
    [int]$Seed = 1,
    [string]$Resume = "",
    [int]$LogEvery = 10
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvExe = Join-Path $Root "build\bin\Release\aibf_env.exe"
$Python = Join-Path $Root "venv\Scripts\python.exe"

if (-not (Test-Path $EnvExe)) { throw "missing $EnvExe - run scripts\build.ps1" }
if (-not (Test-Path $Python)) { $Python = "python" }

$serverArgs = @("--headless", "--quiet", "--port", "$Port", "--envs", "$Envs", "--seed", "$Seed")
if ($EnvConfig -ne "") {
    $envConfigPath = if (Test-Path $EnvConfig) { $EnvConfig } else { Join-Path $Root $EnvConfig }
    if (-not (Test-Path $envConfigPath)) { throw "no such environment config: $EnvConfig" }
    $serverArgs += @("--config", $envConfigPath)
    Write-Host "environment config: $envConfigPath" -ForegroundColor Cyan
}

Write-Host "starting the environment server on port $Port with $Envs environments" -ForegroundColor Cyan
# Launched from the repo root so relative paths inside an environment config -
# a motion clip, most importantly - resolve the same way regardless of where
# this script was invoked from.
$server = Start-Process -FilePath $EnvExe -ArgumentList $serverArgs -PassThru `
    -WindowStyle Hidden -WorkingDirectory $Root
Start-Sleep -Milliseconds 800

if ($server.HasExited) { throw "the environment server exited immediately" }

$arguments = @((Join-Path $Root "python\train.py"), "--config", (Join-Path $Root $Config),
               "--port", "$Port", "--envs", "$Envs", "--seed", "$Seed", "--log-every", "$LogEvery")
if ($Name -ne "") { $arguments += @("--name", $Name) }
if ($Steps -gt 0) { $arguments += @("--steps", "$Steps") }
if ($Resume -ne "") { $arguments += @("--resume", $Resume) }

try {
    & $Python @arguments
} finally {
    if (-not $server.HasExited) {
        Write-Host "stopping the environment server" -ForegroundColor Cyan
        Stop-Process -Id $server.Id -Force
    }
}
