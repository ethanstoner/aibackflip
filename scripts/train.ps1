# Starts the environment server and a training run together, and shuts the
# server down when training ends.
#
#   .\scripts\train.ps1
#   .\scripts\train.ps1 -Config configs\ppo_stand.json -Steps 5000000 -Name stand_v2
#   .\scripts\train.ps1 -Port 51500 -Envs 32

param(
    [string]$Config = "configs\ppo_stand.json",
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

Write-Host "starting the environment server on port $Port with $Envs environments" -ForegroundColor Cyan
$server = Start-Process -FilePath $EnvExe `
    -ArgumentList "--headless", "--quiet", "--port", "$Port", "--envs", "$Envs", "--seed", "$Seed" `
    -PassThru -WindowStyle Hidden
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
