# Evaluates a checkpoint and captures a strip of frames showing what it does.
#
# Runs twice: once headless to gather episode statistics over many episodes, and
# once with the renderer to write screenshots. A number without a picture does
# not establish that a policy is doing what its reward says.
#
#   .\scripts\evaluate.ps1 -Model checkpoints\stand_best.pt
#   .\scripts\evaluate.ps1 -Model checkpoints\stand_best.pt -Shots out\stand -Frames 6

param(
    [Parameter(Mandatory = $true)][string]$Model,
    # Environment config, which for an imitation policy is not optional: without
    # it the simulator runs the standing task with the standing motor gains, and
    # the checkpoint is evaluated against a figure it was never trained for.
    [string]$EnvConfig = "",
    [string]$Shots = "",
    [int]$Frames = 6,
    [int]$FirstFrameStep = 150,
    [int]$FrameIntervalSteps = 90,
    [int]$Episodes = 30,
    [int]$Envs = 4,
    [int]$Port = 51999,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvExe = Join-Path $Root "build\bin\Release\aibf_env.exe"
$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
if (-not (Test-Path $Model)) { $Model = Join-Path $Root $Model }

# Quoted, because the repo lives under a path with spaces in it and
# -ArgumentList splits on whitespace.
$configArgs = @()
if ($EnvConfig -ne "") {
    $configPath = if (Test-Path $EnvConfig) { $EnvConfig } else { Join-Path $Root $EnvConfig }
    if (-not (Test-Path $configPath)) { throw "no such environment config: $EnvConfig" }
    $configArgs = @("--config", "`"$configPath`"")
    Write-Host "environment config: $configPath" -ForegroundColor Cyan
}

Write-Host "`n=== episode statistics ===" -ForegroundColor Cyan
$server = Start-Process -FilePath $EnvExe `
    -ArgumentList (@("--headless", "--quiet", "--port", "$Port", "--envs", "$Envs") + $configArgs) `
    -PassThru -WindowStyle Hidden -WorkingDirectory $Root
Start-Sleep -Milliseconds 800
try {
    & $Python (Join-Path $Root "python\test.py") --model $Model --port $Port --envs $Envs `
        --episodes $Episodes --seed $Seed
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}

if ($Shots -eq "") { exit 0 }

$shotDir = Split-Path -Parent $Shots
if ($shotDir -ne "" -and -not (Test-Path $shotDir)) {
    New-Item -ItemType Directory -Force -Path $shotDir | Out-Null
}

Write-Host "`n=== frames ===" -ForegroundColor Cyan
$capturePort = $Port + 1
$server = Start-Process -FilePath $EnvExe `
    -ArgumentList (@("--quiet", "--port", "$capturePort", "--envs", "1",
                     "--capture", "`"$Shots.png`"", "--capture-after", "$FirstFrameStep",
                     "--capture-count", "$Frames", "--capture-every", "$FrameIntervalSteps") +
                   $configArgs) `
    -PassThru -RedirectStandardOutput "$env:TEMP\aibf_capture.log" -WorkingDirectory $Root
Start-Sleep -Milliseconds 900
try {
    # The server exits after the last capture, which ends this client too. That
    # is expected, so its failure is not treated as an error.
    & $Python (Join-Path $Root "python\test.py") --model $Model --port $capturePort --envs 1 `
        --episodes 999 --max-steps 100000 --seed $Seed 2>$null | Out-Null
} catch {
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
Get-Content "$env:TEMP\aibf_capture.log"
