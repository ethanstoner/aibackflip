# Renders each learned behaviour to an animated GIF for the README.
#
#   .\scripts\demo.ps1                       # every motion
#   .\scripts\demo.ps1 -Only backflip
#   .\scripts\demo.ps1 -Only backflip -Fps 25
#
# Frames come out of the simulator's own renderer (aibf_env --capture writes
# PNGs), then ffmpeg assembles them. Nothing is re-simulated for the picture:
# these are screenshots of the same policy, config and physics the numbers in
# docs/PROGRESS.md were measured from.

param(
    [string]$Only = "",
    [string]$OutDir = "docs\media",
    [int]$Fps = 20,
    # Control steps between captured frames. The simulator runs at 60 Hz, so 3
    # gives 20 rendered frames per simulated second.
    [int]$Every = 3,
    [int]$Frames = 34,
    [int]$After = 2,
    [int]$Width = 480,
    [int]$Port = 51700
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvExe = Join-Path $Root "build\bin\Release\aibf_env.exe"
$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
if (-not (Test-Path $EnvExe)) { throw "missing $EnvExe - run scripts\build.ps1" }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg is not on PATH" }

# The capture configs are the ones with reference-state-init switched off, so a
# clip starts at its beginning and the GIF shows the whole motion rather than
# whichever half an episode happened to start in.
$demos = @(
    @{ name = "backflip";  model = "checkpoints\imit_backflip_best.pt"; config = "configs\env2d_backflip_capture.json" },
    @{ name = "roll";      model = "checkpoints\imit_roll_best.pt";     config = "configs\env2d_forward_roll_capture.json" },
    @{ name = "jump";      model = "checkpoints\imit_jump_v2_best.pt";  config = "configs\env2d_jump_capture.json" },
    @{ name = "push";      model = "checkpoints\robust_v1_best.pt";     config = "configs\env2d_push_demo.json" },
    # The shove-trained policy, taking a 300 N.s hit at the apex. Deliberately
    # the magnitude where the two backflip policies separate.
    @{ name = "shoved";    model = "checkpoints\imit_backflip_robust_latest.pt";
       config = "configs\env2d_backflip_shove_demo.json" }
)

if ($Only -ne "") { $demos = $demos | Where-Object { $_.name -eq $Only } }
if ($demos.Count -eq 0) { throw "no demo named '$Only'" }

$outPath = Join-Path $Root $OutDir
New-Item -ItemType Directory -Force -Path $outPath | Out-Null

foreach ($demo in $demos) {
    $name = $demo.name
    Write-Host "`n=== $name ===" -ForegroundColor Cyan

    $model = Join-Path $Root $demo.model
    $config = Join-Path $Root $demo.config
    if (-not (Test-Path $model)) { Write-Host "  no checkpoint: $model" -ForegroundColor Yellow; continue }
    if (-not (Test-Path $config)) { Write-Host "  no config: $config" -ForegroundColor Yellow; continue }

    $frameDir = Join-Path $env:TEMP "aibf_demo_$name"
    if (Test-Path $frameDir) { Remove-Item -Recurse -Force $frameDir }
    New-Item -ItemType Directory -Force -Path $frameDir | Out-Null
    $stem = Join-Path $frameDir "f"

    # Quoted individually: the repo path contains spaces and -ArgumentList
    # splits on whitespace.
    $server = Start-Process -FilePath $EnvExe -PassThru -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $frameDir "server.log") `
        -ArgumentList @("--quiet", "--port", "$Port", "--envs", "1",
                        "--config", "`"$config`"",
                        "--capture", "`"$stem.png`"", "--capture-after", "$After",
                        "--capture-count", "$Frames", "--capture-every", "$Every")
    Start-Sleep -Milliseconds 900
    try {
        # The server exits after the last frame, which ends the client too. That
        # is the intended shutdown, so the client's complaint is discarded.
        & $Python (Join-Path $Root "python\test.py") --model $model --port $Port --envs 1 `
            --episodes 999 --max-steps 100000 --seed 0 2>$null | Out-Null
    } catch {
    } finally {
        if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    }

    $captured = @(Get-ChildItem $frameDir -Filter "f_*.png" | Sort-Object Name)
    if ($captured.Count -eq 0) { Write-Host "  no frames captured" -ForegroundColor Red; continue }
    Write-Host "  $($captured.Count) frames" -ForegroundColor Gray

    # Two passes so the palette is built from the whole clip. A single-pass GIF
    # quantises per frame and the figure's limb colours crawl between frames.
    $gif = Join-Path $outPath "$name.gif"
    $palette = Join-Path $frameDir "palette.png"
    $filters = "fps=$Fps,scale=${Width}:-1:flags=lanczos"
    & ffmpeg -y -loglevel error -framerate $Fps -i (Join-Path $frameDir "f_%02d.png") `
        -vf "$filters,palettegen=stats_mode=diff" $palette
    & ffmpeg -y -loglevel error -framerate $Fps -i (Join-Path $frameDir "f_%02d.png") -i $palette `
        -lavfi "$filters [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3" $gif

    if (Test-Path $gif) {
        $kb = [math]::Round((Get-Item $gif).Length / 1KB)
        Write-Host "  $gif  ($kb KB)" -ForegroundColor Green
    } else {
        Write-Host "  ffmpeg produced nothing" -ForegroundColor Red
    }
    $Port += 1
}
