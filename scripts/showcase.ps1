# Renders the showcase media: what the policies do, and how one of them learned.
#
#   .\scripts\showcase.ps1                    # everything
#   .\scripts\showcase.ps1 -Only progression
#   .\scripts\showcase.ps1 -Only results
#
# Frames come from the simulator's own renderer. Nothing is re-simulated for the
# picture: these are screenshots of the same policy, config and physics the
# numbers in docs/PROGRESS.md were measured from.
#
# Two pieces:
#
#   results      one clip per learned behaviour, plus a combined reel.
#   progression  the *same* reference motion attempted by checkpoints taken at
#                intervals through a single training run, side by side. This is
#                the one that needs planning ahead: `_best` is overwritten every
#                time the return improves, so a finished run keeps no trace of
#                its early policy. The numbered snapshots exist for this.

param(
    [string]$Only = "",
    [string]$OutDir = "docs\media",
    [int]$Fps = 20,
    [int]$Every = 3,
    [int]$Frames = 34,
    [int]$Width = 480,
    [int]$Port = 51600
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvExe = Join-Path $Root "build\bin\Release\aibf_env.exe"
$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
if (-not (Test-Path $EnvExe)) { throw "missing $EnvExe - run scripts\build.ps1" }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg is not on PATH" }

$outPath = Join-Path $Root $OutDir
New-Item -ItemType Directory -Force -Path $outPath | Out-Null

# Captures one policy against one environment config and returns the frame
# directory. The capture configs are the ones with reference-state-init off, so
# a clip starts at its beginning and the footage shows the whole motion rather
# than whichever half an episode happened to start in.
function Capture-Policy {
    param(
        [string]$Model, [string]$Config, [string]$Tag,
        [int]$FrameCount = 34, [int]$Spacing = 3, [int]$After = 2, [int]$UsePort = 51600
    )
    $frameDir = Join-Path $env:TEMP "aibf_show_$Tag"
    if (Test-Path $frameDir) { Remove-Item -Recurse -Force $frameDir }
    New-Item -ItemType Directory -Force -Path $frameDir | Out-Null
    $stem = Join-Path $frameDir "f"

    # Hidden, because this runs once per snapshot and a progression pass is two
    # dozen of them. Without it every capture flashes a console window across
    # the desktop, which is unusable while anything else is going on.
    #
    # The renderer itself is already invisible: the capture path sets
    # GLFW_VISIBLE false before creating the window. It is the host process's
    # own console that has to be suppressed here.
    #
    # Quoted individually: the repo path contains spaces and -ArgumentList
    # splits on whitespace.
    $server = Start-Process -FilePath $EnvExe -PassThru -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $frameDir "server.log") `
        -ArgumentList @("--quiet", "--port", "$UsePort", "--envs", "1",
                        "--config", "`"$Config`"",
                        "--capture", "`"$stem.png`"", "--capture-after", "$After",
                        "--capture-count", "$FrameCount", "--capture-every", "$Spacing")
    Start-Sleep -Milliseconds 900
    try {
        # The server exits after the last frame, which ends the client too. That
        # is the intended shutdown, so the client's complaint is discarded.
        & $Python (Join-Path $Root "python\test.py") --model $Model --port $UsePort --envs 1 `
            --episodes 999 --max-steps 100000 --seed 0 2>$null | Out-Null
    } catch {
    } finally {
        if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    }
    return $frameDir
}

# Two passes so the palette is built from the whole clip. A single-pass GIF
# quantises per frame and the figure's limb colours crawl between frames.
function Write-Gif {
    param([string]$FrameDir, [string]$OutFile, [int]$GifFps = 20, [int]$GifWidth = 480)
    $palette = Join-Path $FrameDir "palette.png"
    $filters = "fps=$GifFps,scale=${GifWidth}:-1:flags=lanczos"
    & ffmpeg -y -loglevel error -framerate $GifFps -i (Join-Path $FrameDir "f_%02d.png") `
        -vf "$filters,palettegen=stats_mode=diff" $palette
    & ffmpeg -y -loglevel error -framerate $GifFps -i (Join-Path $FrameDir "f_%02d.png") -i $palette `
        -lavfi "$filters [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3" $OutFile
}

function Write-Mp4 {
    param([string]$FrameDir, [string]$OutFile, [int]$VidFps = 30, [int]$VidWidth = 960)
    # yuv420p and an even width, or the file plays in ffplay and nowhere else.
    & ffmpeg -y -loglevel error -framerate $VidFps -i (Join-Path $FrameDir "f_%02d.png") `
        -vf "scale=${VidWidth}:-2:flags=lanczos" -c:v libx264 -profile:v high -crf 20 `
        -pix_fmt yuv420p $OutFile
}

# ---------------------------------------------------------------- results

$results = @(
    @{ tag = "backflip"; model = "checkpoints\imit_backflip_best.pt";  config = "configs\env2d_backflip_capture.json";    label = "backflip" },
    @{ tag = "roll";     model = "checkpoints\imit_roll_best.pt";      config = "configs\env2d_forward_roll_capture.json"; label = "forward roll" },
    @{ tag = "jump";     model = "checkpoints\imit_jump_v2_best.pt";   config = "configs\env2d_jump_capture.json";         label = "jump" },
    @{ tag = "push";     model = "checkpoints\robust_v1_best.pt";      config = "configs\env2d_push_demo.json";            label = "push recovery" },
    @{ tag = "shoved";   model = "checkpoints\imit_backflip_robust_latest.pt"; config = "configs\env2d_backflip_shove_demo.json"; label = "shoved at the apex" }
)

if ($Only -eq "" -or $Only -eq "results") {
    Write-Host "`n=== results ===" -ForegroundColor Cyan
    $usePort = $Port
    foreach ($item in $results) {
        $model = Join-Path $Root $item.model
        $config = Join-Path $Root $item.config
        if (-not (Test-Path $model)) { Write-Host "  skip $($item.tag): no checkpoint" -ForegroundColor Yellow; continue }
        if (-not (Test-Path $config)) { Write-Host "  skip $($item.tag): no config" -ForegroundColor Yellow; continue }

        $frameDir = Capture-Policy -Model $model -Config $config -Tag $item.tag `
            -FrameCount $Frames -Spacing $Every -UsePort $usePort
        $count = @(Get-ChildItem $frameDir -Filter "f_*.png").Count
        if ($count -eq 0) { Write-Host "  $($item.tag): no frames" -ForegroundColor Red; continue }

        Write-Gif -FrameDir $frameDir -OutFile (Join-Path $outPath "$($item.tag).gif") -GifFps $Fps -GifWidth $Width
        Write-Mp4 -FrameDir $frameDir -OutFile (Join-Path $outPath "$($item.tag).mp4")
        Write-Host "  $($item.label): $count frames" -ForegroundColor Green
        $usePort += 1
    }
}

# ---------------------------------------------------------------- progression

if ($Only -eq "" -or $Only -eq "progression") {
    Write-Host "`n=== progression ===" -ForegroundColor Cyan

    # Snapshots from one run, in order. The name carries the update number, so
    # sorting by name sorts by training time.
    $snaps = @(Get-ChildItem (Join-Path $Root "checkpoints") -Filter "flipshow_snap*.pt" |
               Sort-Object Name)
    if ($snaps.Count -eq 0) {
        Write-Host "  no snapshots; train with snapshot_every_updates set" -ForegroundColor Yellow
    } else {
        $config = Join-Path $Root "configs\env2d_backflip_capture.json"
        $usePort = $Port + 20
        $stages = @()
        foreach ($snap in $snaps) {
            $tag = $snap.BaseName
            $frameDir = Capture-Policy -Model $snap.FullName -Config $config -Tag $tag `
                -FrameCount $Frames -Spacing $Every -UsePort $usePort
            $count = @(Get-ChildItem $frameDir -Filter "f_*.png").Count
            if ($count -lt $Frames) {
                Write-Host "  ${tag}: only $count frames" -ForegroundColor Yellow
            }
            if ($count -gt 0) { $stages += $frameDir }
            $usePort += 1
        }

        # Four stages side by side, chosen by *measured* behaviour rather than
        # by even spacing, so the panel shows the arc instead of three panels of
        # a solved policy. Measured with flight_test.py, 12 episodes each:
        #
        #   update  120   0% of flips   89 deg   barely leaves the ground
        #   update  720   0%           207 deg   rotates, cannot finish
        #   update  960  33%           226 deg   sometimes
        #   update 2880 100%           358 deg   lands it every time
        #
        # The full curve is not monotonic. Update 1080 already reached 94% and
        # update 1440 fell back to 42% before recovering, which is worth knowing
        # about PPO and worth not hiding by picking only improving stages.
        $wanted = @("flipshow_snap000120", "flipshow_snap000720",
                    "flipshow_snap000960", "flipshow_snap002880")
        $picked = @()
        foreach ($name in $wanted) {
            $match = $stages | Where-Object { (Split-Path $_ -Leaf) -eq "aibf_show_$name" }
            if ($match) { $picked += $match }
        }
        if ($picked.Count -lt 4 -and $stages.Count -ge 4) {
            # Fall back to even spacing if the named snapshots are absent, so a
            # run with different settings still produces something.
            $picked = @($stages[0],
                        $stages[[int]($stages.Count / 3)],
                        $stages[[int](2 * $stages.Count / 3)],
                        $stages[$stages.Count - 1])
        }
        if ($picked.Count -ge 4) {
            $inputs = @()
            foreach ($dir in $picked) { $inputs += @("-framerate", "$Fps", "-i", (Join-Path $dir "f_%02d.png")) }
            $filter = "[0:v]scale=360:-2[a];[1:v]scale=360:-2[b];[2:v]scale=360:-2[c];[3:v]scale=360:-2[d];" +
                      "[a][b][c][d]hstack=inputs=4[v]"
            & ffmpeg -y -loglevel error @inputs -filter_complex $filter -map "[v]" `
                -c:v libx264 -crf 20 -pix_fmt yuv420p (Join-Path $outPath "progression.mp4")

            $palette = Join-Path $env:TEMP "aibf_prog_palette.png"
            & ffmpeg -y -loglevel error -i (Join-Path $outPath "progression.mp4") `
                -vf "fps=$Fps,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff" $palette
            & ffmpeg -y -loglevel error -i (Join-Path $outPath "progression.mp4") -i $palette `
                -lavfi "fps=$Fps,scale=960:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3" `
                (Join-Path $outPath "progression.gif")
            Write-Host "  progression from $($stages.Count) snapshots, 4 shown" -ForegroundColor Green
        } else {
            Write-Host "  need at least 4 usable snapshots, have $($stages.Count)" -ForegroundColor Yellow
        }
    }
}

Write-Host "`nwrote to $outPath" -ForegroundColor Cyan
Get-ChildItem $outPath | Select-Object Name, @{n='KB';e={[math]::Round($_.Length/1KB)}}
