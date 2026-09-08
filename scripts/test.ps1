# Runs both suites and fails if either does. Builds first, so a stale binary
# can never pass a test for code that no longer compiles.
#
#   .\scripts\test.ps1
#   .\scripts\test.ps1 -Filter Quat      C++ cases whose "Suite.name" contains Quat
#   .\scripts\test.ps1 -SkipBuild

param(
    [string]$Config = "Release",
    [string]$Filter = "",
    [switch]$SkipBuild,
    [switch]$CppOnly,
    [switch]$PyOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$failures = @()

if (-not $SkipBuild -and -not $PyOnly) {
    & (Join-Path $PSScriptRoot "build.ps1") -Config $Config
}

if (-not $PyOnly) {
    Write-Host "`n=== C++ ===" -ForegroundColor Cyan
    $exe = Join-Path $Root "build\bin\$Config\aibf_tests.exe"
    if (-not (Test-Path $exe)) { throw "missing $exe - build first" }
    if ($Filter -ne "") { & $exe $Filter } else { & $exe }
    if ($LASTEXITCODE -ne 0) { $failures += "C++" }
}

if (-not $CppOnly) {
    Write-Host "`n=== Python ===" -ForegroundColor Cyan
    $py = Join-Path $Root "venv\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = "python" }
    Push-Location $Root
    try {
        & $py -m pytest
        if ($LASTEXITCODE -ne 0) { $failures += "Python" }
    } finally {
        Pop-Location
    }
}

if ($failures.Count -gt 0) {
    Write-Host "`nFAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nall suites passed" -ForegroundColor Green
