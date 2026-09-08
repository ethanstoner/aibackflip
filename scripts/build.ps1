# Configure + build. The Visual Studio generator locates the 2022 BuildTools
# toolchain itself, so no vcvars64.bat sourcing is required.
#
#   .\scripts\build.ps1                 Release
#   .\scripts\build.ps1 -Config Debug
#   .\scripts\build.ps1 -Clean          wipe build/ first (forces a re-fetch of GLFW)

param(
    [ValidateSet("Release", "Debug", "RelWithDebInfo")]
    [string]$Config = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build"

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "removing $BuildDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

if (-not (Test-Path (Join-Path $BuildDir "CMakeCache.txt"))) {
    Write-Host "configuring..." -ForegroundColor Cyan
    cmake -S $Root -B $BuildDir -G "Visual Studio 17 2022" -A x64
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
}

Write-Host "building $Config..." -ForegroundColor Cyan
cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "binaries in $BuildDir\bin\$Config" -ForegroundColor Green
