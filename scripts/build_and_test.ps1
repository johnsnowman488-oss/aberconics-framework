<#
.SYNOPSIS
    Build and test the aberconics-framework project.
.DESCRIPTION
    Sets up the Python venv, builds the C++ core (static + shared),
    and runs all tests (C++ ctest + Python pytest).
.PARAMETER SkipVenv
    Skip venv creation and dependency installation.
.PARAMETER SkipCpp
    Skip C++ build steps.
.PARAMETER PythonOnly
    Only run Python tests (skip all C++ builds).
.PARAMETER Clean
    Remove build directories before building.
#>
param(
    [switch]$SkipVenv,
    [switch]$SkipCpp,
    [switch]$PythonOnly,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# ── Locate MinGW toolchain ────────────────────────────────────────────
$MingwBin = "C:\msys64\mingw64\bin"
$Gpp      = Join-Path $MingwBin "g++.exe"
$Cmake    = Join-Path $MingwBin "cmake.exe"
$Ninja    = Join-Path $MingwBin "ninja.exe"
$Ctest    = Join-Path $MingwBin "ctest.exe"

if (-not (Test-Path $Gpp)) {
    Write-Error "MinGW g++ not found at $Gpp. Install MSYS2 + mingw-w64-x86_64-toolchain."
    exit 1
}

# ── Python venv ───────────────────────────────────────────────────────
$VenvPython = ".\.venv\Scripts\python.exe"

if (-not $SkipVenv) {
    if (-not (Test-Path ".venv")) {
        Write-Host "→ Creating Python venv..." -ForegroundColor Cyan
        python -m venv .venv
    }
    Write-Host "→ Installing Python dependencies..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPython -m pip install -r requirements.txt --quiet
}

# ── C++ build ─────────────────────────────────────────────────────────
if (-not $SkipCpp -and -not $PythonOnly) {
    $env:PATH = "$MingwBin;$env:PATH"

    # Static build (library + tests + apps)
    $StaticBuildDir = "code\c++ core\build"
    if ($Clean -and (Test-Path $StaticBuildDir)) {
        Write-Host "→ Cleaning static build dir..." -ForegroundColor Yellow
        Remove-Item $StaticBuildDir -Recurse -Force
    }
    Write-Host "→ Configuring C++ static build..." -ForegroundColor Cyan
    & $Cmake -S "code\c++ core" -B $StaticBuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -G Ninja `
        -DCMAKE_CXX_COMPILER="$Gpp" 2>&1 | Out-Null

    Write-Host "→ Building C++ static..." -ForegroundColor Cyan
    & $Cmake --build $StaticBuildDir -j 2>&1 | Out-Null

    Write-Host "→ Running C++ tests (ctest)..." -ForegroundColor Cyan
    & $Ctest --test-dir $StaticBuildDir --output-on-failure 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Error "C++ tests failed."; exit 1 }

    # Shared build (for Python/Julia wrappers, no tests)
    $SharedBuildDir = "code\c++ core\build-shared"
    if ($Clean -and (Test-Path $SharedBuildDir)) {
        Write-Host "→ Cleaning shared build dir..." -ForegroundColor Yellow
        Remove-Item $SharedBuildDir -Recurse -Force
    }
    Write-Host "→ Building C++ shared library..." -ForegroundColor Cyan
    & $Cmake -S "code\c++ core" -B $SharedBuildDir `
        -DBUILD_SHARED_LIBS=ON `
        -DGFE_BUILD_TESTS=OFF `
        -DGFE_BUILD_APPS=OFF `
        -G Ninja `
        -DCMAKE_CXX_COMPILER="$Gpp" 2>&1 | Out-Null
    & $Cmake --build $SharedBuildDir -j 2>&1 | Out-Null
}

# ── Python tests ──────────────────────────────────────────────────────
Write-Host "→ Running Python tests..." -ForegroundColor Cyan
$env:PYTHONPATH = "code\python"
& $VenvPython -m pytest code\python\tests -q 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "Python tests failed."; exit 1 }

# ── Done ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "✓ All builds and tests passed." -ForegroundColor Green