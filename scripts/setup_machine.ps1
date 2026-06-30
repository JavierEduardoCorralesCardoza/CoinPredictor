# One-shot setup for a fresh machine.
# Creates the virtual environment, installs dependencies, and registers the
# daily scheduled task that runs the paper-trading bot.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_machine.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_machine.ps1 -Time 09:30 -Profile defensive -Capital 5000
#
# Requirements: Python 3.10+ available as `python` on PATH.
param(
    [string]$Time = "08:00",
    [string]$Profile = "aggressive",
    [double]$Capital = 1000,
    [switch]$SkipTask
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> Project root: $root"

# --- 1. Python ---
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python not found on PATH. Install Python 3.10+ from https://www.python.org/downloads/ and re-run."
}
Write-Host "==> Using $((& python --version) 2>&1)"

# --- 2. Virtual environment ---
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "==> Creating virtual environment (.venv)"
    & python -m venv $venv
} else {
    Write-Host "==> Virtual environment already exists"
}

# --- 3. Dependencies ---
Write-Host "==> Upgrading pip"
& $py -m pip install --upgrade pip --quiet
Write-Host "==> Installing project (editable) + dependencies"
& $py -m pip install -e . --quiet

# --- 4. Smoke test ---
Write-Host "==> Running a quick import check"
& $py -c "import coinpredictor; print('coinpredictor OK')"

# --- 5. Scheduled task ---
if ($SkipTask) {
    Write-Host "==> Skipping scheduled task registration (-SkipTask)"
} else {
    Write-Host "==> Registering scheduled task"
    & (Join-Path $PSScriptRoot "setup_scheduled_task.ps1") -Time $Time -Profile $Profile -Capital $Capital
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "  Run the bot manually:  powershell -ExecutionPolicy Bypass -File scripts\run_bot.ps1 -Profile $Profile -Capital $Capital"
Write-Host "  View logs:             Get-Content logs\bot.log -Tail 20"
