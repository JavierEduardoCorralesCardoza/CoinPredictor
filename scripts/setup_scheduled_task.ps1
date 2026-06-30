# Registers (or updates) a daily Windows scheduled task that runs the
# CoinPredictor paper-trading bot. Re-run to change time/profile/capital.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_scheduled_task.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_scheduled_task.ps1 -Time 09:30 -Profile defensive
#
# Remove with:
#   Unregister-ScheduledTask -TaskName CoinPredictorBot -Confirm:$false
param(
    [string]$Time = "08:00",
    [string]$Profile = "aggressive",
    [double]$Capital = 1000,
    [string]$TaskName = "CoinPredictorBot"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $root "scripts\run_bot.ps1"

if (-not (Test-Path $runner)) {
    throw "Runner not found: $runner"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -Profile $Profile -Capital $Capital"

$trigger = New-ScheduledTaskTrigger -Daily -At $Time

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Daily CoinPredictor paper-trading bot ($Profile profile)" `
    -Force | Out-Null

Write-Host "Scheduled task '$TaskName' registered: daily at $Time, profile=$Profile, capital=$Capital"
Write-Host "Logs will be written to: $(Join-Path $root 'logs\bot.log')"
