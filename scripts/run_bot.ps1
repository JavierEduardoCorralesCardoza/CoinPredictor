# Runs one paper-trading cycle and appends output to logs/bot.log.
# Invoked by the Windows scheduled task (see setup_scheduled_task.ps1).
param(
    [string]$Profile = "aggressive",
    [double]$Capital = 1000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "bot.log"

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"`n===== $stamp (profile=$Profile) =====" | Out-File -FilePath $log -Append -Encoding utf8
& $py -m coinpredictor.trading.bot --profile $Profile --capital $Capital *>> $log
