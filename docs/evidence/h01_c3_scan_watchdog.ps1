# Detached launcher for the 3,000-sample C3 rescan under the stall watchdog (SP7a).
# Bash background jobs die at 10 minutes; this uses Start-Process so the job outlives the shell.
# Usage (from anywhere):
#   pwsh -File docs/evidence/h01_c3_scan_watchdog.ps1                 # full resume, 78 remaining cells (needs approval)
#   pwsh -File docs/evidence/h01_c3_scan_watchdog.ps1 -Cells 1072605926 -Output docs/evidence/h01-dry-run-3000.json
# Stop:  Stop-Process -Id (Get-Content docs/evidence/h01-c3-scan-3000.pid) -Force
param(
    [string]$Python = "",
    [string]$Cache = "",
    [string]$Output = "docs/evidence/h01-resolved-edge-list-3000.json",
    [string]$Record = "docs/evidence/h01-c3-scan-3000.watchdog.json",
    [int]$StallSeconds = 600,
    [int]$MaxRelaunches = 3,
    [string[]]$Cells = @()
)
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($Python -eq "") { $Python = (Resolve-Path (Join-Path $root "..\h01-braincell\.cache\h01-inspect\Scripts\python.exe")).Path }
if ($Cache -eq "") { $Cache = (Resolve-Path (Join-Path $root "..\h01-braincell\.cache\h01")).Path }
Set-Location $root
$stem = [System.IO.Path]::GetFileNameWithoutExtension($Output)
$argumentList = @("-u", "docs/evidence/h01_c3_scan_watchdog.py", "--python", $Python, "--cache", $Cache,
    "--output", $Output, "--record", $Record, "--root", $root,
    "--stall-seconds", $StallSeconds, "--max-relaunches", $MaxRelaunches)
if ($Cells.Count -gt 0) { $argumentList += @("--cells") + $Cells }
$proc = Start-Process -FilePath $Python -ArgumentList $argumentList -NoNewWindow -PassThru `
    -RedirectStandardOutput "docs/evidence/$stem.watchdog.log" -RedirectStandardError "docs/evidence/$stem.watchdog.err"
$proc.Id | Set-Content "docs/evidence/$stem.pid"
Write-Output ("watchdog pid {0} started {1}; log docs/evidence/{2}.watchdog.log; record {3}" -f $proc.Id, (Get-Date -Format o), $stem, $Record)
