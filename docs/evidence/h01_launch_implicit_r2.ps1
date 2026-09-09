$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h01_status_writer.ps1')
$workRoot = (Get-Location).Path
$python = Join-Path $workRoot '.cache/validation/Scripts/python.exe'
$planPath = Join-Path $workRoot 'docs/evidence/h01-ready-104-implicit-ei-10ms-r2-plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
$launchPath = Join-Path $workRoot 'docs/evidence/h01-ready-104-implicit-ei-10ms-r2-launch.json'
$logPath = Join-Path $workRoot 'docs/evidence/h01-ready-104-implicit-ei-10ms-r2.log'
$errorPath = Join-Path $workRoot 'docs/evidence/h01-ready-104-implicit-ei-10ms-r2.err'
if (Test-Path -LiteralPath $launchPath) { throw 'Existing launch record: inspect before relaunch.' }
$available = [long](& $python -c 'import psutil; print(psutil.virtual_memory().available)')
if ($LASTEXITCODE -ne 0 -or $available -lt ($plan.projected_peak_bytes + $plan.reserve_memory_bytes)) { throw 'Memory preflight failed.' }
$counterProbe = [long]0
foreach ($sampleBytes in @([long]1048576, [long]5368709120)) { $counterProbe = [Math]::Max([long]$counterProbe, [long]$sampleBytes) }
if ($counterProbe -ne 5368709120) { throw 'Int64 peak counter regression.' }
$arguments = @('-u','-m','examples.h01_verified_network','--topology','docs/evidence/h01-verified-network.json','--cache','.cache/h01','--solver','h01_staggered_calcium_implicit','--include-isolated','--cells','104','--current-na','1','--components','docs/evidence/h01-population-components.json','--output','.cache/h01/readiness/104-implicit-ei-10ms-r2','--duration-ms','10','--dt-ms','0.000625','--control','ei','--heartbeat-s','30')
$started = [DateTime]::UtcNow
$child = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $workRoot -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errorPath -PassThru
$record = [ordered]@{status='running'; pid=$child.Id; started_utc=$started.ToString('o'); source_commit=$plan.source_commit; plan='h01-ready-104-implicit-ei-10ms-r2-plan.json'; arguments=$arguments; wall_cap_seconds=$plan.wall_cap_seconds; silence_cap_seconds=$plan.silence_cap_seconds; memory_reserve_bytes=$plan.reserve_memory_bytes; available_bytes_at_launch=$available; peak_tree_rss_bytes=[long]0; peak_counter_type='Int64 on both Math.Max arguments'; monitor_errors=0}
$sampleCode = @'
import json,sys,psutil
p=psutil.Process(int(sys.argv[1]))
members=[p]+p.children(recursive=True)
rss=0
for member in members:
    try: rss+=member.memory_info().rss
    except psutil.NoSuchProcess: pass
print(json.dumps(dict(rss=rss,available=psutil.virtual_memory().available)))
'@
$stopCode = @'
import psutil,sys
try:
    p=psutil.Process(int(sys.argv[1]))
    members=p.children(recursive=True)+[p]
    for member in members:
        try: member.kill()
        except psutil.NoSuchProcess: pass
except psutil.NoSuchProcess: pass
'@
$consecutiveErrors = 0
try {
    while ($true) {
        $child.Refresh()
        if ($child.HasExited) { break }
        $elapsed = ([DateTime]::UtcNow - $started).TotalSeconds
        $latest = $started
        foreach ($path in @($logPath,$errorPath)) {
            if (Test-Path -LiteralPath $path) {
                $written = (Get-Item -LiteralPath $path).LastWriteTimeUtc
                if ($written -gt $latest) { $latest = $written }
            }
        }
        $reason = $null
        if ($elapsed -gt $plan.wall_cap_seconds) { $reason = 'wall cap' }
        if (([DateTime]::UtcNow - $latest).TotalSeconds -gt $plan.silence_cap_seconds) { $reason = 'silence cap' }
        try {
            $sampleText = & $python -c $sampleCode $child.Id
            if ($LASTEXITCODE -ne 0) { throw 'Process sample failed.' }
            $sample = $sampleText | ConvertFrom-Json
            $record.peak_tree_rss_bytes = [Math]::Max([long]$record.peak_tree_rss_bytes,[long]$sample.rss)
            $record.available_bytes_latest = [long]$sample.available
            $consecutiveErrors = 0
            if ($sample.available -lt $plan.reserve_memory_bytes) { $reason = 'free memory reserve' }
        } catch {
            $child.Refresh()
            if ($child.HasExited) { break }
            $record.monitor_errors += 1
            $consecutiveErrors += 1
            if ($consecutiveErrors -ge 3) { $reason = 'three consecutive monitor failures' }
        }
        if ($reason) {
            $record.status = 'aborted; untested'
            $record.abort_reason = $reason
            & $python -c $stopCode $child.Id
            $child.WaitForExit()
            break
        }
        $published = Write-H01Status $launchPath $record
        Start-Sleep -Seconds 5
    }
    $child.WaitForExit()
    $record.exit_code = $child.ExitCode
    if ($record.status -eq 'running') { $record.status = if ($child.ExitCode -eq 0) { 'completed' } else { 'failed' } }
} catch {
    $record.status = 'supervisor failure; untested'
    $record.supervisor_error = $_.Exception.Message
    & $python -c $stopCode $child.Id
    throw
} finally {
    $record.seconds = ([DateTime]::UtcNow - $started).TotalSeconds
    if (-not (Write-H01Status $launchPath $record)) {
        $fallback = $launchPath + '.terminal-' + [Guid]::NewGuid().ToString('N') + '.json'
        if (-not (Write-H01Status $fallback $record)) { throw 'Terminal record publication failed' }
    }
    $record | ConvertTo-Json -Depth 8
}
