$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h01_status_writer.ps1')
$path = Join-Path ([IO.Path]::GetTempPath()) ('h01-status-' + [Guid]::NewGuid().ToString('N') + '.json')
try {
    if (-not (Write-H01Status $path @{status='old'})) { throw 'Initial write failed' }
    $reader = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        if (Write-H01Status $path @{status='new'}) { throw 'Locked replacement unexpectedly succeeded' }
        if (([IO.File]::ReadAllText($path) | ConvertFrom-Json).status -ne 'old') { throw 'Prior JSON damaged' }
    } finally { $reader.Dispose() }
    if (-not (Write-H01Status $path @{status='new'})) { throw 'Recovery failed' }
    if (([IO.File]::ReadAllText($path) | ConvertFrom-Json).status -ne 'new') { throw 'New JSON missing' }
    if (@(Get-ChildItem -LiteralPath ([IO.Path]::GetDirectoryName($path)) -Filter ([IO.Path]::GetFileName($path) + '*.tmp')).Count) { throw 'Temporary files leaked' }
    'PASS: creation, locked replacement, preserved JSON, recovery, cleanup'
} finally {
    if ([IO.File]::Exists($path)) { [IO.File]::Delete($path) }
}
