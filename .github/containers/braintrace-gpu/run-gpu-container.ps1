<#
.SYNOPSIS
    Launch a braintrace-gpu container with the shared XLA compilation cache mounted.

.DESCRIPTION
    The image sets JAX_COMPILATION_CACHE_DIR=/cache/jax, but that path is
    container-local unless a host directory is bind-mounted over it. Without the
    mount every run recompiles and re-autotunes every XLA kernel from scratch.
    This wrapper always supplies the mount so compiled kernels and per-fusion
    autotune results are reused across runs and shared between concurrent
    containers.

.PARAMETER CacheDirectory
    Host directory backing the shared cache. Created if missing.

.PARAMETER Image
    Container image to run.

.PARAMETER Mount
    Extra bind mounts, each formatted as "hostPath:containerPath".

.PARAMETER Command
    Command and arguments to run inside the container.

.EXAMPLE
    .\run-gpu-container.ps1 -Mount "C:\results:/results" -Command python examples/pp_prop/17-temporal-credit-benchmark.py
#>
# PositionalBinding=$false keeps the in-container command from being bound to
# -CacheDirectory/-Image/-Mount positionally; everything unmatched lands in
# -Command instead.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$CacheDirectory = "$env:LOCALAPPDATA\braintrace\jax-cache",
    [string]$Image = 'braintrace-gpu:0.11.0-py314',
    [string[]]$Mount = @(),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Stop'

# PowerShell forwards the '--' end-of-parameters marker through
# ValueFromRemainingArguments, and would otherwise try to bind command flags
# such as 'python -u' as parameters of this script. Callers separate the
# in-container command with '--'; drop it before handing the rest to docker.
if ($Command -and $Command[0] -eq '--') {
    $Command = $Command | Select-Object -Skip 1
}

if (-not $Command) {
    throw 'No command supplied. Separate the in-container command with ''--'', e.g. run-gpu-container.ps1 -- python script.py'
}

New-Item -ItemType Directory -Force -Path $CacheDirectory | Out-Null

$dockerArgs = @(
    'run', '--rm', '--gpus', 'all',
    '-v', "${CacheDirectory}:/cache/jax"
)
foreach ($bind in $Mount) {
    $dockerArgs += @('-v', $bind)
}
$dockerArgs += $Image
$dockerArgs += $Command

Write-Verbose "docker $($dockerArgs -join ' ')"
& docker @dockerArgs
exit $LASTEXITCODE
