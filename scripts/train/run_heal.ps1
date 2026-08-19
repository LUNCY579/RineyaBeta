# Launch a healing run with the environment determinism requires.
#
# These variables MUST be set before python starts. CUBLAS_WORKSPACE_CONFIG in
# particular is read when CUDA initialises; setting it from inside a running
# process is too late and cuBLAS reductions become irreproducible. heal.py checks
# for it and refuses to start rather than producing a run that cannot be replayed.
#
#   .\scripts\train\run_heal.ps1 -Config configs\heal_b0.yaml
#   .\scripts\train\run_heal.ps1 -Config configs\heal_b0.yaml -DryRun
#   .\scripts\train\run_heal.ps1 -Config configs\heal_b0.yaml -Resume auto

param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$Resume = "auto",
    [int]$MaxSteps = 0,
    [string]$Out = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Deterministic cuBLAS workspace. Without this, matmul reductions vary run to run.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# Stable hashing for any set/dict iteration in helper code.
$env:PYTHONHASHSEED = "0"
# Reduces allocator fragmentation, which is the residual OOM risk once the
# pre-flight probe has bounded activation memory.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# The console is cp1252 by default and dies printing byte-level token strings.
$env:PYTHONIOENCODING = "utf-8"

Write-Host "repo   : $repo"
Write-Host "config : $Config"
Write-Host "env    : CUBLAS_WORKSPACE_CONFIG=$($env:CUBLAS_WORKSPACE_CONFIG) PYTHONHASHSEED=$($env:PYTHONHASHSEED)"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $vram = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    Write-Host "vram   : $vram (close other apps if free memory is below the cap)"
}

$argv = @(
    (Join-Path $PSScriptRoot "heal.py"),
    "--config", (Join-Path $repo $Config),
    "--resume", $Resume
)
if ($MaxSteps -gt 0) { $argv += @("--max-steps", $MaxSteps) }
if ($Out -ne "")     { $argv += @("--out", $Out) }
if ($DryRun)         { $argv += "--dry-run" }

& python @argv
exit $LASTEXITCODE
