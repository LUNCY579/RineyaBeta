param(
    [switch]$Offline,
    [switch]$SkipModelSmoke,
    [switch]$DoctorOnly,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $RepoRoot
try {
    $arguments = @(
        "scripts/critical_core/run_discovery_local.py"
    )
    if ($Offline) { $arguments += "--offline" }
    if ($SkipModelSmoke) { $arguments += "--skip-model-smoke" }
    if ($DoctorOnly) { $arguments += "--doctor-only" }

    Write-Host "CRCC v1 local discovery"
    Write-Host "Repository: $RepoRoot"
    Write-Host "Python: $Python"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "CRCC local runner exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
