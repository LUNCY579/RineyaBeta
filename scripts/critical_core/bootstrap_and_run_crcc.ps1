param(
    [switch]$Offline,
    [switch]$SkipModelSmoke,
    [switch]$DoctorOnly,
    [switch]$RecreateEnv,
    [string]$PythonLauncher = "py"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Venv = Join-Path $RepoRoot ".venv-crcc-v1"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

Push-Location $RepoRoot
try {
    if ($RecreateEnv -and (Test-Path $Venv)) {
        Remove-Item -Recurse -Force $Venv
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating isolated CRCC v1 Python 3.11 environment..."
        & $PythonLauncher -3.11 -m venv $Venv
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create Python 3.11 virtual environment. Install Python 3.11 or pass a compatible launcher."
        }
    }

    & $VenvPython -m pip install --disable-pip-version-check --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

    & $VenvPython -m pip install --disable-pip-version-check -r requirements-crcc-v1.txt
    if ($LASTEXITCODE -ne 0) { throw "CRCC dependency installation failed" }

    $arguments = @("scripts/critical_core/run_discovery_local.py")
    if ($Offline) { $arguments += "--offline" }
    if ($SkipModelSmoke) { $arguments += "--skip-model-smoke" }
    if ($DoctorOnly) { $arguments += "--doctor-only" }

    Write-Host "Running CRCC v1 with isolated environment: $Venv"
    & $VenvPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "CRCC local runner exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
