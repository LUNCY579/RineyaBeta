param(
    [string]$Config,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$Repo = [System.IO.Path]::GetFullPath((Join-Path $Here "..\..\.."))
if (-not $Config) {
    $Config = Join-Path $Here "configs\phase_c_down_proj_0p1.yaml"
}
if (-not $Python) {
    $Candidate = Join-Path $Repo ".venv\Scripts\python.exe"
    $Python = if (Test-Path $Candidate) { $Candidate } else { "python" }
}
$Runner = Join-Path $Here "src\phase_c_runner.py"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Invoke-CheckedPython @($Runner, "prepare", "--config", $Config)
Invoke-CheckedPython @($Runner, "run", "--config", $Config, "--resume")
Invoke-CheckedPython @($Runner, "verify", "--config", $Config)
