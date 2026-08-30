param(
    [string]$Python = "",
    [switch]$SkipStress
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = (Resolve-Path (Join-Path $Here "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $Repo ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python interpreter not found: $Python"
}

& $Python -m compileall (Join-Path $Here "src") (Join-Path $Here "tests")
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

& $Python -m unittest discover -s (Join-Path $Here "tests") -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }

& $Python (Join-Path $Here "src\feasibility.py") `
    --out (Join-Path $Here "results\feasibility.json")
if ($LASTEXITCODE -ne 0) { throw "feasibility study failed" }

if (-not $SkipStress) {
    & $Python (Join-Path $Here "src\stress_feasibility.py") `
        --train-steps 40 --eval-tasks 32 `
        --out (Join-Path $Here "results\feasibility_multiseed.json")
    if ($LASTEXITCODE -ne 0) { throw "multi-seed feasibility stress failed" }
}

Write-Host "All peer-stabilizer checks passed."
