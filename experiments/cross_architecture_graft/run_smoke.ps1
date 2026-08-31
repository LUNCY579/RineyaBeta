param(
    [string]$Out = "experiments/cross_architecture_graft/results/smoke.json"
)

$ErrorActionPreference = "Stop"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $Python -m experiments.cross_architecture_graft.src.runner `
    --config experiments/cross_architecture_graft/configs/smoke.json `
    --command smoke `
    --out $Out
if ($LASTEXITCODE -ne 0) { throw "smoke run failed" }
& $Python -m experiments.cross_architecture_graft.src.verify_evidence $Out
if ($LASTEXITCODE -ne 0) { throw "evidence verification failed" }
