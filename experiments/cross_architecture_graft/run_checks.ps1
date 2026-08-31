$ErrorActionPreference = "Stop"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Root = "experiments/cross_architecture_graft"
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("cag-checks-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    & $Python -m compileall -q $Root
    if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

    & $Python -m unittest discover -s "$Root/tests" -p "test_*.py" -v
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }

    & $Python -m experiments.cross_architecture_graft.src.verify_plans |
        Out-File -Encoding utf8 (Join-Path $TempDir "plan-bundle.json")
    if ($LASTEXITCODE -ne 0) { throw "plan verification failed" }

    $Evidence = Join-Path $TempDir "smoke-evidence.json"
    & $Python -m experiments.cross_architecture_graft.src.runner `
        --config "$Root/configs/smoke.json" `
        --command smoke `
        --out $Evidence |
        Out-File -Encoding utf8 (Join-Path $TempDir "smoke-stdout.json")
    if ($LASTEXITCODE -ne 0) { throw "smoke run failed" }

    & $Python -m experiments.cross_architecture_graft.src.verify_evidence $Evidence
    if ($LASTEXITCODE -ne 0) { throw "evidence verification failed" }

    Write-Host "CAG v1 checks: PASS"
}
finally {
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
