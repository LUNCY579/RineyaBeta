$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
python -m dast.cli validate-data fixtures/tiny_reasoning.jsonl
python -m dast.cli smoke --output-dir runs/smoke
pytest -q
python -m compileall -q src tests
