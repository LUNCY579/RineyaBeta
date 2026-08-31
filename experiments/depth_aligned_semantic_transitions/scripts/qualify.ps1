# Authoritative local qualification gate. See scripts/qualify.py for why this
# replaces GitHub Actions and why it is stronger than the workflow it replaces.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python (Join-Path $PSScriptRoot "qualify.py") @args
exit $LASTEXITCODE
