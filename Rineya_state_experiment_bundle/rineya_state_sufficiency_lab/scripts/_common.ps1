# Shared helpers for the PowerShell launchers.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
if (-not $env:PYTHONHASHSEED) { $env:PYTHONHASHSEED = "0" }

function Invoke-Lab {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$LabArgs)
    & python -m rineya_lab.cli @LabArgs
    if ($LASTEXITCODE -ne 0) { throw "rineya-lab exited with code $LASTEXITCODE" }
}
