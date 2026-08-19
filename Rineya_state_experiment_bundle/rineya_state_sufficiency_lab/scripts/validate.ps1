# Strictly validate the bundled dataset before any training run.
param([string[]]$Splits = @("dev", "test"))
. "$PSScriptRoot\_common.ps1"
New-Item -ItemType Directory -Force -Path runs | Out-Null
Invoke-Lab validate-data --splits @Splits -o runs/validation_report.json
Write-Host "Validation report: runs/validation_report.json"
