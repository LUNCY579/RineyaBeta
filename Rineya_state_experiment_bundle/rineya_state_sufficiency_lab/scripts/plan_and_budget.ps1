# Show both view plans and compute matched budgets.
param([int]$Sample = 2000)
. "$PSScriptRoot\_common.ps1"
New-Item -ItemType Directory -Force -Path runs | Out-Null
Invoke-Lab plan -c configs/train_full.yaml -o runs/plan_full.json
Invoke-Lab plan -c configs/train_split_stateful.yaml -o runs/plan_split.json
Invoke-Lab budget -c configs/train_full.yaml --config2 configs/train_split_stateful.yaml `
    --sample $Sample -o runs/budget.json
Write-Host "Copy matched_budget into configs/budgets/*.yaml and apply the SAME file to both arms."
