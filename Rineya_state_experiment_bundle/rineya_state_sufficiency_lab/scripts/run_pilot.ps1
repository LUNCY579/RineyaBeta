# Whole pilot: validate -> plan/budget -> train A/B/C -> evaluate -> compare.
param([int[]]$Seeds = @(20260818))
. "$PSScriptRoot\_common.ps1"

& "$PSScriptRoot\validate.ps1"
& "$PSScriptRoot\plan_and_budget.ps1"

foreach ($Seed in $Seeds) {
    Write-Host "=== seed $Seed ==="
    & "$PSScriptRoot\train_full.ps1" -Seed $Seed
    & "$PSScriptRoot\train_split.ps1" -Seed $Seed
    & "$PSScriptRoot\train_ordered_ablation.ps1" -Seed $Seed
    & "$PSScriptRoot\evaluate.ps1" -Seed $Seed -Split dev
    & "$PSScriptRoot\compare.ps1" -Seed $Seed -Split dev
}
Write-Host ""
Write-Host "DEV pilot complete. Freeze docs/PREREGISTRATION.md (margin_frozen: true) before touching TEST."
