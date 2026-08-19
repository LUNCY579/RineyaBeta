# Paired FULL vs SPLIT comparison.
param([int]$Seed = 20260818, [string]$Split = "dev")
. "$PSScriptRoot\_common.ps1"
Invoke-Lab compare -c configs/compare_full_vs_split.yaml `
    --baseline "runs/eval_full_seed${Seed}_$Split" `
    --candidate "runs/eval_split_seed${Seed}_$Split" `
    --output-dir "runs/comparison_seed${Seed}_$Split"
