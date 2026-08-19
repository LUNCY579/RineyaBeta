# Condition C -- SPLIT (globally shuffled transitions).
param([int]$Seed = 20260818, [string]$Config = "configs/train_split_stateful.yaml",
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra)
. "$PSScriptRoot\_common.ps1"
Invoke-Lab train -c $Config --set "experiment.seed=$Seed" `
    --set "experiment.output_dir=runs/train_split_seed$Seed" @Extra
