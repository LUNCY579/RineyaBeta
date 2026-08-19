# Condition B -- same records as C, trajectory order.
param([int]$Seed = 20260818, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra)
. "$PSScriptRoot\_common.ps1"
Invoke-Lab train -c configs/train_split_ordered.yaml --set "experiment.seed=$Seed" `
    --set "experiment.output_dir=runs/train_split_ordered_seed$Seed" @Extra
