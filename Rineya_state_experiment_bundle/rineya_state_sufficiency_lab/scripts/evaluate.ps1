# Evaluate both arms plus the untouched base checkpoint.
param([int]$Seed = 20260818, [string]$Split = "dev")
. "$PSScriptRoot\_common.ps1"

Invoke-Lab evaluate -c configs/eval_base_model.yaml `
    --set "evaluation.split=$Split" `
    --set "experiment.output_dir=runs/eval_base_$Split"

Invoke-Lab evaluate -c configs/eval_full.yaml `
    --set "evaluation.split=$Split" `
    --set "evaluation.backend.adapter_path=runs/train_full_seed$Seed/adapter" `
    --set "experiment.output_dir=runs/eval_full_seed${Seed}_$Split"

Invoke-Lab evaluate -c configs/eval_split_stateful.yaml `
    --set "evaluation.split=$Split" `
    --set "evaluation.backend.adapter_path=runs/train_split_seed$Seed/adapter" `
    --set "experiment.output_dir=runs/eval_split_seed${Seed}_$Split"

Invoke-Lab evaluate -c configs/eval_perturbations.yaml `
    --set "evaluation.split=$Split" `
    --set "evaluation.backend.adapter_path=runs/train_split_seed$Seed/adapter" `
    --set "experiment.output_dir=runs/eval_perturbations_seed${Seed}_$Split"
