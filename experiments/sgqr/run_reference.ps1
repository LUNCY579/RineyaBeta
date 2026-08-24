param(
    [int]$Layer = 14,
    [int]$GroupSize = 128,
    [double]$BudgetFraction = 0.5,
    [int]$CalibrationLimit = 64,
    [int]$EvaluationLimit = 0,
    [double]$CapMiB = 5120
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$Calib = "artifacts/sgqr_runs/calibration_l${Layer}_g${GroupSize}"
$Router = "artifacts/sgqr_runs/router_l${Layer}_g${GroupSize}.pt"
$EvalRoot = "artifacts/sgqr_runs/evaluation/l${Layer}_g${GroupSize}"

python scripts/verify/verify_sgqr.py

python scripts/sgqr/calibrate.py `
    --dataset chat_dev `
    --layer $Layer `
    --bits 4 `
    --group-size $GroupSize `
    --limit $CalibrationLimit `
    --cap-mib $CapMiB `
    --out $Calib

$GroupCount = [math]::Ceiling(3072 / $GroupSize)
$BudgetGroups = [math]::Ceiling($GroupCount * $BudgetFraction)

python scripts/sgqr/train_router.py `
    --trace "$Calib/utility_trace.pt" `
    --out $Router `
    --budget-groups $BudgetGroups

$LimitArgs = @()
if ($EvaluationLimit -gt 0) {
    $LimitArgs = @("--limit", "$EvaluationLimit")
}

python scripts/sgqr/evaluate.py --dataset chat_dev --mode dense `
    --layer $Layer --cap-mib $CapMiB --out "$EvalRoot/dense" @LimitArgs

python scripts/sgqr/evaluate.py --dataset chat_dev --mode deleted `
    --layer $Layer --cap-mib $CapMiB --out "$EvalRoot/deleted" @LimitArgs

python scripts/sgqr/evaluate.py --dataset chat_dev --mode int8_all `
    --layer $Layer --group-size $GroupSize --cap-mib $CapMiB `
    --out "$EvalRoot/int8_all" @LimitArgs

python scripts/sgqr/evaluate.py --dataset chat_dev --mode int4_all `
    --layer $Layer --group-size $GroupSize --cap-mib $CapMiB `
    --out "$EvalRoot/int4_all" @LimitArgs

python scripts/sgqr/evaluate.py --dataset chat_dev --mode random_topk `
    --layer $Layer --bits 4 --group-size $GroupSize `
    --budget-fraction $BudgetFraction --cap-mib $CapMiB `
    --out "$EvalRoot/random" @LimitArgs

python scripts/sgqr/evaluate.py --dataset chat_dev --mode magnitude_topk `
    --layer $Layer --bits 4 --group-size $GroupSize `
    --budget-fraction $BudgetFraction --cap-mib $CapMiB `
    --out "$EvalRoot/magnitude" @LimitArgs

python scripts/sgqr/evaluate.py --dataset chat_dev --mode static_topk `
    --layer $Layer --bits 4 --group-size $GroupSize `
    --budget-fraction $BudgetFraction --scores "$Calib/summary.json" `
    --cap-mib $CapMiB --out "$EvalRoot/static" @LimitArgs

python scripts/sgqr/evaluate.py --dataset chat_dev --mode dynamic_topk `
    --layer $Layer --bits 4 --group-size $GroupSize `
    --budget-fraction $BudgetFraction --router $Router `
    --cap-mib $CapMiB --out "$EvalRoot/dynamic" @LimitArgs

python scripts/sgqr/report.py `
    --dense "$EvalRoot/dense/summary.json" `
    --deleted "$EvalRoot/deleted/summary.json" `
    --run "int8_all=$EvalRoot/int8_all/summary.json" `
    --run "int4_all=$EvalRoot/int4_all/summary.json" `
    --run "random=$EvalRoot/random/summary.json" `
    --run "magnitude=$EvalRoot/magnitude/summary.json" `
    --run "static=$EvalRoot/static/summary.json" `
    --run "dynamic=$EvalRoot/dynamic/summary.json" `
    --out "$EvalRoot/comparison.json"

Write-Host "SGQR reference workflow complete: $EvalRoot"
