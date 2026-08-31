$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Arms = @("cot_baseline", "state_ordered", "state_shuffled", "state_reverse", "state_final_repeated", "literal_ordered")
foreach ($Arm in $Arms) {
    python -m dast.cli train --config "configs/$Arm.yaml"
}
# Operator and feedback phases are gated; see docs/RUNBOOK.md.
