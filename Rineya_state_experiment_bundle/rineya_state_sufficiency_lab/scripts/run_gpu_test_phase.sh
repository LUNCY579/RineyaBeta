#!/usr/bin/env bash
# PHASE 7 -- open the TEST split. Runs exactly once, and refuses until the
# non-inferiority margin has been frozen and justified.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SEEDS=("${@:-1 2 3}")
if [ "$#" -gt 0 ]; then SEEDS=("$@"); else SEEDS=(1 2 3); fi
LAB="python -m rineya_lab.cli"

python - <<'PY'
import sys
from rineya_lab.config import load_config
cfg = load_config("configs/compare_full_vs_split.yaml")
if not cfg.comparison.margin_frozen:
    sys.exit(
        "REFUSING to open TEST: comparison.margin_frozen is false.\n"
        "Freeze and justify the margin in docs/PREREGISTRATION.md section 6, then set\n"
        "  comparison:\n    margin_frozen: true\n    margin_justification: \"...\"\n"
        "in configs/compare_full_vs_split.yaml."
    )
if not cfg.comparison.margin_justification.strip() or "REPLACE" in cfg.comparison.margin_justification:
    sys.exit("REFUSING to open TEST: comparison.margin_justification is still the placeholder.")
print(f"Margin frozen at {cfg.comparison.non_inferiority_margin_pp} pp. TEST may be opened once.")
PY

for SEED in "${SEEDS[@]}"; do
  $LAB evaluate -c configs/gpu_eval_full.yaml --set evaluation.split=test \
      --set evaluation.backend.adapter_path="runs/T_full_s${SEED}/adapter" \
      --output-dir "runs/TEST_full_s${SEED}"
  $LAB evaluate -c configs/gpu_eval_split.yaml --set evaluation.split=test \
      --set evaluation.backend.adapter_path="runs/T_split_s${SEED}/adapter" \
      --output-dir "runs/TEST_split_s${SEED}"
  $LAB compare -c configs/compare_full_vs_split.yaml \
      --baseline "runs/TEST_full_s${SEED}" --candidate "runs/TEST_split_s${SEED}" \
      --output-dir "runs/TEST_cmp_s${SEED}"
done
$LAB aggregate runs/TEST_cmp_s*/comparison_report.json --output-dir runs/agg_TEST
echo
echo "TEST result of record: runs/agg_TEST/multiseed_report.md"
echo "TEST has now been used. Do not re-run against a different configuration."
