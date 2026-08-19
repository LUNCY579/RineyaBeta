#!/usr/bin/env bash
# Show the training view plans and compute matched budgets for both arms.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
mkdir -p runs
python -m rineya_lab.cli plan -c configs/train_full.yaml           -o runs/plan_full.json
python -m rineya_lab.cli plan -c configs/train_split_stateful.yaml -o runs/plan_split.json
python -m rineya_lab.cli budget \
  -c configs/train_full.yaml \
  --config2 configs/train_split_stateful.yaml \
  --sample "${SAMPLE:-2000}" \
  -o runs/budget.json
echo "Copy the matched_budget values into configs/budgets/*.yaml and apply the same file to BOTH arms."
