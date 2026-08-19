#!/usr/bin/env bash
# Whole pilot end to end: validate -> plan/budget -> train A/B/C -> evaluate -> compare.
# Usage: scripts/run_pilot.sh [seed...]      e.g. scripts/run_pilot.sh 1 2 3
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SEEDS=("${@:-20260818}")

scripts/validate.sh
scripts/plan_and_budget.sh

for SEED in "${SEEDS[@]}"; do
  echo "=== seed ${SEED} ==="
  scripts/train_full.sh "${SEED}"
  scripts/train_split.sh "${SEED}"
  scripts/train_ordered_ablation.sh "${SEED}"
  scripts/evaluate.sh "${SEED}" dev
  scripts/compare.sh "${SEED}" dev
done

echo
echo "DEV pilot complete. Freeze docs/PREREGISTRATION.md (margin_frozen: true) before touching TEST."
