#!/usr/bin/env bash
# Paired FULL vs SPLIT comparison. Usage: scripts/compare.sh [seed] [split]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SEED=${1:-20260818}
SPLIT=${2:-dev}
python -m rineya_lab.cli compare \
  -c configs/compare_full_vs_split.yaml \
  --baseline "runs/eval_full_seed${SEED}_${SPLIT}" \
  --candidate "runs/eval_split_seed${SEED}_${SPLIT}" \
  --output-dir "runs/comparison_seed${SEED}_${SPLIT}"
