#!/usr/bin/env bash
# Condition A -- FULL. Usage: scripts/train_full.sh [seed] [extra --set overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SEED=${1:-20260818}; shift || true
python -m rineya_lab.cli train \
  -c "${CONFIG:-configs/train_full.yaml}" \
  --set "experiment.seed=${SEED}" \
  --set "experiment.output_dir=runs/train_full_seed${SEED}" \
  "$@"
