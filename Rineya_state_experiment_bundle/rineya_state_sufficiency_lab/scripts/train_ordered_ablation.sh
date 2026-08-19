#!/usr/bin/env bash
# Condition B -- identical records to condition C, trajectory order. B vs C isolates order.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SEED=${1:-20260818}; shift || true
python -m rineya_lab.cli train \
  -c configs/train_split_ordered.yaml \
  --set "experiment.seed=${SEED}" \
  --set "experiment.output_dir=runs/train_split_ordered_seed${SEED}" \
  "$@"
