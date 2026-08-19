#!/usr/bin/env bash
# Condition C -- SPLIT (globally shuffled). Usage: scripts/train_split.sh [seed] [overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SEED=${1:-20260818}; shift || true
python -m rineya_lab.cli train \
  -c "${CONFIG:-configs/train_split_stateful.yaml}" \
  --set "experiment.seed=${SEED}" \
  --set "experiment.output_dir=runs/train_split_seed${SEED}" \
  "$@"
