#!/usr/bin/env bash
# Evaluate both arms plus the untouched base checkpoint.
# Usage: scripts/evaluate.sh [seed] [split]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SEED=${1:-20260818}
SPLIT=${2:-dev}

python -m rineya_lab.cli evaluate -c configs/eval_base_model.yaml \
  --set "evaluation.split=${SPLIT}" \
  --set "experiment.output_dir=runs/eval_base_${SPLIT}"

python -m rineya_lab.cli evaluate -c configs/eval_full.yaml \
  --set "evaluation.split=${SPLIT}" \
  --set "evaluation.backend.adapter_path=runs/train_full_seed${SEED}/adapter" \
  --set "experiment.output_dir=runs/eval_full_seed${SEED}_${SPLIT}"

python -m rineya_lab.cli evaluate -c configs/eval_split_stateful.yaml \
  --set "evaluation.split=${SPLIT}" \
  --set "evaluation.backend.adapter_path=runs/train_split_seed${SEED}/adapter" \
  --set "experiment.output_dir=runs/eval_split_seed${SEED}_${SPLIT}"

python -m rineya_lab.cli evaluate -c configs/eval_perturbations.yaml \
  --set "evaluation.split=${SPLIT}" \
  --set "evaluation.backend.adapter_path=runs/train_split_seed${SEED}/adapter" \
  --set "experiment.output_dir=runs/eval_perturbations_seed${SEED}_${SPLIT}"
