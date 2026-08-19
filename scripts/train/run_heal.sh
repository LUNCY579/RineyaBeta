#!/usr/bin/env bash
# POSIX twin of run_heal.ps1, for the CUDA host. Same contract: the determinism
# environment must be set before python starts. CUBLAS_WORKSPACE_CONFIG is read
# when CUDA initialises, so it cannot be fixed from inside the process -- heal.py
# checks for it and refuses to start rather than produce an unreplayable run.
#
#   ./scripts/train/run_heal.sh configs/heal_b0.yaml
#   ./scripts/train/run_heal.sh configs/heal_b0.yaml --dry-run
#   ./scripts/train/run_heal.sh configs/heal_b0.yaml --resume auto
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: run_heal.sh <config.yaml> [extra heal.py args...]}"
shift || true

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONIOENCODING="utf-8"

echo "repo   : ${REPO}"
echo "config : ${CONFIG}"
echo "env    : CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG} PYTHONHASHSEED=${PYTHONHASHSEED}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "vram   : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
fi

exec python "${REPO}/scripts/train/heal.py" --config "${REPO}/${CONFIG}" "$@"
