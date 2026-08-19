#!/usr/bin/env bash
# Strictly validate the bundled dataset before any training run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SPLITS=${SPLITS:-"dev test"}
mkdir -p runs
python -m rineya_lab.cli validate-data --splits ${SPLITS} -o runs/validation_report.json "$@"
echo "Validation report: runs/validation_report.json"
