#!/usr/bin/env bash
# Offline end-to-end smoke test. No GPU, no network, no checkpoint download.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
python -m rineya_lab.cli smoke --all "$@"
