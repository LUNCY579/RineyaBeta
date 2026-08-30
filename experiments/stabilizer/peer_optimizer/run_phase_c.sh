#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi
CONFIG="${1:-$HERE/configs/phase_c_down_proj_0p1.yaml}"
RUNNER="$HERE/src/phase_c_runner.py"

"$PYTHON" "$RUNNER" prepare --config "$CONFIG"
"$PYTHON" "$RUNNER" run --config "$CONFIG" --resume
"$PYTHON" "$RUNNER" verify --config "$CONFIG"
