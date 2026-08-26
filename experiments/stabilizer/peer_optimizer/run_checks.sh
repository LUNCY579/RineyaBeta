#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi
"$PYTHON" -m compileall "$HERE/src" "$HERE/tests"
"$PYTHON" -m unittest discover -s "$HERE/tests" -p 'test_*.py' -v
"$PYTHON" "$HERE/src/feasibility.py" --out "$HERE/results/feasibility.json"
"$PYTHON" "$HERE/src/stress_feasibility.py" --train-steps 40 --eval-tasks 32 \
  --out "$HERE/results/feasibility_multiseed.json"
echo "All peer-stabilizer checks passed."
