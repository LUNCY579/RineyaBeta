#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
ROOT="experiments/cross_architecture_graft"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

"$PYTHON_BIN" -m compileall -q "$ROOT"
"$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
"$PYTHON_BIN" -m experiments.cross_architecture_graft.src.verify_plans \
  > "$TMP_DIR/plan-bundle.json"
"$PYTHON_BIN" -m experiments.cross_architecture_graft.src.runner \
  --config "$ROOT/configs/smoke.json" \
  --command smoke \
  --out "$TMP_DIR/smoke-evidence.json" \
  > "$TMP_DIR/smoke-stdout.json"
"$PYTHON_BIN" -m experiments.cross_architecture_graft.src.verify_evidence \
  "$TMP_DIR/smoke-evidence.json"

echo "CAG v1 checks: PASS"
