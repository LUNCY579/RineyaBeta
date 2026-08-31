#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
OUT="${1:-experiments/cross_architecture_graft/results/smoke.json}"
"$PYTHON_BIN" -m experiments.cross_architecture_graft.src.runner \
  --config experiments/cross_architecture_graft/configs/smoke.json \
  --command smoke \
  --out "$OUT"
"$PYTHON_BIN" -m experiments.cross_architecture_graft.src.verify_evidence "$OUT"
