#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m dast.cli validate-data fixtures/tiny_reasoning.jsonl
python -m dast.cli smoke --output-dir runs/smoke
pytest -q
python -m compileall -q src tests
