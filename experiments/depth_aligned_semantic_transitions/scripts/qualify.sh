#!/usr/bin/env bash
# Authoritative local qualification gate. See scripts/qualify.py for why this
# replaces GitHub Actions and why it is stronger than the workflow it replaces.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$ROOT/scripts/qualify.py" "$@"
