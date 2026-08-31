#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Phase 1: the minimum causal claim ladder. TEST remains closed in every config.
arms=(cot_baseline state_ordered state_shuffled state_reverse state_final_repeated literal_ordered)
for arm in "${arms[@]}"; do
  python -m dast.cli train --config "configs/${arm}.yaml"
done

# Phase 2 is intentionally not automatic. Run operator_ordered and
# operator_shuffled only after the ordered-state gate passes on all frozen seeds.
