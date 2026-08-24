#!/usr/bin/env bash
set -euo pipefail

LAYER="${LAYER:-14}"
GROUP_SIZE="${GROUP_SIZE:-128}"
BUDGET_FRACTION="${BUDGET_FRACTION:-0.5}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-64}"
EVALUATION_LIMIT="${EVALUATION_LIMIT:-0}"
CAP_MIB="${CAP_MIB:-5120}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CALIB="artifacts/sgqr_runs/calibration_l${LAYER}_g${GROUP_SIZE}"
ROUTER="artifacts/sgqr_runs/router_l${LAYER}_g${GROUP_SIZE}.pt"
EVAL_ROOT="artifacts/sgqr_runs/evaluation/l${LAYER}_g${GROUP_SIZE}"

python scripts/verify/verify_sgqr.py

python scripts/sgqr/calibrate.py \
  --dataset chat_dev \
  --layer "$LAYER" \
  --bits 4 \
  --group-size "$GROUP_SIZE" \
  --limit "$CALIBRATION_LIMIT" \
  --cap-mib "$CAP_MIB" \
  --out "$CALIB"

GROUP_COUNT=$(( (3072 + GROUP_SIZE - 1) / GROUP_SIZE ))
BUDGET_GROUPS="$(python - <<PY
import math
print(math.ceil(${GROUP_COUNT} * ${BUDGET_FRACTION}))
PY
)"

python scripts/sgqr/train_router.py \
  --trace "$CALIB/utility_trace.pt" \
  --out "$ROUTER" \
  --budget-groups "$BUDGET_GROUPS"

LIMIT_ARGS=()
if [[ "$EVALUATION_LIMIT" -gt 0 ]]; then
  LIMIT_ARGS=(--limit "$EVALUATION_LIMIT")
fi

python scripts/sgqr/evaluate.py --dataset chat_dev --mode dense \
  --layer "$LAYER" --cap-mib "$CAP_MIB" --out "$EVAL_ROOT/dense" "${LIMIT_ARGS[@]}"

python scripts/sgqr/evaluate.py --dataset chat_dev --mode deleted \
  --layer "$LAYER" --cap-mib "$CAP_MIB" --out "$EVAL_ROOT/deleted" "${LIMIT_ARGS[@]}"

python scripts/sgqr/evaluate.py --dataset chat_dev --mode int8_all \
  --layer "$LAYER" --group-size "$GROUP_SIZE" --cap-mib "$CAP_MIB" \
  --out "$EVAL_ROOT/int8_all" "${LIMIT_ARGS[@]}"

python scripts/sgqr/evaluate.py --dataset chat_dev --mode int4_all \
  --layer "$LAYER" --group-size "$GROUP_SIZE" --cap-mib "$CAP_MIB" \
  --out "$EVAL_ROOT/int4_all" "${LIMIT_ARGS[@]}"

python scripts/sgqr/evaluate.py --dataset chat_dev --mode random_topk \
  --layer "$LAYER" --bits 4 --group-size "$GROUP_SIZE" \
  --budget-fraction "$BUDGET_FRACTION" --cap-mib "$CAP_MIB" \
  --out "$EVAL_ROOT/random" "${LIMIT_ARGS[@]}"

python scripts/sgqr/evaluate.py --dataset chat_dev --mode magnitude_topk \
  --layer "$LAYER" --bits 4 --group-size "$GROUP_SIZE" \
  --budget-fraction "$BUDGET_FRACTION" --cap-mib "$CAP_MIB" \
  --out "$EVAL_ROOT/magnitude" "${LIMIT_ARGS[@]}"

python scripts/sgqr/evaluate.py --dataset chat_dev --mode static_topk \
  --layer "$LAYER" --bits 4 --group-size "$GROUP_SIZE" \
  --budget-fraction "$BUDGET_FRACTION" --scores "$CALIB/summary.json" \
  --cap-mib "$CAP_MIB" --out "$EVAL_ROOT/static" "${LIMIT_ARGS[@]}"

python scripts/sgqr/evaluate.py --dataset chat_dev --mode dynamic_topk \
  --layer "$LAYER" --bits 4 --group-size "$GROUP_SIZE" \
  --budget-fraction "$BUDGET_FRACTION" --router "$ROUTER" \
  --cap-mib "$CAP_MIB" --out "$EVAL_ROOT/dynamic" "${LIMIT_ARGS[@]}"

python scripts/sgqr/report.py \
  --dense "$EVAL_ROOT/dense/summary.json" \
  --deleted "$EVAL_ROOT/deleted/summary.json" \
  --run "int8_all=$EVAL_ROOT/int8_all/summary.json" \
  --run "int4_all=$EVAL_ROOT/int4_all/summary.json" \
  --run "random=$EVAL_ROOT/random/summary.json" \
  --run "magnitude=$EVAL_ROOT/magnitude/summary.json" \
  --run "static=$EVAL_ROOT/static/summary.json" \
  --run "dynamic=$EVAL_ROOT/dynamic/summary.json" \
  --out "$EVAL_ROOT/comparison.json"

echo "SGQR reference workflow complete: $EVAL_ROOT"
