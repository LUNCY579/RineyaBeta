#!/usr/bin/env bash
# Tier-1 (plus two Tier-2) arm sweep at the selected layer, chat_dev, full 510 cases.
#
# Run b0-dense with --self-check FIRST and confirm |delta| = 0 against
# artifacts/gate_runs/r2_base_chat_dev before trusting anything here. A harness
# that silently differs from the one that made the baselines measures the
# harness, not the arms. (Confirmed for this campaign: |delta| = 0.000e+00.)
#
# Order matters: b2-fp-all establishes the noise floor (same weights, remnant
# code path) and must exist before any quantized arm is interpreted.
#
# Each arm writes its own log so a killed sweep can be diagnosed and resumed
# per arm rather than restarted whole; arms already holding a summary.json are
# skipped.
set -u

PY=".venv/Scripts/python.exe"
L=20
DS=chat_dev
RANK=experiments/csgqr/results/trace_L20_int4/significance_analysis.json
DIR=experiments/csgqr/results
LOG="$DIR/tier1.log"

run() {
  local arm="$1"; shift
  local out="$DIR/eval_${arm}__${DS}__L${L}"
  if [ -f "$out/summary.json" ]; then
    echo "[skip] $arm already has $out/summary.json" | tee -a "$LOG"
    return
  fi
  echo "[start] $arm $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
  $PY experiments/csgqr/src/run_eval.py --arm "$arm" --layer "$L" \
      --dataset "$DS" --ranking-from "$RANK" "$@" > "$DIR/${arm}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL $rc] $arm -- see $DIR/${arm}.log" | tee -a "$LOG"
    tail -5 "$DIR/${arm}.log" | tee -a "$LOG"
    return
  fi
  grep -E "mean_quality_score|active_fraction|packed |peak allocated" \
      "$DIR/${arm}.log" | tee -a "$LOG"
  echo "[done] $arm $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
}

echo "tier-1 sweep started $(date -u +%Y-%m-%dT%H:%M:%SZ)  layer $L  $DS" >> "$LOG"

run b1-delete
run b2-fp-all
run b3-int8-all
run b4-int4-all
run b5-random-25
run b6-static-25
run b7-conf-whole --tau-skip 0.9
run b10-csgqr --tau-skip 0.9 --tau-full 0.3
run b13-wrong-layer

echo "tier-1 sweep finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
