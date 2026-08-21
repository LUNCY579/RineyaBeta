#!/usr/bin/env bash
# Quality vs KV budget, AGAINST THE FIXED IMPLEMENTATION.
# The previous run of this file measured a sliding window because eviction fired
# during prefill before any scores existed; those artefacts are kept under
# results/_invalidated_prefill_bug/ for comparison.
#
# Two datasets, chosen for contrast:
#   chat_dev  conversational -- context largely redundant; also a metric that
#             CANNOT detect this damage (see THEORY.md §8), kept as a control
#   rag_dev   evidence-dense -- almost every token load-bearing, the worst case
set -e
PY=../../.venv/Scripts/python.exe

for DS in chat_dev rag_dev; do
  echo "### $DS: reference, eviction off"
  $PY -u src/run_eval_sig.py --budget 0 --dataset $DS
  for F in 0.75 0.50 0.25; do
    echo "### $DS: keep $F of prompt"
    $PY -u src/run_eval_sig.py --keep-frac $F --dataset $DS --compare-to off
  done
  echo "### $DS: keep 0.25 with recent scaled to the budget (not fixed at 32)"
  $PY -u src/run_eval_sig.py --keep-frac 0.25 --recent-frac 0.25 --dataset $DS --compare-to off
  echo "### $DS: keep 0.50 with SnapKV pooling (width 7)"
  $PY -u src/run_eval_sig.py --keep-frac 0.50 --pool 7 --dataset $DS --compare-to off
done
$PY src/summarize.py
