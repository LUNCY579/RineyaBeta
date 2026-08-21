#!/usr/bin/env bash
# Quality sweep, ordered so each step's validity depends only on earlier ones.
# Run from experiments/attention_shape/.  ~45 min on this card.
#
# Two kernel modes are used deliberately (see the comment block in run_eval.py):
#   step 0  --no-compat, the only mode comparable to the stored gate baselines
#   step 1+ compat on, so every variant is compared against a base that ran the
#           same SDPA kernel. Sliding-window variants pass an explicit mask and
#           take the repeat_kv path regardless, so patching the base too is what
#           makes this a comparison of attention shapes rather than of kernels.
set -e
PY=../../.venv/Scripts/python.exe

echo "### 0. protocol fidelity: does this harness reproduce the stored baseline exactly? (P4)"
$PY -u src/run_eval.py --self-check --dataset chat_dev --out results/_selfcheck_chat_dev

echo "### 1. reference runs in sweep mode -- everything below diffs against these"
$PY -u src/run_eval.py --variant base-8KV --dataset chat_dev
$PY -u src/run_eval.py --variant base-8KV --dataset rag_dev

echo "### 2. P1: w=512 exceeds every chat_dev prompt, so sliding is a no-op."
echo "        Correct result: 510/510 byte-identical."
$PY -u src/run_eval.py --variant swa-w512-g4 --dataset chat_dev --compare-to base-8KV

echo "### 3. P2: unhealed head merging should break even at short context"
$PY -u src/run_eval.py --variant gqa-2KV --dataset chat_dev --compare-to base-8KV
$PY -u src/run_eval.py --variant gqa-1KV --dataset chat_dev --compare-to base-8KV

echo "### 4. P3: w=128 is below the mean rag_dev prompt (232 tok) -> graded damage"
$PY -u src/run_eval.py --variant swa-w128-g4 --dataset rag_dev --compare-to base-8KV

echo "### 5. the head axis at comparable cache, same eval"
$PY -u src/run_eval.py --variant gqa-2KV --dataset rag_dev --compare-to base-8KV

$PY src/summarize.py
