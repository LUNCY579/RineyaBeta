#!/usr/bin/env bash
# Staged rehearsal: a SMALL but fully REAL run, before committing to the 3-seed design.
#
#   scripts/run_gpu_rehearsal.sh
#
# ~30-45 minutes of GPU time. It exercises every path the full run takes -- real
# checkpoint, real tokenizer, real LoRA, real batched decoding, real rollouts --
# at a fraction of the scale, and answers the two questions that decide whether
# the full run is worth launching:
#
#   1. Does an adapter actually learn this task on this checkpoint? (headroom)
#   2. How long will the full design really take? (timing extrapolation)
#
# This is a rehearsal, NOT a result. Its runs are written under runs/rehearsal_*
# and must never be reported as findings.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
LAB="python -m rineya_lab.cli"

# ~10% of the primary token budget, and a 200-problem DEV subset.
REHEARSAL_TARGET_TOKENS=${REHEARSAL_TARGET_TOKENS:-340000}
DEV_LIMIT=${DEV_LIMIT:-200}

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

banner "gates"
$LAB validate-data --splits dev -o runs/rehearsal_validation.json
$LAB plan -c configs/gpu_train_split_stateful.yaml -o runs/rehearsal_plan.json
python - <<'PY'
import json
assert json.load(open("runs/rehearsal_validation.json"))["status"] == "PASS"
assert json.load(open("runs/rehearsal_plan.json"))["order_diagnostics"]["epoch_0"]["adjacent_same_problem"] == 0
print("gates OK")
PY

banner "baseline: untouched checkpoint"
$LAB evaluate -c configs/gpu_eval_base.yaml \
    --set evaluation.limit="$DEV_LIMIT" \
    --output-dir runs/rehearsal_base

banner "train FULL and SPLIT at 10% budget"
for ARM in full split_stateful; do
  case "$ARM" in
    full)           OUT=runs/rehearsal_full ;;
    split_stateful) OUT=runs/rehearsal_split ;;
  esac
  $LAB train -c "configs/gpu_train_${ARM}.yaml" \
      --set budget.mode=matched_target_tokens \
      --set budget.max_target_tokens="$REHEARSAL_TARGET_TOKENS" \
      --set training.eval_steps=25 \
      --set experiment.seed=1 \
      --set experiment.output_dir="$OUT"
done

banner "evaluate both arms, all protocols"
$LAB evaluate -c configs/gpu_eval_full.yaml \
    --set evaluation.limit="$DEV_LIMIT" \
    --set evaluation.backend.adapter_path=runs/rehearsal_full/adapter \
    --output-dir runs/rehearsal_eval_full
$LAB evaluate -c configs/gpu_eval_split.yaml \
    --set evaluation.limit="$DEV_LIMIT" \
    --set evaluation.backend.adapter_path=runs/rehearsal_split/adapter \
    --output-dir runs/rehearsal_eval_split

banner "verdict"
python - <<'PY'
import json

def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

base = load("runs/rehearsal_base/eval_report.json")
full = load("runs/rehearsal_eval_full/eval_report.json")
split = load("runs/rehearsal_eval_split/eval_report.json")

def acc(report, protocol):
    summary = report["protocols"].get(protocol)
    return None if summary is None else summary["strict"]["accuracy"]

def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"

base_full = acc(base, "full_direct") or 0.0
full_full = acc(full, "full_direct") or 0.0
print(f"base   full_direct       : {pct(base_full)}")
print(f"FULL   full_direct       : {pct(full_full)}   (lift {100 * (full_full - base_full):+.2f} pp)")
print(f"SPLIT  oracle_transition : {pct(acc(split, 'oracle_transition'))}")
print(f"SPLIT  generated_rollout : {pct(acc(split, 'generated_rollout'))}")
gap = split.get("derived", {}).get("free_rollout_gap") or {}
print(f"SPLIT  free-rollout gap  : {pct(gap.get('gap'))}")
for name, report in (("FULL", full), ("SPLIT", split)):
    for protocol, summary in report["protocols"].items():
        groups = summary["failure_groups"]
        print(f"  {name:5s} {protocol:24s} format={pct(groups['format_rate'])} "
              f"control={pct(groups['control_rate'])} reasoning={pct(groups['reasoning_rate'])}")

print()
if 100 * full_full < 10:
    print("VERDICT: FULL barely moved at 10% budget. Before the full run, raise LoRA rank or")
    print("         epochs, or reconsider the checkpoint. A full run risks a floor-effect null.")
else:
    print("VERDICT: the arms are learning. Extrapolate wall-clock from the run manifests,")
    print("         fill the preregistration blanks, then launch scripts/run_gpu_experiment.sh.")
print("\nThis was a REHEARSAL at reduced budget on DEV. Not a result.")
PY
