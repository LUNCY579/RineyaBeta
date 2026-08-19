#!/usr/bin/env bash
# The full experiment, in the order the protocol requires.
#
#   scripts/run_gpu_experiment.sh 1 2 3
#
# Phases are numbered and each one stops on failure. Nothing here touches TEST:
# that is a separate, deliberate step after the preregistration is frozen
# (see PHASE 7 at the bottom, which refuses to run until margin_frozen is true).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [ "$#" -gt 0 ]; then SEEDS=("$@"); else SEEDS=(1 2 3); fi
LAB="python -m rineya_lab.cli"

# PRIMARY regime T: both arms stop at FULL's native supervised-token count.
# Measured on this checkpoint's tokenizer; see configs/budgets/matched_target_tokens.yaml.
MATCHED_TARGET_TOKENS=3425404

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

# --------------------------------------------------------------------------------
banner "PHASE 0  pre-flight gates"
mkdir -p runs
$LAB info                                                  > runs/environment.json
$LAB validate-data --splits dev test -o runs/validation.json
$LAB plan -c configs/gpu_train_split_stateful.yaml -o runs/plan_shuffled.json
$LAB plan -c configs/gpu_train_split_ordered.yaml  -o runs/plan_ordered.json
$LAB budget -c configs/gpu_train_full.yaml \
            --config2 configs/gpu_train_split_stateful.yaml \
            --sample 6000 -o runs/budget.json

python - <<'PY'
import json, sys
val = json.load(open("runs/validation.json"))
assert val["status"] == "PASS", f"G1 dataset validation failed: {val['errors'][:3]}"
shuffled = json.load(open("runs/plan_shuffled.json"))["order_diagnostics"]["epoch_0"]
ordered  = json.load(open("runs/plan_ordered.json"))["order_diagnostics"]["epoch_0"]

# G2. A uniform shuffle does NOT produce zero adjacencies: over n rows drawn
# from problems with k_p transitions each it produces sum_p k_p(k_p-1)/n
# same-problem pairs and sum_p (k_p-1)/n consecutive-step pairs. On the real
# TRAIN epoch that is 3.32 and 0.75. Asserting == 0 would demand a constrained
# shuffle -- reintroducing exactly the ordering dependence condition C removes
# -- and it fails on a correct shuffle (measured: 5 and 1). Gate against the
# uniform-shuffle ceiling instead; a shuffle that never happened overshoots it
# by four orders of magnitude.
ref = shuffled["uniform_shuffle_reference"]
assert shuffled["adjacent_same_problem"] <= ref["adjacent_same_problem_upper"], (
    f"G2 failed: condition C has {shuffled['adjacent_same_problem']} adjacent same-problem pairs, "
    f"above the uniform-shuffle ceiling {ref['adjacent_same_problem_upper']:.1f} "
    f"(expected {ref['adjacent_same_problem_expected']:.2f}) -- the rows are not globally shuffled"
)
assert shuffled["adjacent_consecutive_steps"] <= ref["adjacent_consecutive_steps_upper"], (
    f"G2 failed: condition C has {shuffled['adjacent_consecutive_steps']} chronologically adjacent pairs, "
    f"above the ceiling {ref['adjacent_consecutive_steps_upper']:.1f} "
    f"(expected {ref['adjacent_consecutive_steps_expected']:.2f}) -- chronology survived the shuffle"
)
print(
    f"G2: adjacent_same_problem={shuffled['adjacent_same_problem']} "
    f"(expected {ref['adjacent_same_problem_expected']:.2f}, ceiling {ref['adjacent_same_problem_upper']:.1f}), "
    f"adjacent_consecutive_steps={shuffled['adjacent_consecutive_steps']} "
    f"(expected {ref['adjacent_consecutive_steps_expected']:.2f})"
)

# G3. Condition B is the opposite claim: essentially every consecutive pair that
# exists must be adjacent, so require most of them rather than merely > 0.
available = ordered["uniform_shuffle_reference"]["consecutive_pairs_available"]
assert ordered["adjacent_consecutive_steps"] >= 0.99 * available, (
    f"G3 failed: condition B keeps only {ordered['adjacent_consecutive_steps']} of {available} "
    "consecutive step pairs adjacent; it is not trajectory-ordered"
)
budget = json.load(open("runs/budget.json"))
for arm in ("primary", "secondary"):
    truncated = budget[arm]["measured"]["truncated_targets"]
    assert truncated == 0, f"G5 failed: {arm} truncates {truncated} targets; raise training.max_seq_len"
print("Pre-flight gates G1, G2, G3, G5 PASSED")
PY

banner "PHASE 1  untouched base-model baseline (the zero point)"
$LAB evaluate -c configs/gpu_eval_base.yaml --output-dir runs/eval_base_dev

# --------------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  banner "PHASE 2  seed ${SEED} -- train A / B / C under the PRIMARY regime (matched target tokens)"

  for ARM in full split_stateful split_ordered; do
    case "$ARM" in
      full)          OUT="runs/T_full_s${SEED}" ;;
      split_stateful) OUT="runs/T_split_s${SEED}" ;;
      split_ordered)  OUT="runs/T_ordered_s${SEED}" ;;
    esac
    $LAB train -c "configs/gpu_train_${ARM}.yaml" \
        --set budget.mode=matched_target_tokens \
        --set budget.max_target_tokens="${MATCHED_TARGET_TOKENS}" \
        --set experiment.seed="$SEED" \
        --set experiment.output_dir="$OUT"
  done

  banner "PHASE 3  seed ${SEED} -- evaluate on DEV"
  $LAB evaluate -c configs/gpu_eval_full.yaml \
      --set evaluation.backend.adapter_path="runs/T_full_s${SEED}/adapter" \
      --output-dir "runs/T_eval_full_s${SEED}"

  $LAB evaluate -c configs/gpu_eval_split.yaml \
      --set evaluation.backend.adapter_path="runs/T_split_s${SEED}/adapter" \
      --output-dir "runs/T_eval_split_s${SEED}"

  $LAB evaluate -c configs/gpu_eval_split.yaml \
      --set evaluation.backend.adapter_path="runs/T_ordered_s${SEED}/adapter" \
      --set evaluation.protocols='[oracle_transition, generated_rollout]' \
      --output-dir "runs/T_eval_ordered_s${SEED}"

  if [ "$SEED" = "${SEEDS[0]}" ]; then
    banner "GATE G9  learnability / headroom"
    LEARNABILITY_FLOOR_PCT=${LEARNABILITY_FLOOR_PCT:-25} \
    LEARNABILITY_MIN_LIFT_PP=${LEARNABILITY_MIN_LIFT_PP:-15} \
    python scripts/check_learnability.py \
        "runs/T_eval_full_s${SEED}/eval_report.json" \
        "runs/eval_base_dev/eval_report.json"
  fi

  banner "PHASE 4  seed ${SEED} -- paired comparisons"
  # A vs C : the practical hypothesis
  $LAB compare -c configs/compare_full_vs_split.yaml \
      --baseline "runs/T_eval_full_s${SEED}" --candidate "runs/T_eval_split_s${SEED}" \
      --output-dir "runs/T_cmp_A_vs_C_s${SEED}"
  # B vs C : does chronological presentation matter at all
  $LAB compare -c configs/compare_full_vs_split.yaml \
      --set comparison.protocol_pairs='[generated_rollout:generated_rollout, oracle_transition:oracle_transition]' \
      --baseline "runs/T_eval_ordered_s${SEED}" --candidate "runs/T_eval_split_s${SEED}" \
      --output-dir "runs/T_cmp_B_vs_C_s${SEED}"
done

# --------------------------------------------------------------------------------
banner "PHASE 5  confirmatory regime P (matched source problems) and context regime N"
for SEED in "${SEEDS[@]}"; do
  $LAB train -c configs/gpu_train_full.yaml \
      --set experiment.seed="$SEED" --set experiment.output_dir="runs/P_full_s${SEED}"
  $LAB train -c configs/gpu_train_split_stateful.yaml \
      --set data_view.one_transition_per_problem=true \
      --set budget.mode=matched_problems --set budget.max_unique_problems=43200 \
      --set experiment.seed="$SEED" --set experiment.output_dir="runs/P_split_s${SEED}"

  $LAB evaluate -c configs/gpu_eval_full.yaml \
      --set evaluation.backend.adapter_path="runs/P_full_s${SEED}/adapter" \
      --output-dir "runs/P_eval_full_s${SEED}"
  $LAB evaluate -c configs/gpu_eval_split.yaml \
      --set evaluation.backend.adapter_path="runs/P_split_s${SEED}/adapter" \
      --set evaluation.protocols='[oracle_transition, generated_rollout, oracle_trajectory]' \
      --output-dir "runs/P_eval_split_s${SEED}"
  $LAB compare -c configs/compare_full_vs_split.yaml \
      --baseline "runs/P_eval_full_s${SEED}" --candidate "runs/P_eval_split_s${SEED}" \
      --output-dir "runs/P_cmp_A_vs_C_s${SEED}"
done

# --------------------------------------------------------------------------------
banner "PHASE 6  cross-seed aggregation"
$LAB aggregate runs/T_cmp_A_vs_C_s*/comparison_report.json --output-dir runs/agg_T_A_vs_C
$LAB aggregate runs/T_cmp_B_vs_C_s*/comparison_report.json --output-dir runs/agg_T_B_vs_C
$LAB aggregate runs/P_cmp_A_vs_C_s*/comparison_report.json --output-dir runs/agg_P_A_vs_C

banner "DEV PHASE COMPLETE"
cat <<'MSG'
Read, in this order:
  runs/eval_base_dev/eval_report.md          the zero point
  runs/agg_T_A_vs_C/multiseed_report.md      PRIMARY  : full-sequence vs sequence-free
  runs/agg_T_B_vs_C/multiseed_report.md      ORDER    : does chronology matter
  runs/agg_P_A_vs_C/multiseed_report.md      CONFIRM  : matched source problems
  runs/T_eval_split_s*/eval_report.md        state sufficiency + free-rollout gap

TEST is NOT open. Before PHASE 7:
  1. Fill docs/PREREGISTRATION.md section 6 with the real margin and its justification.
  2. Set comparison.margin_frozen: true in configs/compare_full_vs_split.yaml.
  3. Re-read section 5 of docs/READINESS_ASSESSMENT.md and decide the claim wording.
MSG
