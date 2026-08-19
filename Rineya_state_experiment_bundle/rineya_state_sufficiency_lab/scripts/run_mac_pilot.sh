#!/usr/bin/env bash
# Scaled pilot for a CPU-only, memory-constrained host.
#
#   scripts/run_mac_pilot.sh            # run / resume the whole pilot
#   TRAIN_PROBLEMS=2000 scripts/run_mac_pilot.sh
#   STAGE=calibrate scripts/run_mac_pilot.sh   # just measure throughput and stop
#
# RESUMABLE BY DESIGN. A multi-day run on a swapping host will be interrupted.
# Every stage checks for its own completed output and skips it, and interrupted
# training resumes from its latest checkpoint with the example order unchanged.
# Re-running this script after a crash, a reboot, or a Ctrl-C is always safe.
#
# What this pilot can and cannot answer is stated at the end of the run, and in
# docs/READINESS_ASSESSMENT.md. Read that before reporting anything from it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

LAB="python -m rineya_lab.cli"
TRAIN_PROBLEMS=${TRAIN_PROBLEMS:-1000}
SEED=${SEED:-1}
STAGE=${STAGE:-all}
mkdir -p runs

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
done_marker() { [ -f "$1/.stage_complete" ]; }
mark_done()  { touch "$1/.stage_complete"; }

# Train, resuming from the newest checkpoint if one exists.
train_arm() {
  local config="$1" out="$2" tokens="$3"
  if done_marker "$out"; then banner "SKIP $out (already complete)"; return 0; fi
  local resume=""
  if [ -d "$out/checkpoints" ]; then
    local latest
    latest=$(ls -d "$out"/checkpoints/*/ 2>/dev/null | sort | tail -1)
    [ -n "$latest" ] && { resume="--set training.resume_from=${latest%/}"; banner "RESUMING $out from ${latest%/}"; }
  fi
  banner "TRAIN $out  (problems=$TRAIN_PROBLEMS, matched target tokens=$tokens)"
  # shellcheck disable=SC2086
  $LAB train -c "$config" \
      --set data_view.max_unique_problems="$TRAIN_PROBLEMS" \
      --set budget.mode=matched_target_tokens \
      --set budget.max_target_tokens="$tokens" \
      --set experiment.seed="$SEED" \
      --set experiment.output_dir="$out" $resume || return 1
  mark_done "$out"
}

evaluate_arm() {
  local config="$1" out="$2" adapter="$3"
  if done_marker "$out"; then banner "SKIP $out (already complete)"; return 0; fi
  banner "EVAL $out"
  local extra=""
  [ -n "$adapter" ] && extra="--set evaluation.backend.adapter_path=$adapter"
  # shellcheck disable=SC2086
  $LAB evaluate -c "$config" --set experiment.output_dir="$out" $extra || return 1
  mark_done "$out"
}

# ---------------------------------------------------------------------------
banner "STAGE 0  calibrate: measure real throughput on this host"
if [ ! -f runs/pilot_calibration.json ]; then
  $LAB train -c configs/pilot_train_full.yaml \
      --set data_view.max_unique_problems=16 \
      --set experiment.output_dir=runs/pilot_calibrate \
      --set training.save_steps=0 --set training.eval_steps=0 \
      --set training.checkpoint_selection=last
  python - <<'PY'
import json
m = json.load(open("runs/pilot_calibrate/run_manifest.json"))
t = m["training"]
rows = t["exposure"]["records_seen"]
rate = t["records_per_second"] or 1e-9
calib = {"rows": rows, "wall_clock_seconds": t["wall_clock_seconds"], "rows_per_second": rate,
         "target_tokens_per_second": t["target_tokens_per_second"]}
json.dump(calib, open("runs/pilot_calibration.json", "w"), indent=2)
print(json.dumps(calib, indent=2))
for problems in (500, 1000, 2000, 4000):
    full_rows = problems
    split_rows = problems * 1.49          # matched target tokens -> ~1.49x rows
    hours = (full_rows + 2 * split_rows) / rate / 3600
    print(f"  {problems:>5} train problems -> ~{hours:6.1f} h for all three arms (A + B + C), one seed")
PY
fi
cat runs/pilot_calibration.json
[ "$STAGE" = "calibrate" ] && { echo; echo "Calibration only. Re-run with TRAIN_PROBLEMS=<n> to continue."; exit 0; }

# Matched target-token budget, scaled to the chosen training-problem count.
# 79.3 target tokens per FULL record, measured with this tokenizer.
MATCHED_TOKENS=$(python -c "print(int($TRAIN_PROBLEMS * 79.3))")

# ---------------------------------------------------------------------------
banner "STAGE 1  gates"
if [ ! -f runs/pilot_validation.json ]; then
  $LAB validate-data --splits dev -o runs/pilot_validation.json || exit 1
fi
$LAB plan -c configs/pilot_train_split.yaml -o runs/pilot_plan.json >/dev/null
python - <<'PY'
import json, sys
if json.load(open("runs/pilot_validation.json"))["status"] != "PASS":
    sys.exit("dataset validation FAILED")
# Ceiling, not equality: a uniform shuffle produces sum_p k_p(k_p-1)/n adjacent
# same-problem pairs, never reliably zero. See order_diagnostics() in
# rineya_lab/data/views.py.
shuffled = json.load(open("runs/pilot_plan.json"))["order_diagnostics"]["epoch_0"]
ref = shuffled["uniform_shuffle_reference"]
if shuffled["adjacent_same_problem"] > ref["adjacent_same_problem_upper"]:
    sys.exit(
        f"condition C is not globally shuffled: {shuffled['adjacent_same_problem']} adjacent "
        f"same-problem pairs vs ceiling {ref['adjacent_same_problem_upper']:.1f}"
    )
if shuffled["adjacent_consecutive_steps"] > ref["adjacent_consecutive_steps_upper"]:
    sys.exit(
        f"chronology survived the shuffle: {shuffled['adjacent_consecutive_steps']} adjacent "
        f"consecutive pairs vs ceiling {ref['adjacent_consecutive_steps_upper']:.1f}"
    )
print("gates OK")
PY
[ $? -ne 0 ] && exit 1

banner "STAGE 2  baseline (untouched checkpoint, full DEV)"
evaluate_arm configs/pilot_eval_base.yaml runs/pilot_eval_base "" || exit 1

banner "STAGE 3  train condition A (FULL) and evaluate it"
# A is trained and scored FIRST so the learnability gate can stop the run before
# conditions B and C are trained. If FULL cannot learn the task, B and C are
# wasted compute and an A-vs-C null would be uninterpretable anyway.
train_arm configs/pilot_train_full.yaml   runs/pilot_full      "$MATCHED_TOKENS" || exit 1
evaluate_arm configs/pilot_eval_full.yaml runs/pilot_eval_full runs/pilot_full/adapter || exit 1

banner "GATE G9  learnability / headroom  (go / no-go)"
LEARNABILITY_FLOOR_PCT=${LEARNABILITY_FLOOR_PCT:-25} \
LEARNABILITY_MIN_LIFT_PP=${LEARNABILITY_MIN_LIFT_PP:-15} \
python scripts/check_learnability.py runs/pilot_eval_full/eval_report.json runs/pilot_eval_base/eval_report.json
G9=$?
if [ $G9 -ne 0 ] && [ "${IGNORE_G9:-0}" != "1" ]; then
  cat <<'MSG'

STOPPING before conditions B and C.
FULL did not clear the declared learnability floor, so an A-vs-C comparison could
not be interpreted: "sequence-free training is just as good" and "neither
representation learned anything" would produce the same number.

Options:
  * raise TRAIN_PROBLEMS and re-run (this script resumes; A retrains only if you
    delete runs/pilot_full/.stage_complete)
  * raise LoRA capacity:  --set lora.r=32 --set lora.alpha=64
  * train longer:         --set training.epochs=2
  * override deliberately: IGNORE_G9=1 scripts/run_mac_pilot.sh
MSG
  exit 2
fi

banner "STAGE 4  train conditions C and B, then evaluate C"
train_arm configs/pilot_train_split.yaml   runs/pilot_split   "$MATCHED_TOKENS" || exit 1
train_arm configs/pilot_train_ordered.yaml runs/pilot_ordered "$MATCHED_TOKENS" || exit 1
evaluate_arm configs/pilot_eval_split.yaml runs/pilot_eval_split runs/pilot_split/adapter || exit 1

banner "STAGE 5  paired comparison (A vs C)"
$LAB compare -c configs/compare_full_vs_split.yaml \
    --set comparison.protocol_pairs='[full_direct:generated_rollout]' \
    --baseline runs/pilot_eval_full --candidate runs/pilot_eval_split \
    --output-dir runs/pilot_comparison || exit 1

banner "PILOT COMPLETE"
cat <<MSG
Artifacts:
  runs/pilot_eval_base/eval_report.md      the zero point
  runs/pilot_eval_full/eval_report.md      condition A
  runs/pilot_eval_split/eval_report.md     condition C, incl. the free-rollout gap
  runs/pilot_comparison/comparison_report.md

Scope of this pilot -- state it wherever the numbers appear:
  * trained on ${TRAIN_PROBLEMS} of 43,200 source problems, one seed, matched target tokens
  * evaluated on the FULL DEV split, so the interval is not power-limited
  * a single seed gives NO estimate of run-to-run variance; run seeds 2 and 3
    (SEED=2, SEED=3) and combine with 'rineya-lab aggregate' before any verdict
  * TEST has not been touched
MSG
[ $G9 -ne 0 ] && echo && echo "NOTE: Gate G9 did not pass. Read its output above before interpreting anything."
exit 0
