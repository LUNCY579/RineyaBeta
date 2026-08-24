# PDRR Runbook

All commands are run from the repository root.

## 1. Checkout the remote branch

First checkout:

```bash
git fetch origin
git switch --create research/pre-deletion-recovery-routing-v0.3 \
  --track origin/research/pre-deletion-recovery-routing-v0.3
```

When the local branch already exists:

```bash
git switch research/pre-deletion-recovery-routing-v0.3
git pull --ff-only
```

Verify the branch and head before running anything:

```bash
git branch --show-current
git log -1 --oneline
git status --short
```

## 2. CPU contract suite

Linux/macOS:

```bash
python -m pip install -r experiments/predeletion_recovery/requirements-cpu.txt
export PYTHONPATH="$PWD/experiments/predeletion_recovery/src"
python -m compileall -q experiments/predeletion_recovery
python -m pytest -q experiments/predeletion_recovery/tests
```

PowerShell:

```powershell
python -m pip install -r experiments/predeletion_recovery/requirements-cpu.txt
$env:PYTHONPATH = "$PWD\experiments\predeletion_recovery\src"
python -m compileall -q experiments/predeletion_recovery
python -m pytest -q experiments/predeletion_recovery/tests
```

## 3. Controlled CPU reference

Quick smoke run:

```bash
python experiments/predeletion_recovery/scripts/run_cpu_reference.py \
  --quick \
  --out experiments/predeletion_recovery/results/local_cpu_quick
```

Full committed protocol:

```bash
python experiments/predeletion_recovery/scripts/run_cpu_reference.py \
  --config experiments/predeletion_recovery/configs/cpu_reference.yaml \
  --out experiments/predeletion_recovery/results/local_cpu_reference
```

The output contains per-arm/per-seed records, aggregate metrics, Recovery Ratios,
handoff metrics, route-alignment diagnostics, and a machine-readable validation
report. This is a controlled mechanism study, not evidence about a trained Rineya
checkpoint.

## 4. Required local assets for the real Rineya study

The repository intentionally ignores large or upstream-owned artifacts. Before a
real run, confirm that these exist locally:

```text
artifacts/rineya-core-en32k-28L/model*.safetensors
artifacts/eval_sets/chat_dev.jsonl
artifacts/eval_sets/rag_dev.jsonl
artifacts/heal_mixes/B0/heal_mix_b0.jsonl
```

The branch does not download model weights or regenerate sealed evaluation data.
Keep `rag_test_SEALED` untouched until a model rung and analysis plan are frozen.

Install the real-model dependencies:

```bash
python -m pip install -r experiments/predeletion_recovery/requirements-hf.txt
export PYTHONPATH="$PWD/experiments/predeletion_recovery/src"
```

PowerShell:

```powershell
python -m pip install -r experiments/predeletion_recovery/requirements-hf.txt
$env:PYTHONPATH = "$PWD\experiments\predeletion_recovery\src"
```

## 5. Causal layer calibration

The calibration command reads the model and training data paths from the pilot
configuration. It measures exact identity-ablation damage for every eligible
layer and writes a single JSON result containing the low-, medium-, and
high-significance targets.

```bash
python experiments/predeletion_recovery/scripts/calibrate_layers.py \
  --config experiments/predeletion_recovery/configs/rineya_pilot.yaml \
  --out artifacts/pdrr/calibration.json \
  --limit 256 \
  --batch-size 4
```

For a CPU-only calibration smoke test, add:

```text
--device cpu --limit 8 --batch-size 1 --candidates 12,13,14
```

The confirmatory target rule is frozen at the 20th, 50th, and 80th percentile
ranks by mean intact-versus-ablated final-logit KL, excluding the first and final
blocks. Do not change targets after viewing recovery outcomes.

## 6. Single-arm preflight

A dry run verifies paths, arm resolution, target selection, recipient selection,
parameter accounting, data loading, and the deterministic plan without updating
weights:

```bash
python experiments/predeletion_recovery/scripts/train_hf.py \
  --config experiments/predeletion_recovery/configs/rineya_pilot.yaml \
  --arm causal_state_position_recovery \
  --target-layer 14 \
  --seed 20260823 \
  --out artifacts/pdrr/preflight \
  --max-steps 2 \
  --dry-run
```

Then remove `--dry-run` for a two-step tensor-path check. The real preflight must
exercise:

- the intact frozen-teacher path;
- the exact target-ablation teacher path;
- the scheduled student path;
- the permanently compact shadow path;
- source and recipient residual capture on the same inputs;
- router normalization and finite component-recovery loss;
- checkpoint save/resume;
- compact-only consolidation;
- physical deletion and scar-free equivalence.

The trainer exports either `export_full/` for the intact anchor or
`export_compact/` for every pruning arm. Physical surgery is part of training
completion; no separate export script is required.

## 7. Ordered pilot

Run one medium-significance target and one seed through this minimal mechanism
ladder before launching the complete matrix:

```text
full
hard_prune
hard_prune_kd
gradual_kd
matrix_no_recovery
random_matrix_recovery
causal_state_position_recovery
```

A convenient dry-run matrix can be generated by making a pilot copy of the
confirmatory YAML with those seven arms and one seed, then running `orchestrate.py`
with `--dry-run`. Advance only if every cell passes scar-free export checks and
the full method is not clearly dominated.

## 8. Confirmatory matrix

```bash
python experiments/predeletion_recovery/scripts/orchestrate.py \
  --config experiments/predeletion_recovery/configs/rineya_confirmatory.yaml \
  --calibration artifacts/pdrr/calibration.json \
  --out artifacts/pdrr/confirmatory
```

Useful controls:

```text
--dry-run          write all cell manifests without training
--cpu              force CPU for a very small diagnostic run
--eval-limit 20    restrict each DEV benchmark during a smoke test
--skip-eval        train/export only; evaluate later
--targets 8,14,21  explicitly supply already-frozen target indices
```

The orchestrator runs cells sequentially by default, creates isolated directories,
records exact commands and hashes, resumes trainer checkpoints, and writes:

```text
artifacts/pdrr/confirmatory/experiment_index.json
```

## 9. Benchmark one exported checkpoint

`evaluate_export.py` delegates generation and scoring to the repository's existing
versioned `scripts/eval/run_gates.py`, preserving its chat template, greedy
protocol, response control, and quality scorer.

```bash
python experiments/predeletion_recovery/scripts/evaluate_export.py \
  --model artifacts/pdrr/confirmatory/target_14/seed_20260823/causal_state_position_recovery/export_compact \
  --run-manifest artifacts/pdrr/confirmatory/target_14/seed_20260823/causal_state_position_recovery/run_manifest.json \
  --datasets chat_dev,rag_dev \
  --out artifacts/pdrr/confirmatory/target_14/seed_20260823/causal_state_position_recovery/evaluation
```

For a CPU smoke check, add `--cpu --limit 10`. Each evaluation directory contains
standard gate outputs plus a normalized `benchmark_rows.jsonl` used by the paired
confirmatory analysis.

## 10. Mechanistic handoff probe

After both a recovered compact checkpoint and the matched hard-pruned compact
checkpoint exist:

```bash
python experiments/predeletion_recovery/scripts/mechanistic_probe.py \
  --recovered-model artifacts/pdrr/confirmatory/target_14/seed_20260823/causal_state_position_recovery/export_compact \
  --hard-model artifacts/pdrr/confirmatory/target_14/seed_20260823/hard_prune/export_compact \
  --run-manifest artifacts/pdrr/confirmatory/target_14/seed_20260823/causal_state_position_recovery/run_manifest.json \
  --train-log artifacts/pdrr/confirmatory/target_14/seed_20260823/causal_state_position_recovery/train_log.jsonl \
  --data artifacts/eval_sets/chat_dev.jsonl \
  --out artifacts/pdrr/mechanistic/target_14_seed_20260823.json \
  --limit 128
```

This measures whether training-time route responsibility predicts the increase in
post-recovery causal importance of the designated surviving layers.

## 11. Confirmatory analysis

```bash
python experiments/predeletion_recovery/scripts/analyze_benchmarks.py \
  --index artifacts/pdrr/confirmatory/experiment_index.json \
  --out artifacts/pdrr/analysis
```

To include one or more mechanistic probe files, pass a comma-separated list:

```text
--mechanistic artifacts/pdrr/mechanistic/a.json,artifacts/pdrr/mechanistic/b.json
```

The analyzer requires exact target/seed/example pairing, computes Recovery Ratio
against matched full and hard-pruned anchors, applies clustered paired inference
and Holm correction, checks practical-effect and category-regression gates, and
emits an explicit supported/unsupported claim ladder.

## 12. Reading results

The committed controlled reference study is located at:

```text
experiments/predeletion_recovery/results/cpu_reference/report.md
experiments/predeletion_recovery/results/cpu_reference/summary.json
experiments/predeletion_recovery/results/cpu_reference/validation.json
```

Real Rineya outputs are intentionally written under `artifacts/pdrr/` and are not
committed automatically. Commit only compact summaries, manifests, and evidence
needed to reproduce claims; do not commit model weights, private data, sealed
per-example outputs, or temporary checkpoints.
