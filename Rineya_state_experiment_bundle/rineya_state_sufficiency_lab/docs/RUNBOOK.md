# Operational runbook

Ordered procedure for running the experiment. Every step is CPU-checkable before
GPU time is spent.

## 0. Install

```bash
pip install -e ".[all]"          # or ".[train]" for GPU work without dev tools
export PYTHONHASHSEED=0
```

Point the lab at the dataset once:

```bash
export RINEYA_DATASET_ROOT=/abs/path/to/rineya_sequence_independence_dataset_v1
```

An explicit root is strict: if it does not exist the run fails rather than
quietly loading a different dataset.

## 1. Verify the dataset

```bash
rineya-lab validate-data --splits dev test
rineya-lab validate-data                   # full TRAIN pass, several minutes
```

Must report `"status": "PASS"`. This re-derives the FULL↔SPLIT pairing rather
than trusting the released manifest, and checks the causal safeguard (no state
entering step *t* may state an answer only derived later).

## 2. Check the data path with no model

```bash
rineya-lab plan -c configs/train_split_stateful.yaml
```

Confirm in `order_diagnostics`, against `uniform_shuffle_reference` in the same
block rather than against zero:

- shuffled → `adjacent_same_problem ≤ adjacent_same_problem_upper` **and**
  `adjacent_consecutive_steps ≤ adjacent_consecutive_steps_upper`
- trajectory → `adjacent_consecutive_steps ≥ 0.99 × consecutive_pairs_available`

Zero is the wrong expectation for a shuffled epoch. A uniform permutation of
`n` rows over problems with `k_p` transitions each still leaves
`Σ k_p(k_p−1)/n` same-problem pairs and `Σ (k_p−1)/n` consecutive-step pairs
side by side — **3.32** and **0.75** on the real TRAIN epoch, where the measured
counts are 5 and 1. A shuffle that silently never happened produces 127,741
consecutive pairs, so the ceiling has four orders of magnitude of headroom.

Then tokenize without building a model:

```bash
rineya-lab train -c configs/dry_run_full.yaml
rineya-lab train -c configs/dry_run_split.yaml
```

Inspect `truncated_targets`. It should be 0. A non-zero value means
`max_seq_len` is cutting supervision, which damages FULL more than SPLIT.

## 3. Fix the budget

```bash
rineya-lab budget -c configs/train_full.yaml \
                  --config2 configs/train_split_stateful.yaml \
                  --sample 0 -o runs/budget.json
```

Copy the recommended values into `configs/budgets/*.yaml` and apply the same file
to **both** arms. Declare the primary regime in the preregistration.

## 4. Freeze the preregistration

Fill in `docs/PREREGISTRATION.md`, especially the non-inferiority margin, set
`comparison.margin_frozen: true`, and commit. Until then every comparison report
carries a not-frozen warning.

## 5. Train

```bash
rineya-lab train -c configs/train_full.yaml            --set experiment.seed=1
rineya-lab train -c configs/train_split_stateful.yaml  --set experiment.seed=1
rineya-lab train -c configs/train_split_ordered.yaml   --set experiment.seed=1   # condition B
```

Repeat for at least three seeds. Multi-GPU:

```bash
accelerate launch --config_file configs/accelerate/multi_gpu.yaml \
  -m rineya_lab.cli train -c configs/train_split_stateful.yaml
```

Resume after an interruption:

```bash
rineya-lab train -c configs/train_full.yaml \
  --set training.resume_from=runs/train_full/checkpoints/update_0000500
```

Resume restores the optimizer, the scheduler and the exact position inside the
planned epoch, so the example order is unchanged.

## 6. Evaluate on DEV

```bash
rineya-lab evaluate -c configs/eval_base_model.yaml
rineya-lab evaluate -c configs/eval_full.yaml
rineya-lab evaluate -c configs/eval_split_stateful.yaml
rineya-lab evaluate -c configs/eval_perturbations.yaml
```

Read the four questions separately:

| Question | Protocol |
|---|---|
| local transition competence | `oracle_transition` |
| state sufficiency | `oracle_transition` vs `full_history_transition`, with `no_state_transition` as the floor |
| error propagation | `oracle_trajectory` vs `generated_rollout` |
| end-to-end capability | `full_direct` (FULL) vs `generated_rollout` (SPLIT) |

Do not report "the compact state is sufficient" from `oracle_transition` alone.
That claim needs `generated_rollout` too.

## 7. Compare

```bash
rineya-lab compare -c configs/compare_full_vs_split.yaml \
  --baseline runs/eval_full --candidate runs/eval_split_stateful \
  --output-dir runs/comparison
```

Check `validity_warnings` first. Then read the CI, not the p-value alone.

## 7b. Cross-seed aggregation

A single seed is not a result. After every seed's comparison exists:

```bash
rineya-lab aggregate runs/T_cmp_A_vs_C_s*/comparison_report.json --output-dir runs/agg_T_A_vs_C
```

Read the verdict field, not just the mean:

| Verdict | Meaning |
|---|---|
| `candidate_superior` / `baseline_superior` | the seed-level interval excludes 0 |
| `non_inferior` | the interval's lower bound clears the -margin |
| `not_established` | the interval spans the margin |
| `inconclusive_at_this_power` | **between-seed SD exceeds the margin** — run-to-run variation is larger than the effect being tested. Report as inconclusive; do not report a direction |
| `insufficient_seeds` | one seed; no variance estimate exists |

## 7c. The scripted full run

```bash
scripts/run_gpu_experiment.sh 1 2 3      # pre-flight, base, A/B/C, DEV eval, compare, aggregate
scripts/run_gpu_test_phase.sh 1 2 3      # refuses to run until the margin is frozen
```

Phase 0 of the first script asserts gates G1, G2, G3 and G5 and aborts on failure,
so a misconfigured run stops before it consumes GPU time.

## 8. Open TEST once

Only after steps 4–7 are complete and the preregistration is committed:

```bash
rineya-lab evaluate -c configs/eval_full.yaml           --set evaluation.split=test
rineya-lab evaluate -c configs/eval_split_stateful.yaml --set evaluation.split=test
rineya-lab compare  -c configs/compare_full_vs_split.yaml --baseline ... --candidate ...
```

## Budget regimes

| Regime | What it holds constant | What it cannot hold constant |
|---|---|---|
| N native | each recipe's natural epoch | rows, tokens, updates |
| U matched updates | optimizer updates | tokens, problem coverage |
| T matched target tokens | supervised tokens | updates, problem coverage |
| P matched problems | unique source problems per epoch | tokens, trajectory coverage |

Report more than one. A difference that survives every regime is a finding; one
that appears in only a single regime is a fact about that regime.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `Could not locate the ... dataset` | explicit root missing | fix the path; explicit roots never fall back |
| `Dataset checksum verification failed` | corrupt or modified data | re-extract; do not train on it |
| `truncated_targets > 0` | `max_seq_len` too small | raise it, or accept and report it |
| High `format` failures in SPLIT only | contract compliance, not reasoning | report the lenient track alongside |
| `nontermination` dominates rollouts | model never emits `done` | check the adapter learned the stateful view, not next-step-only |
| McNemar reported not-applicable | fractional per-problem scores | expected for transition protocols; use the bootstrap CI |
| Comparison warns about differing problem ids | arms evaluated different sets | re-run both with the same `evaluation.limit` and seed |
