# Rineya State Sufficiency Lab

A controlled research harness for one question:

> Can a model acquire reasoning behaviour from **independent, unordered
> sufficient-state transitions**, instead of learning every solution as one
> uninterrupted textual trajectory?

Two adapters are trained from the **same immutable base checkpoint**:

```
FULL   :  problem  ->  complete reasoning trajectory and answer
SPLIT  :  (problem, compact current state)  ->  (next step, updated state, done, answer)
```

The SPLIT training records are **globally shuffled**. The optimizer never relies
on seeing transition *t−1* immediately before transition *t*. Step 7 is not
presented after step 6; the model receives the state required for step 7 directly.

This is a capability-learning experiment. It is not an efficiency experiment, not
a brevity experiment, and it does not restrict response length at evaluation time.

## The hypothesis

Let a complete solution history at step *t* be `H_t`, and let `s_t = g(q, H_t)` be
a compact state derived from that complete trajectory. The claim under test is
approximate Markov sufficiency in reasoning *supervision*:

```
P(r_{t+1} | q, H_t)  ≈  P(r_{t+1} | q, s_t)
```

The sequence still exists in the structure of the problem. What is being
questioned is whether it must be replayed inside every training target, retained
as complete preceding context, and presented to the optimizer in chronological
order.

## The three conditions

| | Condition | Training representation | Order | Config |
|---|---|---|---|---|
| **A** | FULL | problem → complete rationale + answer | shuffled problems | `configs/train_full.yaml` |
| **B** | Ordered transitions | (problem, state) → transition | steps preserved within a problem | `configs/train_split_ordered.yaml` |
| **C** | Unordered transitions | *the same records as B* | globally shuffled | `configs/train_split_stateful.yaml` |

- **A vs B** — does decomposed state-transition supervision differ from
  full-rationale supervision?
- **B vs C** — does chronological presentation itself matter?
- **A vs C** — the practical hypothesis: can full-sequence training be replaced
  by sequence-free compact-state training at comparable capability?

Condition B is not optional. Without it, an A-vs-C difference could be caused by
removing the rationale, changing the target to local transitions, compacting the
context, *or* destroying order — four confounded explanations.

## Install

```bash
pip install -e ".[all]"
export PYTHONHASHSEED=0
export RINEYA_DATASET_ROOT=/abs/path/to/rineya_sequence_independence_dataset_v1
```

`pip install -e .` alone gives the data, evaluation and analysis stack with no
torch dependency. Add `".[train]"` for GPU training, `".[quant]"` for QLoRA.

## Sixty-second check (no GPU, no network, no checkpoint)

```bash
rineya-lab smoke --all
```

Validates the dataset, builds all three view plans, tokenizes and accounts both
arms, runs every evaluation protocol against a scripted reference backend,
confirms the harness detects seven distinct failure modes, runs the paired
comparison, and — if torch is present — trains two real LoRA adapters on a
randomly initialised tiny model.

## Running the experiment

```bash
rineya-lab validate-data --splits dev test         # 1. verify the data contract
rineya-lab plan   -c configs/train_split_stateful.yaml   # 2. verify the order really is shuffled
rineya-lab budget -c configs/train_full.yaml \
                  --config2 configs/train_split_stateful.yaml   # 3. compute matched budgets
# 4. freeze docs/PREREGISTRATION.md, then:
rineya-lab train    -c configs/train_full.yaml           --set experiment.seed=1
rineya-lab train    -c configs/train_split_stateful.yaml --set experiment.seed=1
rineya-lab evaluate -c configs/eval_full.yaml
rineya-lab evaluate -c configs/eval_split_stateful.yaml
rineya-lab compare  -c configs/compare_full_vs_split.yaml \
                    --baseline runs/eval_full --candidate runs/eval_split_stateful
rineya-lab aggregate runs/cmp_s*/comparison_report.json   # combine seeds
```

For the checkpoint in `../model`, the whole design is scripted:

```bash
scripts/run_gpu_experiment.sh 1 2 3    # pre-flight gates, base, A/B/C, DEV, compare, aggregate
scripts/run_gpu_test_phase.sh  1 2 3   # refuses to run until the margin is frozen
```

Or the whole pilot: `scripts/run_pilot.sh 1 2 3` (`scripts\run_pilot.ps1 -Seeds 1,2,3`).

## Training controls

Because SPLIT expands one problem into ~3.96 rows, a naive comparison confounds
*representation* with *quantity*. Available controls:

- native epoch exposure, matched optimizer updates, matched target-token budgets,
  matched unique source-problem exposure, one-transition-per-problem sampling;
- globally shuffled / trajectory-ordered / reverse / curriculum orderings;
- per-problem loss weighting (`inverse_transitions`) so each source problem
  carries equal gradient mass regardless of how many transitions it produced;
- assistant-only loss masking, target-preserving truncation, gradient
  checkpointing, automatic BF16/FP16 selection, QLoRA, deterministic seeding,
  exact checkpoint resume, and full exposure/token accounting.

## Evaluation architecture

Four questions the harness refuses to conflate:

| Question | Protocol |
|---|---|
| 1. local transition competence | `oracle_transition` |
| 2. state sufficiency | `oracle_transition` vs `full_history_transition`, floored by `no_state_transition` / `last_state_only` |
| 3. error propagation | `oracle_trajectory` (teacher forced) vs `generated_rollout` (autonomous) |
| 4. end-to-end capability | `full_direct` for FULL, `generated_rollout` for SPLIT |

Plus `shuffled_state` and `distractor_state` perturbations, and detection of
loops, stagnation, premature completion and nontermination, with first-error-depth
analysis and the free-rollout gap.

This is what separates *"the compact state is sufficient when correct states are
supplied"* from the much stronger *"the model can autonomously construct and
maintain sufficient states throughout a complete solution."*

## Handling arbitrary responses

The evaluator never assumes the model obeys the requested format. Every raw
response is retained and routed through: exact JSON, fenced JSON, JSON embedded
in prose, conservative malformed-JSON repair, explicit final-answer tags, boxed
LaTeX, labelled answers, and a final-line fallback. Empty output, refusals,
conflicting answers, missing fields, invalid completion flags, backend failures
and nonterminating rollouts each get their own label.

Two tracks: **strict** (primary — only unambiguous, contract-compliant answers
receive credit) and **lenient** (diagnostic — recoverable knowledge despite
formatting noise). Responses that cannot be interpreted safely **remain in the
denominator and score as failures**. They are never silently discarded.

## Scoring

Deterministic typed scorers — integers, floats, scientific notation, fractions,
percentages, booleans, exact and normalised text, multiple choice, ordered and
unordered collections, JSON, dates and times, symbolic expressions, quantities
with units, and configurable regex tasks. Tolerances, unit synonyms and the
answer-type → scorer map are all configurable, and every record stores which
scorer and extraction route decided it.

A general-purpose LLM judge is deliberately not the primary correctness measure:
these tasks have verifiable canonical answers. A judge may be added later as a
supplementary qualitative diagnostic without replacing deterministic scoring.

## Statistics

Results are paired by **original problem id**, never by expanded transition row.
Reported: paired accuracy differences, paired cluster-bootstrap intervals, exact
McNemar, sign-flip randomization, superiority / non-inferiority / equivalence
decisions against a configurable practical-equivalence margin, category /
trajectory-depth / failure-type stratification with Holm-Bonferroni correction,
strict-versus-lenient disagreement analysis, and deterministic manual-audit
samples.

The shipped **2.0 pp non-inferiority margin is a placeholder**. Every comparison
report warns until you freeze and justify the real margin in
`docs/PREREGISTRATION.md` and set `comparison.margin_frozen: true`.

## Layout

```
configs/     base + condition + ablation + budget + eval + compare + launcher configs
docs/        preregistration, metric card, failure taxonomy, runbook, evaluation
             standard, parsing contract, dataset assumptions, extending,
             reproducibility, Windows notes
schemas/     machine-readable schemas for run manifests, records and reports
scripts/     Unix (.sh) and Windows (.ps1) launchers
src/rineya_lab/
  config.py        one validated config tree; unknown keys are a hard error
  data/            loading, views/ordering, budget accounting, strict validation
  training/        chat templates, collation, LoRA/QLoRA, the training loop
  evaluation/      prompts, parsing, scoring, backends, rollouts, protocols
  analysis/        paired statistics and the comparison report
tests/       unit + integration + artifact-schema + torch-gated training tests
```

## Status of results

The complete 48,000-problem GPU training experiment has **not** been run here.
The offline suite proves both pipelines are executable and that the evaluation
harness behaves correctly on scripted inputs. Those are engineering results, not
scientific results about the hypothesis.

Any report produced with `evaluation.backend.kind: reference` carries a warning
field, and comparisons involving one raise a validity warning.

## Documentation

| Document | Read it when |
|---|---|
| [RUNBOOK](docs/RUNBOOK.md) | you are about to run something |
| [PREREGISTRATION](docs/PREREGISTRATION.md) | before any TEST result exists |
| [EVALUATION_STANDARD](docs/EVALUATION_STANDARD.md) | you want to know what a number means |
| [METRIC_CARD](docs/METRIC_CARD.md) | you are reporting a number |
| [RESPONSE_PARSING_CONTRACT](docs/RESPONSE_PARSING_CONTRACT.md) | a response parsed unexpectedly |
| [FAILURE_TAXONOMY](docs/FAILURE_TAXONOMY.md) | you are reading a failure distribution |
| [DATASET_ASSUMPTIONS](docs/DATASET_ASSUMPTIONS.md) | you are swapping in other data |
| [REPRODUCIBILITY](docs/REPRODUCIBILITY.md) | you need to reproduce a past run |
| [EXTENDING](docs/EXTENDING.md) | you are adding a scorer, protocol or backend |
| [WINDOWS](docs/WINDOWS.md) | you are on Windows |

## Licence

Apache 2.0. See `LICENSE`.
