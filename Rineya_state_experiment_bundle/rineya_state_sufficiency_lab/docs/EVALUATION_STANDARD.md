# Evaluation standard

Rules the evaluation code enforces, and the reasoning behind each.

## 1. Four questions, four protocols

These must never be conflated:

1. **Local transition competence** — given a *correct* compact state, can the
   model produce the correct next transition? → `oracle_transition`
2. **State sufficiency** — does a compact state support the next decision as well
   as the complete history does? → `oracle_transition` vs
   `full_history_transition`, floored by `no_state_transition`
3. **Error propagation** — how much is lost when the model must continue from its
   *own* states? → `oracle_trajectory` vs `generated_rollout`
4. **End-to-end capability** — can the arm solve the original problem?
   → `full_direct` (FULL) and `generated_rollout` (SPLIT)

The claim "the compact state is sufficient when correct states are supplied" is
supported by (1) and (2). The much stronger claim "the model can autonomously
construct and maintain sufficient states" additionally requires (3) and (4).

## 2. Nothing is silently dropped

`scored_items == total_items` for every protocol. Empty output, refusals,
malformed structure and backend failures stay in the denominator and score as
failures. Any harness that drops them inflates accuracy by exactly the rate at
which its own contract is violated.

## 3. Strict is primary

Strict credit requires a contract-compliant extraction route, no conflicting
candidates, and acceptance by the typed scorer. Lenient is reported beside it as
a diagnostic, and the disagreement between them is reported explicitly.

## 4. Every verdict is auditable

Each record stores the prompt, the raw response, the extraction route, the parse
status, the scorer, the normalised prediction and gold, the failure type and the
per-component results. Any disputed item can be re-adjudicated from the record
alone. A deterministic audit sample — over-weighted toward track disagreements
and format failures — is included in every report.

## 5. Prompts are minimal edits of one template

The transition prompt reproduces the dataset's training prompt verbatim, so
`oracle_transition` is in-distribution for a SPLIT adapter. Every perturbation
protocol removes or replaces exactly one component of that same string, so a
change in accuracy is attributable to that component and not to rewording.

Prompts that are knowingly out-of-distribution for an arm carry
`in_distribution_for` in the record, and reports say so.

## 6. Rollouts are bounded and diagnosed

The step budget is the constant `max_steps_absolute` (16, against a dataset
whose longest trajectory is 9 steps), identical for every problem.

It is deliberately *not* derived from the trajectory's own length. Doing so --
`use_dataset_step_bound: true`, which was the shipped default -- hands the SPLIT
arm gold information the FULL arm never receives: how long the problem is, and
therefore when a `done` is plausible. `full_direct` gets one fixed-length call.
The option remains available for a declared secondary analysis and must be
reported as such. `oracle_trajectory` is unaffected: there the gold step count
is the protocol.

Four control failures are
detected and distinguished: `loop`, `stagnation`, `premature_completion`,
`nontermination`. First-error depth is recorded by comparing each emitted state
against the canonical state at that depth.

## 7. Determinism by default

Greedy decoding, seeded ordering, SHA-256-based shuffles, and a recorded
`problem_ids_sha256` for the evaluated set. Two runs of the same config on the
same data produce the same records.

## 7b. Reasoning scratch is not an answer

Checkpoints with a thinking-enabled chat template may emit `<think>…</think>`.
Those blocks are working notes: a value the model considers and then discards must
not be scored, and must not create a false "conflicting answers" verdict. Closed
blocks are removed before extraction, and a response that ends inside an
*unterminated* block is treated as having produced no answer, because it did not.
Every removal is recorded as a note on the record. Controlled by
`evaluation.strip_reasoning_blocks` (default true).

## 8. The reference backend is never a result

`evaluation.backend.kind: reference` produces scripted output for harness
testing. Its reports carry a warning field, and any comparison involving one
raises a validity warning. Never present these numbers as evidence about the
hypothesis.

## 9. Deterministic scoring over model judging

These tasks have canonical answers. A model judge would introduce an
uncontrolled, non-reproducible source of variance into the primary endpoint. A
judge may be added later as a supplementary qualitative diagnostic; it must not
replace the deterministic scorers.

## 10. Pair on problems, not rows

One problem expands into ~3.96 transitions, which are not independent
observations. Every paired statistic collapses a problem's items to one score
first. Only problems present in both runs are compared, and the asymmetric
remainder is reported rather than dropped.
