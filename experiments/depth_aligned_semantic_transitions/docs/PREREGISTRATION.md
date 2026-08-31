# DAST preregistration — draft, TEST CLOSED

**Status:** DRAFT. This document is not frozen and its placeholder values cannot
open the sealed TEST split.

## 1. Primary question

Does ordered alignment of cumulative compact reasoning states to increasing
Transformer depth improve held-out reasoning relative to matched shuffled and
reversed depth assignments?

## 2. Base checkpoint

- model: `Qwen/Qwen3-0.6B` or audited Rineya/R2 equivalent: **TBD before freeze**
- exact local path/revision/hash: **TBD**
- architecture: expected 28 decoder layers; runtime-resolved checkpoints must be
  7, 14, 21, 28 or the run is invalid for the initial protocol
- tokenizer/chat template revision: **TBD**

## 3. Data

- source: `Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1`
- immutable dataset SHA-256: **TBD from validation command**
- train split size used: **TBD**
- DEV split size used: **TBD**
- TEST: sealed
- OOD generator/category strata: **TBD**
- deeper-than-training trace stratum: **TBD**

## 4. Arms

Required Phase 1:

1. `cot_baseline`
2. `state_ordered`
3. `state_shuffled`
4. `state_reverse`
5. `state_final_repeated`
6. `literal_ordered`

Required seeds: proposed `17, 29, 43`; **TBD/freeze**.

No arm-specific hyperparameter tuning after comparative runs begin. The only
per-arm differences are those specified by the arm contract.

## 5. Training budget

- LoRA rank/alpha/targets: 16/32, Q/K/V/O + gate/up/down; **confirm/freeze**
- epochs or matched optimizer updates: **TBD**
- source-problem exposure: matched
- prompt/assistant tokens: matched
- gradient accumulation: **TBD**
- learning rate and warmup: **TBD**
- state/contrastive weights: **TBD after symmetric calibration**
- hardware: **TBD**

## 6. Primary outcome

**TBD before freeze.** Candidate recommendation:

- primary: strict generated-answer accuracy on paired DEV problems;
- decisive treatment contrast: ordered minus the better of shuffled and reversed;
- practical minimum effect: **TBD** percentage points;
- uncertainty: paired problem-cluster bootstrap, 95% interval;
- multiplicity: ordered-vs-shuffled and ordered-vs-reverse are jointly required,
  not selectively reported.

## 7. Secondary outcomes

- final-state cosine;
- canonical order margin;
- checkpoint top-1 alignment;
- deeper-trace accuracy;
- held-out generator/category accuracy;
- teacher-forced LM loss/perplexity;
- formatting/refusal/nontermination rates;
- peak VRAM and elapsed time.

Secondary outcomes cannot rescue a failed primary gate.

## 8. Phase gates

### G0 — Engineering

All tests/smoke pass; full dataset validates; exact assignment inspection passes;
real tokenizer prefix and prompt-mask seam test passes.

### G1 — Learnability

Ordered state loss improves from initialization and canonical order margin is
positive on DEV without unacceptable LM degradation. Threshold: **TBD**.

### G2 — Depth-order causality

Ordered passes both shuffled and reverse contrasts on the primary metric, with the
frozen practical margin and uncertainty rule. Threshold: **TBD**.

### G3 — Progressive-state specificity

Ordered beats final-state repeated and is not explained by generic deep
supervision. Threshold: **TBD**.

### G4 — Literal-vs-semantic

Report whether compact states outperform literal reasoning steps; this comparison
is interpretive and does not replace G2.

### G5 — Operators

Only after G2/G3. Operator ordered must beat operator shuffled on held-out
transitions and transfer strata. Threshold: **TBD**.

### G6 — Causality

Only after G2/G3. Semantic counterfactual interventions must change answers in the
predicted direction more often than wrong-stage and norm-matched random controls.
Threshold: **TBD**.

## 9. Exclusion and failure handling

- No generated response is dropped for formatting failure.
- OOM, NaN, interrupted, and zero-output runs are retained and reported.
- A seed may be rerun only for a documented infrastructure failure, never because
  its result is unfavorable.
- Leakage/budget mismatches invalidate the affected comparison and require
  withdrawal, not reinterpretation.

## 10. Stop rules

- Stop Phase 1 and close the tested hypothesis if ordered does not pass G2/G3.
- Stop operators if router collapse or no transfer occurs under matched controls.
- Stop feedback if interventions are non-directional or degrade preservation beyond
  the frozen boundary.
- Do not open TEST to resolve an ambiguous DEV result.

## 11. Freeze block

Fill only when all placeholders are resolved:

```text
status: FROZEN | DRAFT
commit: TBD
sha256_of_this_file: TBD
frozen_at_utc: TBD
approved_by: TBD
```
