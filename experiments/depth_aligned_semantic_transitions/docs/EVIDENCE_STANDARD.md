# Evidence and claim standard

## Status vocabulary

- **BUILT — UNRUN:** code, configs, tests, and smoke exist; no qualifying real checkpoint result.
- **ENGINEERING PASS:** software and artifact contracts execute as designed.
- **EXPLORATORY DEV:** real-model DEV observations; thresholds may still be tuned.
- **CONFIRMATORY DEV:** frozen seeds, arms, budgets, metrics, and gates completed on DEV.
- **TEST OPENED:** permitted only by a frozen preregistration hash and explicit approval.
- **POSITIVE — BOUNDED:** a specific preregistered claim passes; no broader capability claim.
- **CLOSED — NEGATIVE:** the tested formulation fails its decisive controls.
- **WITHDRAWN:** evidence is invalidated by leakage, mismatch, or implementation defect.

## Evidence hierarchy

1. raw predictions and per-example mechanism records;
2. run manifest, config hash, dataset hash, checkpoint layers, and environment;
3. deterministic aggregate report generated from those records;
4. branch progress/finding document;
5. prose plan or preregistration.

A plan is never evidence that a run occurred. A smoke test is never a model result.
A directory name is never evidence that its contents are valid.

## Primary causal comparisons

The main V1 claim depends on `state_ordered` versus **both** `state_shuffled` and
`state_reverse`. The treatment and controls have identical semantic target count,
state-head architecture, LoRA capacity, prompt/response tokens, source problems,
and optimizer schedule. `cot_baseline` provides context but is not the only causal
control because it lacks active auxiliary losses.

## Required reporting

Every real run must report:

- exact base model path/revision and parameter counts;
- exact resolved checkpoint layers;
- train/eval problem IDs or their cryptographic manifest;
- token, update, and source-problem exposure;
- strict generated-answer accuracy and failure denominator;
- teacher-forced LM loss and depth diagnostics;
- peak allocated/reserved VRAM and elapsed time;
- all seeds, including failures and interrupted runs;
- whether TEST was opened;
- deviations from preregistration.

## Leakage invalidators

Withdraw a result if:

- semantic pooling includes assistant target tokens;
- the target encoder or fixed projection changes during training;
- a problem/source template crosses split boundaries contrary to the dataset contract;
- shuffled/reversed controls receive a different target multiset or update budget;
- TEST is inspected before the frozen decision gate;
- failed seeds are omitted;
- evaluation extracts or drops malformed responses asymmetrically by arm.

## Causal intervention standard

A random perturbation changing logits is only an engineering check. Scientific
causality requires a semantic counterfactual: replace an intermediate state with a
state from a matched problem whose correct conclusion is known to differ, then
show that the output changes in the predicted direction more often than random,
wrong-stage, and norm-matched noise controls.

## Operator transfer standard

Operator evidence requires more than clustered router weights. A reusable operator
must improve held-out transitions where content/entities differ and preferably
where operator compositions are novel. The required ordering is:

\[
\operatorname{sim}(O_{same\ operation,different\ content})
>
\operatorname{sim}(O_{different\ operation,same\ content}).
\]

Report both positive and negative operator categories and router collapse metrics.
