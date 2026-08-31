# RGE-PEFT Findings

**Governing SSOT:** `docs/PLAN_RGE_PEFT.md`
**Program version:** v1.1

This document records evidence-backed findings only. Implementation status and
planned hypotheses are not scientific results.

## M1 — verified experimental substrate

M1 passed on training-source commit `840b04e`.

Established:

- The RGE AdamW path is bit-identical to the legacy trainer under the CPU smoke,
  rank-16 all-target GPU smoke, and bounded actual-B0 comparisons.
- Effective-delta norms use low-rank Gram algebra; successive deltas use a
  rank-at-most-2r concatenation. Radius and angle use the base/delta inner product
  without materializing dense `B @ A` in the runtime measurement path.
- Delta metrics are invariant under tested diagonal and nontrivial
  well-conditioned LoRA gauge transformations across three aspect ratios. Raw
  factor norms are not invariant.
- Geometry-enabled and geometry-disabled AdamW trajectories are bit-identical,
  including adapter tensors, optimizer state, position, loss/LR sequence, and RNG.
- RGE exact resume restores generic optimizer and geometry measurement state.
- Full geometry every 10 steps has 6.79% paired within-trajectory amortized
  overhead on the principal GPU smoke and no post-warmup reserved-memory growth.

Not established:

- Muon improves Rineya capability or token/GPU-hour efficiency.
- DeltaBall, curriculum ordering, constant-tail continuation, or Delta-CMA helps.
- Any RGE candidate improves a frozen capability or retention gate.

Timing caveat: two cross-process overhead attempts showed materially drifting
disabled references (+13.7% and -37.9%). Those comparisons are not used for the
cadence decision; the same-process measured/unmeasured cadence contrast is used.

## M2 — exploratory optimizer-pilot status

Established so far:

- Muon LR `4e-4` was selected without evaluation by a paired effective-step
  bracket: median AdamW-relative ratios `0.472`, `0.944`, and `1.889` at raw LR
  `2e-4`, `4e-4`, and `8e-4`.
- At 104,571 matched supervised tokens, Muon reduced token-weighted training loss
  by `0.15530` and final-step loss by `0.36331` versus AdamW. Muon had lower loss
  on 75 of 101 paired plan steps, tied one, and was higher on 25.
- Muon optimizer-state tensor bytes were about half AdamW's. Peak reserved memory
  was 434 MiB higher in the repaired pair summaries.

Not established:

- Frozen retention or capability improvement. `chat_dev`/`rag_dev` endpoints
  have not yet been evaluated.
- Any speed or GPU-hour advantage. External user inference workloads repeatedly
  contended with both arms, invalidating timing comparisons.

Infrastructure negative: an asynchronously killed process could leave a
`train_log.jsonl` row newer than its checkpoint. Exact resume replayed the step
correctly but the stale row caused the post-run token-axis audit to fail. Resume
now reconciles logs to checkpoint position and keeps the last record per step;
an injected-tail regression passes. No model or optimizer tensor required repair.

### M2 decision

Muon is rejected for the general Rineya v1 recipe.

- `chat_dev`: mean quality `0.8580` versus AdamW `0.8732`, delta `-0.01524`,
  paired 95% bootstrap CI `[-0.02857, -0.00195]`. This fails the `-0.01`
  non-inferiority guardrail.
- `rag_dev`: mean quality delta `-0.00542`, but four critical
  `compound_decoding_corruption` cases appeared (control zero), intelligibility
  fell to `0.9947`, and contract violations rose from `0.00133` to `0.00266`.
- Timing/GPU-hour comparison is invalid because unrelated user inference jobs
  repeatedly contended with both training arms.

Thus the lower training loss is optimizer-dynamics evidence, not a useful-model
or efficiency win. M3 may still test DeltaBall under AdamW because H3 is
preregistered as logically separable from H1.

## M3 — exploratory DeltaBall result

The frozen surviving geometry recipe is AdamW plus DeltaBall epsilon
`0.014355268562212586` with a 40-step linear warm-in.

- The 0.5x radius was active on 37.14% of post-warm-in matrix checks and worsened
  training loss, so it was rejected as overconstrained.
- The 1x radius was active on 1.054% of checks and 34.43% of steps. Training loss
  was essentially neutral/slightly worse, but it was the best eligible radius.
- The 2x radius produced zero hits through fully warmed checkpoint 40 and was
  stopped as inactive; it receives no causal credit.
- Selected 1x retention was slightly positive: `chat_dev +0.00149` and
  `rag_dev +0.000806`, with no critical regression and RAG contract violations
  reduced to zero. Chat's CI crosses zero; this is exploratory, not confirmed.

M3 passes only as mechanism/named-retention evidence and authorizes M4. No
targeted capability, replicated, speed, or sealed claim is established.

## M4 — curriculum ordering status

A 12,875-row component-stratified subset was consumed completely under both
reference-shuffled and component-local order. Membership, 105,402 supervised
tokens, initial tensors, and token-indexed LR/DeltaBall schedules matched.

Component-local order reduced token-weighted training loss by only `0.00137` and
worsened final-step loss by `0.145`. Retention was mixed: `chat_dev -0.00368`,
`rag_dev +0.00081`, both non-inferior with confidence intervals crossing zero and
no critical regression. Curriculum is retained only as weak exploratory input to
the constant-tail test; an ordering benefit is not established.

The 12-step constant-LR continuation was also endpoint-neutral: `chat_dev`
changed by `-0.000884` and `rag_dev` was identical on all 752 cases. It receives
no standalone credit, but its six equally spaced checkpoints are valid M5 CMA
inputs.

## M5 — Delta-CMA exploratory result

Equal-weight Delta-space averaging of tail steps `{2,4,6,8,10,12}` produced an
exact rank-96 diagnostic and a primary rank-16 adapter. Rank-16 retained
`0.99999999968` of global delta energy with `1.79e-5` relative reconstruction
error; both adapters passed normal PEFT load/merge.

CMA-exact changed chat by `+0.00128` and RAG by exactly zero versus tail-final.
CMA-r16 changed chat by `+0.000873` and RAG by zero. Compression cost only
`-0.000409` chat versus exact and no RAG case. Rank-16 CMA is frozen for M6, but
the gain is tiny and exploratory; no useful capability improvement is established.

## Sealed boundary

The sealed RAG test set has not been accessed by RGE-PEFT. It remains prohibited
for implementation, debugging, hyperparameter selection, and candidate selection.
