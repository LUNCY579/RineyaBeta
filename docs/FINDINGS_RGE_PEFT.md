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

## Sealed boundary

The sealed RAG test set has not been accessed by RGE-PEFT. It remains prohibited
for implementation, debugging, hyperparameter selection, and candidate selection.
