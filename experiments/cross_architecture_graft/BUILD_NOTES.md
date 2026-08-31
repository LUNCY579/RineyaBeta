# Cross-Architecture Graft v1 — Build Notes

## Current build result

The branch contains a complete synthetic end-to-end path and a ready-to-run
local Rineya campaign path. The real checkpoint/data artifacts are not available
inside the GitHub editing environment, so no real Rineya training result is
claimed.

Current authorized label:

```text
BUILT — SYNTHETICALLY VERIFIED — REAL CAMPAIGN UNRUN
```

## Implemented proof obligations

- Exact-zero scalar/channel/token gates.
- Exact evaluation bypass for a zero gate.
- Explicit input/output representation bridges.
- Causal NoPE and latent-KV attention reference implementations.
- Deterministic routed-MoE reference path.
- Generic adapter for future already-loaded foreign decoder blocks.
- Additive wrapping of selected Hugging Face decoder layers.
- Graft-only state export and strict reload.
- Local-only model loading.
- Frozen-host optimizer boundary.
- Multi-row pre-training logit equivalence proof.
- Frozen-parameter mutation detection.
- Deterministic grouped train/dev splitting.
- Split-manifest rehashing before every real run.
- Complete config snapshot and tamper-evident evidence payload.
- Same-seed manifest comparator with host/data/training-contract checks.
- Five-seed matched campaign materializer.

## Bugs found and fixed before commit

### 1. Dead first-step gate gradient

An early implementation used a conditional zero branch around the gated donor
contribution. Although it protected the host from NaNs, the conditional also
removed the gradient into an exactly zero gate. The final implementation instead
sanitizes only the donor delta:

```python
safe_delta = torch.nan_to_num(host_delta, nan=0.0, posinf=0.0, neginf=0.0)
contribution = gate * safe_delta
```

This preserves an exact zero contribution while retaining the first-step gate
gradient.

### 2. `unsealed` falsely matched `sealed`

The first forbidden-path guard used substring matching. A valid path such as
`unsealed_train.jsonl` therefore matched `sealed`. The corrected guard explicitly
normalizes the word `unsealed` to an open-development marker before testing for
forbidden assets. Regression coverage now proves both behaviors:

- `unsealed_*` is allowed;
- `sealed`, hidden/final holdouts, showcase, and long-context-demo paths are
  rejected.

### 3. Underidentified primary control

The initial design included only a parameter-matched SwiGLU branch. That could
show a benefit over an MLP but could not distinguish latent K/V compression from
the generic effect of adding attention. A full NoPE/GQA control was added and
dimensioned to within 0.51% of the primary trainable budget.

### 4. Nondeterministic MoE accumulation risk

A dispatch implementation used indexed accumulation, which can be nondeterministic
on GPU. The reference MoE now evaluates compact experts, gathers top-k outputs,
and sums them in a fixed tensor order. This favors experimental reproducibility
over serving efficiency.

## Synthetic smoke interpretation

The smoke task creates a frozen toy host and a frozen latent-KV teacher. The
candidate receives an exact copy of that teacher behind a zero gate and trainable
bridges; a matched SwiGLU sidecar receives the same targets. The test proves that:

- zero gate reproduces the host bit-for-bit;
- the host and frozen donor do not mutate;
- the candidate path can learn and export;
- the matched control runs through the same optimizer path;
- the evidence record is produced and verifiable.

Because the candidate is deliberately handed the exact synthetic teacher, its
lower MSE is expected and carries no model-quality or donor-transfer meaning.

## Known v1 limitations

- The latent-KV block is an MLA-inspired mechanism, not a checkpoint-compatible
  DeepSeek implementation and not a cache-aware production kernel.
- NoPE is evaluated as an added sidecar, not as periodic replacement throughout
  the model.
- Routed MoE does not implement the complete DeepSeekMoE balancing/training
  recipe.
- The real runner performs a short fixed-budget graft tune and unsealed NLL
  screen; it does not invoke semantic gates automatically.
- V1 does not implement exact checkpoint resume. Interrupted scientific arms
  restart from step zero.
- Actual cross-family donor weights and learned nonlinear stitching are deferred
  to a separately controlled functional-transfer phase.

These limitations are intentional boundaries, not hidden claims.
