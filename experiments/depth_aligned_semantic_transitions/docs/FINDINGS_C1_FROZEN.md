# Frozen-backbone trajectory study (C1) — not causal

**Status:** CPU analysis of committed artifacts. This is **decodability**, not
computation, not a capability result, and not a reason to reopen V1.

**Artifacts:**
[`../artifacts/c1a_layer_stage_profile_dev.json`](../artifacts/c1a_layer_stage_profile_dev.json),
[`../artifacts/c1b_permutation_k4_dev.json`](../artifacts/c1b_permutation_k4_dev.json)

**Cache provenance (from JSON, tensors not committed):** Qwen3-0.6B, DEV 2400,
`final_norm` readout, layers 7/14/21/27 for K=4, config_sha256
`8256507551ca66884724d9b188e95c86d1c35c76dd53fd284c4366e91b31f8d5`, git
`c023758`, `lora_loaded: false`, `require_distinct: true`.

---

## C1b — exhaustive K=4 permutations (24 × 20 splits)

| assignment | median rank | times #1 | mean `L_state` |
|---|---:|---:|---:|
| identity (0,1,2,3) | **2** | 6 / 20 | 0.1240 |
| reverse (3,2,1,0) | **10** | 0 / 20 | 0.1260 |
| best: (1,2,3,0) | 1 | 14 / 20 | 0.1236 |

Mean within-split `L_state` spread: **0.00724** (about **5.8%** of identity
`L_state`). The frozen backbone mildly ranks forward above reverse and is close
to indifferent among permutations. That **refutes** “the untouched model prefers
to reason backwards.” It does **not** show that identity is a depth clock.

An older FINDINGS_G3 §2b paragraph cited identity median rank 3 and spread
~4.3% from a prior script. Both estimates agree the landscape is flat. This
file uses the committed `c1b` JSON.

## C1a — layer × stage profile (layers 1–27)

`best_layer_per_stage` is **[1, 1, 1, 1]**. Spearman of stage vs best layer is
**0**. There is no diagonal “early stage early layer” ridge under a shared
linear readout. All four centred stages are best decoded at **layer 1**.

Do not read this as “reasoning happens at layer 1.” It says the frozen
absolute-state targets, after within-problem centering, do not induce a
chronological depth axis at this estimator.

## K=8

Not run. On this corpus only ~4.6% of DEV traces have eight distinct compact
states, so K=8 repeats targets and flattens the landscape by construction
([`../scripts/permutation_landscape.py`](../scripts/permutation_landscape.py)).

Optional K=5/6 permutation was **not** run. Local caches are K=4 and all-layer
profile tensors; permutation mode requires `hidden.shape[1] == K`. This pass
does not rebuild a K=6 cache on GPU.

## C2

Not run. Pooled C1 cannot decide whether π*(x) is stable by problem family.

## Claim boundary

Readable across depth ≠ DAST created a trajectory ≠ useful as a training target.
V1 remains CLOSED — NEGATIVE. TEST sealed.
