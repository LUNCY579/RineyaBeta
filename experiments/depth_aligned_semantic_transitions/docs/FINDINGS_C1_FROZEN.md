# Frozen-backbone trajectory study (C1) — not causal

**Status:** CPU analysis of committed artifacts. This is **decodability**, not
computation, not a capability result, and not a reason to reopen V1.

**Artifacts:**
[`../artifacts/c1a_layer_stage_profile_dev.json`](../artifacts/c1a_layer_stage_profile_dev.json),
[`../artifacts/c1a_layer_stage_profile_with_embedding.json`](../artifacts/c1a_layer_stage_profile_with_embedding.json),
[`../artifacts/c1b_permutation_k4_dev.json`](../artifacts/c1b_permutation_k4_dev.json),
[`../artifacts/c1c_depth_information_controls.json`](../artifacts/c1c_depth_information_controls.json),
[`../artifacts/c2_trajectory_heterogeneity.json`](../artifacts/c2_trajectory_heterogeneity.json)

**Cache provenance (from JSON, tensors not committed):** Qwen3-0.6B, DEV 2400,
`final_norm` readout, layers 7/14/21/27 for K=4, config_sha256
`8256507551ca66884724d9b188e95c86d1c35c76dd53fd284c4366e91b31f8d5`, git
`c023758`, `lora_loaded: false`, `require_distinct: true`.

---

## C1b — exhaustive K=4 permutations (24 × 20 splits), n = 1379

**`--require-distinct` now works.** It was a no-op until the
`DepthCache.trajectory_lengths` fix, so the first pass ran on all 2,400 problems
including the 1,021 (42.5%) whose depth targets are **not** all distinct. For those,
permutations that swap two depths holding the same target are literal no-ops, which
flattens the landscape by target construction. Corrected run drops them:

| assignment | median rank | times #1 | mean `L_state` |
|---|---:|---:|---:|
| best: (0,2,1,3) odd/even interleave | 3 | 4 / 20 | 0.14250 |
| **identity (0,1,2,3)** | **7** | 4 / 20 | 0.14280 |
| **reverse (3,2,1,0)** | **18** | 0 / 20 | 0.14369 |
| worst: (3,2,0,1) | 23 | 0 / 20 | 0.14466 |

Mean within-split `L_state` spread: **0.00269**, about **1.89%** of identity
`L_state`. For comparison, the unfiltered pass reported identity rank 2, reverse 10,
spread 5.84% — so removing the duplicate-target artifact makes the frozen model
look **more** indifferent, not less.

Two claims must be separated, because the correction breaks them apart:

- **"The untouched model prefers to reason backwards" is refuted.** Reverse ranks
  **18 of 24** and is never first, in the filtered run and the unfiltered one and
  the older single-split study alike. Three independent passes, same direction.
- **"Forward is the preferred assignment" is NOT supported.** Identity is
  **7 of 24** on median and first in only 4/20 splits. The best assignment is
  odd/even interleave, ahead of identity by 0.0003 `L_state` — 0.2% — which is
  not a preference worth naming. Identity is mid-pack.

Neither shows identity is a depth clock.

### Superseding the older §2b numbers

`FINDINGS_G3_SEEDS.md` §2b reports "20 random half/half splits", identity median
rank **3** against reverse **12**, reverse "never ranks first", and a spread of
**0.0096 (4.3%)**. Those numbers are **not reproducible and not supported by any
committed artifact**:

- [`../scripts/permutation_preference.py`](../scripts/permutation_preference.py),
  the only script that produced them, contains **no RNG and no split loop**. It
  does exactly one deterministic split (`half = N // 2`). A median over 20 random
  splits was never computable from it.
- Its committed output
  [`../qualification/permutation_preference.json`](../qualification/permutation_preference.json)
  records identity rank **1** and reverse rank **4** of 24 on 600 problems, with
  spread 0.22287→0.23103 = **0.00816**, i.e. **3.66%** of `L_state`.
- That script also consumed a hidden-state cache keyed `SYM27` that no committed
  script produced, so its input was unreproducible until
  [`../scripts/build_depth_cache.py`](../scripts/build_depth_cache.py) closed the gap.

The **direction** survives all three passes — identity ahead of reverse on the
frozen model in the old single split (1 vs 4), in the unfiltered 20-split run
(2 vs 10), and in the corrected filtered run (7 vs 18). The specific rank and spread
figures in §2b should not be cited. This file's `c1b` numbers supersede them.

## C1a — layer × stage profile (layers 1–27)

`best_layer_per_stage` is **[1, 1, 1, 1]**. Spearman of stage vs best layer is
**0**. There is no diagonal “early stage early layer” ridge under a shared
linear readout. All four centred stages are best decoded at **layer 1**.

Do not read this as “reasoning happens at layer 1.” It says the frozen
absolute-state targets, after within-problem centering, do not induce a
chronological depth axis at this estimator.

Two caveats on the summary fields, so they are not over-read.
`monotone_in_stage: true` is **vacuous** when every stage shares one argmax
layer — a tie satisfies `<=` — and the Spearman is 0 for the same reason: no
variance, not a measured absence of correlation.

## C1c — the signal is present *before* any decoder layer

[`../artifacts/c1a_layer_stage_profile_with_embedding.json`](../artifacts/c1a_layer_stage_profile_with_embedding.json),
[`../artifacts/c1c_depth_information_controls.json`](../artifacts/c1c_depth_information_controls.json),
[`../scripts/depth_information_controls.py`](../scripts/depth_information_controls.py)

C1a could not separate *“shallow layers decode best”* from *“the targets live in
input-embedding space.”* `FrozenTextTargetEncoder` builds every target from the
frozen **input embedding**, so a probe reading a near-embedding hidden state may
be doing embedding-to-embedding regression rather than reading computation.
Adding **layer 0** — the token-embedding centroid, before any decoder layer runs
— decides it.

| layer | stage0 | stage1 | stage2 | stage3 |
|---|---:|---:|---:|---:|
| **0 (embeddings)** | 0.8662 | 0.8935 | 0.9275 | 0.9450 |
| 1 | 0.8695 | 0.8960 | 0.9290 | 0.9452 |
| 9 (minimum) | 0.8433 | 0.8756 | 0.9126 | 0.9309 |
| 27 (deepest DAST site) | 0.8498 | 0.8798 | 0.9136 | 0.9333 |

Layer 0 is within 0.003 of the best layer and **beats every layer from 3 upward**.
The 28-layer stack adds nothing to stage decodability and slightly degrades it.

Four matched conditions, shared readout, **n = 1379** (distinctness filter applied),
20 splits:

| condition | `L_state` | order margin | top-1 |
|---|---:|---:|---:|
| `real` — layers 7/14/21/27 (the V1 sites) | 0.14280 | +1.1151 | 0.9733 |
| `embedding_x4` — layers 0/1/2/3 | **0.13138** | +1.1294 | 0.9631 |
| `embedding_only` — layer 0 at all four depths | 0.89084 | +0.0000 | **0.2500** |
| `shuffled_pairing` — right depths, wrong problem | 0.71088 | +0.3743 | 0.6211 |
| *trained* `state_ordered`, 250 steps | *0.26290* | *+0.9350* | *0.8332* |

Read in order:

1. **`embedding_only` is the control that keeps the rest honest.** A shared readout
   fed the *identical* vector at every depth cannot separate stages, and it lands
   exactly on the 0.25 chance top-1 with a margin of 0.0000. So the 0.973 top-1
   elsewhere is genuine four-way discrimination, not a metric that scores well on
   anything.
2. **But that discrimination needs only four *distinguishable* representations, not
   a chronological axis.** The four *shallowest* states (0/1/2/3) support the
   objective slightly **better** than V1's sites spread across the stack
   (`L_state` 0.131 vs 0.143). Spanning depth is not merely unnecessary here; it is
   marginally counterproductive.
3. **`shuffled_pairing` sets the floor.** Breaking the problem↔target correspondence
   while preserving both marginal geometries costs most but not all of the signal —
   within-problem centering leaves some structure — so the real conditions are well
   clear of chance.
4. Against all of this, the *trained* projector reaches `L_state` 0.263: roughly
   **~1.8× worse at DAST's own objective** than a closed-form ridge on the untouched
   bottom of the network.

The honest summary of C1a+C1c is stronger than “no chronological ridge”:

> The quantity `L_state` supervises is essentially available in the prompt's
> token-embedding centroid. It does not require the depth stack, it is not improved
> by it, and 250 steps of training the real mechanism fits it substantially worse
> than a ridge probe does.

This is a statement about the **target design**, and it is the most likely reason
`L_state` is easy to drive while being uninformative about reasoning. Estimator
caveat: the ridge is fitted on one DEV half and scored on the other, whereas the
projector was trained on TRAIN and scored on DEV. Both are held-out, and train/DEV
share generators, so the comparison is fair in kind — but it is not an identical
procedure, and the gap is far larger than that asymmetry plausibly explains.

## K=8

Not run. On this corpus only ~4.6% of DEV traces have eight distinct compact
states, so K=8 repeats targets and flattens the landscape by construction
([`../scripts/permutation_landscape.py`](../scripts/permutation_landscape.py)).

Optional K=5/6 permutation was **not** run. Local caches are K=4 and all-layer
profile tensors; permutation mode requires `hidden.shape[1] == K`. This pass
does not rebuild a K=6 cache on GPU.

## C2 — real but tiny example-family heterogeneity

[`../artifacts/c2_trajectory_heterogeneity.json`](../artifacts/c2_trajectory_heterogeneity.json),
[`../scripts/trajectory_heterogeneity.py`](../scripts/trajectory_heterogeneity.py)

A flat pooled landscape can hide stable per-example preference. The trap is that
with 24 permutations and a noisy per-problem estimate, `argmin` **always** looks
heterogeneous — every problem has some favourite. So the question is not "do
preferences differ?" but "do they **replicate**?"

Design: each replicate splits problems FIT (50%) / EVAL_A (25%) / EVAL_B (25%); the
shared readout is fitted on FIT only; per-problem losses are scored on A and B
independently; a family's preference is its mean loss vector, centred across
permutations so it measures preference *shape* rather than difficulty. Reported
against a **label-shuffled null** that preserves every marginal and destroys only
family structure.

| stratum | r(A,B) | label null | argmin agree | argmin null | chance |
|---|---:|---:|---:|---:|---:|
| generator | **+0.3701** | +0.0814 | 0.1363 | 0.0463 | 0.0417 |
| answer_type | **+0.3893** | +0.1024 | 0.1501 | 0.0618 | 0.0417 |
| step_count | +0.3650 | +0.2076 | 0.1250 | 0.0625 | 0.0417 |

Per-problem `argmin` agreement across independent refits: **0.1650** against a
chance of 0.0417.

**The heterogeneity is real.** Split-half reliability clears its label-shuffled null
by roughly 4×, on two strata whose nulls are near zero. Families do reliably differ
in which assignment they prefer.

**And it is far too small to act on.** The entire permutation spread is 1.89% of
`L_state` (C1b), so what replicates reliably is a difference of about **0.003 in
`L_state`**, with no evidence that it maps to capability at all. Modal preference
replicates only 13–15% of the time.

> **C2 passes the reliability test and fails the magnitude test. It is NOT a green
> light for Adaptive DAST.** A router would be trained to select among options that
> differ by 0.2% of an objective which C1c already shows is satisfiable from the
> token embedding. Reliable is not the same as consequential.

Caveats. `step_count` has the highest null (+0.2076) because step count also tracks
overall difficulty, so part of its apparent structure is a difficulty confound.
`category` is not reported: it is 1:1 with `generator` in this corpus (30 of each,
30 distinct pairs), so showing both would read as two confirmations of one
measurement.

## Claim boundary

Readable across depth ≠ DAST created a trajectory ≠ useful as a training target.
V1 remains CLOSED — NEGATIVE. TEST sealed.
