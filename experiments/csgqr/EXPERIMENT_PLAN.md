# CSGQR — Experiment Plan

The design as executed, with the reasoning that fixed each choice. Written
retrospectively for the parts already run and prospectively for what remains, and
it says which choices were **pre-registered** and which were made after seeing
data — because that distinction is what the controls are worth.

---

## 1. Endpoints, and why there are two

| endpoint | unit | power here | role |
| --- | --- | --- | --- |
| `run_eval.py` — generation + rubric | case (510) | **none** | protocol fidelity to the repository's existing baselines |
| `eval_forced.py` — teacher-forced paired | reply position (3,830), clustered by prompt | high | **primary** |

The generation endpoint is the repository's own protocol and reproduces its stored
baselines at `|Δ| = 0.000e+00`, which is why it is kept: it establishes that this
harness is the same harness. But it cannot resolve the question. Paired sd
**0.14901** against a deletion effect of **0.01135**; 80% power needs ~1,350 cases
for deletion and ~10,900 for arm-vs-arm gaps; 510 exist. After Holm–Bonferroni
across nine pre-registered contrasts, none was significant.

That was **discovered, not anticipated** — the Tier-1 sweep ran first and the power
analysis was done to explain two anomalies in its output. The teacher-forced
endpoint was built in response. Anyone reading the arm table must know which
endpoint a number came from.

**Metrics.** `KL(dense ‖ arm)` and `gold_top1` are primary; `d_nll` is reported and
**not relied on**, because the wrong-layer control scores *better* than dense on it
while being significantly worse on `gold_top1` — flattening an over-confident
distribution lowers average NLL on a model that is right 28% of the time. A metric
that makes a control look like an improvement cannot be primary.

Sequence-level `sequence_flip_rate` and `first_flip` are always reported alongside
token-level means, because INT4-all agrees with dense on 94.4% of tokens and still
changes 39.6% of replies.

---

## 2. Data roles

| split | source | used for |
| --- | --- | --- |
| calibration / trace | `chat_dev` prompts **0–95** | utility labels, probe fitting, `δ`, static ranking, `α` |
| held-out | `chat_dev` prompts **96–509** (414 usable) | every number in `FINDINGS.md` |
| development (generation) | `chat_dev` all 510 | Tier-1 sweep, retained as provenance only |
| **sealed** | `rag_test_SEALED`, `puzzle_test_SEALED`, `puzzle_challenge_SEALED` | **never read** |

Disjointness is **enforced, not documented**: `eval_forced.py` refuses a
probe-gated run whose prompts overlap the probe's training trace and names the
required `--skip-prompts` value. Demonstrated by running it and being refused.
`calibrate_partial.py` likewise fits and evaluates on prompt-disjoint splits.

Both eval scripts refuse any path containing `SEALED`.

`rag_dev` (752 cases) exists and was **not** swept. That is a gap, not a choice.

---

## 3. Layer selection — pre-registered rule

A 28-layer scan measured deletion damage and recovery for `fp`, `int8`, `int4`
(`results/layer_scan.json`, 64 prompts, token-weighted mean
`KL(dense ‖ arm)` at reply positions). The rule — **take the median-deletion-damage
layer** — was fixed before the scan and selected **L20** (delete KL 0.3194,
deleted top-1 agreement 0.750, INT4 recovery 0.970 sitting at the population
median).

Chosen by rule, not by preference. The scan also recorded that MLP-only deletion
damage is *not* the block-level angular-distance prior: L26 sits near the top of
the prunable list by angular distance and mid-range here.

---

## 4. Arms

46 in `variants.CATALOGUE`, each a serialized `RoutingPolicy` so that an
arm-to-arm difference is a difference of policy and never of implementation.
Capability flags (`needs_lens`, `needs_probe`, `needs_group_utility`) are derived
from the policy and cross-checked against explicit lists, so an arm cannot
silently become a different arm — the failure mode that made **S1** invisible.

### 4.1 Pre-registered before any CSGQR result was inspected

* references: dense, exact deletion, FP-frozen-through-the-remnant-path
* precision ladder: INT8-all, INT4-all
* matched-budget group controls: random, magnitude, shuffled, reversed
* donor identity: wrong-layer donor at matched bytes
* confidence features: max-prob, entropy, probability margin, logit margin
* the contrast list in `analyze_arms.py`
* the analytic ceilings in `results/budget.json`

### 4.2 Added during this session, and why — stated so they are not mistaken for pre-registered

| added | reason |
| --- | --- |
| **random-token gate** (5 matched + 4 ladder) | Every *group*-level arm had a random partner; the *token*-level gate did not, so **H2 was untestable** and the gap had been filled by inference from a classifier metric. This is a control that should have existed from the start. |
| **dynamic-significance arms** (ρ=0.25, 0.50) | Previously unrunnable — **S1**. |
| **probe-gated arms** | Previously unrunnable — no serialized probe existed. |
| **threshold ladders** (6 lens, 6 probe, 4 random) | Comparing a lens gate at τ=0.9 against a probe gate at τ=0.5 compares two points on two curves. Frontiers are needed, not points. |

The distinction matters: 4.2 arms were added *knowing* the earlier results, so
their value rests on being controls that make the method look **worse** or on being
the method's own missing components — not on being favourable.

---

## 5. Statistical design

* **Paired** at the reply position; every arm sees identical prompts, positions and
  gold tokens.
* **Prompt-clustered** bootstrap, 4,000 draws, seed fixed. Positions within a
  prompt are dependent; treating them as independent would shrink intervals ~3×.
* Paired differences formed **per position before resampling**.
* Multiple comparisons: Holm–Bonferroni over a contrast list fixed in source. On
  the teacher-forced endpoint, orderings are claimed from **non-overlapping 95%
  intervals**, which is conservative relative to a paired test.
* One seed. Trainable components are closed-form ridge or deterministic full-batch
  L-BFGS from a fixed zero init, so a rerun is bit-identical and there is no
  optimiser seed to vary. This does not satisfy the brief's three-seed request for
  anything genuinely stochastic.

---

## 6. Execution order, and the gate that stopped things

1. Repository state audit → **found the remote branch empty**; committed and pushed
   the at-risk evidence before touching implementation.
2. Anomaly chase on the Tier-1 output → **found the endpoint has no power**.
3. Stage I: 19 defects, 86 → 123 verification checks.
4. Stage II readiness gate → `READY_FOR_SCIENTIFIC_EXPERIMENTS` for the mechanism,
   with **Gate F explicitly carved out** as not assessable.
5. Stage III in five batches: A baselines, B matched budget, C gates, D
   random-token controls, E frontiers. Then the GPU benchmark.

**What the gate actually stopped:** no systems claim was made, because
`bench_gpu.py` could not get an exclusive card. It was run anyway — memory is
valid under contention, latency is not — and the report says which is which.

---

## 7. Confounds handled, and how

| confound | handling |
| --- | --- |
| shape-dependent GEMM (cuBLAS tile selection) | `b2-fp-all` travels the identical code path with identical weights. On the teacher-forced endpoint it is **numerically identical** to dense, so the noise floor is **zero** and any divergence is attributable. (On the generation endpoint it was 0.00084.) |
| calibration leakage | `--skip-prompts` enforcement; disjoint fit/eval in `calibrate_partial.py`; `α` never fitted on evaluation records in any committed run (α = 1.00000 everywhere) |
| re-quantization between configurations | remnant installed once, policy swapped, so every group is measured against the same packed weights |
| arm-dependent confidence | the pre-remnant residual is arm-independent by construction, and measured at max \|Δ\| = **0.000e+00** over 798 positions |
| GPU contention | recorded; latency not quoted; three-run spread reported (3.7×) |
| a metric that flatters a control | `d_nll` demoted after the wrong-layer arm scored better than dense on it |
| a surrogate that flatters the method | `capture_at_k` overstates by 4× for the dynamic selection; replaced by direct measurement |

---

## 8. What remains, gated

`MAXIMUM_POTENTIAL_ROADMAP.md` has the full ordering. In short: **multi-layer
CSGQR with the reduced method** (binary probe gate, static interventional ranking)
and the **same-byte capacity controls** are Tier 1, ~3 GPU-hours, each with a
predeclared prediction so both outcomes are informative. Nothing else starts until
those report.

The reduced method drops the exact lens (unaffordable), the per-state router (no
advantage over static) and the three-level ramp (no useful interior point). The
plan for the next campaign is therefore a test of what survived, not a retest of
the specification.
