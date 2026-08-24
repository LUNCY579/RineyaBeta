# CSGQR — Calibration and Routing

Every fitted quantity in the method: what it is, what it was fitted on, what it
came out as, and whether it earned its place.

Four things are fitted. Two are worth keeping, one is measurably inert at this
bitwidth, and one is a signal source that had to be replaced.

---

## 1. The quantization gain `α`

**What it is.** A least-squares output gain correcting the systematic shrinkage
blockwise quantization introduces. Not a trained parameter; fitted on calibration
inputs and frozen.

### 1.1 Scalar, whole remnant

```
α = E⟨F^Q(h), F(h)⟩ / E‖F^Q(h)‖²
```

**Measured on this checkpoint: α ≈ 0.997 at asymmetric INT4.**

So the gain buys almost nothing here — the shrinkage it exists to correct is
small once a zero point is used. That is a real finding about asymmetric
quantization, not a failure of the calibration, and it is why every committed
arm ran at `α = 1.00000` without penalty.

**But it took two bugs to establish.** The design writes the denominator as
`E‖F^Q‖² + ε`. Implemented literally with `ε = 1e-8` it is scale-dependent: MLP
output magnitudes are checkpoint-dependent, and on a small-activation layer the
sums were ~1e-7, making ε **13% of the denominator**. It pulled a true α of
0.9979 down to 0.8797 and made the "calibrated" remnant *worse* than the
uncalibrated one (relative error 0.1148 → 0.1644). A fixed epsilon looks
unimpeachable in source; it was found by testing across activation scales.
Now handled as the zero-check it actually is, with float64 accumulation, and
verified against the closed form to < 1e-5 across a **2500×** activation-scale
span.

The second: an earlier `run_eval.py` fitted α against the *quantized remnant's
own* output, which returns exactly 1.0 by construction and looks like a working
calibration.

### 1.2 Per group

```
α_g = E⟨F^Q_g(h), F_g(h)⟩ / E‖F^Q_g(h)‖²
```

The target is the donor's **own group output** `F_g`, not the whole MLP. That
distinction is not pedantic. Fitting each group against `F` gives

```
α_g = 1 + ⟨F_g, Σ_{h≠g} F_h⟩ / ‖F_g‖²
```

which is inflated by the cross-group inner product and **does not return 1 even
for a lossless remnant**. Measured on a bit-exact remnant, the wrong form gave
gains spanning **0.978–1.024** and injected **1.32%** relative error into a
reconstruction that was exact. It is a different estimator, not a looser one.

Regression test: a bit-exact remnant must return `α_g = 1` for every group, to
< 1e-9.

**Consequence for a previously recorded result.** The campaign had recorded
"per-group α is worse than a single scalar α" (relative error 0.1246 vs 0.1148)
as a finding. That was measured with the wrong estimator, so **it is not evidence
about per-group α** and is withdrawn as a finding. The scalar form is still what
is used, on the grounds that α ≈ 0.997 makes the whole question inert here.

### 1.3 Partial execution — the gain no routed arm had

At `ρ < 1` the surviving model receives `h + Σ_{g∈A} F_g` where it would have
received `h + F`. Nothing was rescaling that, and every routed arm ran at
`α = 1`. Two targets, answering different questions:

| target | fits against | asks |
| --- | --- | --- |
| `rows` | `Σ_{g∈A} F_g` | reproduce faithfully what I chose to spend |
| `whole` | `F` | use what I chose to best approximate the layer |

`whole` is the least-squares optimum for the perturbation the residual stream
sees, and its magnitude is itself a measurement:

```
α = 1 + ⟨F_A, F − F_A⟩ / ‖F_A‖²
```

exactly 1 when the active groups are orthogonal to the inactive ones, rising
toward `1/ρ` as they become positively correlated.

**Measured** (`results/partial_gain_L20.json`, layer 20, INT4, 24 groups, fit and
eval splits disjoint **by prompt**):

| ρ | selection | ‖F_A‖/‖F‖ | α (rows) | α (whole) | rel err α=1 | rel err α=whole |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.125 | static top-k | 0.3501 | 0.9906 | 1.0998 | 0.92372 | 0.92308 |
| 0.25 | static top-k | 0.6344 | 0.9685 | 1.0477 | 0.74543 | 0.74471 |
| 0.50 | static top-k | 0.7669 | 0.9711 | 1.0250 | 0.61898 | 0.61870 |
| 0.75 | static top-k | 0.9040 | 0.9724 | 0.9887 | 0.45160 | 0.45145 |
| 1.00 | all | 1.0106 | 0.9731 | 0.9732 | 0.18155 | 0.17955 |

**Three conclusions.**

1. **α_whole is 0.99–1.10, never near `1/ρ`.** So the groups are close to
   orthogonal and the partial sum does not overshoot. I had expected the opposite
   and had used the expected `1/ρ` inflation to justify *refusing* to offer this
   target; that justification was wrong and is corrected in the source.
2. **Calibration is worth ~0.1%.** At ρ=0.25 the relative output error moves
   0.74543 → 0.74471. The missing partial-execution gain is therefore **not** an
   explanation for anything, and in particular it was not the explanation for the
   Tier-1 impression that static-25 underperformed. (That was noise; see
   `STATISTICAL_ANALYSIS.md` §1.)
3. **The relative output error at ρ=0.25 is 0.745.** A quarter of the groups
   reconstructs a quarter of the layer, roughly as one would expect and nothing
   better. Any recovery at low ρ comes from *which* quarter, not from rescaling.

---

## 2. The confidence temperature `T`

Fitted by golden-section search on NLL, on calibration data only, and kept
strictly separate from the generation temperature — the module never reads a
generation config, and generation here is greedy anyway.

**It cannot help, and the reason is structural rather than empirical.** A
temperature is monotone in the logits, so it cannot change the *ranking* of
positions by confidence. AUROC depends only on that ranking. So temperature can
fix `ECE = 0.3557` and cannot move `AUROC = 0.5515` by construction.

Calibrating a signal that does not discriminate produces a well-calibrated
non-discriminative signal. Recorded so nobody spends time on it.

---

## 3. The gate: what "how much" is read from

### 3.1 The exact vocabulary lens — falsified, twice over

| feature | AUROC | AUPRC | gap (low − high) | Pearson(1−C, U) | Spearman(1−C, U) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `max_prob` | 0.5515 | 0.6421 | **−0.00115** | +0.0173 | +0.1282 |
| `neg_entropy` | 0.5649 | 0.6489 | +0.00892 | +0.0026 | +0.1313 |
| `prob_margin` | 0.5468 | 0.6488 | +0.00543 | +0.0190 | +0.1267 |
| `logit_margin` | 0.5374 | 0.6497 | +0.02393 | −0.0146 | +0.1104 |

Against a mean utility of 0.30347: the primary specified feature has the **wrong
sign**, the best gap is 7.9% of the mean, and AUROC 0.537–0.565 is not
deployable.

**Mechanism, measured in the same trace:** the raw lens's top-1 token agrees with
the dense model's top-1 only **26.94%** of the time at this depth. The gate is
answering a well-formed question about a different decision.

**And it could not be afforded even if it worked:**

```
LM-head MACs / gated-MLP MACs = |V| / (3·d_ff) = 32768 / 9216 = 3.556×
```

A gate that skipped on *every* token would still cost 3.56× what it saved. The
viability condition is `|V| < 3·d_ff`; on this checkpoint `d_ff` would need to
exceed 10,923 instead of 3,072, and for production models (`|V|` ≈ 128–256k,
`d_ff` ≈ 8–28k) it is worse.

**One honest nuance.** Spearman is small but *consistently positive* across all
four features (+0.11 to +0.13) while Pearson is ≈ 0. A weak monotone effect in the
predicted direction does exist, invisible in the linear correlation because the
utility distribution is skewed. The verdict is "two orders of magnitude too weak
to act on", not "absent".

### 3.2 The hidden-state need probe — this is the gate

```
q = σ(w·h + b)
```

Fitted on the pre-remnant residual, split **by prompt** (48/19/29 prompts →
394/150/254 positions), λ chosen on validation, test touched once.

| predictor | AUROC | AUPRC |
| --- | ---: | ---: |
| raw `max_prob` | 0.5515 | 0.6421 |
| raw `neg_entropy` | 0.5649 | 0.6489 |
| logistic on all four lens features | 0.580 | 0.648 |
| **linear probe on the residual** | **0.9358** | **0.9492** |
| permuted-label control | 0.5065 | — |
| reply-position-only control | 0.601 | 0.676 |
| prompt-mean **oracle** | 0.6976 | 0.771 |

The probe clears the prompt-mean oracle, so it resolves individual tokens rather
than prompt-level structure; the permuted control at 0.5065 rules out leakage; val
0.9313 against test 0.9358 rules out selection on the test split.

**Caveat that the controls carry, not the sample size:** 1,025 parameters against
394 training positions.

**Cost, per token, as a share of the MLP it gates:**

| monitor | MACs | share | bytes (fp32) |
| --- | --- | ---: | ---: |
| need probe | `2·d_model` | **0.0109%** | 4,100 |
| group router | `2·d_model·\|G\|` | **0.2604%** | 98,400 |
| exact lens | `2·\|V\|·d_model` | **355.6%** | 0 (borrowed) |

Four orders of magnitude. **The probe is not an optimisation of the method; it is
the only viable form of the "how much" gate.**

Standardisation is folded into the weights at load time so inference is a single
matmul, and the fold's exactness is asserted:
`w·((h−μ)/σ) + b == (w/σ)·h + (b − (w/σ)·μ)`.

### 3.3 The need threshold `δ`

`y_t = 1[U_t > δ]`, with `δ = 0.1`.

Chosen because `P(U_t > 0) = 0.9987`. At a layer with real deletion damage, "does
this token need the remnant?" is essentially always yes, so `δ = 0` is
degenerate and **any binary skip gate loses utility by construction**. At
`δ = 0.1` the base rate is 0.62 and AUROC is meaningful.

This is a data-driven choice made on the calibration trace. Its sensitivity has
**not** been swept — a gap, recorded.

The deeper consequence: the useful framing at this layer is *how much*, not
*whether*, and an invocation rate near 100% is the correct behaviour rather than a
gate collapse to be tuned away.

---

## 4. The router: which groups

### 4.1 The utility label is interventional

```
U_{t,g} = KL(p^T ‖ p^0) − KL(p^T ‖ p^{+g})
```

Measured by actually running the model with group `g` on and comparing against it
off — not an activation-magnitude proxy. Larger is better; a useless group scores
≈ 0.

That distinction is what the `magnitude` control exists to test, and it earns its
keep. At matched ρ=0.25 on 414 held-out prompts, KL vs dense:

| selection | KL | 95% CI |
| --- | ---: | --- |
| static significance | **0.17822** | [0.16990, 0.18710] |
| dynamic (probe) | 0.18253 | [0.17356, 0.19252] |
| random | 0.21563 | [0.20570, 0.22695] |
| **magnitude** | 0.23719 | [0.22653, 0.24901] |
| reversed significance | 0.24338 | [0.23270, 0.25507] |
| deletion (ρ=0) | 0.28055 | [0.26861, 0.29380] |

**The magnitude heuristic is worse than random.** So the interventional label
carries information that a weight-norm proxy does not — the routing signal is not
"this group has big weights". And **reversing** the ranking is worse than random
too, so the *order* carries the information rather than merely the identity of the
selected set.

### 4.2 The global ranking

Fitted by accumulating `U_{t,g}` with separate exposure counts and ranking on
mean utility. The result is dominated by one group: group 23 has mean utility
0.05889, **2.6×** the next (group 17 at 0.02255). Two groups have *negative* mean
utility (group 21 at −0.00567, group 11 at −0.00681) — running them hurts on
average.

### 4.3 Dynamic per-state routing — works, and buys nothing

A ridge readout of the residual predicting all 24 group utilities. It is
genuinely state-dependent, not degenerate: per-group selection rates run from
0.548 down to 0.018 across all 24 groups, against `random`'s flat 0.246–0.254 and
`static`'s hard 1.0/0.0.

And it does not beat static:

| ρ | static | dynamic | verdict |
| ---: | ---: | ---: | --- |
| 0.25 | **0.17822** | 0.18253 | static slightly ahead, intervals overlap |
| 0.50 | 0.13565 | **0.12943** | dynamic slightly ahead, intervals overlap |

Neither difference is significant. **The dynamic branch is `NOT_ESTABLISHED`.**

### 4.4 Why the offline analysis said otherwise — the campaign's sharpest lesson

`capture_at_k` scores a selection by **summing singleton marginal utilities**,
which assumes utility is additive over groups. Additivity was verified at `k = 2`
(pairwise interaction 2.1% of `|U(A)|+|U(B)|`) and at `k = 24` (sum within 1.7% of
the whole-remnant utility) — and **assumed in between**.

Measured against true utility, `KL(dense‖delete) − KL(dense‖arm)`, at ρ=0.25 on
held-out prompts:

| selection | true `U` | proxy | ratio |
| --- | ---: | ---: | ---: |
| random | 0.06492 | 0.07398 | 1.14× |
| static significance | 0.10234 | 0.16097 | 1.57× |
| dynamic (probe) | 0.09802 | 0.39209 | **4.00×** |

And the proxy's per-token *oracle* at `k = 12` is **0.63376** — larger than the
true utility of **all 24 groups** (0.27006), which is impossible for a real
utility.

Per-token cherry-picking exploits non-additivity, so **the more tailored the
selection, the more the additive sum overstates it**. The proxy therefore flatters
precisely the method that optimises it hardest. The "probe wins 66–73% of the
static-to-oracle headroom" result is an artefact of the metric, and the headroom
argument for the dynamic branch does not survive contact with a direct
measurement.

### 4.5 The cost term

`R_g = Û_g / (c_g + ε)`. Every group has the same shape on this checkpoint, so
`R_g` is a monotone rescaling of `Û_g` and selection is identical to ranking on
utility alone. Implemented anyway because mixed-size groups are a planned
follow-up and a cost model that appears late tends to appear wrong.

The `(1−C)^γ` factor in the brief's `R_g` is **omitted from the ranking**: it is a
per-token scalar and provably cannot reorder a top-k. Including it created a real
degeneracy — at `C = 1.0` every value became exactly 0.0 and `argsort` returned
groups `0..k−1`, silently turning the significance selector into a contiguous
selector. The factor's actual job, setting `k`, is unchanged in `ρ()`.

---

## 5. Summary: what earned its place

| fitted quantity | verdict |
| --- | --- |
| scalar `α` | **Inert here** (0.997 at INT4). Correct, cheap, keep — but it is not doing work at this bitwidth. |
| per-group `α` | **Correct now**; the prior "worse than scalar" finding was measured with the wrong estimator and is withdrawn. |
| partial-execution `α` | **Measured and inert** (~0.1%). Its real value was falsifying a hypothesis: the groups are near-orthogonal, so partial sums do not overshoot. |
| temperature `T` | **Cannot help.** Monotone, so it cannot move a rank-based metric. |
| exact-lens confidence | **Falsified.** Wrong sign on the primary feature, AUROC 0.54–0.57, and 3.56× the cost of what it gates. |
| need probe | **The gate.** AUROC 0.9358 at 0.0109% of the gated MLP. |
| interventional group ranking | **Works.** Beats random, beats magnitude, and reversing it is worse than random. |
| dynamic per-state router | **`NOT_ESTABLISHED`.** Genuinely dynamic, genuinely informative, no advantage over a fixed ranking. |
| `capture_at_k` as a selection criterion | **Not predictive.** Overstates by 4× for the selection that optimises it. Should not be used to justify a routing design again. |
