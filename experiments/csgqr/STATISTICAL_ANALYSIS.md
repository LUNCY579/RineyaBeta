# CSGQR — Statistical Analysis

How every interval and every p-value in this campaign was produced, what the unit
of analysis is, and where the inference is weak.

---

## 1. The finding that governs everything else: the generation endpoint has no power

This was not assumed. It was measured on the campaign's own Tier-1 output
(`results/arms_comparison.json`).

**Design.** 510 `chat_dev` cases, greedy, `max_new_tokens=32`, scored by the
repository's `quality_evaluation.py` v4.5. Every arm sees the same cases in the
same order, so all contrasts are paired at the case level.

**Measured, for the largest effect in the study — deleting the MLP entirely:**

| quantity | value |
| --- | ---: |
| paired mean delta | −0.01135 |
| paired **sd** | **0.14901** |
| standard error | 0.00660 |
| t (509 df) | **−1.72** |
| cluster-bootstrap 95% CI | [−0.02427, +0.00195] |
| Wilcoxon signed-rank p | 0.0965 |
| Holm-adjusted p | 0.4825 → **retain H₀** |

The sd is **13×** the effect. Required n for 80% power at α = 0.05 two-sided,
`n = ((1.96 + 0.8416)·sd / Δ)²`:

| target effect | required n | available |
| --- | ---: | ---: |
| deletion damage, 0.01135 | **1,352** | 510 |
| static-vs-random gap, 0.00872 | **2,292** | 510 |
| typical arm-vs-arm gap, ~0.004 | **10,892** | 510 |

**Why the sd is so large.** The scorer is a coarse, saturated rubric: p10 = median
= 0.720 on the dense arm. Deletion changed **94.5%** of replies, yet **55.5%** of
cases scored *exactly* the same. Among the 227 cases that did move, the mean delta
was −0.02551 with sd 0.22281 — the metric jumps in discrete steps rather than
responding smoothly.

**Consequence.** After Holm–Bonferroni across the nine pre-registered contrasts,
**not one was significant, including deletion damage itself.** The Tier-1 sweep
executed correctly; it measured the instrument. Its arm ordering — including
INT4-all above dense and static-25 below deletion — is entirely inside the noise
and is not used as evidence anywhere in this campaign.

This is an `EXPERIMENTAL_LIMITATION`, not an `ALGORITHMIC_LIMITATION`, and the
distinction is the whole reason for §2.

---

## 2. The primary endpoint, and its demonstrated power

`eval_forced.py`. Same prompts, same protocol boundary, different unit of
analysis.

**Unit of analysis:** one reply position. **Inference unit:** one prompt.

Positions within a prompt share a context and a truncated prefix and are not
independent, so every interval is a **prompt-clustered percentile bootstrap** —
whole prompts are resampled with replacement and every position of a drawn prompt
comes along. Treating positions as independent would shrink intervals by roughly
`sqrt(mean reply length)` ≈ 3× here, which is the standard way to manufacture
significance on this kind of data.

**Held-out design.** 414 prompts, 3,830 reply positions, `--skip-prompts 96` so
the evaluation is disjoint from the 96 prompts the probes were fitted from. The
script **refuses** to run a probe-gated arm on overlapping prompts and names the
required offset.

**Power, demonstrated rather than claimed.** The endpoint separates arms the
generation scorer could not distinguish at all:

| arm | KL vs dense | 95% CI | CI half-width |
| --- | ---: | --- | ---: |
| b2-fp-all | −0.00000 | [−0.00001, 0.00001] | 1e-5 |
| b3-int8-all | 0.00088 | [0.00085, 0.00090] | 2.5e-5 |
| b4-int4-all | 0.01049 | [0.01009, 0.01090] | 4.0e-4 |
| b6-static-25 | 0.17822 | [0.16990, 0.18710] | 8.6e-3 |
| b1-delete | 0.28055 | [0.26861, 0.29380] | 1.3e-2 |

INT8 and INT4 are separated by 12× their combined interval width. On the
generation endpoint they were 0.005 apart with an sd of 0.149.

**The noise floor is exactly zero.** `b2-fp-all` — the same weights travelling the
remnant code path — is numerically identical to dense (argmax agreement 1.0000).
On the generation endpoint the same comparison gave 0.00084. So on this endpoint
there is no floor to subtract, and any non-zero divergence is attributable.

### 2.1 Bootstrap implementation

`stats.cluster_bootstrap_mean`, 4,000 draws, seed fixed. Computed in closed form
from per-cluster sums and counts rather than by resampling index tensors:

```
mean of a cluster resample = Σ(drawn cluster sums) / Σ(drawn cluster sizes)
```

Verified **bit-identical** to the general `cluster_bootstrap` — point, lo and hi
all agree to 0.0 — and 15× faster. `verify_csgqr.py` also asserts that clustering
by prompt genuinely **widens** the interval versus treating positions as
independent, so the clustering cannot silently become a no-op.

Paired contrasts form the per-position difference **before** resampling
(`cluster_bootstrap_mean_pair`). Resampling the two arms independently would
discard the pairing and inflate the interval substantially, since two arms agree
on most tokens.

---

## 3. Metric choice, and one metric that must not be primary

Three quantities are reported per position, and they answer different questions.

| metric | question | status |
| --- | --- | --- |
| `KL(dense ‖ arm)` | fidelity to the unmodified model | **primary** |
| `gold_top1` | does the arm predict the reference reply token | **primary** |
| `d_nll` | probabilistic prediction of the gold token | **reported, not relied on** |

**Why `d_nll` is demoted, with the evidence.** The wrong-layer donor arm scores
`d_nll = −0.08685` — *better* than dense — while its `gold_top1` is
**significantly worse** (−0.02689, CI [−0.03482, −0.01883]) and its KL is
0.78649. NLL is a proper scoring rule, so this is not an error: injecting a wrong
MLP's output flattens an over-confident distribution, and on a model that is right
only 28.2% of the time, hedging lowers average NLL while losing argmax accuracy.

Reporting `d_nll` as "quality" would have made the wrong-layer control look like
an *improvement*. It is reported alongside, and the confound is stated where the
number appears.

**And a reframing that the metrics force.** `gold_top1` is 0.2822 for dense and
**0.2822** for full deletion — ΔgoldTop1 = +0.00000, CI [−0.00751, +0.00758] —
while KL is 0.28 nats and 88.7% of replies flip an argmax. At this layer the MLP
contributes model-specific behaviour and nothing measurable in
reference-predictive accuracy. So the Recovery Ratio, defined on quality, has a
denominator of approximately zero here, and what the method recovers is
**fidelity, not quality**. Recovery is therefore reported on KL, with the
denominator stated.

---

## 4. Sequence level, because token agreement is not sequence quality

Reported per prompt alongside the token-level means:

| arm | token agreement | replies with ≥1 argmax flip | mean first flip |
| --- | ---: | ---: | ---: |
| b3-int8-all | 0.9883 | 0.1063 | 4.09 |
| b4-int4-all | 0.9439 | 0.3961 | 3.49 |
| b6-static-25 | 0.7851 | 0.8382 | — |
| b1-delete | 0.7342 | 0.8865 | 2.07 |
| b13-wrong-layer | 0.5898 | 0.9686 | 1.27 |

INT4-all agrees with dense on 94.4% of tokens and still changes **39.6%** of
replies. Divergence begins around token 2–4 against a median reply length of 7, so
most of a divergent reply is generated *after* the divergence. Any claim resting
on token-level agreement alone would be overstating the case, which is why both
are always reported.

---

## 5. Multiple comparisons

`analyze_arms.py` applies **Holm–Bonferroni** across a contrast list fixed in
source (`CONTRASTS`) before any number was seen. Nine planned contrasts on the
generation endpoint; none survived.

For the teacher-forced endpoint the contrasts are all-arms-versus-one-baseline
within a batch, and are reported as intervals rather than as a family of
p-values. Where an ordering is claimed — significance beats random, reversed is
worse than random — the claim rests on **non-overlapping 95% intervals**, which is
conservative relative to a paired test and does not require a correction to be
interpreted.

Adding a contrast after seeing the results would invalidate the correction. None
was added.

---

## 6. Tests used, and why each

| test | where | why |
| --- | --- | --- |
| prompt-clustered percentile bootstrap | every interval on the primary endpoint | positions within a prompt are dependent |
| paired cluster bootstrap | every arm-vs-baseline contrast | pairing is free from the design and strictly tighter |
| exact McNemar | paired binary `quality_pass` on the generation endpoint | exact, and the outcome is paired binary |
| Wilcoxon signed-rank | paired per-case quality deltas | the delta distribution is discrete and heavy at zero (55.5% ties), so a t-test's normality assumption is poor |
| Holm–Bonferroni | across the fixed contrast list | step-down, no independence assumption |
| Spearman / Kendall τ-b | confidence-vs-utility, per-token ranking agreement | the relationships are monotone but not linear, and quantization creates exact ties that need average-rank handling |
| exact binomial | inside McNemar | small discordant counts |

All hand-rolled in `stats.py` so results do not depend on scipy being installed,
with tie handling visible in the source. Kendall τ-b is O(n²) but n = 24 groups.

---

## 7. Where the inference is weak

**7.1 One seed.** The trainable components are closed-form ridge or deterministic
full-batch L-BFGS from a fixed zero initialisation, so there is no optimiser seed
to vary and a rerun is bit-identical. The brief asks for three seeds where
feasible; for anything with genuine stochasticity, that has not been done. The
`random` selector *is* stochastic and is averaged over 200 draws in the offline
analysis, but the generation and teacher-forced arms use a single seeded
realisation.

**7.2 The probes are fitted on 394 positions.** 48/19/29 prompts →
394/150/254 positions. The need probe has 1,025 parameters against 394 training
rows. The result rests on its controls, not its n:

| control | AUROC | reads as |
| --- | ---: | --- |
| permuted labels | 0.5065 | no pipeline leakage |
| reply position only | 0.601 | not merely a counter |
| prompt-mean **oracle** | 0.6976 | ceiling on any prompt-level signal |
| **probe on the state** | **0.9358** | clears the oracle → resolves tokens |

λ chosen on a disjoint validation split (val 0.9313, test 0.9358), so the test
figure is not selected on.

**7.3 `δ = 0.1` was chosen from the data.** `P(U > 0) = 0.9987`, so `δ = 0` is
non-discriminative and any binary need label at `δ = 0` is degenerate. `δ = 0.1`
gives a base rate of 0.62. That choice was made on the calibration trace, and its
sensitivity has **not** been swept.

**7.4 A proxy metric that is not a predictor — the campaign's sharpest statistical
lesson.** `capture_at_k` scores a group selection by **summing singleton marginal
utilities**, which assumes utility is additive over groups. Additivity was
verified at `k = 2` (pairwise interaction 2.1% of `|U(A)|+|U(B)|`) and at
`k = 24` (sum within 1.7% of the whole-remnant utility) — and **assumed** in
between. Measured against true utility on held-out prompts:

| selection | true `U` | proxy | ratio |
| --- | ---: | ---: | ---: |
| random | 0.06492 | 0.07398 | 1.14× |
| static significance | 0.10234 | 0.16097 | 1.57× |
| dynamic (probe) | 0.09802 | 0.39209 | **4.00×** |

The proxy's per-token *oracle* at `k = 12` is 0.63376 — larger than the true
utility of **all 24 groups** (0.27006). Per-token cherry-picking exploits
non-additivity, so the more tailored a selection is, the more the additive sum
overstates it. The "probe wins 66–73% of the static-to-oracle headroom" result is
an artefact of the metric, and the direct measurement shows the dynamic router
buys nothing over a fixed ranking.

The general lesson: **a surrogate objective validated at the endpoints of its
range can be badly non-predictive in the middle, and the bias can be
selection-dependent** — so it flatters exactly the method that optimises it
hardest.

**7.5 One layer, one checkpoint, one split, short replies.** Layer 20 was chosen
by rule (median deletion damage over a committed 28-layer scan), not by
preference. It is still one layer of one 0.47B model, on `chat_dev` only, at
`max_new_tokens = 32` with a median reply of 7 tokens. `rag_dev` was not swept.

**7.6 Not an independent audit.** The implementation was written as part of the
campaign; see `LIMITATIONS.md` §4. The statistically relevant compensation is that
the contrast list, the adversarial controls and the analytic ceilings were all
fixed in source before any result was inspected.
