# CSGQR — Findings

All quality/fidelity numbers below are from the **teacher-forced paired endpoint**
on **414 held-out `chat_dev` prompts / 3,830 reply positions**, disjoint from the
96 prompts the probes were fitted from, with prompt-clustered bootstrap intervals.
Layer 20, INT4 asymmetric block-128, 24 groups unless stated.

`Recovery` is on KL, not quality, and the reason is F1.

> **Why not the generation scorer.** It has no power here: paired sd 0.14901
> against a deletion effect of 0.01135, and after Holm–Bonferroni **none** of nine
> pre-registered contrasts was significant, including deletion damage itself.
> Required n is ~1,350 for deletion and ~10,900 for arm-vs-arm gaps; 510 were
> available. See `STATISTICAL_ANALYSIS.md` §1. The Tier-1 table is retained as
> provenance and is used as evidence nowhere.

---

## F1. Deleting this MLP costs no measurable quality — only fidelity

| arm | KL vs dense | gold-token top-1 | Δ gold vs dense |
| --- | ---: | ---: | --- |
| b0-dense | 0.00000 | 0.2822 | — |
| b1-delete | 0.28055 | **0.2822** | **+0.00000**, CI [−0.00751, +0.00758] |

Deleting layer 20's MLP entirely diverges **0.28 nats** from the dense model,
disagrees with its argmax on **26.6%** of tokens, and changes **88.7%** of
replies — while predicting the reference reply token *exactly as well as dense*.
Absolute NLL on the gold token moves 0.1% (4.81435 vs 4.80959).

**This reframes the whole method.** The Recovery Ratio as the brief defines it —
on quality — has a denominator of approximately **zero** at this layer. There is
no quality damage to recover. Everything CSGQR recovers is *fidelity to the
original model's own behaviour*.

That is a coherent and useful objective (it is what you want when replacing a
deployed model with a compressed one), but it is not the objective the brief's
Recovery Ratio assumes, and conflating them would let a fidelity result be
reported as a quality result. It also explains cleanly why the generation scorer
saw nothing: there was nothing to see.

---

## F2. Precision is monotone, and the noise floor is exactly zero

| arm | KL vs dense | 95% CI | recovery | argmax agree |
| --- | ---: | --- | ---: | ---: |
| b2-fp-all | **−0.00000** | [−0.00001, 0.00001] | 1.0000 | **1.0000** |
| b3-int8-all | 0.00088 | [0.00085, 0.00090] | 0.9969 | 0.9883 |
| b4-int4-all | 0.01049 | [0.01009, 0.01090] | 0.9626 | 0.9439 |
| b1-delete | 0.28055 | [0.26861, 0.29380] | 0.0000 | 0.7342 |

The FP donor travelling the remnant code path is **numerically identical to
dense**, so `|b0 − b2|` is zero on this endpoint and any divergence is
attributable. INT8 recovers **99.7%** of the deletion damage, INT4 **96.3%**, with
non-overlapping intervals 12× narrower than their separation.

**Gate B passes decisively.** And the Tier-1 anomaly where INT4 appeared to beat
INT8 *and* dense is fully accounted for as scorer noise.

Note the token/sequence gap: INT4-all agrees with dense on 94.4% of tokens and
still changes **39.6%** of replies. Divergence begins around token 3.5 against a
median reply of 7, so most of a divergent reply is generated after the divergence.

---

## F3. Donor identity matters, and the wrong donor is worse than none

| arm | KL | recovery | Δ gold top-1 |
| --- | ---: | ---: | --- |
| b4-int4-all (correct donor) | 0.01049 | 0.9626 | −0.00261, ns |
| b1-delete (no donor) | 0.28055 | 0.0000 | +0.00000, ns |
| **b13-wrong-layer** (L26 donor at L20) | **0.78649** | **−1.8034** | **−0.02689, CI [−0.03482, −0.01883], SIG** |

A frozen INT4 copy of a *different* layer, at an identical byte budget, is **75×**
worse than the correct donor and **2.8×** worse than installing nothing. It is
also the **only** arm in the entire campaign whose reference-prediction accuracy
degrades significantly.

So the preserved remnant is doing donor-specific work, not supplying generic
capacity. **H4 is supported.** This is the strongest positive result in the
campaign, and it is the one that justifies "retain the retired component" over
"add an equivalent number of fresh bytes".

*Caveat:* the matched same-byte **LoRA** control (B21) and random-frozen-weights
control (B20) were not run, so "beats generic trainable capacity" remains
untested. What is established is "beats a wrong frozen donor of identical size".

---

## F4. Confidence gating works as a compute allocator — and fails as a need classifier

Both are true, and the campaign had previously recorded only the second.

**As a classifier of "does this token need the remnant?", it fails.** Against
`1[U > 0.1]`: AUROC 0.5374–0.5649 across four features; `max_prob` — the primary
specified feature — has the **wrong sign** on the mean-utility gap (−0.00115); ECE
0.3557. The raw lens's top-1 agrees with dense's only **26.94%** of the time, so
it is scoring a different decision. Even at `τ = 0.99`, skipping one token in ten,
15 of 798 positions are confidently-wrong skips. See
`CONFIDENTLY_WRONG_ANALYSIS.md`.

**As a way to decide where to spend compute, it works.** Against a matched
random-token gate at the same activation:

| activation | gate | gate KL | random-token KL | advantage |
| ---: | --- | ---: | ---: | ---: |
| 0.8702 | logit-margin | 0.02988 | 0.04778 | **1.60×** |
| 0.8356 | prob-margin | 0.03410 | 0.05701 | **1.67×** |
| 0.7889 | max_prob | 0.04212 | 0.06572 | **1.56×** |
| 0.6772 | **probe** | 0.05703 | 0.09469 | **1.66×** |
| 0.6494 | entropy | 0.07128 | 0.09975 | **1.40×** |

Every gate beats random skipping at matched compute, with **non-overlapping 95%
intervals** at every point. **H2 is supported.**

**Why both hold.** Spearman(1−C, U) is small but *consistently positive* across
all four features (+0.1104 to +0.1313) while Pearson is ≈ 0. A weak monotone
signal on a heavily skewed utility distribution is a poor binary classifier and a
useful allocator. The earlier conclusion "confidence gating does not work" was
drawn from the classifier metric and was too strong.

---

## F5. The probe matches the best lens on skill and is ~110× cheaper — but only below ~78% activation

Full frontiers, six points each (`results/forced_heldout_E_frontier.json`).
Interpolated to matched activation:

| activation | probe KL | lens KL @ same | probe advantage | random-token @ same |
| ---: | ---: | ---: | ---: | ---: |
| 0.504 | 0.08456 | 0.10988 | **−23.0%** | — |
| 0.604 | 0.07038 | 0.08554 | **−17.7%** | 0.11276 |
| 0.677 | 0.05703 | 0.06611 | **−13.7%** | 0.09471 |
| 0.754 | 0.04755 | 0.04924 | −3.4% | 0.07586 |
| 0.826 | 0.03668 | 0.03582 | +2.4% | 0.05654 |
| 0.894 | 0.02724 | 0.02493 | +9.3% | 0.03792 |

The curves **cross at ≈ 0.78 activation**. The probe dominates in the
aggressive-skipping regime — the regime where compute is actually being saved —
and the lens is marginally better when barely skipping at all. Both beat random
by 25–40% everywhere.

Cost, **measured in wall-clock** (not MACs) per 512 tokens against the dense MLP:

| monitor | measured share | MAC share |
| --- | ---: | ---: |
| need probe | **12.50%** | 0.0109% |
| group router | 12.28% | 0.2604% |
| exact lens + features | **1395.8%** | 355.6% |

**H5 is supported**: the probe retains the routing quality (better below 78%
activation, within 9% above it) at **~110×** lower monitor cost. But the widely
repeated "four orders of magnitude cheaper" figure is a FLOPs ratio and is
**withdrawn as a latency claim** — see `GPU_BENCHMARKS.md` §2.

And the exact lens remains structurally unaffordable regardless of skill:
`|V|/(3·d_ff) = 3.556×` the gated MLP for the projection alone, 1395.8% measured
with the feature reduction included. A gate that skipped on *every* token could
not pay for itself.

---

## F6. Interventional significance routing works; the magnitude heuristic is worse than random

Matched budget, ρ = 0.25 (6 of 24 groups), so the only difference is *which*
groups run:

| selection | KL | 95% CI | recovery |
| --- | ---: | --- | ---: |
| **static significance** | **0.17822** | [0.16990, 0.18710] | 0.3648 |
| dynamic (per-state probe) | 0.18253 | [0.17356, 0.19252] | 0.3494 |
| random | 0.21563 | [0.20570, 0.22695] | 0.2314 |
| **magnitude** | 0.23719 | [0.22653, 0.24901] | 0.1546 |
| reversed significance | 0.24338 | [0.23270, 0.25507] | 0.1325 |
| deletion (ρ=0) | 0.28055 | [0.26861, 0.29380] | 0.0000 |

**H3 is supported, with the full control set behaving as the method predicts:**

* significance beats random with **non-overlapping** intervals;
* **reversing** the ranking is worse than random — the *order* carries the
  information, not merely the identity of the selected set;
* the **magnitude** heuristic is *also worse than random* — so an interventional
  KL-difference label carries information that a weight-norm proxy does not.

Static significance is additionally the only ρ=0.25 arm whose gold-token accuracy
does not degrade (+0.00104 ns, against random −0.00574 SIG and reversed −0.01070
SIG).

At ρ=0.50 the ordering is preserved: dynamic 0.12943 < static 0.13565 < random
0.14763.

---

## F7. Per-state dynamic routing is genuinely dynamic and buys nothing

| ρ | static | dynamic | verdict |
| ---: | ---: | ---: | --- |
| 0.25 | **0.17822** | 0.18253 | static ahead, intervals overlap |
| 0.50 | 0.13565 | **0.12943** | dynamic ahead, intervals overlap |

Neither difference is significant, and the sign flips with budget. The router is
**not** degenerate — its per-group selection rates run from 0.548 down to 0.018
across all 24 groups, against `random`'s flat 0.246–0.254 and `static`'s hard
1.0/0.0. It is genuinely state-conditioned, genuinely informative (it beats random
and magnitude clearly), and no better than a fixed global ranking.

**The dynamic branch is `NOT_ESTABLISHED`.**

### F7.1 And the offline analysis that promised otherwise was measuring an artefact

`capture_at_k` scores a selection by **summing singleton marginal utilities** —
i.e. assuming utility is additive over groups. Additivity was verified at `k = 2`
(pairwise interaction 2.1%) and `k = 24` (sum within 1.7% of whole-remnant
utility) and **assumed in between**. Against true utility at ρ=0.25:

| selection | true `U` | additive proxy | overstatement |
| --- | ---: | ---: | ---: |
| random | 0.06492 | 0.07398 | 1.14× |
| static significance | 0.10234 | 0.16097 | 1.57× |
| dynamic (probe) | 0.09802 | 0.39209 | **4.00×** |

The proxy's per-token *oracle* at `k = 12` is **0.63376** — larger than the true
utility of **all 24 groups** (0.27006), which is impossible for a real utility.

**Per-token cherry-picking exploits non-additivity, so the more tailored a
selection is, the more the additive sum overstates it — the proxy flatters
precisely the method that optimises it hardest.** The prior "the probe wins 66–73%
of the static-to-oracle headroom" result is an artefact, and **H8 is falsified for
intermediate `k`** even though it holds at both endpoints.

This is the campaign's most transferable methodological finding: a surrogate
objective validated at the *ends* of its range can be badly non-predictive in the
middle, with a bias that depends on the selection being scored.

---

## F8. The three-level interior policy buys nothing over a binary gate

| arm | active | skip | full | partial | KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| b12-probe-whole (binary) | 0.6772 | 0.3228 | 0.6772 | 0.0000 | **0.05703** |
| b16-csgqr-dyn (three-level) | 0.6819 | 0.1745 | 0.5218 | 0.3037 | 0.05839 |

The full method reaches a genuine interior operating point — 17.5% skipped, 52.2%
full, 30.3% partial — and lands slightly *worse* for slightly *more* compute than
simply running the whole remnant or nothing. Overlapping intervals; a tie.

Same at the lens gate: b10-csgqr (three-level, static) 0.09732 and
b16-csgqr-lens-dyn (three-level, dynamic) 0.09824 both at 0.5247 active, against
b8-conf-random 0.10874 — so at the three-level operating point the *group
selector* separates the arms by only ~10% while the *gate* choice moves KL by 2×.

**Gate E is therefore only half-passed.** A nontrivial interior activation rate
exists and is reachable (0 < ρ < 1 across the whole frontier), but it is not
*useful*: no interior point beats the binary gate at matched compute.

---

## F9. Hard group skipping is real and fast; the quantized remnant is 5× slower than the layer it replaces

Remnant sub-block alone, 512 tokens (`GPU_BENCHMARKS.md`):

| path | P50 | vs dense MLP |
| --- | ---: | ---: |
| dense `nn.Linear` | 0.4649 ms | 1.00× |
| FP passthrough, ρ=0.25 | **0.2570 ms** | **0.55×** |
| FP passthrough, ρ=1.0 | 0.6592 ms | 1.42× |
| INT4, ρ=0.125 | 1.1105 ms | 2.39× |
| INT4, ρ=1.0 | 2.3221 ms | 5.00× |

**The mechanism is vindicated and the compression is not.** Executed sparsity
delivers a genuine compute reduction — the FP remnant at ρ=0.25 runs in 55% of
the dense MLP's time, monotone in ρ — which confirms that inactive groups really
are never dequantized and never enter a matmul. But dequantization costs ~3.5×
what the saved matmul buys, so *every* quantized operating point is 2.4–5.0×
slower than simply not compressing.

End-to-end, every CSGQR arm is slower than dense and uses **more** peak VRAM
(956 MiB → 1226 MiB for the full method, +28%). `b1-delete` matches dense at
956 MiB, so the packed storage is not the problem: the whole increase is
*execution transient*. What the method saves is persistent (13.3 MiB, confirmed
freed) and what it adds is transient (~200 MiB), and peak is what the card gives
up.

**Gate F fails on measured evidence**, as the pre-registered analytic ceiling
(1.47% of model VRAM, 1.99% of model MACs — both below noise) had already implied.

---

## F10. Two structural results, fixed before measurement

**The exact-confidence monitor cannot pay for itself.**
`|V| / (3·d_ff) = 32768 / 9216 = 3.556×`. A gate skipping on every token would
still cost 3.56× what it saved. Viability needs `|V| < 3·d_ff`; for production
models (`|V|` ≈ 128–256k, `d_ff` ≈ 8–28k) it is worse, not better. Measured: the
projection alone is 3.04× (within contention noise of the closed form) and the
full monitor including the fp32 feature reduction is **13.96×**, so a design that
budgeted for the projection alone would have under-budgeted by 4.6×.

**The single-layer systems ceiling is below measurement noise.** One MLP is 1.99%
of the model. INT4 saves 13.29 MiB = 1.47% of model VRAM; `ρ = 0` removes 1.99% of
model MACs. Measured effective compression is **2.90×**, not the accounted 3.82×,
because measured resident bytes exceed the analytic model by 32% for reasons that
remain unattributed (`GPU_BENCHMARKS.md` §3).

---

## Summary table (44 arms, held-out, sorted by fidelity)

| arm | KL | 95% CI | active | recovery | gold top-1 | seq flip |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| b0-dense | 0.00000 | [0, 0] | — | 1.0000 | 0.2822 | 0.0000 |
| b2-fp-all | −0.00000 | [−1e-5, 1e-5] | 1.000 | 1.0000 | 0.2822 | 0.0000 |
| b3-int8-all | 0.00088 | [0.00085, 0.00090] | 1.000 | 0.9969 | 0.2809 | 0.1063 |
| b4-int4-all | 0.01049 | [0.01009, 0.01090] | 1.000 | 0.9626 | 0.2796 | 0.3961 |
| lens-t0.99 | 0.02070 | [0.01847, 0.02310] | 0.923 | 0.9262 | 0.2825 | 0.4396 |
| probe-t0.80 | 0.02724 | [0.02438, 0.03042] | 0.894 | 0.9029 | 0.2833 | 0.4903 |
| b11-logitmargin-whole | 0.02988 | [0.02709, 0.03276] | 0.870 | 0.8935 | 0.2841 | 0.5097 |
| b10-margin-whole | 0.03410 | [0.03092, 0.03748] | 0.836 | 0.8785 | 0.2841 | 0.5242 |
| probe-t0.70 | 0.03668 | [0.03295, 0.04048] | 0.826 | 0.8693 | 0.2856 | 0.5386 |
| b7-conf-whole | 0.04212 | [0.03844, 0.04596] | 0.789 | 0.8499 | 0.2838 | 0.5580 |
| probe-t0.60 | 0.04755 | [0.04238, 0.05337] | 0.754 | 0.8305 | 0.2859 | 0.5556 |
| b12-probe-whole | 0.05703 | [0.05143, 0.06342] | 0.677 | 0.7967 | 0.2877 | 0.5918 |
| b16-csgqr-dyn | 0.05839 | [0.05259, 0.06487] | 0.682 | 0.7919 | 0.2877 | 0.6256 |
| b9-entropy-whole | 0.07128 | [0.06629, 0.07689] | 0.649 | 0.7459 | 0.2830 | 0.6860 |
| b10-csgqr | 0.09732 | [0.09238, 0.10250] | 0.525 | 0.6531 | 0.2822 | 0.7415 |
| b16-csgqr-lens-dyn | 0.09824 | [0.09313, 0.10367] | 0.525 | 0.6498 | 0.2828 | 0.7319 |
| b8-conf-random | 0.10874 | [0.10200, 0.11699] | 0.525 | 0.6124 | 0.2828 | 0.7754 |
| b9-sig-dyn-50 | 0.12943 | [0.12264, 0.13712] | 0.500 | 0.5387 | 0.2815 | 0.7995 |
| b6-static-50 | 0.13565 | [0.12808, 0.14409] | 0.500 | 0.5165 | 0.2841 | 0.7754 |
| b5-random-50 | 0.14763 | [0.14017, 0.15596] | 0.500 | 0.4738 | 0.2804 | 0.8164 |
| b6-static-25 | 0.17822 | [0.16990, 0.18710] | 0.250 | 0.3648 | 0.2833 | 0.8382 |
| b9-sig-dyn-25 | 0.18253 | [0.17356, 0.19252] | 0.250 | 0.3494 | 0.2768 | 0.8333 |
| b5-random-25 | 0.21563 | [0.20570, 0.22695] | 0.250 | 0.2314 | 0.2765 | 0.8623 |
| b6-magnitude-25 | 0.23719 | [0.22653, 0.24901] | 0.250 | 0.1546 | 0.2828 | 0.8816 |
| b6-reversed-25 | 0.24338 | [0.23270, 0.25507] | 0.250 | 0.1325 | 0.2715 | 0.8623 |
| b1-delete | 0.28055 | [0.26861, 0.29380] | 0.000 | 0.0000 | 0.2822 | 0.8865 |
| b13-wrong-layer | 0.78649 | [0.75297, 0.81941] | 1.000 | **−1.8034** | 0.2554 | 0.9686 |

Random-token controls (`r-tok-*`, `rtok-t*`) and the remaining ladder points are
in `results/forced_heldout_*.json`; the matched comparisons are in F4 and F5.
