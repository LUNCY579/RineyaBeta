# CSGQR — Negative Results

Findings that did not go the method's way. Preserved because they are the
expensive part of the knowledge, and because two of them are structural rather
than empirical — they cannot be fixed by better engineering and would waste the
next person's time if omitted.

Every number here is checkable by
`.venv/Scripts/python.exe experiments/csgqr/src/verify_claims.py`.

---

## N1. The exact-confidence gate cannot pay for itself. Structural.

`results/budget.json`, exact arithmetic on the checkpoint's dimensions.

The LM-head projection needed for intermediate confidence costs `2·|V|·d_model`
MACs. The MLP it is deciding about costs `2·3·d_model·d_ff`. So the ratio is

```
|V| / (3 · d_ff)  =  32768 / 9216  =  3.556×
```

A gate that skipped the remnant on **every single token** would still cost
3.56× more than it saved. No threshold, calibration, lens, or feature choice
changes this — it is a property of the shapes.

The viability condition is `|V| < 3·d_ff`. On this checkpoint `d_ff` would have
to exceed 10,923 instead of 3,072. For current production models the inequality
is worse, not better: `|V|` ≈ 128–256k against `d_ff` ≈ 8–28k.

**Consequence.** The "Version A exact probability gate" in the design is viable
only as an *offline labelling teacher*, never as a deployed gate at single-layer
scope. This was known qualitatively from the early-exit literature; the closed
form makes it quantitative and settles it.

## N2. The single-layer systems ceiling is below measurement noise. Structural.

One MLP layer is **1.99%** of this model (9,437,184 of 474,021,888 parameters).

| remnant format | saved vs bf16 | as share of model VRAM |
| --- | ---: | ---: |
| int8-asym | 8.79 MiB | 0.97% |
| int4-asym | 13.29 MiB | **1.47%** |
| int2-asym | 15.54 MiB | 1.72% |

Routing is capped just as low: even `ρ = 0` — never running the remnant at all —
removes only **1.99%** of model MACs per token. At `ρ = 0.25` it is 1.49%.

**Consequence.** At single-layer scope no routing policy can produce an
end-to-end systems win that clears allocator and clock variation on a consumer
GPU. This is why Gate E is assessed against the per-layer microbenchmark and the
analytic ceiling rather than against a whole-model speedup, and why "CSGQR is
faster" is not a claim this campaign can make. The method's systems case requires
many retired layers, which was not run.

## N3. H1 as specified is falsified: intermediate confidence does not predict remnant need.

Layer 20, 798 reply positions from 96 prompts, `δ = 0.1`
(`results/trace_L20_int4/confidence_analysis_d0.1.json`).

H1 predicts `E[U | C low] − E[U | C high] > 0`. Measured, against a mean utility
of 0.303:

| feature | gap (low − high) | AUROC | Pearson corr(1−C, U) |
| --- | ---: | ---: | ---: |
| `max_prob` | **−0.00115** | 0.552 | +0.017 |
| `neg_entropy` | +0.00892 | 0.565 | +0.003 |
| `prob_margin` | +0.00543 | 0.547 | +0.019 |
| `logit_margin` | +0.02393 | 0.537 | −0.015 |

`max_prob` — the primary specified feature — has the **wrong sign**. The largest
gap is 8% of the mean utility, and every cluster-bootstrap interval overlaps
heavily (`max_prob` low `[0.287, 0.397]` vs high `[0.235, 0.463]`). AUROC 0.54–0.57
is not deployable.

**Why**, from the same trace: the raw logit lens's top-1 token agrees with the
dense model's top-1 only **26.9%** of the time at this depth. The lens is not
measuring the decision it is supposed to gate. `ECE(max_prob) = 0.356`.

This is the failure mode the early-exit literature reports for untrained
intermediate readouts, reproduced here with the mechanism visible.

*Scope of the claim:* falsified for the **raw** (rank-0) lens. A trained
alignment lens was not run, so that variant is NOT ESTABLISHED rather than
falsified — but see N4, which makes training the lens the wrong repair.

## N4. Confidently-wrong skips are frequent at every useful threshold.

Same trace. A skip is "confidently wrong" when confidence exceeds the threshold,
the deleted model disagrees with the dense teacher, **and** the full remnant
would have repaired it.

| `τ_skip` (max_prob) | skip rate | P(skip \| needed) | confidently wrong |
| ---: | ---: | ---: | ---: |
| 0.90 | 0.254 | 0.221 | 37 |
| 0.95 | 0.186 | 0.162 | 25 |
| 0.99 | 0.105 | 0.105 | 15 |

Even at `τ = 0.99` — skipping only a tenth of tokens — 15 of 798 positions
(1.9%) are catastrophic false skips. For context, the deleted model disagrees
with dense on 24.8% of tokens and the remnant repairs 85.9% of those, so the
remnant is genuinely load-bearing; confidence simply cannot find which tokens
need it.

## N5. The remnant is needed almost everywhere, so "whether" is barely a question.

`P(U > 0) = 0.9987` over 798 positions. The utility distribution is
overwhelmingly positive (q01 = +0.0002, median +0.144, min −0.0157).

**Consequence.** At a layer with real deletion damage, "does this token need the
remnant?" is not a discriminative binary decision — the answer is essentially
always yes. Any binary skip gate loses utility by construction. This is why the
labelling threshold had to be raised to `δ = 0.1` (base rate 0.62) to make AUROC
meaningful at all, and why the useful framing is *how much* rather than
*whether*.

It also means a reported "invocation rate" near 100% is the correct behaviour
here, not a gate collapse to be tuned away.

## N6. A static global significance ranking captures only a third of what is available.

`results/trace_L20_int4/significance_analysis.json`.

| ρ | static ranking | per-token oracle | headroom unclaimed |
| ---: | ---: | ---: | ---: |
| 0.125 | 0.1037 | 0.3285 | 68.4% |
| 0.25 | 0.1610 | 0.4831 | **66.7%** |
| 0.50 | 0.2454 | 0.6338 | 61.3% |

Per-token group orderings agree with the global ranking at mean Spearman
**0.070** (median 0.087, p05 −0.29). Which groups matter is strongly
token-specific.

This is negative *for the static form of the method* — a fixed global ranking is
far from sufficient — while being the positive case for the dynamic form. See
F5 in `FINDINGS.md`.

## N7. Per-group α calibration is worse than a single scalar α.

Per-group least-squares solves each group's 1-D problem independently and so
ignores cross-group correlation; it is not the optimum for the sum. Measured on a
synthetic gated MLP: scalar α gives relative error 0.1148, per-group 0.1246,
uncalibrated 0.1148.

Recorded so the per-group form is not mistaken for a refinement later. Also note
α ≈ 0.998 for asymmetric INT4 here, so the gain buys almost nothing at this
bitwidth — the shrinkage it exists to correct is small once zero-point
quantization is used.

## N8. cuBLAS makes an arithmetically identical path shape-dependently non-identical.

The remnant reconstructs the donor's weight matrices bit-identically
(`torch.equal` on all three) and reproduces `nn.Linear` bit-identically at
`[64,1024]`, `[1,1,1024]`, `[1,64,1024]`, `[2,8,1024]` — but differs by
`3.9e-3` max absolute at `[1,22,1024]`. Same weights, same arithmetic, different
bf16 accumulation order because cuBLAS picks a different tile for that shape.

**Consequence for every quality comparison:** `b2-fp-all` (identical weights,
remnant code path) is the correct reference for quantized arms, and `|b0 − b2|`
is a noise floor below which no difference is attributable to CSGQR. The same
confound is on record in `experiments/attention_shape`, where a kernel change
moved 23% of greedy replies while shifting mean quality by +0.00138.

In divergence measurements the floor is zero — the FP arm's KL was **exactly
0.0 at all 28 layers**. It is generation, where one flipped argmax diverges an
entire reply, that exposes it.

## N9. INT4 group-utility ranking is not the whole story that FP would tell.

Not yet measured — flagged as an open negative risk rather than a result. The
rank-stability comparison between FP, INT8 and INT4 group utilities
(`§15` of the brief) requires a trace per bitwidth; only the INT4 trace was
generated. If INT4 reorders which groups matter, the significance ranking would
have to be recomputed per bitwidth, and a ranking measured at FP would be the
wrong one to deploy at INT4.

## N10. Engineering defects found, each of which would have produced a wrong number

1. **`calibrate_alpha` absolute epsilon.** `ε = 1e-8` was 13% of the denominator
   on a small-activation layer, pulling a true α of 0.9979 down to 0.8797 and
   making the calibrated remnant *worse* than the uncalibrated one. Scale-dependent
   and invisible in source review.
2. **α fitted against the quantized remnant's own output** in a draft of
   `run_eval.py` — returns 1.0 by construction and looks like a working
   calibration.
3. **`remove()` reinstalled a hollow MLP** whose projections had been freed.
4. **Autograd graph over `[positions, 32768]`** because `lm_head.weight`
   requires grad.
5. **Non-ASCII output crashed the Windows cp1252 console** *after* the expensive
   computation, in two scripts.
6. **A 24-group ranking passed to a 1-group arm** raised a permutation error on
   the first arm of the sweep. Silently truncating it would have invented an
   ordering the significance run never measured.

## What is NOT claimed

- Not claimed: that CSGQR improves measured VRAM or latency end-to-end. N2 says
  the ceiling forbids it at this scope.
- Not claimed: that confidence gating works. N3 falsifies it for the raw lens.
- Not claimed: that a trained lens would fail. It was not run — though N1 makes
  it economically pointless at this scope regardless of its accuracy.
- Not claimed: anything about attention remnants, KV-cache behaviour, or
  request-level residency. Not implemented.
- Not claimed: anything about the sealed test sets. Untouched.

---

# Session 2 revisions and additions — 2026-08-24

Measured on the **teacher-forced paired endpoint**, 414 held-out `chat_dev`
prompts / 3,830 reply positions, prompt-clustered intervals. Several entries above
were written from the generation endpoint, which turned out to have no power here
(N11), and three of them are **revised** below rather than left standing.

## Revisions to entries above

### N3 revised — H1 is falsified as a classifier, but H2 is SUPPORTED

N3 concluded that intermediate confidence does not predict remnant need, from
AUROC 0.54–0.57 against a binary `1[U > 0.1]` label, and the campaign treated
that as "confidence gating does not work". **The first half stands; the conclusion
was too strong.**

Against a matched **random-token** gate at the same activation, every gate wins
with non-overlapping 95% intervals:

| activation | gate | gate KL | random-token KL | advantage |
| ---: | --- | ---: | ---: | ---: |
| 0.8702 | logit-margin | 0.02988 | 0.04778 | 1.60× |
| 0.8356 | prob-margin | 0.03410 | 0.05701 | 1.67× |
| 0.7889 | max_prob | 0.04212 | 0.06572 | 1.56× |
| 0.6772 | probe | 0.05703 | 0.09469 | 1.66× |
| 0.6494 | entropy | 0.07128 | 0.09975 | 1.40× |

Both hold at once because they are different questions. Spearman(1−C, U) is small
but *consistently positive* across all four features (+0.1104 to +0.1313) while
Pearson is ≈ 0. A weak monotone signal on a heavily skewed utility distribution is
a poor binary classifier and a useful **compute allocator**.

**The matched control did not exist before this session.** Every group-level arm
had a random partner; the token-level gate did not, so H2 was untestable and the
gap was filled with an inference from the classifier metric. That was the error.

### N7 withdrawn — measured with the wrong estimator

N7 recorded "per-group α calibration is worse than a single scalar α"
(relative error 0.1246 vs 0.1148). That was measured with an estimator that fits
each group against the **whole** MLP output, which is inflated by the cross-group
inner product and does not return 1 even for a lossless remnant — on a bit-exact
remnant it gave gains spanning 0.978–1.024 and injected 1.32% relative error into
an exact reconstruction (`BUG_LEDGER.md` **S2**).

**It is not evidence about per-group α.** The scalar form is still used, on the
different and better ground that α ≈ 0.997 at asymmetric INT4 makes the whole
question inert at this bitwidth.

### N6 revised — the headroom it reported was a metric artefact

N6 recorded that a static global ranking captures only a third of the per-token
oracle's utility (66.7% unclaimed at ρ=0.25), and treated that as the positive
case for the dynamic branch. **The headroom does not exist.** See N14.

## New negative results

### N11. The repository's generation endpoint cannot resolve this question. Structural for this model scale.

Paired sd of the per-case quality delta is **0.14901** against a deletion effect
of **0.01135** — a factor of 13. Deletion changed **94.5%** of replies while
**55.5%** of cases scored *exactly* the same, because the rubric moves in discrete
steps (p10 = median = 0.720).

Required n for 80% power: **1,352** for deletion damage, **2,292** for the
static-vs-random gap, **10,892** for typical arm-vs-arm gaps of ~0.004. `chat_dev`
has 510.

After Holm–Bonferroni across nine pre-registered contrasts, **not one was
significant — including deletion damage itself** (CI [−0.02427, +0.00195]).

**Consequence.** The entire Tier-1 sweep measured its own instrument. Its arm
ordering — INT4-all above dense, static-25 below deletion — is noise, and both
"anomalies" recorded in `REPOSITORY_STATE_AUDIT.md` §12 are resolved as such. No
amount of GPU time on that endpoint would have helped. Any single-layer
compression study using a coarse rubric on a few hundred generations is measuring
its instrument, and that generalises beyond this campaign.

### N12. Deleting this MLP costs zero reference-prediction accuracy

`gold_top1` is **0.2822** for dense and **0.2822** for full deletion —
Δ = **+0.00000**, CI [−0.00751, +0.00758] — while KL is 0.28 nats, argmax
disagreement 26.6%, and 88.7% of replies change. Absolute NLL on the gold token
moves 0.1%.

**Consequence.** The Recovery Ratio as the brief defines it, on quality, has a
denominator of approximately **zero** at this layer. There is no quality damage to
recover; what the method recovers is fidelity to the original model's behaviour.
That is a legitimate objective but it is not the one the metric assumes, and
reporting a fidelity result as a quality result would be a category error.

### N13. The per-state dynamic router works and buys nothing

| ρ | static | dynamic | verdict |
| ---: | ---: | ---: | --- |
| 0.25 | **0.17822** | 0.18253 | static ahead, overlapping |
| 0.50 | 0.13565 | **0.12943** | dynamic ahead, overlapping |

Neither difference is significant and the sign flips with budget. The router is
**not** degenerate: its per-group selection rates run 0.548 → 0.018 across all 24
groups, against `random`'s flat 0.246–0.254. It is genuinely state-conditioned,
clearly better than random and magnitude, and no better than a fixed ranking.

### N14. The additive utility proxy overstates by 4× for exactly the method that optimises it

`capture_at_k` sums singleton marginal utilities. Additivity was verified at
`k = 2` (interaction 2.1%) and `k = 24` (sum within 1.7% of whole-remnant utility)
and **assumed in between**. Against true utility at ρ=0.25:

| selection | true `U` | proxy | overstatement |
| --- | ---: | ---: | ---: |
| random | 0.06492 | 0.07398 | 1.14× |
| static significance | 0.10234 | 0.16097 | 1.57× |
| dynamic (probe) | 0.09802 | 0.39209 | **4.00×** |

And the proxy's per-token oracle at `k = 12` is **0.63376**, larger than the true
utility of **all 24 groups** (0.27006) — impossible for a real utility.

Per-token cherry-picking exploits non-additivity, so the bias grows with how
tailored the selection is: **the proxy flatters precisely the method that
optimises it hardest**. H8 is falsified for intermediate `k`.

The transferable lesson: a surrogate objective validated at the *ends* of its range
can be badly non-predictive in the middle, with a selection-dependent bias.

### N15. No interior operating point is useful

| arm | active | skip | full | partial | KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| binary probe gate | 0.6772 | 0.3228 | 0.6772 | 0.0000 | **0.05703** |
| three-level, dynamic | 0.6819 | 0.1745 | 0.5218 | 0.3037 | 0.05839 |

Interior points exist across the whole frontier (0 < ρ < 1 everywhere), and none
beats a binary all-or-nothing gate at matched compute. **Gate E is half-passed:
the interior exists and is not worth occupying.** H6 falsified as "useful".

### N16. The quantized remnant is 5× slower than the layer it replaces

Remnant sub-block alone, 512 tokens: dense `nn.Linear` 0.4649 ms, INT4 all groups
**2.3221 ms** (5.00×), INT8 2.2656 ms (4.87×), FP passthrough 0.6592 ms (1.42×).
At ρ=0.125 INT4 is still 2.39× dense.

Dequantization costs about **3.5×** what the saved matmul buys, and no routing
policy addresses that — it is the price of reconstructing weights every forward.
`REFERENCE_BACKEND_LIMITATION`: a fused dequantize-GEMM is the untested repair, and
H12 is `NOT_ESTABLISHED` rather than supported.

### N17. Peak VRAM goes up, not down

End-to-end peak reserved: dense **956 MiB**, full method **1226 MiB** (+28%).
`b1-delete` matches dense at 956 MiB, so packed storage is not the cause — the
entire increase is **execution transient** (dequantized blocks, per-bucket gathers,
monitor workspace).

What the method saves is persistent (13.3 MiB, donor freeing confirmed at
−12.414 MiB) and what it adds is transient (~200 MiB). Peak is what the card
actually gives up.

### N18. Per-token routing costs 16× in the reference backend

`b5-random-25` took **622 s** against **39 s** for `b1-delete` and ~27 s for
`b6-static-25` over the same 414 prompts. Per-token bucketing issues one
dequantize-and-GEMM per distinct group set, and for a selection that changes every
token the bucket count approaches the token count. Visible end-to-end as the full
method's 1.90× prefill latency.

### N19. Measured resident bytes exceed the analytic model by 32%, unexplained

| format | measured | analytic | ratio |
| --- | ---: | ---: | ---: |
| INT8 | 10.211 MiB | 9.211 MiB | 1.1086 |
| INT4 | 6.211 MiB | 4.711 MiB | **1.3185** |

Investigated and **not** resolved. On a clean allocator the accounting is exact to
the byte (diff +0, no buffer is a view into a larger storage). The excess appears
only with the 904 MiB model resident, and is exactly one third of each `qweight`.
It is **not** allocator power-of-two rounding — a direct 1,572,864-byte allocation
reports exactly that.

No explanation is offered, so the **measured** figure is quoted: INT4's effective
compression against bf16 is **2.90×**, not the accounted 3.82×.

### N20. Latency on a shared GPU varies 3.7×, so absolute timings are not usable

A different experiment (`experiments/kv_significance/src/passkey.py --lengths
131072`) ran concurrently on the same 8 GiB card for most of this session. Across
three runs of the *same* benchmark the dense MLP measured **0.4649, 0.4782 and
1.7218 ms**, and the exact-lens monitor's share moved between **391% and 1396%**.
One run died with `CUBLAS_STATUS_EXECUTION_FAILED`.

Orderings and same-run ratios reproduced across all three runs and are used;
absolute latencies are not quoted. Memory is unaffected (`memory_allocated` tracks
this process only). Recorded because a latency number taken opportunistically
under contention would have looked exactly like evidence.

### N21. The probe's cost advantage is ~110×, not ~32,000×

The campaign repeatedly quoted "the need probe costs 0.0109% of the MLP it gates,
against 355.6% for the exact lens". Those are **FLOP and byte ratios** and are
correct as such. Measured in wall-clock per 512 tokens: need probe **12.50%**,
group router 12.28%, exact lens + features **1395.8%**.

A `[512, 1024] × [1024, 1]` GEMM is small enough to be launch-bound, so its MAC
share understates its latency by nearly three orders of magnitude. The deployment
argument survives at **~110×**; the magnitude was a FLOPs claim presented as a
latency claim.

Also: the closed form predicts the *projection* well (3.04× measured against
3.556× predicted) and understates the **full** monitor by 4.6×, because the fp32
log-softmax, entropy and top-2 over 32,768 classes is memory-bound and costs 3.6×
the projection itself. A design budgeting for the projection alone under-budgets
by 4.6×.

### N22. Two of three mechanisms have close prior art

`NOVELTY_AND_RELATED_WORK.md`. Deja Vu (ICML 2023) already trains a predictor on
the pre-block hidden state to select which MLP neurons to compute per token, at
85% structured sparsity and 2× speedup. QTALE (2026) already gates token-adaptive
layer execution on confidence, studied for quantization robustness. The
residual-gate line (Laitenberger et al. 2025) is a learned residual readout gating
branches per token — structurally the need probe.

What survives as distinct is the **interventional** utility label (a KL-difference
obtained by intervening, not an activation-magnitude proxy) applied to a
**structurally retired** component, and the donor-identity control. The mechanism
is a recombination.

### N23. Claims withdrawn after larger samples

Recorded because the pattern matters more than the individual items. Each was
first observed at n=8 or n=24 prompts and did not survive n=414:

* "static top-k reconstructs `F` better than random yet produces a worse
  next-token distribution" — a sign inversion. **Withdrawn**: on held-out prompts
  the ordering is ordinary (dynamic 0.137 < static 0.180 < random 0.213 at n=24;
  static < dynamic at n=414). The n=8 KL and the corroborating generation figure
  were both noise.
* "dynamic significance beats static at matched budget" — **withdrawn** (N13).
* "the three-level interior point beats the binary gate" — **withdrawn** (N15).

**Every claim that favoured the method at small n dissolved at large n, while
every claim that favoured the controls survived.** That asymmetry is what sample-size
discipline looks like from the inside, and it is the reason the held-out design
with `--skip-prompts` enforcement exists.
