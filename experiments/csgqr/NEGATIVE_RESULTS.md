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
