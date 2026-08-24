# CSGQR — Method and Code Audit

This is the file the campaign brief names in its required-documentation list. It
is the same audit as `METHOD_AUDIT.md`, which was written in session 1 under the
shorter name and is kept at that path because other documents already cite it.

**Read it together with the files that supersede parts of it**, because session 2
revised several of its conclusions:

* `BUG_LEDGER.md` — 19 defects with a regression test each. Section 3 below lists
  five; fourteen more were found in session 2, four of them `MAJOR` or `CRITICAL`.
* `SPEC_TO_CODE_MATRIX.md` — every requirement classified, replacing the
  intended-versus-implemented table in section 1 below at finer grain.
* Section 6 below reports the raw-lens confidence failure and concludes that
  confidence gating does not work. **That conclusion is revised.** With a matched
  random-token control — which did not exist when section 6 was written — every
  gate beats random skipping at matched compute by 1.40–1.67× with non-overlapping
  intervals. Confidence fails as a needed/not-needed classifier and works as a
  compute allocator. See `NEGATIVE_RESULTS.md` N3 revised.
* Section 4.2 below records "per-group α is worse than scalar α". That was
  measured with an estimator fitting each group against the whole MLP rather than
  against its own donor group, and is **withdrawn** — see `BUG_LEDGER.md` **S2**
  and `NEGATIVE_RESULTS.md` N7.
* Section 6's claim that the probe's cost advantage over the exact lens is four
  orders of magnitude is a FLOP ratio. Measured in wall-clock it is **~110×**.
  See `GPU_BENCHMARKS.md` §2.

Everything else stands, including its section 0 caveat that the implementation
audited here was written during the campaign rather than inherited — which remains
the single largest limitation on all of it.

---

# CSGQR — Method Audit

Intended method, what is actually computed, every discrepancy found, and what was
changed. Written against the design specification supplied for this campaign
(three-level policy, alignment lens, calibration temperature, cost-aware routing,
Policy C), not against the branch's earlier prose.

---

## 0. The audit's own limitation, first

**This branch contained no CSGQR implementation.** Verified across every local and
remote ref, the working tree, `git stash`, the reflog, and dangling objects:
zero files matching `*csgqr*`, zero occurrences of the string `remnant` in any
tracked file on `main`. `research/csgqr-quantized-remnant` pointed at `1664f54`,
a significance-matrix paper commit shared with
`research/pre-deletion-recovery-routing-v0.1` and `-v0.2`, eleven commits behind
`main`. `docs/CSGQR_DESIGN.md`, `scripts/csgqr/`, `scripts/verify/verify_csgqr.py`
and the CSGQR configs did not exist.

`scripts/common/response_control.py` — which the brief instructed me to inspect —
exists only on `main`, which is what identified the branch label as stale rather
than the brief as wrong. `origin/main` was merged in (commit `3c051f1`); no
history was rewritten and no other branch was touched.

**So the implementation audited below was written during this campaign.** That
weakens audit independence in a way testing cannot repair. Two compensations,
both structural rather than promissory:

1. The adversarial controls — random routing, shuffled and reversed significance,
   magnitude ranking, wrong-layer donor, permuted-label probe, prompt-mean
   oracle — are in `variants.py` and `probe_utility.py` and were written before
   any CSGQR result was inspected.
2. The analytic ceilings in `results/budget.json` were computed before any GPU
   run, so the systems verdict is measured against a number fixed in advance.

---

## 1. Intended vs. implemented

| Design element | Intended | Implemented | Status |
| --- | --- | --- | --- |
| Retired component | frozen quantized remnant of a retiring block | `QuantizedMlpRemnant`, packed INT2/3/4/8 or bf16 passthrough | as specified |
| Deleted state | `h' = h` | `mode="none"` — verified bit-identical to a true MLP bypass (0.0 logit deviation) | as specified |
| Full remnant | `h' = h + F^Q(h)` | `mode="all"` | as specified |
| Adaptive | `h' = h + Σ_{g∈A} α_g F_g^Q(h)` | `forward_groups`, slicing before dequantization | as specified |
| Groups | architecture-aligned MLP channel groups after `down_proj` | contiguous slices of the intermediate axis; all three projections stored `[d_ff, d_model]` | as specified |
| Additivity | `F = Σ_g F_g` | **exact**, verified in float64 to `<1e-14` | as specified |
| Confidence | `z̃ = W_LM · N_f(A_j h)` | `ConfidenceHead`, borrows the model's own norm and LM head | as specified |
| Capture point | residual immediately before the remnant | pre-norm post-attention residual, via forward-pre-hook on `post_attention_layernorm` | as specified |
| Alignment lens `A_j` | optional affine / low-rank | `AlignmentLens(rank)`, zero-init so rank>0 starts as exact identity | implemented, **rank 0 only was tested** |
| Features | max prob, normalized entropy, top-2 prob and logit margins, stability | all five implemented; entropy normalized by `log|V|` over the full vocabulary | as specified |
| Calibration `T_j` | fitted on held-out data, separate from generation temperature | `fit_temperature`, golden-section on NLL; module never reads a generation config | as specified |
| Three-level policy | `τ_full < τ_skip`, `ρ = ((τ_skip−C)/(τ_skip−τ_full))^γ`, `k = ⌈ρ·|G|⌉` | `RoutingPolicy.rho` / `.budget` | as specified |
| Cost-aware value | `R_g = (1−C)^γ · Û_g / (c_g+ε)` | `RoutingPolicy.select` | as specified; **all groups are equal-cost on this checkpoint, so `R_g` is a monotone rescaling of `Û_g`** |
| Utility | `U_{t,g} = KL(p^T‖p^0) − KL(p^T‖p^{+g})` | `significance.marginal_utility` | as specified |
| Output gain | `α = E⟨F^Q,F⟩ / (E‖F^Q‖²+ε)` | `calibrate_alpha` | **corrected — see §3.1** |
| Attention remnant | Policy C: MLP gated per token, attention request-level or always-on | **not implemented.** Only the MLP is retired; attention is untouched | **not implemented** |
| Survivor adaptation | Phase 5 LoRA on neighbours | not implemented (Tier 3) | not implemented |

---

## 2. What the code actually computes

For the installed layer `j`, per token `t`:

```
h        = post-attention residual at layer j          (pre-norm)
x        = post_attention_layernorm(h)                 (what Qwen3 feeds its MLP)
C_t      = feature( softmax( W_LM · N_f(A(h_t)) / T ) )
k_t      = ceil( rho(C_t) * |G| )
A_t      = top-k_t groups by  (1-C_t)^gamma * U_g / (c_g + eps)
out_t    = sum_{g in A_t} alpha_g * F_g^Q(x_t)
h_{j+1}  = h + out
```

with `F_g^Q(x) = (silu(x·Ŵ_gate[I_g]ᵀ) ⊙ x·Ŵ_up[I_g]ᵀ) · Ŵ_downᵀ[I_g]`, hats
denoting dequantized-on-demand rows.

Three properties worth stating because they are easy to get wrong and were
tested rather than assumed:

**Group execution is real, not masked.** `PackedMatrix.dequantize(lo, hi)` slices
the packed bytes, the scales and the zero points before any arithmetic. At
`ρ=0.25` the three matmuls are genuinely a quarter-size. Verified: row-slice
dequantization is bit-identical to the corresponding rows of the full
dequantization.

**Confidence is arm-independent, exactly.** Only layer `j`'s MLP changes, so
layers `0..j-1` are identical across dense, deleted and remnant arms; `h_{j,t}`
depends only on those layers applied to positions `≤ t`. Measured: max |Δ| =
**0.000e+00** over 798 positions (`trace_manifest.json:residual_invariance_max_abs`).
This is why the trace computes confidence once.

**Prefill is gated too.** Skipping the remnant at a prefill position changes that
position's hidden state and therefore its keys and values for every layer above
`j`. Gating only decode positions would be a different method.

---

## 3. Bugs found, and what each would have done

### 3.1 `calibrate_alpha` — absolute epsilon (would have silently degraded every quantized arm)

The design writes the denominator as `E‖F^Q(h)‖² + ε`. Implemented literally with
`ε = 1e-8`, this is scale-dependent: MLP output magnitudes depend on the
checkpoint's weight scale, and on a small-activation layer the sums were ~1e-7,
making `ε` **13% of the denominator**. It pulled a true `α = 0.9979` down to
`0.8797` and made the calibrated remnant *worse* than an uncalibrated one
(relative error 0.1148 → 0.1644).

Fixed: the degenerate case is handled as the zero-check it actually is, with no
additive term, and the sums accumulate in float64 because they run over hundreds
of thousands of tokens. `α` now matches the closed form to `<1e-5` at activation
scales spanning 2500×, and never increases reconstruction error.

Found by testing, not by reading. A fixed `ε` looks unimpeachable in source.

### 3.2 `InstalledCSGQR.remove()` — restored a hollow module

With `keep_donor_resident=False` the full-precision projections are deleted, so
`remove()` reinstalled an MLP with no `gate_proj`, failing later and elsewhere.
Now refuses with an explicit message.

### 3.3 α fitted against the wrong reference (caught before it ran)

A draft of `run_eval.py` fitted `α` against the quantized remnant's own output,
which returns exactly 1.0 by construction and would have looked like a working
calibration. Now fitted against the unquantized donor while it is still resident.

### 3.4 Autograd graph over the vocabulary projection

`lm_head.weight.requires_grad` is True, so the confidence projection built a
graph over a `[positions, 32768]` tensor and held it for the whole prompt.
Wrapped in `no_grad`.

### 3.5 Non-ASCII output crashed the Windows console

`analyze_significance.py` and `budget.py` printed `∪` and a stray combining
character; the Windows cp1252 console raised `UnicodeEncodeError` mid-run, after
the expensive computation. Sources are now ASCII-only.

---

## 4. Discrepancies that are *not* bugs, but change how results must be read

### 4.1 cuBLAS makes the remnant path shape-dependently non-bit-identical

The remnant reconstructs the donor's weight matrices **bit-identically**
(`torch.equal` on all three) and reproduces `nn.Linear` bit-identically at
shapes `[64,1024]`, `[1,1,1024]`, `[1,64,1024]`, `[2,8,1024]` — but **not** at
`[1,22,1024]`, where it differs by `3.9e-3` max absolute. Identical weights,
identical arithmetic, different accumulation order: cuBLAS selects a different
bf16 GEMM tile for that shape.

Consequence for the experimental design, and it is load-bearing:

- `b0-dense` is the true model, comparable to the stored gate baselines.
- **`b2-fp-all` — the same weights through the remnant path — is the correct
  reference for every quantized arm.**
- `|b0 − b2|` is the noise floor. A quality difference smaller than it is not
  attributable to CSGQR.

This is the same confound `experiments/attention_shape` recorded, where a kernel
change moved 23% of greedy replies while shifting mean quality by +0.00138.

In the layer scan (single forward, no generation) the FP arm's KL was **exactly
0.0 at all 28 layers**, so the floor is zero for divergence measurements; it is
generation, where a single flipped argmax diverges a whole reply, that exposes it.

### 4.2 Per-group `α` is worse than scalar `α`, as the algebra predicts

Per-group least-squares solves each group's 1-D problem independently and
therefore ignores cross-group correlation, so it is not the optimum for the sum.
Measured: scalar α gives relative error 0.1148, per-group 0.1246. The scalar form
is used. This is not a defect — it is the reason the design specifies a scalar
gain first — but it is worth recording so nobody "improves" it later.

### 4.3 `k = ⌈ρ·|G|⌉` means the interior region never activates zero groups

Any confidence strictly below `τ_skip` activates at least one group. Deliberate:
the interior region should not contain a hidden second skip region. It means
`k=0` is reachable only through the `τ_skip` branch, which is what the
activation-rate accounting assumes.

### 4.4 2- and 3-bit are stored one value per byte

Only 4- and 8-bit get real bit-packing. `QuantSpec(2)` and `QuantSpec(3)` exist
for the Tier-3 bitwidth sweep and their *storage* claim is therefore not
validated — `values_per_byte` says so. Inventing a 3-bit packer whose byte count
nothing checks would be worse than storing them loosely and saying so.

---

## 5. Two structural results, exact arithmetic, fixed before measurement

### 5.1 Group decomposition is exactly additive — because the projections are bias-free

`Qwen3MLP` has `bias=False` on `gate_proj`, `up_proj` and `down_proj`. The
elementwise product is per-intermediate-unit and `down_proj` is linear over that
axis, so for any partition `{I_g}`, `F(x) = Σ_g F_g(x)` exactly. Partial group
execution therefore *is* intermediate-neuron selection, with no interaction term
to model.

The constructor refuses a biased `down_proj`, because a `down_proj` bias would be
added once per active group instead of once overall — an error that would scale
with `k_t` and look like a routing effect.

### 5.2 The exact-confidence monitor costs more than the MLP it gates

```
LM head MACs / MLP MACs  =  (2·|V|·d_model) / (2·3·d_model·d_ff)  =  |V| / (3·d_ff)
                         =  32768 / 9216  =  3.556×
```

A gate that skipped the remnant on **every token** would still cost 3.56× what it
saved. The condition for viability is `|V| < 3·d_ff`; for current models
(`|V|` ≈ 128–256k, `d_ff` ≈ 8–28k) it essentially never holds.

This is not an argument for optimising the lens. Combined with §6 below — the
raw lens carries almost no routing signal anyway — it means the exact-confidence
path is both unaffordable *and* uninformative, and the hidden-state probe is the
only viable form of the "how much" gate.

---

## 6. The finding that changes the method

Measured at L20 over 798 positions from 96 prompts, `δ = 0.1`:

| predictor of "remnant needed" | AUROC | AUPRC |
| --- | ---: | ---: |
| raw `max_prob` (the specified gate) | 0.593 | 0.660 |
| raw `neg_entropy` | 0.600 | 0.659 |
| logistic on all four lens features | 0.580 | 0.648 |
| **logistic on the pre-remnant residual** | **0.936** | **0.949** |
| permuted-label control | 0.506 | 0.631 |
| reply-position-only control | 0.601 | 0.676 |
| prompt-mean *oracle* (prompt-level ceiling) | 0.698 | 0.771 |

The information is in the state. **The vocabulary projection destroys it.** The
mechanism is visible in the same trace: the raw lens's top-1 token agrees with
the dense model's top-1 only **27%** of the time at this depth, so it is not
measuring the decision it is meant to gate — exactly the failure mode the
early-exit literature reports for untrained intermediate readouts.

The probe clears the prompt-level oracle ceiling (0.936 vs 0.698), so it is
resolving individual tokens rather than prompt-level structure, and the
permuted-label control at 0.506 rules out pipeline leakage. λ was chosen on a
disjoint validation split (val 0.931, test 0.936).

**Caveat, stated plainly:** the probe has 1025 parameters against 394 training
positions. The controls are what make the result credible, not the sample size.
Confirming it at a larger `n` and on `rag_dev` is the first item in
`RESEARCH_VERDICT.md`.

---

## 7. Final architecture as built

- One retired **MLP** (Policy C's MLP half). Attention is untouched.
- Frozen packed remnant, INT4 asymmetric block-128 by default, stored
  `[d_ff, d_model]` for all three projections so a group is a contiguous
  byte-aligned row slice.
- Scalar output gain fitted against the unquantized donor on calibration prompts.
- Confidence at the pre-norm post-attention residual, via the model's own norm
  and LM head, rank-0 lens, temperature separate from generation temperature.
- Three-level routing with per-token bucketing: tokens sharing a selected group
  set share a matmul. For a static ranking that is ≤ `|G|+1` buckets at any
  sequence length; for per-state dynamic selection the bucket count approaches
  the token count, which is a real cost of the dynamic branch.
- Every arm is a `RoutingPolicy` configuration, so an arm-to-arm difference is a
  difference of policy and never of implementation.

## 8. Limitations

1. **Not an independent audit** (§0).
2. **Attention remnant unimplemented**, so the KV-cache half of Policy C is
   untested and the request-level residency policy — the only mechanism in the
   design that can actually free VRAM dynamically — is unmeasured.
3. **Trained lens untested.** Only rank 0 was run. Given §6, a trained lens is
   the wrong direction anyway (the probe skips the projection entirely), but H7's
   lens variant is formally NOT ESTABLISHED rather than falsified.
4. **Single layer.** The systems ceiling at this scope is 1.47% of VRAM and 1.99%
   of MACs, below allocator noise. Multi-layer was not run.
5. **Single checkpoint, one dev split, short replies** (`max_new_tokens=32`,
   median reply 7 tokens). Out-of-domain and long-generation behaviour untested.
6. **Probe is a linear readout** fitted on 394 positions, evaluated on 254.
7. **Simulated frontier** (`sweep_threshold.py`) reproduces routing quantities
   exactly but approximates utility by assuming per-token independence; the
   cross-token coupling through attention above layer `j` is unquantified.
