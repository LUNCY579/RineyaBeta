# CSGQR — Method Specification

The algorithm as this branch implements it, reconstructed from first principles
rather than transcribed from prose. Where the campaign brief and the arithmetic
disagree, the arithmetic is followed and the disagreement is stated.

Notation follows the brief: `j` is the retired component's layer, `h_{j,t}` the
residual state immediately before it for token `t`, `F_j` the original
full-precision component, `F_j^Q` the frozen quantized remnant, `G` the set of
architecture-aligned groups, `A_t ⊆ G` the groups executed for token `t`.

---

## 1. The three paths, and what distinguishes them

```
dense        h_{j+1,t} = h_{j,t} + F_j( N(h_{j,t}) )
deleted      h_{j+1,t} = h_{j,t}
full remnant h_{j+1,t} = h_{j,t} + F_j^Q( N(h_{j,t}) )
adaptive     h_{j+1,t} = h_{j,t} + Σ_{g ∈ A_t} α_{j,g} · F_{j,g}^Q( N(h_{j,t}) )
```

`N` is the layer's `post_attention_layernorm`. Two distinct tensors are involved
and conflating them is the easiest way to get this wrong:

* **`N(h_{j,t})`** — the post-norm tensor, is what the remnant *consumes*.
* **`h_{j,t}`** — the pre-norm residual, is what the gate and the router *read*.

`h_{j,t}` is the tensor the decoder layer adds the MLP output back onto. It is
obtained by a `forward_pre_hook` on `post_attention_layernorm`, whose input is
exactly it. Reading the post-norm tensor instead would be a different,
RMSNorm-rescaled quantity, and the implementation raises rather than silently
falling back to it.

### 1.1 A structural fact that makes the confidence signal cheap to reason about

Only layer `j` is modified, so layers `0..j−1` are identical across every arm.
`h_{j,t}` depends only on those layers applied to positions `≤ t`, and is
therefore **identical across the dense, deleted and remnant arms**. Confidence
and predicted utility are properties of the surviving prefix, not of the arm.

Measured, not assumed: max `|Δ|` = **0.000e+00** over 798 positions
(`results/trace_L20_int4/trace_manifest.json:residual_invariance_max_abs`). This
is why one forward pass can produce the gate signal for all arms in the trace.

---

## 2. Group decomposition is exact

For a Qwen3-style gated MLP with `bias=False` on all three projections:

```
F(x) = W_down · [ silu(W_gate x) ⊙ (W_up x) ]
```

Index the intermediate dimension by `i`. The elementwise product is per-`i`, and
`W_down` is linear over `i`, so for **any** partition `{I_g}` of the intermediate
axis:

```
F(x) = Σ_g W_down[:, I_g] · [ silu(W_gate[I_g,:] x) ⊙ (W_up[I_g,:] x) ]
     = Σ_g F_g(x)
```

**exactly** — the only discrepancy is floating-point summation order. Verified in
float64 to `< 1e-14`.

Two consequences that shape everything downstream:

1. Executing a subset of groups **is** intermediate-neuron selection. There is no
   approximation and no cross-group interaction term in the *output*.
2. The absence of a bias is what makes it exact. A `down_proj` bias would be
   added once per active group instead of once overall — an error that scales
   with `k_t` and would look exactly like a routing effect. The constructor
   refuses a biased `down_proj` rather than producing that error.

Note carefully what this does **not** say. The outputs are additive; **utility is
not**, because utility is a difference of KL divergences and KL is not linear.
Additivity of utility is a separate empirical question, measured in §7.

### 2.1 Storage layout is a correctness requirement, not an optimisation

`W_gate` and `W_up` are `[d_ff, d_model]`, so a group is a contiguous row slice.
`W_down` is `[d_model, d_ff]`, so the same group is a *column* slice — not
contiguous, and at 4 bits not even byte-aligned. So `W_down` is stored
**transposed**. All three matrices then share the layout `[d_ff, d_model]`,
quantized blockwise along `d_model`, and group `g` is rows `[g·G, (g+1)·G)` of
each: a contiguous, byte-aligned range whose scales are a contiguous slice too.
The down projection is computed as `u_g @ down_qT[I_g]`, which is the same
arithmetic.

Without this layout, "skip a group" could only be implemented as a mask after the
fact, and the compute claim would be void.

---

## 3. Quantization

Blockwise, along the input dimension, `block = 128` by default.

Asymmetric (default): `lo = min(0, min_block)`, `hi = max(0, max_block)`,
`scale = (hi − lo)/(2^b − 1)`, `zero = round(−lo/scale)`. Including zero in the
represented interval keeps an exactly-zero weight exactly zero after the round
trip.

Symmetric: `scale = amax/(2^{b−1} − 1)`, codes in `[−qmax, qmax]` stored biased by
`2^{b−1}`, so the storage range is `[1, 15]` at 4 bits and zero maps to 8
exactly. This costs one representable level to keep zero exact, deliberately.

Stated numerical choices:

* The scale search runs in **float32** regardless of weight dtype. bf16 has 8
  mantissa bits and computing a range in it visibly quantizes the scale itself.
* Scales are stored float16 and **counted in the byte total**. Zero points are
  uint8 and also counted.
* An all-zero block would give `scale = 0` and a division by zero. Such blocks
  get `scale = 1` and quantize to exactly zero — exact rather than merely safe.
* Padding for a non-multiple final block is NaN-filled during the range search so
  it cannot influence a block's min/max, then written as zero so the packed bytes
  are deterministic. It is never read back.

At 4 bits, two values occupy one byte (low nibble = even index). A 1024-value row
is 512 bytes. Measured for one L20 MLP: **4,939,872 B** packed against
**18,874,464 B** bf16 = **3.821×**, and the analytic byte model predicts the
measured figure exactly.

`INT3`/`INT2` are accepted but stored one value per byte. Their *storage* claim is
therefore not validated, and the code says so rather than implying a packer that
does not exist.

### 3.1 Freezing

Packed storage is registered as **buffers**, not parameters. Buffers move with
`.to()` and are saved by `state_dict`, but an optimizer built over
`model.parameters()` cannot see them, so no optimizer state can be created for
the donor by accident. Integer dtypes cannot carry gradients at all, which is a
stronger guarantee than setting a flag.

`∇_{θ_j^Q} L = 0` therefore holds by construction rather than by discipline.

---

## 4. Output gain

The gain corrects the systematic shrinkage blockwise quantization introduces. It
is fitted on calibration data, is not a trained parameter, and comes in four
forms that answer different questions.

**Scalar, whole remnant.** The gain minimising `‖F(h) − α F^Q(h)‖²`:

```
α_j = E⟨F_j^Q(h), F_j(h)⟩ / ( E‖F_j^Q(h)‖² )
```

**Per group.** The same 1-D problem per group, each against **its own** donor
group output:

```
α_{j,g} = E⟨F_{j,g}^Q(h), F_{j,g}(h)⟩ / ( E‖F_{j,g}^Q(h)‖² )
```

Fitting a group against the *whole* `F` instead is a **different estimator**, not
a looser one: since `⟨F_g, F⟩ = ‖F_g‖² + ⟨F_g, Σ_{h≠g} F_h⟩`, it is inflated by
the cross-group inner product and does not return 1 even for a lossless remnant.
Measured: on a bit-exact remnant it gave gains spanning 0.978–1.024 and injected
**1.32%** relative error into an exact reconstruction.

**Partial execution, two targets.** At `ρ < 1` the surviving model receives
`h + α F_A` where it would have received `h + F`, so a gain for the active set is
a real question the brief implies but does not name:

```
against="rows"    α = E⟨F_A^Q, Σ_{g∈A} F_g⟩ / E‖F_A^Q‖²      "reproduce what I spent"
against="whole"   α = E⟨F_A^Q, F⟩ / E‖F_A^Q‖²                "approximate the layer"
```

`"whole"` is the least-squares optimum for the perturbation the residual stream
sees, and its magnitude is itself a measurement:

```
α = 1 + ⟨F_A, F − F_A⟩ / ‖F_A‖²
```

so it is exactly 1 when the active groups are orthogonal to the inactive ones and
rises toward `1/ρ` as they become positively correlated. **Measured on this
checkpoint: 0.99–1.10 across ρ ∈ [0.125, 1.0], never near `1/ρ`.** The groups are
near-orthogonal, and calibration moves the output error by ~0.1%. So the missing
gain is *not* the explanation for partial-execution arms underperforming — a
hypothesis this specification recorded and then discarded on evidence.

**The epsilon.** The brief writes the denominator as `E‖F^Q‖² + ε`. An
*absolute* ε is wrong and measurably so: MLP output magnitudes are
checkpoint-dependent, and on a small-activation layer a fixed `1e-8` was **13% of
the denominator**, pulling a true `α` of 0.9979 down to 0.8797 and making the
"calibrated" remnant worse than the uncalibrated one. The degenerate case is
handled as the zero-check it actually is, with no additive term, and the sums
accumulate in float64.

---

## 5. The "how much" signal

Two sources, and the choice between them is the campaign's central systems
result.

### 5.1 Exact vocabulary lens

A causal LM has no probability distribution at intermediate depth until one is
constructed:

```
z̃_{j,t} = W_LM · N_f( A_j h_{j,t} )
p̃_{j,t} = softmax( z̃_{j,t} / T_j )
```

`W_LM` and `N_f` are borrowed from the model — not copied, because a duplicated
32768×1024 head is 64 MiB and would swamp a 4.7 MiB remnant. `A_j = I` is the raw
logit lens. `T_j` is a **calibration** temperature fitted on held-out data by
golden-section search on NLL; it is not the generation temperature, and this
module never reads a generation config.

Features, all computed in float32 (a bf16 softmax over 32768 classes moves the
entropy term in the third decimal, which is the order of the differences being
resolved):

```
C^max = max_v p̃(v)
C^H   = 1 − H(p̃)/log|V|          normalised by the FULL vocabulary
C^M   = p̃⁽¹⁾ − p̃⁽²⁾
M^z   = z̃⁽¹⁾ − z̃⁽²⁾              not confined to [0,1]
C^S   = 1 − JS(p̃_{j−d}, p̃_j)/log 2
```

**The closed form that decides its fate.** The projection costs `2·|V|·d_model`
MACs; the MLP it decides about costs `2·3·d_model·d_ff`. So

```
cost ratio = |V| / (3·d_ff) = 32768 / 9216 = 3.556×
```

A gate that skipped the remnant on **every single token** would still cost 3.56×
what it saved. The viability condition is `|V| < 3·d_ff`; for current production
models (`|V|` ≈ 128–256k, `d_ff` ≈ 8–28k) it essentially never holds. This is
arithmetic on the shapes, not a measurement, and no threshold, lens or feature
choice changes it.

Deployment discipline, since prefill is gated too (§6.3) and every gated position
needs a signal: project in **chunks** of 256 positions, reduce to scalars inside
the call, and let the vocabulary tensor die there. Unchunked, a 2048-token prompt
materialises ~1.6 GiB of transient on an 8 GiB card — fine on short dev prompts,
OOM on long ones.

### 5.2 Hidden-state probe

```
q_{j,t} = σ( w·h_{j,t} + b )
```

A linear readout of the residual, predicting *need* directly. Converted to the
same maturity orientation as the lens features (`C = 1 − q`) before any threshold
sees it, so one three-level implementation serves both and a `τ` means the same
kind of thing in each.

Cost, per token, as a share of the MLP it gates:

| monitor | MACs | share of gated MLP | bytes |
| --- | --- | ---: | ---: |
| need probe | `2·d_model` | **0.0109%** | 4,100 |
| group router | `2·d_model·|G|` | **0.2604%** | 98,400 |
| exact lens | `2·|V|·d_model` | **355.6%** | 0 (borrowed) |

Four orders of magnitude. Combined with the measured accuracy — the probe reaches
AUROC **0.936** on held-out prompts where the raw lens reaches 0.55–0.60 — the
exact-confidence path is both unaffordable *and* uninformative, and the probe is
not an optimisation of the method but **the only viable deployment form of it**.

Standardisation is folded into the weights at load time, so inference is a single
matmul:

```
w·((h−μ)/σ) + b  ==  (w/σ)·h + (b − (w/σ)·μ)
```

---

## 6. The three-level policy

With `τ_full < τ_skip` and a maturity signal `C`:

```
C ≥ τ_skip                →  A_t = ∅          bypass; nothing dequantized
C ≤ τ_full                →  A_t = G          full remnant
τ_full < C < τ_skip       →  ρ_t = ((τ_skip − C)/(τ_skip − τ_full))^γ
                             k_t = ⌈ρ_t·|G|⌉
                             A_t = top-k_t groups by R_{t,g}
```

`γ < 1` spends compute more readily, `γ > 1` is stingier, `γ = 1` is linear.

`k = ⌈ρ|G|⌉` means any `C` strictly below `τ_skip` activates at least one group.
That is deliberate: the interior region should not silently contain a second skip
region, and it makes `k = 0` reachable only through the `τ_skip` branch — which is
what the activation-rate accounting assumes.

### 6.1 Group selection

```
R_{t,g} = Û_{t,g} / (c_g + ε)
```

The brief writes `R_g = (1−C)^γ · Û_g / (c_g + ε)`. The `(1−C)^γ` factor is a
**per-token scalar**: it multiplies every group's value equally and therefore
cannot change which `k` are largest. It is omitted from the ranking score for two
reasons — it is mathematically inert there, and including it created a genuine
degeneracy: at `C = 1.0` every value became exactly `0.0` and `argsort` returned
groups `0..k−1`, silently turning the significance selector into a contiguous
selector. The factor's real job is setting *how many* groups run, which `ρ` does.

Ties resolve to ascending group id via a **stable** sort. Blockwise quantization
creates exact ties in group utility far more often than continuous data would, and
an unstable sort makes an arm irreproducible for exactly the tokens where the
choice is arbitrary.

On this checkpoint every group has the same shape and therefore the same cost, so
`R_g` is a monotone rescaling of `Û_g` and selection is identical to ranking on
utility alone. The cost term is implemented anyway, because mixed-size groups are
a planned follow-up and a cost model that only appears later tends to appear
wrong.

### 6.2 Hard execution

Inactive groups must avoid dequantization, the gate/up projections, the
activation, the down projection, and the activation allocation. `select` returns
group **ids**; `PackedMatrix.dequantize(lo, hi)` slices the packed bytes, the
scales and the zero points *before* any arithmetic. Nothing downstream ever sees
an inactive group.

The distinction the brief insists on:

* **Mathematical sparsity** — some outputs are zero.
* **Executed sparsity** — the corresponding operations were not performed.

Only the second supports a compute claim, and it is enforced at the interface
rather than trusted to a caller. Contiguous selected groups are coalesced into
maximal runs so that `ρ = 1` costs three dequantization calls rather than 72;
this is bit-identical and is launch overhead only.

### 6.3 Prefill is gated

Skipping the remnant at a prefill position changes that position's hidden state
and therefore its keys and values for every layer above `j`. Gating only decode
positions would be a **different method** with a different quality profile. Both
are gated, per token, on their own signal.

### 6.4 Per-token bucketing

Different tokens select different group sets, which cannot be one matmul. Tokens
are bucketed by their selected set and one matmul runs per distinct set. For a
static ranking the set is determined by `k` alone, so there are at most `|G|+1`
buckets at any sequence length. For per-state dynamic significance the sets
genuinely differ per token and the bucket count approaches the token count. That
is a real cost of the dynamic branch, not an artefact, and it is measured rather
than hidden behind a FLOP count.

---

## 7. Utility

```
U_t     = KL(p^T ‖ p^0) − KL(p^T ‖ p^Q)
U_{t,g} = KL(p^T ‖ p^0) − KL(p^T ‖ p^{+g})
```

`p^T` is the dense teacher, `p^0` the deleted path, `p^Q` the full remnant,
`p^{+g}` the remnant with only group `g` active. The convention is
`damage_without − damage_with`, so **larger is better** and a useless group scores
≈ 0 rather than ≈ 1. Every consumer assumes this, and the direction is stated
once and never reversed.

Utility is measured by **intervening** — actually running the model with the group
on and comparing against it off — not by an activation-magnitude proxy. That is
what the `magnitude` ranking exists to be compared against.

Traces are paired on identical prompts, token states, model weights and decoding
context. The remnant is installed **once** and the policy swapped between
forwards, so every group is measured against exactly the same packed weights
rather than against a freshly quantized copy.

### 7.1 Additivity of utility, measured

Outputs are exactly additive; utility need not be. Measured at L20 over 798
positions:

* sum of 24 singleton utilities = **0.29825**; whole-remnant utility =
  **0.30347**. Gap −0.00521, i.e. **1.7%**.
* pairwise interaction `|U(A∪B) − U(A) − U(B)|` over 40 sampled pairs: mean
  **2.1%** of `|U(A)|+|U(B)|`, max 8.4%; 13/40 have an interaction whose CI
  excludes zero, but all are small.

So independent top-k ranking is a sound selection rule at both endpoints
(`k = 2` and `k = 24`). Intermediate `k` is interpolated, not measured.

### 7.2 Need label

```
y_t = 1[ U_t > δ ]
```

`δ = 0.1` rather than 0, because `P(U_t > 0) = 0.9987`: at a layer with real
deletion damage, "does this token need the remnant?" is essentially always yes,
so `δ = 0` is non-discriminative and any binary skip gate loses utility by
construction. At `δ = 0.1` the base rate is 0.62 and AUROC is meaningful. The
useful framing is therefore *how much*, not *whether*.

---

## 8. What this specification does not claim

* **Nothing about attention.** Only the MLP is retired. The KV-cache problem —
  that skipping attention for an earlier token leaves later tokens needing keys
  and values that were never generated — is unsolved here, so the request-level
  residency policy (the only mechanism in the design that can free VRAM
  dynamically) is unimplemented and unmeasured.
* **Nothing about survivor adaptation.** Donor-frozen router-only is what was
  built. No LoRA on neighbouring blocks.
* **No end-to-end systems win.** One MLP layer is **1.99%** of this model's
  parameters (9,437,184 of 474,021,888). INT4 saves 13.29 MiB = **1.47%** of model
  VRAM; even `ρ = 0` removes only 1.99% of model MACs per token. At single-layer
  scope the ceiling is below allocator and clock noise, whatever the routing does.
  This is arithmetic fixed before any measurement, and it is why the campaign is
  a *mechanism* study with the systems ceiling reported analytically.

---

## 9. The claim the method can attempt to earn

> A structurally retired transformer component is retained as a frozen low-bit
> remnant; a cheap readout of the pre-remnant state controls how much donor
> computation is invoked, and a separate marginal-utility router selects the most
> useful architecture-aligned donor groups before computation continues through
> the surviving model.

Note the deviation from the brief's wording, and it is load-bearing:
*"confidence or predicted need"* has become *"a cheap readout of the pre-remnant
state"*. The exact-probability form of the gate is ruled out by §5.1's closed
form on cost and by measurement on accuracy. If CSGQR is a real method, it is a
**state-probe**-gated method, and calling it confidence-gated would be describing
a variant the evidence rejects.
