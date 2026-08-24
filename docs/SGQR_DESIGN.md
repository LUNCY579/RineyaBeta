# SGQR: Significance-Gated Quantized Remnant

## Status

This branch isolates the **fundamental remnant hypothesis** from the later
confidence-gating hypothesis.

SGQR contains no next-token confidence projection, no entropy threshold, no
top-two probability margin, and no confidence-conditioned compute budget.

The research question is narrower:

> Can a structurally retired transformer component remain useful as a frozen,
> low-bit donor, and can significance identify which additive donor groups
> preserve the most quality under a fixed deployment budget?

Confidence may later become an optional controller that chooses the budget. It is
not required to establish whether the quantized remnant itself works.

---

## 1. Motivation

Deleting a transformer component removes both its storage and its computation,
but may cause a large quality loss. Keeping the same component at full precision
preserves quality but does not provide the desired compression.

SGQR studies an intermediate point:

1. retire the full-precision component;
2. preserve a frozen quantized remnant of its original weights;
3. divide that remnant into additive architecture-aligned groups;
4. allocate a fixed number of groups;
5. use measured significance or predicted marginal recovery utility to select
   which groups execute.

This creates two independent compression axes:

- **persistent storage:** donor bitwidth;
- **active computation:** number of selected remnant groups.

The absence of confidence gating makes the initial experiment easier to
interpret. A result cannot be attributed to a probability lens or confidence
threshold: it tests quantized donor preservation and significance directly.

---

## 2. Formal paths

Let \(j\) be the selected MLP and \(h_{j,t}\) its normalized input at token
position \(t\).

The dense model computes

\[
h_{j+1,t}^{T}=h_{j,t}+F_j(h_{j,t}).
\]

Exact deletion computes

\[
h_{j+1,t}^{0}=h_{j,t}.
\]

The complete frozen quantized remnant computes

\[
h_{j+1,t}^{Q}=h_{j,t}+F_j^Q(h_{j,t}).
\]

SGQR with active group set \(\mathcal A_t\) computes

\[
h_{j+1,t}^{R}
=
h_{j,t}
+
\sum_{g\in\mathcal A_t}
\alpha_{j,g}F_{j,g}^Q(h_{j,t}).
\]

The quantized weights are buffers, not trainable parameters:

\[
\nabla_{\theta_j^Q}\mathcal L=0.
\]

The first implementation uses a fixed group budget \(k\):

\[
|\mathcal A_t|=k.
\]

Significance chooses the members of \(\mathcal A_t\). A later CSGQR layer may
choose \(k\) dynamically from confidence, but SGQR does not.

---

## 3. Additive Qwen MLP groups

For a Qwen gated MLP,

\[
F(h)
=
W_{\mathrm{down}}
[
\operatorname{SiLU}(W_{\mathrm{gate}}h)
\odot
(W_{\mathrm{up}}h)
].
\]

Partition the intermediate dimension into disjoint groups. Group \(g\) is

\[
F_g(h)
=
W_{\mathrm{down},:,g}
[
\operatorname{SiLU}(W_{\mathrm{gate},g,:}h)
\odot
(W_{\mathrm{up},g,:}h)
].
\]

Because the groups partition the intermediate dimension,

\[
F(h)=\sum_g F_g(h).
\]

This identity is a required implementation invariant. It is why the initial
research targets MLP channel groups rather than arbitrary weight entries.

The reference implementation quantizes and stores each coupled group as three
matrices:

- gate rows;
- up rows;
- down columns.

Inactive groups are not dequantized or multiplied in the reference selective
path.

---

## 4. Quantization

The reference backend supports:

- row-wise symmetric INT8;
- row-wise symmetric INT4;
- physical INT4 packing at two values per byte.

For each row \(w\),

\[
s=\frac{\max |w|}{q_{\max}},
\qquad
q=\operatorname{clip}
\left(
\operatorname{round}(w/s),
-q_{\max},
q_{\max}
\right).
\]

For INT4, \(q_{\max}=7\). For INT8, \(q_{\max}=127\).

The implementation reports separately:

1. logical donor parameter count;
2. trainable donor parameter count;
3. packed weight bytes;
4. scale bytes;
5. GPU-resident bytes;
6. temporary dequantization workspace;
7. measured latency.

Quantization is not described as parameter-count reduction. It reduces storage
per logical parameter.

The reference backend is designed to establish correctness. It does not claim
that ordinary PyTorch gather/dequantize/linear operations are an optimized
low-bit kernel.

---

## 5. Significance as causal recovery utility

For one prompt state, obtain:

- dense teacher logits \(z_t^T\);
- deleted logits \(z_t^0\);
- logits \(z_t^{+g}\) with quantized group \(g\) active.

Define

\[
D(T,X)=\operatorname{KL}(p_t^T\|p_t^X).
\]

The marginal recovery utility of group \(g\) is

\[
U_{t,g}
=
D(T,0)-D(T,+g).
\]

Interpretation:

- \(U_{t,g}>0\): the group repairs part of the deletion discrepancy;
- \(U_{t,g}\approx0\): the group adds little value;
- \(U_{t,g}<0\): the group is harmful relative to deletion.

The complete remnant utility is

\[
U_t^{\mathrm{all}}
=
D(T,0)-D(T,Q).
\]

The calibration workflow records accumulated utility and observation count
separately:

\[
S_g\leftarrow S_g+U_{t,g},
\qquad
N_g\leftarrow N_g+1.
\]

A static significance score is

\[
M_g=\frac{S_g}{(N_g+\epsilon)^e}.
\]

The default \(e=1\) is the mean utility endpoint. The exponent is an experimental
choice, not a universal constant.

---

## 6. Static and state-conditioned SGQR

### Static SGQR

A single global ranking is learned from calibration prompts:

\[
\mathcal A
=
\operatorname{TopK}
\left(
M_g/c_g,
k
\right),
\]

where \(c_g\) is measured or estimated group execution cost.

The same group set executes for every token. This is the cleanest first test and
can map to a static compiled subnetwork.

### State-conditioned SGQR

A lightweight router predicts per-state utility:

\[
\widehat U_{t,:}=R_\psi(h_{j,t}).
\]

Then

\[
\mathcal A_t
=
\operatorname{TopK}
\left(
\widehat U_{t,g}/c_g,
k
\right).
\]

This is dynamic significance, not confidence. It predicts *which groups are
useful* under a fixed budget. It does not estimate whether the next-token
decision is confident and does not change \(k\).

The state-conditioned reference path gathers token rows by selected group and
executes a group only for tokens that selected it. This is mathematically sparse
but may incur gather/scatter and small-kernel overhead. GPU measurements must
separate algorithmic quality from backend efficiency.

---

## 7. Baseline ladder

The fundamental experiment must establish:

1. dense original;
2. exact MLP deletion;
3. frozen full-precision MLP;
4. always-on INT8 remnant;
5. always-on INT4 remnant;
6. random groups at matched budget;
7. magnitude-ranked groups;
8. static significance;
9. state-conditioned significance.

The always-on INT4 arm is the quality ceiling for router-only INT4 SGQR:

\[
Q_{\mathrm{SGQR}}
\lesssim
Q_{\mathrm{INT4-all}},
\]

unless later survivor adaptation improves the model beyond the frozen remnant.

If INT4-all preserves little of the deleted capability, routing cannot repair
that missing ceiling.

---

## 8. Candidate-layer selection

The first experiment should not use an arbitrary layer.

Scan for:

- low deletion sensitivity;
- moderate deletion sensitivity;
- high deletion sensitivity.

The primary SGQR experiment should use a moderate layer:

- a nearly redundant layer creates a ceiling effect;
- a critical layer may be outside the remaining architecture's recoverable
  capacity;
- moderate damage provides room to measure recovery.

The scan reports dense-teacher KL for deletion, INT8, and INT4, plus the fraction
of deletion KL recovered by the low-bit remnant.

---

## 9. Recovery metrics

For a higher-is-better task metric,

\[
\operatorname{RecoveryRatio}
=
\frac{
Q_{\mathrm{method}}-Q_{\mathrm{delete}}
}{
Q_{\mathrm{dense}}-Q_{\mathrm{delete}}
}.
\]

Absolute quality must always accompany this ratio. The ratio is unstable when
dense and deleted quality are nearly equal.

Other secondary metrics include:

\[
\operatorname{RecoveryPerCompute}
=
\frac{
Q_{\mathrm{method}}-Q_{\mathrm{delete}}
}{
\text{active remnant compute}
}
\]

and

\[
\operatorname{RecoveryPerByte}
=
\frac{
Q_{\mathrm{method}}-Q_{\mathrm{delete}}
}{
\text{remnant storage bytes}
}.
\]

The primary frontier is:

\[
\operatorname{RecoveryRatio}(b,\rho),
\]

where \(b\) is bitwidth and \(\rho=k/G\) is active group fraction.

---

## 10. Core hypotheses

### H1 — Low-bit donor preservation

INT8 and INT4 retain meaningful donor-specific function relative to exact
deletion.

### H2 — Significance advantage

Utility-ranked groups outperform random and magnitude-ranked groups at matched
active group count.

### H3 — State-conditioned advantage

A hidden-state utility router outperforms or complements static global
significance at matched compute.

### H4 — Interior frontier

There exists \(0<\rho<1\) where SGQR recovers substantial deletion damage
without executing every donor group.

### H5 — Donor identity

The correct retired-layer remnant outperforms a wrong-layer or random frozen
side branch at matched storage.

### H6 — Systems potential

Packed low-bit storage and structured execution produce a useful measured
storage, VRAM, latency, or energy frontier after backend overhead is accounted
for.

Each hypothesis must receive one of:

- SUPPORTED;
- PARTIALLY_SUPPORTED;
- NOT_ESTABLISHED;
- FALSIFIED;
- INCONCLUSIVE.

---

## 11. Required ablations

At matched group budgets compare:

- random group ranking;
- matrix-magnitude ranking;
- static mean utility;
- exposure-normalized utility;
- shuffled utility;
- reversed utility;
- state-conditioned utility.

Also test whether group utility is additive. For groups \(A\) and \(B\),

\[
I(A,B)
=
U(A\cup B)-U(A)-U(B).
\]

Large interactions would indicate that independent top-k selection is not the
best router and motivate later greedy marginal or combinatorial policies.

---

## 12. GPU interpretation

An inactive group must be excluded before its dequantization and matrix
multiplications.

Multiplying a complete MLP output by a sparse mask is not compute skipping.

Report:

- serialized bytes;
- model-load allocated and reserved VRAM;
- prefill peaks;
- decode peaks;
- dequantization workspace;
- group invocation fraction;
- synchronized latency;
- tokens per second.

The reference implementation may be slower than dense execution because dynamic
gather/scatter and small matrices are inefficient. That is a backend limitation,
not automatically an algorithmic quality failure.

A later optimized backend may use:

- fused dequantize-GEMM;
- Triton;
- grouped GEMM;
- contiguous packed group layouts;
- static compiled routes;
- route-based batching.

Optimized outputs must be checked against the reference backend.

---

## 13. Why confidence is deferred

The complete CSGQR concept used confidence to choose *how much* remnant compute
to allocate and significance to choose *where* to allocate it.

SGQR removes the first decision.

It fixes the budget and asks whether the remaining mechanism is useful:

\[
\text{fixed budget}
+
\text{quantized donor}
+
\text{significance selection}.
\]

This provides a cleaner causal decomposition:

1. Does quantized donor function survive?
2. Does significance identify useful donor components?
3. Does partial execution produce a useful quality–cost frontier?

Only after those questions are answered should confidence be added as an
optional dynamic budget controller.

If SGQR fails because the donor or significance mechanism is weak, a confidence
threshold cannot rescue the foundation.

If SGQR succeeds, CSGQR can later test whether confidence improves budget
allocation.

---

## 14. Advancement gates

### Gate A — Implementation

Packed quantization, additive grouping, hard selective execution, serialization,
and evaluation invariants pass.

### Gate B — Donor viability

Always-on INT8 or INT4 recovers meaningful deletion damage.

### Gate C — Significance value

Significance beats random at matched active group count.

### Gate D — Interior frontier

A partial budget retains useful quality.

### Gate E — Systems reporting

Real GPU memory and latency are measured without substituting theoretical FLOPs.

Confidence research should not be reintroduced before Gates A–D are understood.

---

## 15. Initial workflow

1. Run `scripts/verify/verify_sgqr.py`.
2. Scan candidate layers with `scripts/sgqr/scan_layers.py`.
3. Choose a moderate-sensitivity layer.
4. Calibrate group utility with `scripts/sgqr/calibrate.py`.
5. Train the optional state-conditioned utility router.
6. Evaluate all baseline modes.
7. Sweep active group fractions.
8. Compile the quality/storage/compute report.
9. Preserve negative results.
10. Only then consider adding confidence as a separate experimental layer.

SGQR is intended to make the quantized-remnant hypothesis independently
falsifiable before the larger adaptive-compute architecture is attempted.
