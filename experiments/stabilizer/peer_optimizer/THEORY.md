# PCSM mathematical specification

## 1. Problem statement

Let a pretrained transformer contain native parameter matrices

```math
\mathcal W_0=\{W_1,\ldots,W_L\}.
```

For target matrix `W_t`, sparse native-weight fine-tuning seeks

```math
\min_{\Delta W_t}\;\mathcal L(W_t+\Delta W_t)
\quad\text{subject to}\quad
\|\Delta W_t\|_0\le K_t.
```

Ordinary gradient top-`K` chooses the support from the target's own calibration gradient. PCSM tests the stronger hypothesis that other layers contain useful information about the target's appropriate plasticity:

```math
M_t
=
f_\theta\left(
W_t,\nabla_{W_t}\mathcal L,
\{W_j,H_j,G_j,d_j,r_j\}_{j\ne t}
\right),
```

where `H_j` denotes optional activation behavior, `G_j` optional calibration-gradient behavior, `d_j` depth, and `r_j` module role.

The generated object is not an arbitrary replacement weight. PCSM factorizes adaptation authority:

```math
\underbrace{M_t,A_t}_{\text{peer-conditioned plasticity}}
\qquad
\underbrace{\nabla_{W_t}\mathcal L}_{\text{task direction}}
\qquad
\underbrace{\eta,\operatorname{AdamW}}_{\text{optimization dynamics}}.
```

## 2. Layer descriptors

Each layer is compressed into a fixed-dimensional descriptor

```math
z_j=E(W_j,H_j,G_j,d_j).
```

The implementation uses 20 scale-normalized statistics:

- signed mean, standard deviation, absolute mean, and maximum weight magnitude in RMS units;
- mean and coefficient of variation of row and column RMS energy;
- positive-weight fraction and two magnitude-tail fractions;
- logarithmic row and column sizes;
- activation signed mean, standard deviation, and absolute mean in RMS units;
- gradient signed mean, standard deviation, and absolute mean in RMS units;
- normalized depth.

Descriptors are intentionally compact. Feeding every raw matrix into a second transformer would defeat the limited-hardware objective.

## 3. Eligible coworker graph

A target receives messages from an eligible set

```math
\mathcal P_t
=
\{j\ne t:r_j=r_t\}
```

by default. This same-role restriction reflects a real symmetry problem: a query projection and an MLP down-projection do not share obvious coordinate semantics. The implementation fails closed when `P_t` is empty unless fallback is explicitly requested.

Self-advice is forbidden:

```math
\alpha_{tt}=0.
```

## 4. Peer attention

Descriptors are encoded as `e_j`. Attention combines a transparent cosine prior, a learnable pairwise residual, and a depth-distance penalty:

```math
\ell_{jt}
=
\frac{\cos(z_t,z_j)}{\tau}
+
\sigma(\gamma)
\frac{q(e_t)^\top k(e_j)}{\sqrt d}
-
\operatorname{softplus}(\lambda_d)
\frac{|d_j-d_t|}{d_{\max}-d_{\min}}.
```

```math
\alpha_{jt}
=
\frac{\exp(\ell_{jt})}
{\sum_{r\in\mathcal P_t}\exp(\ell_{rt})},
\qquad j\in\mathcal P_t.
```

The learned term begins near zero, so an uncalibrated generator starts from descriptor similarity rather than random routing. The target context is

```math
c_t=\sum_{j\in\mathcal P_t}\alpha_{jt}v(e_j).
```

The attention vector is saved as evidence. It is not treated as a causal explanation without ablation.

## 5. Structural property maps

Directly averaging signed peer weights assumes neuron alignment that may not exist. PCSM instead extracts four non-negative property maps from each matrix:

```math
P_j(W_j)
=
\left[
P_j^{\text{elem}},
P_j^{\text{row}},
P_j^{\text{col}},
P_j^{\text{joint}}
\right].
```

Element magnitude:

```math
P_{j,ab}^{\text{elem}}
=
\left|
\frac{W_{j,ab}-\mu(W_j)}
{\operatorname{RMS}(W_j-\mu(W_j))+\epsilon}
\right|.
```

Row energy:

```math
r_{j,a}
=
\sqrt{\frac{1}{n}\sum_b W_{j,ab}^2},
\qquad
P_{j,ab}^{\text{row}}
=
\frac{r_{j,a}}{\frac1m\sum_u r_{j,u}+\epsilon}.
```

Column energy is defined analogously. Joint leverage is the normalized product of row and column energy.

When peer and target shapes differ, each channel is bilinearly projected to the target shape. This is a coarse structural projection, not a one-to-one neuron correspondence claim.

The target's aggregated peer properties are

```math
\bar P_t
=
\sum_{j\in\mathcal P_t}
\alpha_{jt}\,
\Pi_{j\rightarrow t}(P_j).
```

## 6. Coordinate evidence

For every target coordinate `(a,b)`, the implementation constructs eight maps:

```math
x_{t,ab}=
\begin{bmatrix}
|g_{t,ab}|/\operatorname{RMS}(g_t)\\
\bar P_{t,ab}^{\text{elem}}\\
\bar P_{t,ab}^{\text{row}}\\
\bar P_{t,ab}^{\text{col}}\\
\bar P_{t,ab}^{\text{joint}}\\
P_{t,ab}^{\text{elem}}\\
|\operatorname{mean}_c\bar P_{t,ab}^{c}-\operatorname{mean}_cP_{t,ab}^{c}|\\
(|g_{t,ab}|/\operatorname{RMS}(g_t))\operatorname{mean}_c\bar P_{t,ab}^{c}
\end{bmatrix}.
```

Each map is standardized over the target matrix. A hypernetwork conditioned on the target and peer context produces feature weights

```math
\beta_t
=
\operatorname{softmax}
\left(h_\theta([e_t;c_t])\right),
\qquad
\beta_{t,f}\ge0,
\quad
\sum_f\beta_{t,f}=1.
```

The optimization score matrix is

```math
S_{t,ab}
=
\sum_f \beta_{t,f}\,\widehat{x}_{t,ab,f}.
```

Non-negative normalized feature weights make the generated policy inspectable and prevent arbitrary cancellation among incomparable evidence channels.

## 7. Exact support

For a declared budget `K_t`,

```math
M_t
=
\operatorname{TopKMask}(S_t,K_t),
```

with the invariant

```math
M_t\in\{0,1\}^{m\times n},
\qquad
\|M_t\|_0=K_t.
```

A floating-point fraction `p` resolves once to

```math
K_t=\max(1,\operatorname{round}(p\,mn)).
```

For actual SFT, `M_t` is generated once and frozen.

### Meta-calibration gradient

Top-`K` is non-differentiable. During outer-loop calibration, define a smooth surrogate around the detached `K`-th score threshold:

```math
\widetilde M_{t,ab}
=
\sigma\left(
\frac{S_{t,ab}-\operatorname{stopgrad}(q_K(S_t))}{T}
\right),
```

rescaled toward mass `K`. A custom autograd operator returns `M_t` exactly in the forward pass and routes the backward derivative through `M_tilde`. This matters because ordinary algebraic STE expressions can leave sub-ULP deviations from binary masks.

## 8. Step-scale matrix

A second head outputs three target-level values. Positive slopes are obtained through `softplus`, and the local scale matrix is bounded:

```math
A_{t,ab}
=
a_{\min}
+(a_{\max}-a_{\min})
\sigma\left(
b_t+s_t\widehat S_{t,ab}+p_t\widehat{\bar P}_{t,ab}
\right).
```

Defaults are

```math
a_{\min}=0.20,
\qquad
a_{\max}=1.20.
```

Therefore peers can attenuate or modestly amplify the optimizer but cannot create unbounded steps.

## 9. Native update

For SGD notation,

```math
W_t^{(n+1)}
=
W_t^{(n)}
-
\eta\,
M_t\odot A_t\odot G_t^{(n)}.
```

For AdamW, putting `A_t` inside the raw gradient is not enough: for a positive coordinate-wise constant, the factor is approximately cancelled by `m_t / sqrt(v_t)`. The implementation therefore accumulates **masked, unscaled** moments and applies `A_t` to the realized optimizer step:

```math
\tilde G_t=M_t\odot G_t,
```

```math
m_t
=M_t\odot\left(\beta_1m_{t-1}+(1-\beta_1)\tilde G_t\right),
```

```math
v_t
=M_t\odot\left(\beta_2v_{t-1}+(1-\beta_2)\tilde G_t^2\right),
```

```math
U_t
= M_t\odot A_t\odot
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon},
```

```math
W_t^{(n+1)}
= W_t^{(n)}-\eta U_t-\eta\lambda\,M_t\odot A_t\odot W_t^{(n)}.
```

The positive bounded scale therefore controls realized motion without changing its optimizer-defined sign. Both adaptive state and decoupled weight decay remain restricted to the declared support. Thus, absent external mutation,

```math
M_{t,ab}=0
\implies
W_{t,ab}^{(n)}=W_{t,ab}^{(0)}
\quad\forall n.
```

This is stronger than zeroing the current gradient.

## 10. Outer-loop objective

The generator can be calibrated over tasks or held-out batches by minimizing post-update validation loss:

```math
\min_\theta
\mathbb E_{\tau}
\left[
\mathcal L_{\tau}^{\mathrm{val}}
\left(
W_t
-
\eta M_{t,\theta}^{\mathrm{STE}}
\odot A_{t,\theta}
\odot
\nabla_{W_t}\mathcal L_{\tau}^{\mathrm{cal}}
\right)
\right].
```

A leakage-resistant implementation must separate:

- generator/meta-training tasks;
- mask-calibration data;
- SFT data;
- development evaluation;
- sealed evaluation.

The generator should be frozen before the compared SFT arms begin.

## 11. Optional structural regularizer

A naive term such as

```math
\|W_t-W_j\|_F^2
```

would homogenize layers that are supposed to specialize. The implementation instead offers

```math
\mathcal R_{\text{struct}}
=
\|r(W_t')-\bar r_t\|_2^2
+
\|c(W_t')-\bar c_t\|_2^2,
```

where `r` and `c` are normalized row and column energy profiles. This term is optional and must have its own ablation; it is not silently added to the optimizer.

## 12. Required invariants

A run is invalid if any of the following fail:

1. **Exact budget:** `sum(M_t) = K_t`.
2. **Binary forward mask:** every mask value is exactly 0 or 1.
3. **No self-advice:** `alpha_tt = 0`.
4. **Semantic peer eligibility:** excluded module roles receive zero attention.
5. **Task-owned direction:** selected update signs oppose the task gradient under gradient descent.
6. **Frozen-coordinate preservation:** no coordinate outside `M_t` changes.
7. **Masked optimizer state:** Adam moments outside `M_t` remain zero.
8. **Persisted support:** the saved mask hash matches the mask used by training.
9. **Fixed support:** the SFT support does not change unless the experiment explicitly studies support migration.
10. **Restart:** actual SFT begins from the exact pre-calibration checkpoint.

## 13. Complexity

For one target with shape `m x n` and `P` eligible peers:

- descriptor extraction is linear in peer parameter count but can be cached;
- attention is `O(Pd)` for one target query after descriptors are encoded;
- property projection and aggregation are `O(Pmn)` in the straightforward implementation;
- score, mask, and scale storage are `O(mn)` transiently;
- exact `topk` is backend-dependent and roughly `O(mn log K)` or better.

For a `1024 x 3072` matrix, one fp32 score map is approximately 12 MiB. Targets can be planned sequentially, but a production implementation should add blockwise/streaming top-`K` before claiming low peak memory.

Sparse native tuning also does not automatically reduce dense backward matmul cost. The current scientific claim concerns **where plasticity resides**, not immediate sparse-kernel acceleration.

## 14. What the synthetic study proves—and does not prove

The controlled study constructs a hidden sparse optimum from peer structural properties and corrupts the target calibration gradient with noise. It verifies that:

- peer property maps reach the generated support;
- the STE can train the generator end-to-end;
- real peer coordinates beat coordinate-shuffled peers;
- the fixed support drives a lower true objective than the controls;
- all update invariants remain intact.

Because the optimum is constructed from peer information, success is expected if the implementation is correct. The study does not establish that real transformer peers contain an analogous prior. Only the real-data protocol can test that hypothesis.
