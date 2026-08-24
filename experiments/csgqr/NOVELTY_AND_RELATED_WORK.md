# CSGQR — Novelty and Related Work

Primary-source review conducted 2026-08-24. Every paper below was located by
search and its abstract or full text read; where a detail could not be confirmed
from the source, that is said explicitly rather than glossed.

The brief is direct about what may not be claimed:

> Do not claim novelty for: confidence; quantization; gating; frozen weights;
> structured groups; or utility prediction individually.

and asks whether the implementation and evidence earn this composite:

> A structurally retired transformer component is retained as a frozen low-bit
> remnant; confidence or predicted need controls how much donor computation is
> invoked, and a separate marginal-utility router selects the most useful
> architecture-aligned donor groups before computation continues through the
> surviving model.

**Assessment, stated first: the composite is a recombination of established
mechanisms, and every individual mechanism has close prior art. Two of the three
halves are substantially anticipated by specific papers.** What this campaign can
defensibly claim is not the mechanism but the *measurement* — and specifically
the negative and structural results, which are the parts the prior work does not
report.

---

## 1. The three mechanisms, and who got there first

### 1.1 The "how much" gate — anticipated

Adaptive-depth decoding with a calibrated per-token confidence signal is a
mature line.

**CALM** (Schuster, Fisch, Gupta et al., NeurIPS 2022) is the canonical
treatment. It names the three problems CSGQR also faces — *which* confidence
measure, how to connect per-token exits to sequence-level constraints, and how to
attend back to hidden representations that were never computed — and calibrates
local exit decisions so that sequence-level metrics are provably maintained with
high probability. It reports full performance at a third to a half of the layers
on average. CSGQR's `confidence_whole` mode is a single-layer special case of
this, without the sequence-level conformal guarantee. **CALM's third problem is
also the exact reason CSGQR's attention phases B–D are unimplemented.**

**LayerSkip** (arXiv:2404.16710) trains for early exit and adds self-speculative
decoding on top.

**"What Layers When: Learning to Skip Compute in LLMs with Residual Gates"**
(Laitenberger, Kopiczko, Snoek, Asano, arXiv:2510.13876, Oct 2025) is closer
still to CSGQR's *deployed* gate: a lightweight learned gate reading the residual
stream makes per-token decisions about executing layer branches. That is
structurally the same object as this campaign's `need` probe — a linear readout of
the residual deciding whether to run a branch. I could not confirm from the PDF
whether its gate is strictly linear or whether it gates attention and MLP jointly,
so I do not assert the finer correspondence; the family membership is not in
doubt.

**GateSkip** (surfaced in review, not read in full) is described as equipping each
Attention/MLP branch with a sigmoid-linear gate on the residual stream, ranking
tokens by gate value and skipping low-importance ones under a per-layer budget.
If that description is accurate it is very nearly CSGQR's need-probe gate plus a
budget policy.

**QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs**
(Noh, Choi, Kim, arXiv:2602.10431, 2026) is the nearest whole-system collision:
token-adaptive layer skipping, gated on confidence derived from the model's own
activations, explicitly studied for robustness under weight quantization. It
skips **whole transformer layers** rather than routing within a retained
component, and it does not select channel subsets. But "confidence-gated,
token-adaptive execution of a quantized model" as a framing is taken.

**Also relevant to the negative result specifically:** "Early Exit Is a Natural
Capability in Transformer-based Models" (arXiv:2412.01455) studies early exit
*without* joint optimisation, which is the regime CSGQR's raw rank-0 logit lens
sits in. The literature consensus that untrained intermediate readouts are poor
predictors is what this campaign reproduced, with the mechanism visible (raw lens
top-1 agrees with dense top-1 only 26.9% of the time at L20).

**Verdict on this half: `NOT NOVEL`.** CSGQR's contribution here is a closed-form
cost bound (§3), not a mechanism.

### 1.2 The "which groups" router — substantially anticipated

**Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time**
(Liu, Wang et al., ICML 2023, arXiv:2310.17157) is the closest prior work to
CSGQR's significance router, and the correspondence is close enough that it must
be stated plainly. Deja Vu trains a small MLP predictor that reads the input to
block `j` and predicts the contextual sparsity — which attention heads and which
**MLP neurons** — of block `j+1`. It reports contextual sparsity that is on
average 85% structured sparse, over 95% of MLP parameters zeroable for a given
token, predictor validation accuracy ~93% in middle layers, and over 2× end-to-end
speedup.

CSGQR's group router is the same object at a coarser granularity: a learned
readout of the pre-block hidden state predicting which MLP channels are worth
computing for this token. Two differences are real rather than cosmetic:

1. **The prediction target.** Deja Vu's predictor learns to identify which
   neurons will be *active* — it approximates the dense output. CSGQR's router is
   trained on an **interventional** label,
   `U_{t,g} = KL(p^T‖p^0) − KL(p^T‖p^{+g})`, obtained by actually running the
   model with group `g` on and comparing against it off. That is a causal
   marginal-utility quantity, not a magnitude or activation proxy. This campaign
   also carries the control that makes the distinction testable: a
   `magnitude` ranking that the interventional router has to beat.
2. **What the routing is applied to.** Deja Vu sparsifies a **live** layer.
   CSGQR routes within a layer that has been **structurally retired** and retained
   only as a frozen low-bit remnant. The question is not "which neurons of this
   layer matter" but "how much of a component I have already deleted should I
   resurrect, and which parts".

Difference 2 is partly a framing difference — the arithmetic of "compute a subset
of MLP channels" is identical either way. Difference 1 is a genuine methodological
distinction, and it is the one worth defending.

**LoRAPrune** (Zhang et al., 2024) uses LoRA weights and gradients to score
Transformer components for structured pruning, and **MultiPruner**
(arXiv:2501.09949) compresses along residual blocks, MLP channels and attention
heads in sequence. Both establish MLP channel groups as the standard structured
unit — so "architecture-aligned groups of the intermediate dimension" is
conventional, exactly as the brief warns.

**Verdict on this half: `PARTIALLY NOVEL`** — the interventional utility label and
its controls are a real distinction; the mechanism is not.

### 1.3 The frozen low-bit remnant — not novel as a mechanism

Post-pruning recovery by adding a small trainable set while most weights stay
frozen (usually LoRA) is the standard recipe; "A Free Lunch in LLM Compression:
Revisiting Retraining after Pruning" (arXiv:2510.14444) is a recent survey-style
treatment. Joint structured pruning with mixed-precision quantization
(arXiv:2606.07819) and ensemble pruning-plus-quantization (SPQ, arXiv:2602.18420)
establish the pruning×quantization product space.

What CSGQR does differently is small but specific: instead of *deleting* the
pruned component and *adding* trainable capacity to compensate, it **keeps the
pruned component itself** at low bitwidth and spends it selectively. That is why
the `b13-wrong-layer` control (donor from a different layer, matched byte budget)
and the planned same-byte-LoRA control matter — they are the arms that decide
whether preserved *donor identity* is doing the work, or merely preserved
*capacity*. I did not find a paper running that specific control.

**Verdict: `NOT NOVEL` as a mechanism; the donor-identity control is a
contribution to the evidence, not to the method.**

### 1.4 Intermediate logit lens

Reading intermediate states through the model's own final norm and LM head is the
standard "logit lens" technique from interpretability, long predating this work.
CSGQR uses it as the *reference* measurement instrument and, having measured it,
rejects it as a deployment mechanism.

---

## 2. Where this campaign is on firmer ground

The defensible contributions are measurements, not mechanisms. In descending
order of confidence:

**2.1 A closed-form viability bound for exact-probability gating.** The
intermediate LM-head projection costs `2·|V|·d_model` MACs; a gated MLP costs
`2·3·d_model·d_ff`. So the monitor-to-monitored ratio is

```
|V| / (3·d_ff)
```

and the gate can only pay for itself when `|V| < 3·d_ff`. On this checkpoint that
is `32768 / 9216 = 3.556×` — a gate that skipped on *every* token would still
cost 3.56× what it saved. For current production models (`|V|` ≈ 128–256k,
`d_ff` ≈ 8–28k) the inequality essentially never holds. The qualitative point
("full-vocabulary exits are expensive") is folklore in the early-exit literature;
I did not find it stated as this inequality, with the crossover condition made
explicit. It is cheap, checkable, and it settles a design question before any
experiment.

**2.2 A measured monitor-cost hierarchy on the same card.** Need probe 0.0109% of
the gated MLP, group router 0.2604%, exact lens 355.6% — four orders of magnitude,
with the probe reaching AUROC 0.936 where the raw lens reaches 0.55–0.60. This is
the quantitative form of "use a probe, not a projection".

**2.3 An instrument-power negative result that is likely to generalise.** The
repository's own generation-and-score endpoint cannot resolve the deletion of one
MLP layer on a 0.47B model at n=510: paired sd 0.149 against an effect of 0.0114,
55.5% of cases scoring identically while 94.5% of replies changed, and after
Holm–Bonferroni **no** planned contrast significant — including deletion damage.
Any single-layer compression study using a coarse rubric on a few hundred
generations is measuring its instrument. I have not seen this stated with the
arithmetic attached, and it is the kind of finding that saves other people GPU
time.

**2.4 A candidate sign inversion that did not survive.** I had flagged one here:
static top-k by singleton marginal utility reconstructs `F` strictly better than
random at every budget (relative output error 0.745 vs 0.841 at ρ=0.25) while
appearing to produce a *worse* next-token distribution (KL 0.371 vs 0.287). On
held-out prompts the ordering is the ordinary one — dynamic 0.137 < static 0.180 <
random 0.213 at ρ=0.25 — so the inversion was small-sample noise: the KL figure
came from 8 prompts, and the corroborating generation figure came from an endpoint
with no power (§2.3). **Withdrawn.** It is recorded rather than deleted because
the shape of the error is instructive: two independent measurements agreed, and
both were noise.

**2.4′ And two more small-sample claims that also did not survive.** I recorded
both at n=24 prompts; at n=414 held-out prompts neither holds.

*"Dynamic significance beats static at matched budget."* It does not. At ρ=0.25
static is 0.17822 [0.16990, 0.18710] and dynamic is 0.18253 [0.17356, 0.19252];
at ρ=0.50 the order reverses (0.13565 vs 0.12943). Both intervals overlap at both
budgets. The router is genuinely state-dependent — per-group selection rates run
0.548 down to 0.018 where random is flat at 0.25 — so this is not a degenerate
router. It works and buys nothing over a fixed global ranking.

*"The three-level interior point beats the binary gate."* It does not. The full
method (probe gate + dynamic significance, 17.5% skip / 52.2% full / 30.3%
partial) reaches KL 0.05839 at 0.6819 active, against the binary probe gate's
0.05703 at 0.6772 active. Slightly worse for slightly more compute — a tie.

**What does survive** is narrower and is about the *selector*, not the routing
policy: at matched ρ=0.25 the interventional significance ranking beats random
(0.17822 vs 0.21563, non-overlapping), beats the **magnitude** heuristic (0.23719
— itself worse than random), and **reversing** it is worse than random (0.24338).
So the ranking carries real, order-dependent information that a weight-norm proxy
does not. That is a genuine result about interventional utility labels, and it is
the part of the Deja Vu comparison in §1.2 that this campaign can actually speak
to.

The pattern across 2.4, 2.4′ is worth naming: **every claim that favoured the
method at small n dissolved at large n, while the claims that favoured the
controls survived.** That asymmetry is what an honest sample-size discipline looks
like from the inside.

---

## 3. The claim as it should be worded

The brief's composite is not earned as stated, for one specific reason and one
general one.

Specifically, "confidence or predicted need controls how much donor computation is
invoked" conflates two things this campaign showed are not interchangeable.
Confidence — the exact-lens form — is both unaffordable (§2.1) and uninformative
(AUROC 0.55–0.60, and `max_prob` has the *wrong sign*). Predicted need from the
state works. Writing them as alternatives implies a choice where the evidence
gives an answer.

Generally, the composite reads as a novel architecture when it is a recombination
whose parts are individually standard and whose two main halves are anticipated by
Deja Vu and by the confidence-gated layer-skipping line.

A wording the evidence supports:

> A structurally retired transformer MLP is retained as a frozen, physically
> packed 4-bit remnant. A linear probe on the pre-remnant residual — not an
> intermediate vocabulary projection, which costs `|V|/(3·d_ff)` times the
> component it gates and carries little routing signal — predicts how much of the
> remnant to execute, and a second linear probe trained on *interventional*
> per-group marginal utility selects which architecture-aligned groups to
> execute, before computation continues through the surviving model.

That is narrower, and it is what was built and measured.

---

## 4. What would make the method novel rather than the measurement

Honest list, in the order I would try them:

1. **Multi-layer scope.** Every systems claim here is capped by one MLP being
   1.99% of the model. Deja Vu's 2× speedup comes from sparsifying *every* layer.
   A retired-remnant method that works across many layers is a different and much
   more interesting object, and is the only route to a systems result.
2. **The donor-identity question, answered.** If a correct INT4 donor
   demonstrably beats a wrong-layer donor, random frozen weights, *and* a
   same-byte LoRA, then "preserve the pruned component rather than replace it"
   becomes a claim with evidence rather than a design preference. This is the
   cheapest genuinely-new result available, and `b13-wrong-layer` is already run.
3. **Interventional versus magnitude routing, head to head.** Deja Vu-style
   activation prediction against CSGQR-style marginal-utility prediction, matched
   budget, same layer. If the interventional label wins, that is a contribution to
   the contextual-sparsity line and not merely to this method.
4. **The sequence-level gap.** CALM's conformal machinery connects per-token
   decisions to sequence-level guarantees. Nothing here does. Given that this
   campaign measures 94.5% reply divergence at near-zero mean quality change, the
   per-token/sequence-level gap is where the real risk lives.

---

## Sources

- [Confident Adaptive Language Modeling (CALM), Schuster et al., NeurIPS 2022](https://arxiv.org/abs/2207.07061)
- [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time, Liu et al., ICML 2023](https://arxiv.org/pdf/2310.17157)
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/pdf/2404.16710)
- [QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs, Noh et al., 2026](https://arxiv.org/pdf/2602.10431)
- [What Layers When: Learning to Skip Compute in LLMs with Residual Gates, Laitenberger et al., 2025](https://arxiv.org/pdf/2510.13876)
- [Early Exit Is a Natural Capability in Transformer-based Models](https://arxiv.org/pdf/2412.01455)
- [MultiPruner: Balanced Structure Removal in Foundation Models](https://arxiv.org/pdf/2501.09949)
- [A Free Lunch in LLM Compression: Revisiting Retraining after Pruning](https://arxiv.org/html/2510.14444v2)
- [Joint Structural Pruning and Mixed-Precision Quantization for LLM Compression](https://arxiv.org/html/2606.07819v1)
- [SPQ: An Ensemble Technique for Large Language Model Compression](https://arxiv.org/pdf/2602.18420)
- [Accelerating Large Language Model Inference via Early-Exiting Algorithms](https://arxiv.org/pdf/2509.05915)
