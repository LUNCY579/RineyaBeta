# DAST V2 hypothesis tree (H1–H7)

V1 is closed negative. This file does **not** propose a new architecture to run
next. It records competing explanations for the ordered deficit and the evidence
that would distinguish them. Successor choice happens **after** B1/B2/C1, and
only one successor is implemented.

Claim boundary: [`DAST_V1_CLOSURE.md`](DAST_V1_CLOSURE.md). Ranked protocols:
[`DAST_NEXT_EXPERIMENTS.md`](DAST_NEXT_EXPERIMENTS.md).

Decodability ≠ optimization ease ≠ causal capability. Lower `L_state` is not a
reasoning improvement. Reverse-fit is not reverse reasoning.

---

## Hypotheses

### H1 — Depth-localized interference

One checkpoint/state pairing, especially the deepest coherent checkpoint (layer
27, final-norm readout), causes most of the ≈1 pp ordered deficit.

- **Currently:** unknown. V1 always supervised all four depths together.
- **Test:** B1 single-checkpoint arms (ckpt k → state k) plus mismatch ckpt4→s1.
- **If yes:** Localized DAST (drop or move the bad checkpoint). Inspect output-head conflict (layer 27 readout vs layer-28 LM head).
- **If no:** not a single-site story.

### H2 — Gradient conflict

`L_state` gradients under forward chronological supervision anti-align with
`L_LM` more than reverse or flat-target supervision.

- **Currently:** compatible with trained `L_state` ordered > reverse in every seed, while frozen geometry mildly prefers forward. Not demonstrated.
- **Test:** B2 cosine(g_LM, g_state) on matched minibatches for ordered / reverse / final-repeated (and B1 arms if they run).
- **If yes:** λ reduction (B3) or soft constraint; the parked [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md) matrix becomes eligible **only then**.
- **If no:** do not force the conflict story.

### H3 — Overconstraint

The model already represents semantic progress flexibly. A rigid one-state-per-depth map removes useful degrees of freedom.

- **Currently:** supported by C1/C1c, more strongly than "weakly". Frozen K=4 permutation landscape is nearly flat (identity median rank 2/24, reverse 10/24, spread ~6% of `L_state`). The layer×stage profile peaks at **layer 1 for every stage**, and C1c shows **layer 0 — the token-embedding centroid, before any decoder layer — matches it and beats every layer from 3 up**. Four *consecutive bottom* layers (0/1/2/3) fit the objective slightly **better** (`L_state` 0.111) than V1's spread-out sites (0.124), while the trained projector reaches only 0.263. The `embedding_only` control lands at exactly chance top-1 (0.2500), so this is not a metric that scores well on anything: the objective genuinely needs four distinguishable representations, it just does not need a *chronological depth axis*. Readable ≠ structured as a depth clock.
- **Test:** B1 (all four together hurt, singles do not) plus B3 (mechanism saturates while accuracy falls as λ↑). C2 π*(x) heterogeneity.
- **If yes:** Soft/Adaptive DAST — only after C2 shows stable example-wise routing, not because the landscape is flat.

### H4 — Target-specific conflict

The deepest cumulative/final centred target interferes with output-producing representations.

- **Currently:** unknown. Final-repeated (same final state at every depth) was **capability-baseline-like** on average, which argues against “any deep auxiliary is costly,” but does not isolate deepest→final under ordered mapping.
- **Test:** B1 ckpt4→s4 vs ckpt4→s1 vs earlier checkpoints.
- **If ckpt4→s4 hurts and ckpt4→s1 does not:** target content. If both hurt: deepest-layer auxiliary itself.

### H5 — Fixed-order assumption is wrong

Different problems prefer different state→depth assignments, so a universal π is inappropriate.

- **Currently:** open. C1 is a **pooled** landscape. Flat average can hide stable π*(x).
- **Test:** C2, held-out per-family permutation preference. Not yet run.
- **If yes:** Adaptive DAST with matched-budget controls (uniform α, shuffled α, baseline, final-repeated).

### H6 — Absolute states are the wrong object

Semantic motion Δh_k ↔ Δs_k may capture computation better than absolute identities.

- **Currently:** materially strengthened by C1c, though still not a capability result. PHASE0 noted between-problem same-stage delta cosine 0.94–0.99 vs within-problem cross-stage −0.32. C1c adds the sharper point: because `FrozenTextTargetEncoder` builds targets from the **frozen input embedding**, the absolute-state objective is essentially satisfiable from the prompt's token-embedding centroid, and the depth stack neither supplies nor improves it. That is a defect in the *target object*, not in the depth mapping — which is exactly H6's claim, arrived at independently of the motion geometry. A motion target would at minimum have to demonstrate it is **not** similarly recoverable at layer 0; that becomes a mandatory control, not an optional one.
- **Test:** only after B/C, and only if motion wins on **capability and conflict**, not prettier probes. **Spends the SSOT §13 one-mechanism revision budget.**

### H7 — Scale / depth dependence

The ordered deficit may be peculiar to Qwen3-0.6B.

- **Currently:** untested. Forbidden as the next GPU spend until a mechanism is localized on this stack.
- **If later:** normalized/functional checkpoints on Rineya, not copied 7/14/21/27.

---

## Decision tree (after B1/B2/C1)

```text
B1: is the deficit concentrated at one checkpoint?
  YES, deepest→final → H1/H4 → LOCALIZED DAST (move/drop that site). Need replicate + B2.
  YES, several forward sites each small → H2/H3 → B2 then maybe B3; not a new architecture yet.
  NO, only the four together hurt → H3 overconstraint → C2 before Adaptive.
  deepest→s1 harmless AND deepest→s4 hurts → H4 target content.
  both deepest mappings hurt → deepest auxiliary itself.

B2: is ordered cosine(g_LM, g_state) systematically more negative than reverse/flat?
  YES → H2. λ sweep B3 frozen at {0, 0.1, 0.25, 0.5, 1.0}×λ0. Do not sign V2 until this fires.
  NO → drop H2 as the primary story.

C1/C1c (done): landscape nearly flat; no chronological layer ridge; the whole
  signal is present at layer 0, before any decoder layer, and the bottom four
  layers fit it better than V1's sites.
  Supports H3/H5 as live and raises H6 from speculative to a target-design defect.
  Does not implement Adaptive by itself.
  Refutes "transformers reason backwards" as a frozen-geometry claim.
  NOTE: any successor keeping absolute embedding-derived targets inherits this
  defect. A layer-0 recoverability control is now mandatory for every new target.

C2 (not done): stable π*(x) by category/length?
  YES → SOFT/ADAPTIVE DAST.
  NO → do not build a router to fit noise.

Reverse capability (not L_state) > baseline, replicated?
  YES → consider Bidirectional/Convergent. V1 reverse is baseline-like, so this is currently OFF.
  NO → do not implement reverse streams.

ΔH beats absolute states on generation AND conflict?
  YES → motion DAST (revision budget).
  Probe-only win → do not adopt.

Mechanism localized AND we still need a domain check?
  THEN H7 cross-scale / Rineya. Not before.

B1 unstructured <2 pp AND B2 similar across arms AND C2 null?
  CLOSE the DAST training direction. Keep the probe/instrumentation framework.
  CAUTION: "B1 unstructured <2 pp" is NOT evidence of absence. The one-seed MDE on
  this endpoint is 2.27 pp, so that outcome is what an underpowered test returns
  whatever the truth is. Close on B2 + C2 + the C1c target-design defect, not on a
  null accuracy column. See DAST_NEXT_EXPERIMENTS.md, power gate.
```

**Do not implement more than one successor.** Bidirectional requires reverse **accuracy** advantage. Adaptive requires C2. Motion spends the revision budget. The parked V2 9-run matrix is H2-only and stays unsigned until B2 supports it.
