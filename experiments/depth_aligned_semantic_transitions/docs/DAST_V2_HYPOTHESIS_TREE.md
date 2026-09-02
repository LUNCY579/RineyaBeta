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

- **MEASURED — NOT SUPPORTED.** B2, seed 29, three arms, 32 matched minibatches each:

  | arm | global cosine | LoRA-only cosine | frac modules < 0 |
  |---|---:|---:|---:|
  | ordered | −0.00167 ± 0.00156 | −0.00623 ± 0.00516 | 0.534 |
  | reverse | +0.00168 ± 0.00177 | +0.00475 ± 0.00511 | 0.479 |
  | final_repeated | −0.00063 ± 0.00121 | −0.00364 ± 0.00668 | 0.481 |

  Every contrast is inside 2σ (largest |t| = 1.51). The ordering is consistently in H2's predicted direction — ordered is most negative and most anti-aligned on all three metrics — but the magnitudes settle it: |cosine| ≈ 0.006. **The objectives are not fighting; they are essentially orthogonal**, which is what two unrelated objectives look like in a 10M-dimensional space.
- **Consequence:** per the frozen stop rule, H2 is **not** the primary story and [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md) stays **PARKED and unsigned**.
- **The transient possibility was then TESTED — B2-T, AMBIGUOUS.** 5 init seeds x 3 arms, paired per-step in the EARLY window, one-sample t over seeds ([`FINDINGS_B2T_TRAJECTORY.md`](FINDINGS_B2T_TRAJECTORY.md)):

  | contrast | mean D | t | p | significant |
  |---|---:|---:|---:|---|
  | **ordered − reverse** | **−0.003566** | **−4.57** | **0.0103** | **yes**, 5/5 seeds negative |
  | ordered − final_repeated | +0.001047 | +0.21 | 0.847 | no |

  The frozen rule required BOTH, so the verdict is AMBIGUOUS and no successor is licensed.
- **But the stronger reason not to revive H2 is a dissociation.** Accuracy: ordered−final_repeated is **significant** (−1.17 pp, p=0.0167) while ordered−reverse is **null** (p=0.128). Gradients: exactly the reverse. **The conflict appears where the deficit is absent and is absent where the deficit is.** H2 was proposed as the explanation *for the deficit*; this pattern is anti-correlated with that story, not merely short of significance.
- **What survives as a fact:** forward vs reverse depth assignment differ reliably in early-training gradient geometry at matched target multiset. That is optimisation geometry, not reasoning, and it is dissociated from capability.
- **Caveat on the instrument:** validity check 2 partially FAILED — ordered's LATE window disagrees in sign with the step-250 snapshot, so the trajectory's convergence end is not interpretable. The primary test does not read that window.

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

- **Currently:** **answered, and the answer is "real but far too small to act on."** C2 split-half reliability against a label-shuffled null: generator r = +0.3701 (null +0.0814), answer_type +0.3893 (null +0.1024), step_count +0.3650 (null +0.2076, partly a difficulty confound). Per-problem argmin agreement 0.1650 against 0.0417 chance. Family preference is genuinely reproducible — roughly 4× its null. **But the entire permutation spread is 1.89% of `L_state`**, so what replicates is a ~0.003 `L_state` difference, and modal preference replicates only 13–15% of the time.
- **Test:** C2, done. [`FINDINGS_C1_FROZEN.md`](FINDINGS_C1_FROZEN.md) §C2.
- **Verdict:** C2 passes the reliability test and **fails the magnitude test**. Adaptive DAST stays **forbidden**: a router would select among options differing by 0.2% of an objective C1c shows is satisfiable from the token embedding. Reliable ≠ consequential. Reopening requires evidence that the preferred assignment changes *capability*, not `L_state`.

### H6 — Absolute states are the wrong object

Semantic motion Δh_k ↔ Δs_k may capture computation better than absolute identities.

- **Currently:** materially strengthened by C1c, though still not a capability result. PHASE0 noted between-problem same-stage delta cosine 0.94–0.99 vs within-problem cross-stage −0.32. C1c adds the sharper point: because `FrozenTextTargetEncoder` builds targets from the **frozen input embedding**, the absolute-state objective is essentially satisfiable from the prompt's token-embedding centroid, and the depth stack neither supplies nor improves it. That is a defect in the *target object*, not in the depth mapping — which is exactly H6's claim, arrived at independently of the motion geometry. A motion target would at minimum have to demonstrate it is **not** similarly recoverable at layer 0; that becomes a mandatory control, not an optional one.
- **MEASURED — DOES NOT CLEAR THE GATE.** [`../artifacts/h6_motion_target_control.json`](../artifacts/h6_motion_target_control.json). Two criteria, CPU only, no training, no budget spent.

  | criterion | absolute | motion_raw | motion_centered |
  |---|---:|---:|---:|
  | problem information (`L_shuf − L_true`) | 0.568 | 0.538 (0.95×) | 0.543 (0.96×) |
  | depth advantage (`L(0/1/2/3) − L(7/14/21/27)`) | −0.0114 | **−0.0046** | **−0.0052** |

  Criterion 1 **passes**, contrary to the prediction that PHASE0 §3.8's 0.94–0.99 between-problem delta cosine made deltas per-stage constants: motion retains 95–96% of the problem-specific information absolute states carry. That pessimistic reading is **wrong** and is recorded as such.

  Criterion 2 **fails**, exactly as absolute states do. Layers 0/1/2/3 still fit motion targets better than the V1 sites, and putting motion on the hidden side too (`delta_hidden`) makes every variant *worse*. Motion is mildly less harmed by the real depths (−0.005 vs −0.011) — a real difference, still the wrong sign.
- **Verdict:** motion changes *what* is supervised without changing the fact that the supervised quantity does not require the depth stack. **Revision budget NOT spent.**
- Practical note: 1118 motion targets are exactly zero before filtering, because a duplicated consecutive state gives a zero delta. The distinctness filter is load-bearing for this variant, not optional.

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

B2 (DONE): is ordered cosine(g_LM, g_state) more negative than reverse/flat?
  NO. All arms within 2 sigma of each other and within ~0.01 of orthogonal.
  => H2 dropped as the primary story. V2 matrix stays PARKED.
  => B3 (lambda sweep) loses its trigger: it was conditional on B1/B2 showing
     interference, and B2 shows none.

C1/C1c (done): landscape nearly flat; no chronological layer ridge; the whole
  signal is present at layer 0, before any decoder layer, and the bottom four
  layers fit it better than V1's sites.
  Supports H3/H5 as live and raises H6 from speculative to a target-design defect.
  Does not implement Adaptive by itself.
  Refutes "transformers reason backwards" as a frozen-geometry claim.
  NOTE: any successor keeping absolute embedding-derived targets inherits this
  defect. A layer-0 recoverability control is now mandatory for every new target.

C2 (DONE): stable pi*(x) by family?
  Reliability YES (r ~ +0.37-0.39 vs null ~ +0.08-0.10), magnitude NO (spread is
  1.89% of L_state; argmin replicates 13-15%).
  => do NOT build a router. It would fit a real but 0.003-sized effect on an
     objective that C1c shows needs no depth at all.

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

---

## Where the tree has actually landed (2026-09-01)

Three of the four cheap diagnostics have run, and they converge:

| branch | result | consequence |
|---|---|---|
| **B2** | cosines orthogonal, no arm difference | H2 dropped; V2 matrix stays parked |
| **C2** | reliability passes (r≈0.37–0.39 vs null 0.08–0.10), magnitude fails (spread 1.89% of `L_state`) | Adaptive DAST stays forbidden |
| **C1c** | the objective is satisfiable from the layer-0 token embedding; four bottom layers beat the V1 sites | H6 raised to a diagnosed target-design defect |
| **B1** | not run — the endpoint power analysis shows its accuracy column cannot answer the localization question at any affordable seed count | H1/H4 remain formally open but are not reachable |
| **B2-T** (H2 as a trajectory) | AMBIGUOUS. ordered−reverse replicates (t=−4.57, 5/5 seeds); ordered−final_repeated null. Dissociated from where the accuracy deficit actually is | H2 not revived; V2 stays parked |
| **H6 motion gate** | passes problem-information (0.96×), **fails** the layer-0 control (−0.005) | motion disqualified; revision budget unspent |
| **Stage E** | huge power (p≈1e-48) but magnitude uncorrelated with accuracy (r=+0.188), cannot resolve ordered−reverse (p=0.076), fires at p=9e-45 on a null contrast | **not** a capability screen; B1/B3 stay unanswerable |

**Recommendation: CLOSE the DAST training direction as formulated, and keep the
probe/instrumentation framework.** The tree's close-out branch asked for B2 similar
across arms and C2 null; B2 is similar across arms, and C2 is null *in magnitude*
even though it is positive in reliability. C1c supplies the reason the whole
formulation underdelivers: `L_state` supervises a quantity that needs no depth.

This is a close on the **target object**, not merely on the depth mapping. Any
successor must change what is supervised and must clear a **layer-0 recoverability
control** first. Motion/ΔH was the only remaining candidate with a mechanism-level
argument, and it has now been tested against that control and **failed it** without
spending the revision budget.

**The tree is therefore fully traversed.** Every branch that could have licensed a
successor has been measured and closed:

- H1/H4 → unreachable on any affordable endpoint (power), and Stage E does not fix it.
- H2 → refuted (orthogonal gradients).
- H3 → supported, which is a diagnosis rather than a mechanism to build.
- H5 → reliable but 1.89%-of-`L_state` small; Adaptive forbidden.
- H6 → motion fails the layer-0 control.
- H7 → would replicate a defective objective; not licensed.

CLOSE is now a *complete* negative rather than a stopping point: the alternatives
were excluded, not merely left unfunded. The durable output is the probe and
instrumentation framework, plus the layer-0 recoverability control as a standing
gate on any future depth-supervision target.

**Do not implement more than one successor.** Bidirectional requires reverse **accuracy** advantage. Adaptive requires C2. Motion spends the revision budget. The parked V2 9-run matrix is H2-only and stays unsigned until B2 supports it.
