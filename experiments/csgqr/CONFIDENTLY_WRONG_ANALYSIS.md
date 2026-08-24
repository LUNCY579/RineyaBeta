# CSGQR — Confidently Wrong Skips

The failure mode that decides whether a confidence gate is deployable: the model
is confident, the gate therefore skips the remnant, and the remnant would have
repaired an error the truncated model was about to make.

Source: `results/trace_L20_int4/confidence_analysis_d0.1.json` — layer 20, INT4
remnant, 24 groups, **798 reply positions from 96 prompts**, `δ = 0.1`.

A skip is **confidently wrong** when all three hold at the same position:

1. confidence ≥ `τ_skip`, so the gate skips;
2. the deleted path's top-1 token disagrees with the dense teacher's; and
3. the full remnant's top-1 **agrees** with the dense teacher's — so running it
   would have repaired the disagreement.

Condition 3 is what makes this a gate failure rather than a general model error:
the repair was available and the gate declined it.

---

## 1. The remnant is genuinely load-bearing

Before blaming the gate, establish there is something to lose.

| quantity | value |
| --- | ---: |
| `KL(dense ‖ deleted)` | 0.31259 |
| `KL(dense ‖ full remnant)` | 0.00913 |
| mean utility `U` | 0.30347 |
| KL recovery ratio | **0.9708** |
| positions where deleted top-1 ≠ dense top-1 | **198 / 798 = 24.81%** |
| of those, repaired by the full remnant | **170 / 198 = 85.86%** |

So the INT4 remnant recovers 97% of the divergence damage and repairs 86% of the
argmax disagreements deletion causes. There is a great deal for a gate to get
wrong.

---

## 2. Confidently-wrong skips at every useful threshold

Gating on `max_prob` from the raw rank-0 lens:

| `τ_skip` | skip rate | false skip \| needed | confidently wrong | as % of all positions | utility forgone |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.6178 | 0.5870 | **102** | 12.78% | 134.63 |
| 0.70 | 0.4211 | 0.3866 | **69** | 8.65% | 98.22 |
| 0.80 | 0.3609 | 0.3259 | **57** | 7.14% | 88.41 |
| 0.90 | 0.2544 | 0.2206 | **37** | 4.64% | — |
| 0.95 | 0.1860 | 0.1620 | **25** | 3.13% | — |
| 0.99 | 0.1053 | 0.1053 | **15** | 1.88% | — |

Read the last row carefully. At `τ = 0.99` the gate skips only **one token in
ten** — it has given up almost all of the compute saving that motivates the
method — and it *still* mis-skips 15 positions where the remnant would have
repaired a wrong argmax. There is no threshold at which this gate is both useful
and safe.

`false skip | needed` is `P(gate skips | U > δ)`. At the nominal operating point
`τ = 0.9` it is **0.2206**: more than a fifth of the tokens that genuinely need
the remnant are skipped.

---

## 3. Why: the lens is not measuring the decision it gates

The mechanism is visible in the same trace and it is not subtle.

> **The raw lens's top-1 token agrees with the dense model's top-1 only 26.94% of
> the time at this depth.**

The gate is asked "is this next-token decision mature?" and answers using a
distribution that names a different token three times out of four. Its confidence
is a well-formed number about the wrong question.

Calibration is correspondingly bad:

| metric on `max_prob` | value |
| --- | ---: |
| ECE (15 equal-width bins) | **0.3557** |
| MCE | 0.6961 |
| Brier | 0.3283 |

An ECE of 0.36 on a signal whose range is [0, 1] is not a miscalibration to be
fixed by a temperature; it is a signal that does not track the thing it is being
compared against.

This reproduces what the early-exit literature reports for untrained intermediate
readouts, with the mechanism measured rather than inferred.

---

## 4. Every confidence feature, and the sign of the effect

H1 predicts `E[U | C low] − E[U | C high] > 0`: low confidence should mean the
remnant is more needed. Against a **mean utility of 0.30347**:

| feature | gap (low − high) | AUROC | AUPRC | Pearson(1−C, U) | Spearman(1−C, U) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `max_prob` | **−0.00115** | 0.5515 | 0.6421 | +0.0173 | +0.1282 |
| `neg_entropy` | +0.00892 | 0.5649 | 0.6489 | +0.0026 | +0.1313 |
| `prob_margin` | +0.00543 | 0.5468 | 0.6488 | +0.0190 | +0.1267 |
| `logit_margin` | +0.02393 | 0.5374 | 0.6497 | −0.0146 | +0.1104 |

`max_prob` — the design's primary feature — has the **wrong sign**. The largest
gap belongs to `logit_margin` and is 7.9% of the mean utility. AUROC 0.537–0.565
is not deployable, and all four cluster-bootstrap intervals overlap heavily.

**One honest nuance.** Spearman is small but *consistently positive* (+0.11 to
+0.13) across all four features while Pearson is ≈ 0. So a weak monotone
relationship in the direction H1 predicts does exist — it is simply far too small
to gate on, and it is invisible in the linear correlation because the utility
distribution is heavily skewed. The correct verdict is not "no relationship" but
"a relationship two orders of magnitude too weak to act on".

---

## 5. Category analysis: what is not available

The brief asks for confidently-wrong cases to be categorised (reasoning, numbers,
named entities, abstention, contradiction, long-context retrieval, final answer,
punctuation, semantic token, procedural step).

**This is not reported, and the reason is not effort.** The trace records token
ids, not token roles: `dense_top1`, `deleted_top1`, `remnant_top1`, `gold_token`,
and the position index. Assigning a linguistic role to a single token position
would require either a tagger applied to the decoded context or a hand
adjudication of 102 positions. Doing the former and calling its output "reasoning
tokens" would be inventing a taxonomy and attributing it to measurement.

What *is* available from the recorded fields, and is reported instead:

| quantity | value |
| --- | ---: |
| mean first-flip position, deletion | 2.07 |
| mean first-flip position, INT4-all | 3.49 |
| mean first-flip position, wrong-layer donor | 1.27 |

Divergence starts **early** in a reply — around token 2 for deletion. Combined
with a median reply length of 7 tokens under this protocol, that means most of a
divergent reply is generated after the divergence, which is the sequence-level
half of the problem and is why `sequence_flip_rate` is reported alongside
token-level agreement everywhere.

A proper categorisation needs a longer-generation protocol and an explicit
annotation pass. It is listed in `MAXIMUM_POTENTIAL_ROADMAP.md`.

---

## 6. Do the proposed remedies help?

The brief lists candidate fixes. Each assessed against evidence:

| remedy | verdict | evidence |
| --- | --- | --- |
| **entropy instead of max-prob** | **No.** AUROC 0.5649 vs 0.5515 — the best of four bad options. | table in §4 |
| **margin features** | **No.** 0.5468 and 0.5374, no better. | table in §4 |
| **prediction stability across depths** | **Not established.** `C^S = 1 − JS/log 2` is implemented and unit-tested but was never wired to a gate. | `confidence.py` |
| **conservative thresholds** | **No.** This is the §2 table: `τ = 0.99` still mis-skips 15 positions while saving almost nothing. Raising `τ` trades the method's entire purpose for a residual failure rate. | §2 |
| **significance prior** | **Not applicable as a fix to this failure.** Significance decides *which* groups, not *whether*. It cannot repair a skip. |
| **neighbouring-layer agreement** | **Not run.** Would need traces at more than one layer. |
| **temperature calibration** | **No, and structurally so.** A temperature is monotone in the logits, so it cannot change the *ranking* of positions by confidence — and AUROC depends only on that ranking. Temperature can fix ECE 0.3557; it cannot move AUROC 0.5515 by construction. |
| **a trained alignment lens** | **Not established, and the wrong direction.** Never trained. But the exact projection costs `|V|/(3·d_ff) = 3.556×` the MLP it gates, so even a perfect lens cannot pay for itself at this scope. |
| **a hidden-state probe** | **Yes — this is the fix.** AUROC **0.9358** on held-out prompts against the raw lens's 0.55–0.60, at **0.0109%** of the gated MLP's cost. Permuted-label control 0.5065 (no leakage); prompt-mean *oracle* 0.6976, which the probe clears, so it resolves individual tokens rather than prompt-level structure. |

That last row is the load-bearing one, and note what it implies about the
diagnosis in §3. The information needed to gate this remnant **is present in the
pre-remnant residual** — a linear readout of it reaches AUROC 0.936. It is the
vocabulary projection that destroys it. So the failure is not "the model does not
know whether it needs this layer"; it is "asking the question through the LM head
throws the answer away".

---

## 7. Conclusion

Confidence gating on an intermediate vocabulary projection **fails at this layer,
and fails for a diagnosable reason**:

* the lens names a different token from the dense model 73% of the time, so its
  confidence is about the wrong decision;
* consequently AUROC is 0.537–0.565 and `max_prob` has the wrong sign;
* every threshold that saves meaningful compute mis-skips a substantial fraction
  of the tokens that need the remnant, and even `τ = 0.99` — one token in ten
  skipped — leaves 15 of 798 positions confidently wrong;
* temperature calibration cannot help, because AUROC is rank-based;
* and the projection costs 3.56× the component it gates, so it could not be
  afforded even if it worked.

The same information, read directly from the residual by a linear probe costing
0.0109% of the gated MLP, reaches AUROC 0.9358. **The "how much" gate is viable
only as a state probe, and the exact-confidence form of CSGQR should be regarded
as falsified rather than untuned.**
