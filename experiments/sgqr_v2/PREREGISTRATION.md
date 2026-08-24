# SGQR-v2 — Preregistration

Written **before any SGQR-v2 experiment was run**, so that the decision rules
below cannot be adjusted after seeing results. Committed as its own commit for
that reason; the commit hash is the timestamp.

Branch: `research/sgqr-v2-remnant-recovery`, based on
`research/csgqr-quantized-remnant` at `dc977dd`.

---

## 1. The question

> When a transformer layer is structurally retired, is retaining a low-bit copy
> of **that exact deleted computation** a uniquely efficient recovery mechanism,
> or is its benefit merely equivalent to adding **generic parameter capacity**
> back to the damaged model?

Everything below is designed to make a "generic capacity" answer *easy to
observe* if it is true. That is the point.

---

## 2. What is carried forward, and what is dropped

CSGQR-v1 is a completed experiment (`experiments/csgqr/`). Its history is not
rewritten. From it:

**Carried forward as the core:**

* the exact frozen quantized remnant (INT4 recovers 96.3% of single-layer
  deletion divergence);
* interventional significance ranking (beats random *and* beats magnitude, which
  is itself worse than random);
* a cheap linear probe on the hidden state, **but re-targeted** to predict
  recovery utility rather than model confidence.

**Dropped, on v1 evidence:**

| dropped | why |
| --- | --- |
| vocabulary-head confidence gate | costs `\|V\|/(3·d_ff)` = 3.56× the layer it gates in MACs, 13.96× measured. Cannot pay for itself at any threshold. |
| per-token dynamic significance router | genuinely state-conditioned, and does not beat a fixed global ranking at either budget tested (ρ=0.25 or 0.50), intervals overlapping. |
| three-level skip/partial/full ramp | ties a binary gate at matched compute. |

Dropped components remain in `experiments/csgqr/` and stay runnable as
historical baselines. They are not deleted.

---

## 3. Preregistered layer selection

**Rule, fixed in advance: the four layers with the lowest deletion damage in the
committed 28-layer scan** (`experiments/csgqr/results/layer_scan.json`, 64
prompts, 520 reply positions, token-weighted mean `KL(dense ‖ deleted)`).

That rule selects, in order:

| rank | layer | `delete_kl` | deleted top-1 agreement | INT4 recovery (1-layer) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **L25** | 0.1615 | 0.8558 | 0.9739 |
| 2 | **L13** | 0.1684 | 0.8288 | 0.9447 |
| 3 | **L14** | 0.1935 | 0.8096 | 0.9495 |
| 4 | **L24** | 0.2067 | 0.8288 | 0.9788 |

**Retirement set = {13, 14, 24, 25}.** Sum of individual single-layer damages =
0.7301 nats; whether joint deletion is additive, sublinear or superlinear is
**H5** below and is not assumed.

This is the set a practitioner would actually choose, it is derived from evidence
that predates this campaign, and it was fixed before any four-layer number
existed. Two of the four (13, 14) are adjacent, which is a property of the rule,
not a choice — adjacency is recorded as a possible confound for the interaction
analysis rather than engineered away.

---

## 4. Preregistered byte budget and the matched LoRA rank

One INT4 MLP remnant is **4,939,872 B** (measured, not estimated — the analytic
model and `torch.cuda.memory_allocated` agree exactly). So:

```
four remnants = 4 × 4,939,872 = 19,759,488 B = 18.844 MiB
```

LoRA parameters per surviving layer per unit rank, over all seven projections
(`q,k,v,o,gate,up,down`) with `d_model=1024`, `d_ff=3072`, 16 heads × 128,
8 KV heads:

```
r · [(1024+2048) + (1024+1024) + (1024+1024) + (2048+1024)
     + (1024+3072) + (1024+3072) + (3072+1024)] = 22,528 · r
```

Over the **24 surviving layers**, stored bf16 (2 B/param):

| rank | params | stored bytes | MiB | vs remnant budget |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4,325,376 | 8,650,752 | 8.25 | 43.8% |
| 16 | 8,650,752 | 17,301,504 | 16.50 | 87.6% |
| **18** | **9,732,096** | **19,464,192** | **18.56** | **98.5%** |
| 19 | 10,272,768 | 20,545,536 | 19.59 | 104.0% |
| 24 | 12,976,128 | 25,952,256 | 24.75 | 131.3% |
| 32 | 17,301,504 | 34,603,008 | 33.00 | 175.1% |

**`r = 18` is the byte-matched rank** (98.5% of the remnant budget). It is the
headline comparison. Ranks 8/16/24/32 are swept so that a reviewer cannot argue
the matched point was undertrained or under-ranked, and **the sweep includes
ranks that exceed the remnant's budget** — if LoRA needs 175% of the bytes to
win, that is itself the answer.

Both **parameter count and stored bytes** are reported for every arm, because the
formats differ (packed uint8 + fp16 scales versus bf16 dense).

### 4.1 An asymmetry that must be stated, not hidden

**The remnant is frozen and untrained; LoRA is trained.** LoRA additionally sees
task data the remnant never sees. This favours LoRA, and it is the honest form of
the comparison: the question is whether *free, untrained, exactly-preserved
weights* beat *trained generic capacity at the same storage cost*. Training FLOPs
for the LoRA arms are reported alongside, since they are a real cost the remnant
does not pay.

---

## 5. Preregistered hypotheses and falsification criteria

| | hypothesis | supported if | **falsified / weakened if** |
| --- | --- | --- | --- |
| **H1** | exact remnants beat unrelated parameters of the same size | exact ≫ wrong-layer remnant on capability metrics | exact ≈ wrong-layer |
| **H2** | the benefit is **not** merely capacity | exact > same-byte LoRA **and** > same-byte adapter | **same-byte LoRA ≥ exact** on the capability metrics |
| **H3** | interventional significance beats static proxies | significance top-k > random and > magnitude at matched active compute | significance ≈ random under four-layer conditions |
| **H4** | a utility probe beats random gating | probe > random token gate at matched activation | probe ≈ random |
| **H5** | multi-layer recovery does not collapse | four-layer INT4 recovery within ~10 points of the single-layer 96.3% | recovery collapses superlinearly |

**H2 is the campaign.** If same-byte LoRA matches or beats the exact remnant on
downstream capability, the correct conclusion is
`CAPACITY_EXPLANATION_LIKELY`, and it will be reported as such. A result that
favours the null is not a failed campaign.

---

## 6. Preregistered decision gate (after Phase 3)

Stop and publish a verdict before any systems/kernel work. The verdict label is
determined by these rules, fixed now:

```
STRONGLY_SUPPORTED          H1 and H2 both hold, with non-overlapping intervals
                            on a capability metric, and H5 holds.
PROMISING                   H1 holds; H2 holds on fidelity but not capability,
                            or holds with overlapping intervals.
MIXED                       H1 holds, H2 ambiguous (intervals overlap both ways).
CAPACITY_EXPLANATION_LIKELY same-byte LoRA >= exact remnant on capability.
NOT_SUPPORTED               exact ~= wrong-layer, i.e. H1 itself fails at scale.
```

No other label may be used, and the label is read off these rules rather than
chosen.

---

## 7. Preregistered endpoints

CSGQR-v1 established that the repository's **generation-and-score endpoint has no
statistical power** for single-layer effects: paired sd 0.14901 against a
deletion effect of 0.01135, needing ~1,350 cases where 510 exist, and after
Holm–Bonferroni *no* contrast was significant including deletion damage itself.

Four-layer deletion should be far more damaging, so that endpoint may become
usable. **Whether it is, is an empirical question and is checked first**
(Experiment 0 below). Both endpoints are reported regardless:

| endpoint | unit | role |
| --- | --- | --- |
| teacher-forced paired (`eval_forced`) | reply position, prompt-clustered | fidelity + gold-token accuracy; always reported |
| generation + rubric (`run_eval`) | case | capability; reported with its power stated |

**Primary capability metric:** gold-token top-1 accuracy and cross-entropy on
held-out reply positions, because CSGQR-v1 showed that KL-to-dense can move a
great deal while reference-prediction accuracy does not move at all. A mechanism
that repairs logits but not capability will not be described as recovering the
model.

---

## 8. Preregistered data roles

| split | source | used for |
| --- | --- | --- |
| significance calibration | `chat_dev` prompts 0–95 | interventional group ranking, in the **four-layer-deleted** context |
| probe training | `chat_dev` prompts 0–95 (disjoint sub-split by prompt) | utility probe, Phase 5 only |
| LoRA training | `artifacts/heal_mixes/B0` | every trained recovery arm |
| held-out evaluation | `chat_dev` prompts 96+ | every number in the report |
| sealed | `*_SEALED` | **never read** |

Thresholds and ranks are selected on validation, never on the held-out set. The
existing `--skip-prompts` guard refuses a probe-gated run whose prompts overlap
its own training trace.

---

## 9. Significance in the multi-layer setting — a definition fixed in advance

With four retired layers there are 4 × 24 = **96** selectable groups. The
marginal utility of group `g` in layer `j` depends on what is active elsewhere,
so the conditioning must be stated:

> `U_{j,g}` is measured **with all four layers deleted**, by activating only
> group `g` of layer `j` and measuring
> `KL(p_dense ‖ p_alldeleted) − KL(p_dense ‖ p_{+(j,g)})`.

That is the deployment-relevant conditioning: it asks "starting from the
compressed model, what does turning this one group back on buy?"

Budget is allocated by a **global** top-k over all 96 groups, so significance
decides *where* compute goes including across layers. Per-layer-uniform
allocation is retained as a control.

CSGQR-v1 established that the *additive* proxy for subset utility overstates by
up to 4× and must not be used to select a design (`NEGATIVE_RESULTS.md` N14).
**Every budget point in this campaign is measured by running the arm, not by
summing singleton utilities.**

---

## 10. Preregistered arm matrix

| ID | arm | purpose |
| --- | --- | --- |
| A | dense | upper bound |
| B | 4-layer deletion | damage floor |
| C | 4 exact INT4 remnants, ρ=1 | maximum recoverability retained |
| D | SGQR-v2: significance top-k, ρ ∈ {0.10, 0.25, 0.50, 0.75} | the method |
| E | random groups at matched ρ, ≥3 seeds | routing control |
| F | magnitude groups at matched ρ | static-proxy control |
| G | wrong-layer remnants, ≥3 mappings | layer-identity control |
| **H** | **same-byte LoRA, r ∈ {8,16,18,24,32}** | **the capacity control** |
| I | same-byte generic adapter (bottleneck / low-rank residual) | second capacity control |
| J | FP16 / INT8 / INT4 remnants at ρ=1 | bitwidth ablation |

---

## 11. Known limitations, stated before results exist

* **Shared GPU.** A concurrent experiment has been running on the same 8 GiB card
  throughout; CSGQR-v1 measured a 3.7× spread across identical benchmark runs.
  All timing carries `SHARED_GPU_TIMING — NOT SUITABLE FOR ABSOLUTE PERFORMANCE
  CLAIMS` unless an exclusive window is obtained.
* **Not an independent audit.** The implementation is authored by the same
  campaign that evaluates it.
* **One 0.47B checkpoint, one dev split, short replies** (`max_new_tokens=32`,
  median reply 7 tokens).
* **The remnant is untrained and LoRA is trained** (§4.1).
