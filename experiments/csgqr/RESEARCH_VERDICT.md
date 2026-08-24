# CSGQR — Research Verdict

---

# `CONTINUE_WITH_REDIRECTION`

The mechanism is real and two of its three components earn their place. The
composite as specified does not: the exact-confidence gate is unaffordable by
arithmetic and uninformative by measurement, the per-state dynamic router buys
nothing over a fixed ranking, and the three-level interior policy is a tie with a
binary gate. What survives is a **narrower, cheaper method** than the brief
describes, plus one strong donor-identity result and one negative systems result
that caps the whole direction at single-layer scope.

Redirect to **multi-layer scope with a binary state-probe gate and a static
interventional ranking**, and drop the exact lens, the per-state router and the
three-level ramp until multi-layer evidence justifies re-testing them.

---

## Hypothesis verdicts

| | hypothesis | verdict | basis |
| --- | --- | --- | --- |
| **H1** | `corr(1−C, U) > 0` | **`PARTIALLY_SUPPORTED`** | Spearman +0.1104 to +0.1313 across all four features — consistently the predicted sign — but Pearson ≈ 0, AUROC 0.5374–0.5649, and `max_prob` has the wrong sign on the mean gap. A real relationship two orders of magnitude too weak to classify on. |
| **H2** | confidence gating beats random activation at matched compute | **`SUPPORTED`** | Every gate beats a matched random-token gate with non-overlapping 95% intervals, by 1.40–1.67× across five activations. |
| **H3** | significance beats random, magnitude and static alternatives | **`SUPPORTED`** (vs random and magnitude) / **`NOT_ESTABLISHED`** (vs static) | ρ=0.25: significance 0.17822 vs random 0.21563 (non-overlapping); magnitude 0.23719 is *worse than random*; reversed 0.24338 worse still. But dynamic does not beat static — see H8. |
| **H4** | donor-specific preservation | **`SUPPORTED`** | Wrong-layer donor at identical bytes: KL 0.78649 vs 0.01049, i.e. 75× worse than the correct donor and 2.8× worse than installing nothing, and the only arm with a significant gold-token drop (−0.02689, CI [−0.03482, −0.01883]). *Scope:* beats a wrong frozen donor; same-byte LoRA untested. |
| **H5** | probe retains routing quality at lower overhead | **`SUPPORTED`** | Probe beats the interpolated lens frontier by 13.7–23.0% below 0.78 activation, loses by ≤9.3% above it, at ~110× lower measured monitor cost. |
| **H6** | a useful interior point `0 < ρ < 1` exists | **`FALSIFIED`** as "useful" | Interior points exist across the whole frontier, but none beats a binary gate at matched compute: 0.05839 at 0.6819 active against 0.05703 at 0.6772. |
| **H7** | consumer-hardware benefit without shifting cost | **`FALSIFIED`** | Every arm is slower than dense and uses more peak VRAM (956 → 1226 MiB, +28%). Cost is not merely unshifted, it is increased. |
| **H8** | group utility is sufficiently additive for independent ranking | **`PARTIALLY_SUPPORTED`** at the endpoints, **`FALSIFIED`** in between | Additive to 2.1% at `k=2` and 1.7% at `k=24`; at `k=6` the additive proxy overstates true utility by 1.14× (random), 1.57× (static) and **4.00×** (dynamic), and its oracle exceeds the utility of all 24 groups. Top-k as a *rule* still works (H3); the additive *score* is not a predictor. |
| **H9** | survivor adaptation helps | **`NOT_ESTABLISHED`** | Not run. Router-only, donor frozen throughout. |
| **H10** | multi-layer scalability | **`NOT_ESTABLISHED`** | Not run. This is the redirection. |
| **H11** | calibration behaviour is characterisable | **`PARTIALLY_SUPPORTED`** | Within one layer and one split, thresholds map predictably onto activation (six-point monotone frontiers for both gates). Cross-layer and cross-domain transfer untested. |
| **H12** | optimized kernels preserve the reference frontier | **`NOT_ESTABLISHED`** | No optimized backend exists. The 3.5× dequantization penalty is a reference-backend property and a fused kernel is a hypothesis, not a demonstration. |

## Advancement gates

| gate | status | basis |
| --- | --- | --- |
| **A** — implementation | **PASS** | 144 checks; `READY_FOR_SCIENTIFIC_EXPERIMENTS`; reference backend graded A. |
| **B** — quantized donor viability | **PASS** | INT8 recovers 99.7%, INT4 96.3% of deletion damage. |
| **C** — confidence usefulness | **PASS**, redirected | Beats random at matched compute (H2), but only via the state probe or a lens whose cost forbids deployment. |
| **D** — significance usefulness | **PASS** | Beats random and magnitude at matched budget; order-dependent. |
| **E** — interior frontier | **HALF** | Interior points exist; none is useful (H6). |
| **F** — systems value | **FAIL** | Slower and larger on every axis (H7). |

---

## The eighteen questions

**1. Was the implementation initially complete?**
No, in two distinct senses. The **remote branch contained no CSGQR work at all** —
byte-identical to `origin/r2-healing-harness`, 0/0 ahead-behind, empty diff — while
15 commits sat unpushed locally along with ten uncommitted evaluation runs. And
the local implementation, though substantial and serious, could not run the
dynamic half of its own method.

**2. What major bugs were found?**
Twenty-one this session, in `BUG_LEDGER.md` (plus six carried forward from
the previous campaign). Twenty are code defects; **D1** is a defect in my
own documentation. Twenty-six of the twenty-seven rows are `FIXED` with a
test that runs and passes; **S15** remains `OPEN`. The five that mattered:

* **S1 (CRITICAL)** — an arm requesting dynamic per-state significance silently
  executed the **static** ranking, with no error and correct-looking activation
  accounting. The matched-budget dynamic-vs-static contrast was *unrunnable*.
* **S2 (MAJOR)** — per-group α fitted against the whole MLP instead of its own
  donor group, injecting 1.32% relative error into a *bit-exact* remnant.
* **S8 (MAJOR)** — the gate materialised `[T, |V|]` plus three fp32 tensors for
  the whole prefill: ~1.6 GiB at 2048 tokens on an 8 GiB card. Fine on short dev
  prompts, OOM on long ones.
* **S19 (MAJOR)** — the FP passthrough remnant **aliased** the donor's storage
  instead of copying it, breaking both the byte accounting (3×) and the freezing
  guarantee for the arm that exists to be the clean reference.
* **D1 (MAJOR, documentation)** — the GPU microbenchmark table mixed figures
  from two different benchmark runs, and two of the three derived claims
  flattered the method. Found by re-deriving every quoted number from the
  committed JSON.

**3. What was changed?**
Bug fixes plus four genuinely missing components: a **powered evaluation
endpoint** (the existing one cannot resolve the question), the **trained probe
artifacts** that make the dynamic half executable, the **partial-execution gain**,
and the **matched random-token gate** without which H2 was untestable.
`verify_csgqr.py` grew 86 → 144 checks.

**4. What implementation-confidence grade?**
Reference backend **A**; optimized backend **C** (there isn't one); experimental
interpretability **B**. A verdict is admissible on the mechanism, not on systems.

**5. Does the quantized donor preserve donor-specific function?**
**Yes, decisively.** INT4 recovers 96.3% of deletion damage, and the wrong-layer
donor at identical bytes is 2.8× *worse than installing nothing*.

**6. Does confidence predict usefulness?**
**As a compute allocator, yes** (1.4–1.7× over random at matched activation). **As
a needed/not-needed classifier, no** (AUROC 0.54–0.56, wrong sign on `max_prob`).
The prior campaign recorded only the second and concluded too strongly.

**7. Does significance predict useful groups?**
**Yes.** Static interventional ranking beats random with non-overlapping
intervals; the magnitude heuristic is *worse* than random; reversing the order is
worse still. **Per-state** significance adds nothing over static.

**8. Is there a useful interior operating point?**
**No.** Interior points exist everywhere on the frontier; none beats a binary gate
at matched compute.

**9. Does exact vocabulary confidence cost too much?**
**Yes, structurally.** `|V|/(3·d_ff) = 3.556×` the gated MLP for the projection;
**1395.8%** measured including the fp32 feature reduction. A gate skipping every
token could not pay for itself, and the condition `|V| < 3·d_ff` fails worse on
larger models.

**10. Does the hidden probe solve that?**
**Yes.** Same skill as the best lens feature (better below 0.78 activation) at
~110× lower measured monitor cost. Note: ~110×, not the ~32,000× the FLOP ratio
suggests — a correction this campaign had to make about its own claim.

**11. Does CSGQR beat random and static controls?**
Random: **yes**, at both the gate (1.4–1.7×) and the selector (non-overlapping at
matched ρ). Static: **no** — neither the dynamic router nor the three-level ramp
beats its static/binary counterpart.

**12. Does it beat generic same-budget capacity?**
**Untested.** It beats a *wrong frozen donor* of identical size, which is
suggestive but not the same claim. Same-byte LoRA (B21) and random frozen weights
(B20) were not run. This is the largest gap in the evidence.

**13. Real consumer-hardware benefit?**
**No.** Slower on prefill (up to 1.90×), lower decode throughput, and +28% peak
reserved VRAM. Dequantization costs ~3.5× what the saved matmul buys.

**14. Maximum demonstrated compression?**
**3.82× accounted / 2.90× measured** for one MLP's three matrices (18.000 MiB
bf16 → 4.711 MiB packed, 6.211 MiB measured resident). As a share of the model:
**1.47% accounted, ~1.30% measured**. Donor freeing confirmed at −12.414 MiB.

**15. What quality recovery was achieved?**
On fidelity: INT8 **99.7%**, INT4 **96.3%**, best gated arm at 92.6% of damage
recovered while skipping 7.7% of tokens. On *quality* the question is
ill-posed — deletion costs **zero** reference-prediction accuracy (F1), so there is
no quality damage to recover at this layer.

**16. Which claim is fully supported?**
> A structurally retired Qwen3 MLP retained as a physically packed 4-bit remnant
> recovers **96.3%** of the KL divergence that deleting it causes, and the
> preservation is **donor-specific**: the same byte budget filled with a
> different layer's weights is 2.8× worse than installing nothing.

**17. Which claim would be overstatement?**
Several, and the campaign made versions of each before measuring:

* that CSGQR saves memory or time — it costs both;
* that confidence gating "doesn't work" — it beats random at matched compute;
* that the probe is four orders of magnitude cheaper — ~110× in wall-clock;
* that per-state routing captures 66–73% of available headroom — an artefact of
  an additive proxy that overstates by 4×;
* that a useful interior operating point exists — it exists and is not useful;
* that any of this is *novel as a mechanism* — Deja Vu already routes MLP
  channels per token from a learned predictor, and QTALE already gates quantized
  layer execution on confidence.

**18. Highest-information next experiment?**
See below.

---

## Failure attribution

| finding | attribution |
| --- | --- |
| Generation endpoint resolves nothing | `EXPERIMENTAL_LIMITATION` — fixed by `eval_forced.py` |
| Exact lens unaffordable | `ALGORITHMIC_LIMITATION` — closed-form, model-independent |
| Exact lens uninformative | `ALGORITHMIC_LIMITATION` + `MODEL_SPECIFIC_LIMITATION` (0.47B, one layer) |
| Dynamic router adds nothing | `ALGORITHMIC_LIMITATION` — the router works; the headroom was proxy artefact |
| Interior policy adds nothing | `ALGORITHMIC_LIMITATION` |
| Quantized remnant 5× slower | `REFERENCE_BACKEND_LIMITATION` — a fused dequant-GEMM is the untested repair |
| Peak VRAM rises | `REFERENCE_BACKEND_LIMITATION` — transient reuse is the untested repair |
| Systems ceiling below noise | `MODEL_SPECIFIC_LIMITATION` — single-layer scope, fixed in advance |
| No quality damage to recover | `MODEL_SPECIFIC_LIMITATION` — a property of L20 of this checkpoint |
| Same-byte-capacity control absent | `EXPERIMENTAL_LIMITATION` |

Note the pattern: the two `ALGORITHMIC_LIMITATION` rows that matter most — the
lens's cost and the dynamic router's futility — are **not** repairable by better
engineering, and both were established against controls rather than inferred.

---

## Maximum demonstrated potential

| quantity | value |
| --- | ---: |
| storage reduction, one MLP | 3.82× accounted, **2.90× measured** |
| as a share of model VRAM | 1.47% accounted, ~1.30% measured |
| fidelity retained, INT4 always-on | **96.3%** of deletion damage recovered |
| fidelity retained, best gated arm | 92.6% recovered at 92.3% activation |
| useful activation rate | **1.0** — no gated point beats always-on per unit of fidelity |
| quality retained | **1.000** — quality was never lost (F1) |
| measured latency | **worse** than dense at every operating point |
| measured peak VRAM | **worse** than dense at every operating point |

Uncertainty: single layer, single 0.47B checkpoint, one dev split,
`max_new_tokens = 32`, probes fitted on 394 positions, all latency taken on a
**shared** GPU with a 3.7× run-to-run spread.

---

## Why `CONTINUE_WITH_REDIRECTION` rather than `STOP`

Three things are worth continuing, and they are all specific:

1. **The donor-identity result is strong and cheap to strengthen.** A frozen
   low-bit copy of the *right* layer beating a wrong copy by 2.8× *below deletion*
   is a real claim about what pruning throws away. One more arm — same-byte LoRA —
   converts it into a claim about capacity versus identity.
2. **The interventional-utility label beats the magnitude proxy**, and the
   magnitude proxy is what the contextual-sparsity line uses. That is a
   contribution to someone else's method, not just this one.
3. **Hard group skipping demonstrably works** (FP at ρ=0.25 in 70% of dense
   time). The systems failure is entirely dequantization overhead, which is a
   kernel problem with a known shape of solution.

And `STOP` would be wrong for a reason worth stating: **every negative result here
is either scope-bound (single layer) or backend-bound (no fused kernel), except
two — and those two kill only the parts of the method that were never doing the
work.** Dropping the exact lens and the per-state router leaves a simpler method
whose remaining components are individually supported.

---

## The single highest-information next experiment

> **Multi-layer CSGQR: retire the four least-damaging MLPs simultaneously
> (L25, L21, L20, L18 by the committed layer scan), each as a packed INT4 remnant
> with a per-layer binary state-probe gate and a static interventional ranking —
> and include a same-byte LoRA arm as the matched-capacity control.**

Why this and not anything else:

* It is the **only** experiment that can move Gate F. Every systems number in this
  campaign is capped by one MLP being 1.99% of the model; four layers is 7.9% of
  parameters and 7.9% of MACs, which clears allocator noise. If CSGQR cannot show
  a systems benefit at 4 layers it will not show one at 1.
* It tests **H10**, the only untested hypothesis whose failure would end the
  direction outright — layers may interact, and four independently-gated remnants
  may compound divergence super-linearly.
* It uses **only the components that survived**: binary probe gate (F4, F5),
  static interventional ranking (F6). It drops the exact lens (F10), the per-state
  router (F7) and the three-level ramp (F8). So it is a test of the reduced
  method, not of the specification.
* Adding the **same-byte LoRA** arm closes question 12, the largest remaining
  hole, at marginal cost — the machinery exists in `artifacts/adapters/`.
* And it is affordable: the teacher-forced endpoint costs ~1 minute per arm per
  414 prompts, so the full design is under an hour of GPU time, against ~10 hours
  for the equivalent generation sweep that could not have resolved it anyway.

**Pre-registered prediction, so the result is interpretable either way:** four
INT4 remnants recover **> 90%** of four-layer deletion damage (the single-layer
result composed), the probe gate holds **> 1.3×** over random at matched
activation, and end-to-end latency is still **worse** than dense because the
dequantization penalty scales with the number of remnants. If the last part is
wrong, the systems case reopens; if the first part is wrong, H10 fails and the
direction should stop.

---

## What must not be claimed from this campaign

* Any systems benefit. Gate F failed on measurement.
* Any quality benefit. Quality was never lost at this layer (F1).
* Novelty of the mechanism. Two of three halves are anticipated
  (`NOVELTY_AND_RELATED_WORK.md`).
* Anything about attention remnants, KV-cache behaviour or request-level donor
  residency. Not implemented.
* Anything about the sealed test sets. Never read.
