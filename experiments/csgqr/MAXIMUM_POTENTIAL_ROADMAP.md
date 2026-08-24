# CSGQR — Maximum-Potential Roadmap

What is worth trying next, in priority order, with the reason each item is at that
position and a predeclared budget. Items are gated: an item does not start until
its prerequisites pass, because the brief is explicit that increasingly complex
extensions must not be explored past a failed gate.

Current gate state (`RESEARCH_VERDICT.md`): A pass, B pass, C pass (redirected),
D pass, **E half**, **F fail**.

---

## Tier 1 — the two experiments that decide the direction

### 1. Multi-layer CSGQR with the reduced method *(the single next experiment)*

Retire the four least-damaging MLPs simultaneously — **L25, L21, L20, L18** by the
committed 28-layer scan — each a packed INT4 remnant with a **per-layer binary
state-probe gate** and a **static interventional ranking**. Drop the exact lens,
the per-state router and the three-level ramp; none of them survived.

*Why first.* It is the only design that can move **Gate F**. Every systems number
in this campaign is capped by one MLP being 1.99% of parameters; four layers is
7.9%, which clears allocator and clock noise where 1.99% cannot. It simultaneously
tests **H10**, the only untested hypothesis whose failure ends the direction
outright: four independently-gated remnants may compound divergence
super-linearly, and layers may interact through the residual stream in ways one
layer cannot reveal.

*Predeclared prediction.* Four INT4 remnants recover **> 90%** of four-layer
deletion damage; the probe gate holds **> 1.3×** over a matched random-token gate;
end-to-end latency is **still worse** than dense, because the dequantization
penalty scales with the number of remnants. If the last is wrong the systems case
reopens; if the first is wrong, stop.

*Budget.* ~15 arms × ~1 min on the teacher-forced endpoint, plus one probe fit per
layer and one GPU benchmark. **Under 2 GPU-hours.**

### 2. Same-byte capacity controls — closing the largest evidence hole

Three arms at an identical byte budget to the INT4 remnant (4.711 MiB):

* **B21** same-byte **LoRA** on the surviving neighbours, trained on the same data;
* **B20** **random frozen** weights of the same shape and bitwidth;
* **B23** static frozen-weight fusion (donor folded in, no gating).

*Why second.* Question 12 of the verdict is unanswered: CSGQR beats a *wrong
frozen donor*, which is suggestive but is not "beats generic capacity". Until B21
runs, the strongest positive result (**H4**, donor identity) is a claim about
identity-versus-wrong-identity rather than identity-versus-capacity — and the
latter is what decides whether "retain the pruned component" beats "add fresh
parameters".

*Predeclared prediction.* The correct INT4 donor beats random frozen weights
decisively and beats same-byte LoRA **narrowly or not at all**, because LoRA is
trained and the donor is not. If LoRA wins, the honest conclusion is that CSGQR's
value is capacity, not preservation, and the framing must change.

*Budget.* 3 arms + one LoRA training run. Machinery exists in
`artifacts/adapters/`. **~1 GPU-hour.**

---

## Tier 2 — only if Tier 1 passes

### 3. A fused dequantize-GEMM

The one repair for the campaign's decisive systems failure. Dequantization costs
~3.5× what the saved matmul buys, making the INT4 remnant 5.0× slower than the
dense layer at ρ=1 and still 2.4× slower at ρ=0.125. A kernel that dequantizes
into registers inside the GEMM removes that term in principle.

*Why not Tier 1.* **H12 is `NOT_ESTABLISHED`, and a hypothetical optimised backend
is not demonstrated performance.** But it must not be attempted before item 1,
because a fast kernel on a method with no multi-layer scope is a fast kernel for
nothing. Triton is available in the environment.

*Success criterion, predeclared.* INT4 at ρ=1 within **1.3×** of the dense bf16
MLP, and ρ=0.25 strictly **below** it. Anything less leaves N16 standing.

### 4. Transient reuse — a preallocated dequantization workspace

Peak reserved VRAM rises 100–270 MiB while the persistent saving is 13.3 MiB
(N17). The increase is entirely execution transient. A preallocated workspace
sized to the maximum active group count, reused across tokens and buckets, should
remove most of it.

*Success criterion.* Peak reserved for the full method within **+20 MiB** of
dense. This is the difference between a method that costs VRAM and one that saves
it, and it is cheap.

### 5. Vectorised routing

Per-token bucketing costs **16×** in the reference backend (622 s vs 39 s over 414
prompts, N18). The per-token policy loop is Python with a GPU→CPU sync on the gate
signal. Batched threshold evaluation and a `scatter`-based bucket assignment
should remove most of it. Needed for any per-token policy to be credible, though
note that after N13 the surviving method's ranking is *static*, so this only
matters if a per-state selector is ever revived.

---

## Tier 3 — mechanism questions worth answering regardless

### 6. Interventional versus magnitude routing, head to head

This campaign's cleanest contribution to the wider literature. The
contextual-sparsity line (Deja Vu) predicts which neurons will be *active*; CSGQR's
label is an interventional KL difference. Measured here: at matched ρ=0.25 the
magnitude heuristic (0.23719) is **worse than random** (0.21563) while the
interventional ranking is clearly better (0.17822).

*What to run.* A Deja Vu-style activation-magnitude predictor against the
interventional router, matched budget, same layer, same prompts. If the
interventional label wins on a *live* layer too, that is a result about
contextual sparsity and not merely about remnants.

### 7. Group granularity

`n_groups` ∈ {12, 24, 48, 96} — i.e. 256, 128, 64, 32 intermediate channels.
3072 divides evenly by all four. Only 24 was measured. The trade-off the brief
asks for (routing flexibility versus packed layout versus kernel count) is
unmeasured, and finer groups directly worsen the per-token bucketing cost.

### 8. Bitwidth extension

INT3, INT2, NF4, FP4, and mixed-bit groups (high-utility groups at higher
precision). Prerequisite: a **real bit-packer** for 3 and 2 bits — they are
currently stored one value per byte and their storage claims are explicitly not
validated. Also worth pairing with `α` recalibration, since α ≈ 0.997 at INT4 but
shrinkage grows as bits fall, so the gain that is inert here may not stay inert.

### 9. δ sensitivity and cross-domain calibration

`δ = 0.1` was chosen because `P(U > 0) = 0.9987` makes `δ = 0` degenerate. Its
sensitivity has never been swept, and no threshold has been transferred across
layers or to `rag_dev`. **H11** is only partially supported for that reason.

### 10. A longer-generation protocol, and the confidently-wrong taxonomy

`max_new_tokens = 32` with a median reply of 7 tokens, and divergence starting
around token 2–4, means most of a divergent reply is generated after the
divergence. The brief asks for confidently-wrong cases to be categorised by
linguistic role; the trace records token ids and positions, not roles, so this
needs a longer protocol plus an explicit annotation pass rather than a tagger whose
output would be attributed to measurement.

---

## Tier 4 — the attention half, still blocked

### 11. Phase B: request-level attention remnant

The attention remnant is on or off for a whole request, which sidesteps the
KV-cache problem entirely. **This is the only mechanism in the design that can
free VRAM dynamically**, and it is unimplemented, so the campaign's memory story
is missing its most promising component.

### 12. Phases C and D: chunk-level, then token-level attention routing

Token-level attention gating requires custom cache semantics: skipping attention
for an earlier token leaves later tokens needing keys and values that were never
generated. CALM names this as one of its three core problems. **Do not claim
full-block token-level CSGQR before this is solved and verified.**

---

## Explicitly deprioritised, with reasons

| item | why not |
| --- | --- |
| **Training the alignment lens** | The exact projection costs `\|V\|/(3·d_ff) = 3.556×` the MLP it gates in MACs and **1395.8%** measured. A perfect lens still cannot pay for itself. H7's lens variant stays `NOT_ESTABLISHED` rather than being pursued. |
| **Per-output-channel `α`** | The scalar gain is 0.997 at INT4 and the partial-execution gain moves output error by 0.1%. There is nothing for a more expressive gain to correct at this bitwidth. Revisit only with INT2/INT3. |
| **Learned combined confidence over lens features** | Measured at AUROC 0.580 — no better than the best single raw feature, and it inherits the projection's cost. |
| **The three-level ramp and `γ` sweep** | No interior point beats a binary gate at matched compute (N15). Sweeping `γ` optimises a region that has been shown not to pay. |
| **Per-state group router** | Genuinely state-conditioned, genuinely informative, and no better than a fixed ranking at either budget (N13). Revive only if multi-layer scope changes the picture. |
| **More `capture_at_k`-driven design** | It overstates by 4× for exactly the selection that optimises it (N14). Should not be used to justify a routing design again. |

---

## Budget discipline

The brief requires predeclared compute budgets, so: **Tier 1 is ~3 GPU-hours
total** and is the whole of the next campaign. Tier 2 does not start until Tier 1
reports. Nothing in Tier 3 or 4 starts until Gate F either passes at multi-layer
scope or is formally abandoned in favour of a storage-only claim.

The failure mode this ordering exists to prevent is the one the brief names: tuning
until a positive result appears. Every Tier 1 item above has a **predeclared
prediction**, so both outcomes are informative and neither is a licence to keep
searching.
