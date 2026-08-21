# Does KV-head compression rotate the long-context frontier?

A pre-registration. Written before any number in it exists, so that the design
cannot be revised into agreement with its own results (P5).

**Status:** design frozen pending Phase 0. Nothing here has been run.
**Companion tools:** `scripts/design/kv_budget.py`, `scripts/design/power_analysis.py`.
All numbers quoted below are reproduced by those two scripts on this host.

---

## 1. The claim, stated so it can fail

> **Reducing KV heads does not degrade long-context ability uniformly. It
> *rotates* the length–quality frontier: at a fixed cache budget, fewer KV heads
> buy more tokens, and whether that trade pays depends on the task in a
> direction predictable in advance.**

The prediction that carries the weight is a **crossing**. If retrieval and
reasoning both simply get worse as KV heads are removed, this is an ordinary
compression-cost paper and a weak one. The claim earns its keep only if the
iso-memory optima for the two task families sit at *different* KV counts, so the
curves cross. A crossing is hard to produce by accident and hard to explain by
"compression hurts."

Falsified if: iso-memory accuracy is monotone in `g` in the same direction for
every task family, with confidence intervals excluding a crossing.

### 1a. What "mathematically prove" can and cannot mean here

Worth stating plainly, because the phrase invites overclaiming. Three different
kinds of statement appear below and they carry very different weight:

- **Exact (arithmetic).** The memory model in §3 is an identity, not an estimate.
  `L_max(g) = B / (2·n_layers·g·d_head·bytes)` is exact given the architecture,
  and §2's feasibility matrix follows from it with no empirical content.
- **A model (assumed, testable).** The rate–distortion account in §4 is a *frame*
  that generates predictions. It can be supported or embarrassed; it cannot be
  proven. Its value is that it predicts a crossing and a specific *absent*
  interaction, both of which could easily come out otherwise.
- **Bounded-error inference (statistical).** Everything empirical. Pre-registered
  hypotheses, a fixed analysis, controlled family-wise error, replication across
  seeds and one model scale. This bounds the probability of being wrong under
  stated assumptions. It is not proof, and no amount of seeds makes it proof.

No architecture can be *proven* better. What can be done is to state a
falsifiable model, pre-commit to the test, and report the result whichever way it
lands. That is the standard this document holds itself to.

---

## 2. What the hardware permits — this reframes the question

Run `python scripts/design/kv_budget.py --budget-gb 6.5`. On this host
(RTX 4070 Laptop, 8,187 MiB total, desktop compositor routinely holding 4–5 GiB,
15.2 GB system RAM of which ~1.9 GB is free):

**Matched-length feasibility, bf16 KV, 6.5 GB usable:**

| KV heads | 4K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| 8 (stock) | 2.26 GB | 3.67 GB | 5.55 GB | **9.31 GB ✗** | **16.82 GB ✗** |
| 4 | 1.97 | 2.67 | 3.61 | 5.49 | **9.25 ✗** |
| 2 | 1.82 | 2.17 | 2.64 | 3.58 | 5.46 |
| 1 | 1.75 | 1.92 | 2.16 | 2.63 | 3.57 |

Three consequences, and they are not minor adjustments to the plan:

**(a) The control condition is unmeasurable where the science is.** The 8KV
baseline cannot reach 64K or 128K on this card — not "slowly", not at all. A
matched-length ladder of 4K/16K/32K/64K/128K has no control arm above 32K.
System RAM does not rescue this: 128K at 8KV needs 15.0 GB of cache against
15.2 GB of total RAM, ~1.9 GB of it free. CPU offload is not an escape hatch
here, it is a second impossibility.

**(b) So iso-memory is not merely the more honest framing — it is the only one
that is executable locally.** At a fixed cache budget each configuration reaches
a different length, exactly doubling per halving of `g`:

| KV heads | KV budget | max length (bf16) | vs stock |
|---|---|---|---|
| 8 | 4.71 GB | 41,049 | 1.00× |
| 4 | 4.77 GB | 83,123 | 2.02× |
| 2 | 4.80 GB | 167,270 | 4.07× |
| 1 | 4.81 GB | 335,565 | 8.17× |

(Slightly super-linear because fewer KV heads also frees weight bytes.)

**(c) `determinism.py` cannot follow us into long context.** It pins math/eager
SDPA (`determinism.py:114-117`), and the eager score matrix is quadratic: 0.5 GB
at 4K, **8.6 GB at 16K**, 34.4 GB at 32K, 550 GB at 128K — per layer. Long-context
evaluation is impossible under the current determinism contract. §7a defines a
separate, weaker, and *achievable* contract for the evaluation path.

This is the single most important thing Phase 0 established, and it happened
before any GPU time was spent, which is the point of P7.

---

## 2a. The asset gap, which dominates Phase 0

An audit of every dataset in the tree produced an uncomfortable result that has
to be stated before the measurement plan, because it changes its size:

> **There is no long-context evaluation asset in this repository.** Not a short
> one — none at all. No passkey items, no needle-in-a-haystack items, and
> nothing whose context approaches the lengths under study.

> **SUPERSEDED (2026-08-20).** This was true when written and is no longer. A
> passkey generator with depth × length control now exists at
> `experiments/kv_significance/src/passkey.py`, and retrieval has been measured
> to 128K. What remains unbuilt is the *other* half of the battery: position-
> bucketed perplexity on genuinely held-out text, and long-context multi-hop.
> The reasoning about contamination and floor effects below still applies to
> those. See `docs/FINDINGS_KV_PROGRAM.md` §4b–4c for what was measured.

| Asset | Rows | Actual context length |
|---|---:|---|
| `Rineya_Long_Context_Demo_v1` (frozen) | 256 | **1,130–1,504 tokens** |
| `Rineya_LCR_SFT_v1` | 8,000 | 1,150–2,177 words (**≈2.7K tokens max**) |
| `Rineya-Puzzle-Reasoning-51k` | 51,000 | short |
| RAG DEV / chat DEV gates | 752 / 510 | prompts ≤330 / ≤199 tokens |

The demo suite's `needle_profile` field (`early`/`middle`/`late`/`distributed`)
is **evidence-placement metadata on multi-fact reasoning items**, not
needle-in-a-haystack retrieval; its data card lists retrieval over extremely long
contexts as explicitly out of scope. The name is a false friend.

So the ladder in this design runs from 4K to 128K, and the longest thing ever
measured on this model is 1,504 tokens — a factor of 87 short at the top rung.
**The entire long-context battery must be built.** Phase 0 is therefore corpus
construction plus harness, not harness alone, and §10 budgets it accordingly.

Two usable raw materials exist, and they have different contamination status:

- **Haystack filler** — the 8,000 LCR-SFT contexts concatenate to roughly 13M
  tokens of in-domain prose. Contamination is tolerable here because the needle
  is synthetic and random: memorising filler does not help locate it. Use the
  800-row **dev** split only, and verify exclusion from every heal mix.
- **Perplexity text** — contamination is *not* tolerable, and LCR-SFT is in the
  healing mix (it dominates the epoch). A genuinely held-out long-form corpus
  must be sourced, ≥50M tokens, verified disjoint by the 20-word-shingle overlap
  check the LCR validator already implements.

### The floor-effect problem this exposes

The only long-context baseline on record is Qwen3-0.6B on the 256-item demo
suite: **`strict_exact` = 0.2031**, and by needle position 0.094 early / 0.188
middle / 0.219 distributed / 0.313 late — at ~1.3K tokens.

A 0.6B model already scoring 20% at 1.3K tokens will be at or near chance by
32K, let alone 128K. **Floor effects would make the crossing in H3 undetectable
no matter how many seeds are run**, because a compressed model cannot score
detectably below a control that is already at zero.

This is not a reason to abandon the design; it is a constraint on item
difficulty. Pre-registered requirement: **needle difficulty is calibrated per
length so that the A-sham control lands in 0.40–0.80 accuracy at every rung.**
Calibration happens in Phase 0 against the sham arm only, is frozen before any
treatment arm is evaluated, and is reported as part of the protocol. A rung where
no difficulty setting puts the control in range is declared unmeasurable and
dropped — visibly, not silently.

---

## 3. The memory model (exact)

Per cached token, in bytes:

```
R(g) = 2 · n_layers · g · d_head · bytes_per_elem
     = 2 · 28 · g · 128 · 2          (Qwen3-0.6B, bf16)
     = 14,336 · g bytes  =  14·g KiB
```

so 112 KiB/token at g=8, down to 14 KiB/token at g=1. For a budget `B`:

```
L_max(g) = B / R(g)                  — linear in 1/g
```

This is an identity, and it is what makes the design tractable: the iso-memory
axis is known exactly in advance, with no measurement error, so the only thing
that has to be estimated is accuracy.

Parameters move too, but far less, and in a way that must not be allowed to
masquerade as the KV effect (see the C-param control in §6):

| KV heads | params | Δ vs stock |
|---|---|---|
| 8 | 596,049,920 | — |
| 4 | 566,689,792 | −29,360,128 |
| 2 | 552,009,728 | −44,040,192 |
| 1 | 544,669,696 | −51,380,224 |

---

## 4. The frame: the KV cache as a lossy code

Treat the cache as a rate-constrained channel between context and decoder.
Per token it spends `R(g)` bytes; over `ℓ` tokens, `ℓ·R(g)` total. At a fixed
budget the model chooses a point on a hyperbola: **many tokens at low per-token
fidelity, or few tokens at high fidelity.** GQA compression moves along the head
axis; truncation moves along the time axis; KV quantization moves along a third.

Define, for task τ and budget B,

```
g*_τ(B) = argmax_g  A_τ( g, min(ℓ_τ, B / R(g)) )
```

**H_RD.** `g*_τ` is task-dependent. Tasks limited by *locating* information
(passkey, multi-key retrieval) are helped more by tokens than by fidelity, so
`g*` is small. Tasks limited by *composing* information (multi-hop reasoning,
in-context discrimination and abstention) need per-token fidelity, so `g*` is
large. Hence the crossing.

The abstention result already on record makes this concrete rather than
speculative: the vocabulary trim's only cost was `conflict` abstention
(−0.1898, nine lost abstentions, `MINI_MODEL_ROADMAP.md` §16.3). Abstention under
conflicting evidence is an in-context *discrimination* task — precisely the
fidelity-limited family — so H_RD predicts it degrades early under KV
compression, and earlier than passkey retrieval does.

### 4a. Why YaRN in particular should interact with `g`

YaRN has two parts: a frequency-dependent interpolation of RoPE, and an
**attention-temperature correction** that is a function of the scale factor `s`
*alone*. That correction exists to hold attention entropy roughly constant as
positions are interpolated. But in GQA, entropy also depends on `g`: with `g`
groups, each KV head is shared by `n_q/g` query heads, and shrinking `g` forces
more query heads to discriminate through the same key subspace. A temperature
calibrated at `g=8` therefore has no reason to remain correct at `g=2`.

**H_YARN.** The optimal attention temperature depends on `g` as well as `s`;
formally, a non-zero `g × temperature` interaction. Mediation prediction: the
interaction is accounted for by measured per-layer attention entropy.

**H_DCA (a pre-registered null).** DCA is a purely positional remapping — it
bounds the maximum relative position, and touches nothing about head capacity.
Predict **no** `g × DCA` interaction.

The null is doing real work. If *both* interactions appear, the mechanism story
is wrong and we are looking at generic degradation dressed up as an interaction.
If the YaRN interaction appears and the DCA one does not, the entropy account
survives a test it could have failed. Designing in a way that can embarrass the
hypothesis is the difference between a result and a rationalisation.

---

## 5. Hypotheses, with directions fixed in advance

| ID | Hypothesis | Primary evidence | Falsified by |
|---|---|---|---|
| **H1** | At iso-memory, retrieval accuracy is **non-decreasing** as `g` falls from 8 to 2 | linear contrast in log₂(g), multi-key retrieval | negative trend, CI excluding 0 |
| **H2** | At iso-memory, multi-hop reasoning is **decreasing** in the same range | same contrast, reasoning suite | non-negative trend |
| **H3** (headline) | H1 and H2 have **opposite signs** — the curves cross | interaction contrast `g × task family` | interaction CI includes 0 |
| **H4** | Optimal YaRN temperature depends on `g` | `g × temperature` interaction | CI includes 0 |
| **H5** | DCA's benefit is **independent** of `g` | `g × DCA` interaction | CI excludes 0 (equivalence bound ±2pp) |
| **H6** | Short-context ability is preserved to `g=4` and breaks by `g=1` | chat/RAG DEV gates, per category | monotone collapse from `g=4` |
| **H7** | Calibration-sensitive categories (`conflict`, `prompt_injection`, `stale_evidence`) degrade **before** aggregate scores | per-category deltas (P2) | aggregate moves first |
| **H8** | KV-head compression and KV int8 quantization are **not** equivalent at matched bytes | `g=4`+int8 vs `g=2`+bf16 | equivalent within ±2pp |

H5 and H8 are stated as nulls/equivalences on purpose. A design that can only
confirm is not a design.

---

## 6. Arms and controls

### The confound that would sink the study

Surgery arms receive continued training; a stock baseline does not. Every
measured difference would then confound KV compression with extra training
tokens. **Non-negotiable: a sham-surgery arm** — the unmodified 8KV model pushed
through a byte-identical healing run (same mix, same seed, same steps, same
plan hash inputs). All comparisons are against the sham, never against stock.

### Arms

| Arm | Surgery | Purpose |
|---|---|---|
| **A-sham** | none | control for the training budget. The reference for everything. |
| **A-4** | 8→4 KV, mean-pooled | treatment |
| **A-2** | 8→2 KV, mean-pooled | treatment |
| **A-1** | 8→1 KV (MQA) | treatment, and the frontier's far end |
| **C-param** | 8KV, MLP width trimmed by 29.4M params | separates "fewer parameters" from "fewer KV heads" |
| **C-repair** | 8→4 KV, alternative merge | separates the *compression* from the *merge rule* |

`C-param` matters because A-4 removes 29,360,128 parameters, and without it any
degradation is equally explained by size. `C-repair` is scoped in Phase 0's pilot
(§10, Phase 0.10) and promoted to a full arm only if the pilot shows the merge rule moves the
result — otherwise mean-pooling is fixed by fiat and the arm is dropped.

Stock Qwen3-0.6B and un-healed R2 are reported as landmarks, never as controls.

### Merge rule

Mean-pool over each group of KV heads, per the GQA construction. Pre-registered
as primary. The pilot compares it against an attention-mass-weighted merge at
`g=4` only; if the gap is below the seed-noise floor, the question is closed and
`C-repair` does not run.

### A repair-capacity problem that must be settled first

Healing is **LoRA** (r=16, α=32, 7 projections, 10,092,544 trainable params —
`heal.py:147-150`). LoRA freezes the base weights. But KV surgery damages
`k_proj`/`v_proj` *specifically*, and a rank-16 adapter is a weak instrument for
repairing a rank-collapsed projection. The GQA literature uptrains with full
fine-tuning.

Fortunately the honest fix is cheap: `k_proj`+`v_proj` at `g=4` are 1024×512 each,
so unfreezing both across 28 layers costs ~29M trainable params — affordable, and
far better matched to the damage. **Phase 0 pilot decides this** (LoRA-only vs
LoRA + unfrozen k/v/o), and whichever wins is then applied identically to every
arm *including the sham*. Note that even if repair stays imperfect, the sham
control keeps the comparison valid; imperfect repair costs external validity
("results hold for models healed this way"), not internal validity.

---

## 7. Measurement

Every endpoint below is measured at each length on the ladder, at each `g`, at
each seed. Feasible cells only (§2); infeasible cells are reported as infeasible,
never silently dropped.

| # | Endpoint | Protocol | Exists? |
|---|---|---|---|
| 1 | **Short-context regression** | chat DEV (510) + RAG DEV (752), `max_new_tokens=32`, greedy, v4.5 scorer, **per category** (P2) | ✅ reuse as-is |
| 2 | **Perplexity** | held-out long documents, **position-bucketed** (§7b) | ❌ corpus + scorer |
| 3 | **Passkey retrieval** | single needle, depth × length grid, greedy, exact match | ❌ build |
| 4 | **Multi-key retrieval** | 2/4/8 needles, *all* must be recovered | ❌ build — **primary endpoint** |
| 5 | **Multi-hop reasoning** | LCR-SFT dev + Puzzle-51k items re-hosted in long haystacks | ❌ build (items exist, lengths do not) |
| 6 | **KV-cache memory** | `max_memory_allocated`, measured then cross-checked against §3 | ❌ build |
| 7 | **Prefill latency** | chunked prefill, 5 warmup + 20 timed, median + IQR | ❌ build |
| 8 | **Decode throughput** | tokens/s at 25/50/75/100% cache occupancy | ❌ build |

One inherited protocol decision is deliberately **not** reused: the v4.5 quality
scorer applies to endpoint 1 only. Retrieval endpoints are scored by exact match
against accepted answers, following the demo suite's `strict_exact`. A 13-dimension
weighted quality score with a critical-defect zeroing rule is the wrong instrument
for "did it find the number", and mixing the two would make the primary endpoint
uninterpretable.

`run_gates.py` accepts any repo-relative JSONL path as `--dataset`
(`run_gates.py:86`), so new eval sets plug in without touching it. Two rough
edges to fix first: `dataset_sha256` comes from a hardcoded two-entry map and
will be `null` for anything new, and the sealed-set guard is a filename
substring check with no hash backing.

### 7a. Evaluation-path determinism (a new, weaker, achievable contract)

Training bit-exactness (`determinism.py`) is unreachable at long context (§2c).
Rather than quietly abandon the guarantee, replace it with one that can be held
and verified:

- greedy decoding, fixed seed, fixed item order;
- SDPA mem-efficient backend permitted (forward-only; the measured
  nondeterminism of 1.46e-11 was a *backward*-pass result);
- **verification standard: run-to-run reproducibility of the reported score**,
  not bitwise equality of tensors. Two independent runs of the same eval must
  agree to 0 on every per-item outcome.

This is strictly weaker than the training contract and must be labelled as such
wherever results are reported.

**Implementation constraint:** `code_fingerprint()` hashes every `.py` under
`scripts/train/` and any edit there invalidates every existing checkpoint's
resume path. All new measurement code therefore goes in `scripts/longctx/` and
`scripts/design/` — **never** in `scripts/train/`.

### 7b. Why perplexity is bucketed by position

Aggregate perplexity over a long document is dominated by its early tokens,
where context is short and every configuration looks alike. A model that has
entirely lost the ability to use position 100K can still post a respectable
whole-document number. Report `ppl(t)` in position buckets and treat the *late*
buckets as the endpoint. Pre-registered: perplexity is **secondary**, and a
perplexity result that contradicts retrieval is resolved in favour of retrieval.

---

## 8. Analysis plan (fixed before data)

### Primary test

Not three pairwise comparisons — a **single linear trend contrast in log₂(g)** on
multi-key retrieval accuracy at iso-memory. "Systematic" is a statement about
trend, and encoding it as trend both matches the claim and collapses the
multiplicity family to m=1.

```
primary:  H0: β_log2(g) = 0     on iso-memory multi-key retrieval accuracy
```

### Secondary, in a fixed hierarchy (gatekeeping)

Tested in order; testing stops at the first non-rejection, which controls
family-wise error without paying full Bonferroni:

1. H3 crossing — `g × task-family` interaction contrast
2. H4 — `g × YaRN temperature`
3. H5 — `g × DCA`, as an **equivalence** test against ±2pp
4. H6/H7 — short-context gates, per category
5. H8 — head compression vs KV quantization at matched bytes

### Estimation

- Paired throughout: identical items across arms, differences taken per item.
- **Do not write a new statistics layer.** A complete, stdlib-only, deterministic
  toolkit already exists, vendored and unused, at
  `Rineya_state_experiment_bundle/rineya_state_sufficiency_lab/src/rineya_lab/analysis/stats.py`:
  `paired_bootstrap_ci` (10,000 iterations, seeded), `exact_mcnemar` (whose `b`/`c`
  are exactly the win/loss counters M3 asks for), `sign_flip_test` with the
  Phipson–Smyth correction, `holm_bonferroni`, and `decide` — which reads
  superiority, non-inferiority and TOST equivalence off a CI against a
  pre-registered margin. It depends only on stdlib plus one hashing helper, which
  matters because scipy and statsmodels are not in the environment.
- Two integration notes: it keys on `problem_id` while gate runs key on `id`, so
  an adapter is needed; and `ComparisonConfig` ships `margin_frozen=False`, which
  makes `decide` emit a loud warning. **Set `margin_frozen=True` and fix the
  margins at ±2pp before Phase 1** — that switch is precisely what P5 is about.
- Win/loss counts reported alongside every mean delta. §16 is the precedent:
  +0.0002 with a CI spanning zero and 45/48 wins is *no effect*; −0.0127 with a
  CI excluding zero and 0/14 is a *real* one. Same magnitude, opposite readings.
- Effect sizes in accuracy points, never in relative percent.

### Two error bars, not one

Every plot carries both:
- **within-run** (item sampling), which more items shrink;
- **between-run** (seed), which only more seeds shrink.

Conflating them is how underpowered studies look powered.

---

## 9. Power — and why "at least three seeds" is the wrong economy

From `python scripts/design/power_analysis.py`. Two kinds of contrast, two units
of replication, and they behave very differently.

**Inference-level contrasts** (YaRN factor, DCA, KV dtype, length — all on one
fixed checkpoint) have *no training variance*. Items are the unit; MDE at 80%
power, baseline accuracy 0.75:

| items | m=1 | m=3 | m=12 |
|---|---|---|---|
| 400 | 5.43pp | 6.27pp | 7.18pp |
| 800 | 3.84 | 4.43 | 5.08 |
| 1600 | 2.71 | 3.13 | 3.59 |

Cheap and well powered. **Design consequence: put as much of the science as
possible in the training-free layer.** Phase 1 does exactly that.

These assume 0.75 baseline accuracy. §2a's calibration targets the harder middle
of the range, and binomial variance is maximal at 0.5, so the planning numbers
should use the worst case: at p=0.50, 400 items gives 6.27pp and 800 gives
4.43pp (`--p-base 0.5`). Budget **≥800 items per retrieval cell**, not 400.

**Training-level contrasts** (the KV arms) are governed by the seed count, and
this is where the plan as originally sketched breaks. MDE at 80% power, matched
seeds, 400 items:

| σ_run | k=3 | k=4 | k=5 | k=6 | k=8 |
|---|---|---|---|---|---|
| 0.5pp | 11.35pp | 6.25 | 4.57 | 3.73 | 2.90 |
| 1.0 | 12.33 | 6.80 | 5.01 | 4.09 | 3.16 |
| 1.5 | 13.92 | 7.65 | 5.61 | 4.60 | 3.56 |
| 2.0 | 15.79 | 8.70 | 6.38 | 5.24 | 4.05 |
| 3.0 | 20.25 | 11.12 | 8.17 | 6.73 | 5.17 |

**Three seeds gives 2 degrees of freedom, and under the m=3 multiplicity family
the critical value is 7.649** — not 4.303, which is the *unadjusted* value.
k=4 gives 4.857, k=5 gives 3.961.

Going from k=3 to k=4 cuts the MDE by ~45% for the cost of one extra run — by far
the best-value compute in the entire program. **At k=3 and σ_run=1.5pp the study
can only detect a 13.9-point effect**; effects that large are not the interesting
ones.

> **Corrected.** These figures were previously 7–43% smaller, because `t_crit`
> applied the multiplicity adjustment by scaling the t value by a ratio of
> *normal* quantiles. That is valid only as df→∞ and its error grows as 1/df —
> worst at exactly the df a k=3 design lives at. Measured on 400k null
> simulations it gave α = 0.0340 against a 0.0167 target, i.e. FWER 0.098 for a
> three-member family, about double the declared 0.05. `t_crit` now inverts the
> exact t distribution (validated against published tables to 4 decimals; FWER
> re-measured at 0.0492). The headline was 9.6pp and is 13.9pp.

And more items cannot buy this back. The floor as items → ∞:

| σ_run | k=3 | k=5 | k=8 |
|---|---|---|---|
| 1.0pp | 5.67pp | 2.29pp | 1.45pp |
| 1.5 | 8.48 | 3.44 | 2.18 |
| 3.0 | 16.99 | 6.89 | 4.35 |

At k=3, σ=1.5pp, going from 400 to 3,200 items moves the MDE only 13.8 → 9.3pp.

> Both figures above are post-correction. An earlier version of this section
> regenerated only the 400-item table when `t_crit` was fixed, leaving these two
> adjacent to it at their pre-correction values — so the page contradicted
> itself by a factor of ~1.4. Regenerate the whole section together:
> `python scripts/design/power_analysis.py`.

**Decisions:**
- **k = 5 seeds** for training arms, matched across arms (same seeds, same data
  order). After the `t_crit` correction, matched pairing is *worse* than
  independent seeds at k=3 — 13.92pp against 11.00pp at σ=1.5 — because pairing
  spends 2 degrees of freedom instead of 4 and the exact t critical value is
  punishing there. It turns favourable from k=4 (7.65 against 8.03) and the gap
  widens with k. **The k=3 case is the one to avoid**, not the pairing.
- **≥1,000 items** per retrieval cell (synthetic, so they are nearly free).
- **σ_run is unknown until M1 runs**, and it sets everything above. M1 is
  therefore blocking, not optional — it is the one measurement that decides
  whether the rest of the program is interpretable.
- Pre-registered **sample-size re-estimation**: run k=3, estimate σ_run, and top
  up to k=5 (or stop) by the rule in §9a. Declared now so it is not a
  post-hoc rescue.

### 9a. Stopping rule

After k=3 on the A-sham vs A-2 contrast:
- σ̂_run ≤ 1.0pp and |effect| ≥ 8pp → conclude at k=3;
- σ̂_run > 2.0pp → the design cannot answer at any affordable k; report the
  variance floor as the finding and stop (this is a real result, not a failure);
- otherwise → extend to k=5 and analyse only at k=5.

---

## 10. Phases, sequencing, cost

Ordered so that the cheapest test that can invalidate an expensive one runs
first (P7). Throughput anchor: the live B0 run measures **12.48 s/step, 1,003
steps ≈ 3.5 h/run** at `max_len=384`; long-context healing will be slower and
Phase 0.11 measures it rather than assuming.

### Phase 0 — build the instrument (blocking, ~2–3 weeks)
Larger than first scoped, because §2a showed there is nothing to measure with.

| | Task | Cost |
|---|---|---|
| 0.1 | Eval-path determinism contract + verifier (§7a) | CPU |
| 0.2 | **Haystack corpus**: concatenate LCR-SFT dev contexts, verify heal-mix exclusion by shingle overlap | CPU |
| 0.3 | **Held-out perplexity corpus**, ≥50M tokens, contamination-checked | CPU + sourcing |
| 0.4 | **Needle generators**: passkey and multi-key, depth × length grid, ≥800 items/cell | CPU |
| 0.5 | **Long-form multi-hop set**: re-host LCR dev + Puzzle items in haystacks | CPU |
| 0.6 | **Difficulty calibration** against the sham arm to land 0.40–0.80 (§2a), then freeze | ~6 h |
| 0.7 | Efficiency probes: measured KV bytes, prefill latency, decode throughput | ~2 h |
| 0.8 | Statistics adapter onto the vendored toolkit; `margin_frozen=True` | CPU |
| 0.9 | **M1 seed-variance floor**, k=5 identical configs | ~18 h |
| 0.10 | **Repair pilot**: LoRA vs LoRA+unfrozen k/v/o at g=4; mean-pool vs weighted merge | ~14 h |
| 0.11 | Healing-throughput calibration at the chosen context | ~2 h |

Phase 0 alone answers "is this study possible"; 0.9 answers "at what
resolution"; 0.6 answers "is there room to move". None of it requires committing
to the expensive phase, and any of the three can end the program cheaply.

### Phase 1 — training-free extrapolation map (~3 days, no training)
Full factorial on **both** stock Qwen3-0.6B and R2: YaRN {off, s=2, 4, 8} × DCA
{off, on} × length {4K…128K, feasible cells} × all endpoints. No training
variance, so this is high-powered and cheap.

Two things it settles before any expensive commitment:
- **base-checkpoint choice**: pre-registered equivalence test between stock and
  R2 under YaRN. Equivalent within ±2pp → use R2 (it serves the product);
  otherwise carry both, and report that the fine-tune changed extrapolation
  behaviour, which is itself publishable.
- **σ_item and ρ**, which convert §9's provisional power table into a real one.

### Phase 2 — GQA surgery + healing (the expensive phase, ~2–3 weeks)
4 arms (A-sham, A-4, A-2, A-1) × 5 matched seeds = **20 runs**. C-param and
C-repair promoted only if Phase 0.10 says they are needed. Sequential: A-sham vs
A-2 first (largest expected effect, and the existence claim), then fill in.

### Phase 3 — interaction study
Phase 1's grid re-run over Phase 2's checkpoints. This is where H3/H4/H5 are
tested. No new training.

### Phase 4 — replication at 1.7B
Qwen3-1.7B shares the shape exactly — 28 layers, 16Q/8KV, head_dim 128, tied
embeddings — so it is a scale replication with no architectural confound, and its
KV cost per token is *identical* to 0.6B's 112 KiB. **But** at 3.44 GB of weights
it reaches only ~21K tokens at g=8 here. Locally: training-free replication to
16K only. The full replication needs rented compute — one 80 GB device covers
every matched-length cell, including the 8KV/64K and 8KV/128K cells that are
impossible on this host. (The checkpoint is not in the tree yet; verify with
`kv_budget.py --model Qwen3-1.7B` once it is.)

### Phase 5 — remediation (conditional)
Only if Phase 2 shows real degradation: does identity-initialised depth expansion
recover it *at matched memory*? This is the honest place for block expansion —
as remediation, not as step 2 of a program whose premise is a small model.

### Dropped, with reasons
- **MoE upcycling.** Cuts active FLOPs, not memory. Memory is the binding
  constraint at 0.6B on 8 GB, and MoE does nothing for the KV cache, which is the
  actual long-context bottleneck. It also inflates total parameters, which
  contradicts the program's frontier definition.
- **SSM / linear-attention hybrid.** Deliberately *not* dropped — but it answers a
  different question. It is the only intervention here that changes memory
  *scaling* (O(1) per layer instead of O(n)) rather than characterising the
  existing frontier. If the goal is a genuinely better architecture rather than a
  publishable interaction, it should be promoted above Phase 2. See §12.

---

## 11. Threats to validity

| # | Threat | Mitigation | Residual |
|---|---|---|---|
| 1 | Training-budget confound | A-sham arm | none if sham holds |
| 2 | Parameter-count confound | C-param arm | conditional on running it |
| 3 | LoRA cannot repair KV surgery | Phase 0.10 pilot; unfreeze k/v/o | external validity only |
| 4 | Control unmeasurable >32K | iso-memory primary; rent for matched-length | matched-length claims are rented or absent |
| 5 | Eval nondeterminism | §7a contract, verified per-item | weaker than training contract; labelled |
| 6 | Seed variance swamps effects | M1 first; k=5; §9a stopping rule | may end the study — declared acceptable |
| 7 | LCR-SFT is a *training* set | strict held-out split, contamination check | |
| 8 | Gate saturation (chat DEV already 0.8336) | long-context endpoints carry the signal | short-context may show nothing |
| 9 | Healing context ≠ eval context | fixed identically across arms | limits generalisation, not comparison |
| 10 | Desktop VRAM contention | fixed budget, `cap_mib`, re-probe per session | measured, not assumed |
| 11 | Multiple comparisons | trend contrast + gatekeeping hierarchy | |
| 12 | Position-bucketed ppl disagrees with retrieval | pre-registered: retrieval wins | |
| 13 | **Floor effects** — 0.6B is already at 0.2031 at 1.3K tokens | difficulty calibrated to 0.40–0.80 per rung (§2a) | rungs where no setting works are dropped |
| 14 | **Every eval asset is newly built by us** | generators pre-registered and frozen before treatment arms run | no external validity against published benchmarks |
| 15 | Haystack filler overlaps the healing mix | dev split only, shingle-overlap verified | filler memorisation does not locate a random needle |
| 16 | Per-category n is small (`conflict` n=38) | category claims reported as discrete counts, not means | some categories cannot support any test |

Threat 14 deserves emphasis: because we build the benchmark and the model, a
result that holds only on our generator is worth much less. Mitigation is to fix
the generators before treatment arms are ever run, and to report the sham arm's
absolute scores so others can judge whether the task is reasonable.

### Kill criteria, declared now

- M1 shows σ_run > 2.0pp → stop; report the variance floor as the finding.
- Difficulty calibration cannot put the sham control in 0.40–0.80 at ≥3 rungs →
  the model is too small for this question; report and stop.
- The sham arm does not return to its pre-surgery gate scores → the healing
  recipe is not a valid repair; fix that before any arm is interpreted.
- Phase 1 shows R2 and stock diverge *and* neither extrapolates past 32K → the
  base model cannot support the study; report and stop.

---

## 12. The two decisions this design leaves open

**Which goal dominates?** If it is the publishable interaction result, the order
above is right. If it is actually a better-performing small model, Phase 5's
SSM/linear-attention hybrid should move ahead of Phase 2 — it is the only item
here that changes memory *scaling* rather than measuring the existing frontier.
These point to genuinely different orderings and the choice is not mine to make.
**Default taken here:** the interaction result, because Phase 1 is cheap,
de-risks everything downstream, and produces the harness the hybrid work would
need anyway.

**Which base checkpoint?** Not decided by argument — decided by Phase 1's
equivalence test, which costs almost nothing and answers it with data.

---

## 13. Deliverables

```
scripts/design/kv_budget.py         feasibility + iso-memory frontier   [done]
scripts/design/power_analysis.py    two-level power / MDE               [done]
manifests/kv_budget.json            feasibility matrix                  [done]
docs/EXPERIMENT_DESIGN_KV_LONGCTX.md  this file                         [done]

scripts/longctx/haystack.py         corpus assembly + contamination check
scripts/longctx/needles.py          passkey + multi-key generator, depth x length
scripts/longctx/ppl_buckets.py      position-bucketed perplexity
scripts/longctx/efficiency.py       KV bytes, prefill latency, decode throughput
scripts/longctx/rope_scaling.py     YaRN / DCA configuration surface
scripts/surgery/gqa_merge.py        8->4/2/1 KV, provenance manifest + sha256
scripts/analysis/adapter.py         gate-run rows -> vendored stats.py contract
```

Nothing under `scripts/train/` is touched: `code_fingerprint()` hashes every
`.py` there and an edit would invalidate the resume path of every existing
checkpoint, including the B0 run currently in flight.

Every surgery emits a provenance manifest and a logit-equivalence check on the
untouched sub-space, per the I2 contract the vocab trim already satisfies.
