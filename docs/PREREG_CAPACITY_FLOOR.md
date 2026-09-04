# Preregistration — the capacity floor, and two complementary mechanisms

**Frozen 2026-09-03, before any of its runs.** Follows
[`FINDINGS_COMPRESSION_IMPROVES_QUALITY.md`](FINDINGS_COMPRESSION_IMPROVES_QUALITY.md),
which established that `lexical-doc-bm25-sys` at keep 0.05 beats the uncompressed
model by +5.2 pp pooled (34/3, p = 1.2e-07, n = 600) and by +5.8 pp out-of-sample.

Endpoint, statistics and strata are unchanged from
[`PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md`](PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md):
`contract_fulfillment_score >= 0.999`, paired exact McNemar, answer / abstain
reported separately and pooled. Model `rineya-core-en32k-28L`, `rag_long4k`,
seed 20260826, sinks 4, recent_frac 0.15, pool 7, window (6,0), n = 600.

---

## 1. The question

Quality rose at keep 0.05 and fell at keep 0.02. Those are not arbitrary points: the
selector must retain two spans to work at all, and they have a measured size.

Measured over 120 prompts (mean, tokenizer-exact):

| span | tokens |
|---|---:|
| prompt | 4226 |
| instruction prefix (system + preamble, up to the first `[doc_`) | **100** |
| one evidence record (median) | **63** |
| question tail (`<QUESTION>` onwards) | 29 |

The budget must also pay for 4 sinks and a recent window of `0.15 · B`. So

```
4 + 0.15·B + 100 + 63  ≤  B      ⟹      B ≥ (4 + 100 + 63) / 0.85 = 196
```

**F1 — the floor.** Pooled quality holds at or above **B\* = 196 tokens** and
degrades below it. keep 0.05 is B ≈ 211 (just above); keep 0.02 is B ≈ 85 (far
below). 196 tokens on a 4226-token prompt is a **21.6× reduction**.

This is a point prediction with a mechanism, not a curve fit. It can be wrong in
three distinguishable ways, and each is informative: the floor could sit *higher*
(the model needs more than the minimal spans), *lower* (partial spans suffice), or
the curve could have no knee at all (the account is wrong).

**F2 — which span breaks first.** `protect_instructions` sets the instruction prefix
to weight 1.0 and BM25 sets the top record to 1.0, so they tie and the 1e-3 tie-break
decides between them. Under a shrinking budget both should degrade together, and
because the instruction prefix is the larger span (100 vs 63) **abstention should
fall faster than answering**.

---

## 2. Runs

**Stage 1 — the floor sweep.** `lexical-doc-bm25-sys` at absolute budgets
**B ∈ {512, 256, 196, 160, 128, 96}**, n = 600. Absolute budget is the right axis
because the floor is an absolute token count; `mean_kept_ratio` is recorded per run.
The uncompressed reference `c7-off-n600` already exists.

**Stage 2 — does this need the prompt's structure?** The result so far is
benchmark-scoped: `rag_long` prints `[doc_… ]` delimiters, and record segmentation
depends on them. A **structure-free** variant, `lexical-chunk-bm25-sys`, segments the
prompt into fixed 64-token chunks with no delimiters at all, BM25-scores each chunk
against the question, and protects the instruction prefix.

**S1.** If the structure-free chunker holds within 2 pp of the delimiter-based
selector at B = 256, the method does not depend on printed document structure and the
result generalises past this benchmark. If it collapses, the finding is a fact about
prompts that mark their own records.

**Stage 3 — instruction sinks.** Tying instructions and evidence at 1.0 makes them
compete through a tie-break. Treating the instruction prefix as an *extension of the
sink region* — kept unconditionally, outside the ranked budget — should lower the
floor, because the ranked budget then buys only evidence.

**S2.** Instruction sinks lower the observed floor relative to Stage 1 by roughly the
instruction length, to `B ≈ (4 + 63)/0.85 + 100 ≈ 179`… **no**: with instructions
outside the ranked pool the requirement becomes `4 + 0.15B + 63 ≤ B − 100`, i.e.
**B ≥ 197** — the same. The prediction is therefore that instruction sinks do **not**
move the floor, but do make the failure *graceful*: below the floor, abstention should
be preserved while answering degrades, instead of both degrading together (F2).
Stage 3 runs only at the two budgets that bracket the observed floor.

---

## 3. Decision rules, fixed now

* The floor B\* is the largest swept budget at which pooled quality is **not**
  significantly below the uncompressed reference (paired exact McNemar, α = 0.05),
  with every smaller swept budget significantly below it. If no such point exists the
  sweep is reported as having no knee and F1 is refuted.
* **F1 is confirmed** if B\* ∈ {160, 196, 256} — i.e. the knee brackets the predicted
  196 within one sweep step either side. It is **refuted** if B\* ≥ 512 or B\* ≤ 128.
* **F2 is confirmed** if, at the first budget below B\*, the abstain stratum's drop
  from uncompressed exceeds the answer stratum's.
* **S1 confirmed** if |chunk − doc| ≤ 2 pp pooled at B = 256 and the chunk arm is
  still significantly above uncompressed.
* **S2 confirmed** if instruction sinks leave the abstain stratum not significantly
  below uncompressed at a budget where Stage 1 has it significantly below.
* Any arm with `heavy_zero_count > 0` is void: the policy silently became
  StreamingLLM. Any arm whose `mean_kept_ratio` is not within 10% of `B/4226` is void.

**No verdict from missing data.** A budget that fails to run is reported as not run.
Partial artifacts are deleted, not analysed.

---

## 3a. AMENDMENT, 2026-09-03, written while the sweep was still running and before
## any of its numbers had been read: F1 as stated is CONFOUNDED

The recent window is `0.15 · B`, and the question tail is **29 tokens**. So the
budget at which the recent window stops covering the question is

```
0.15 · B ≥ 29   ⟹   B ≥ 193
```

which is **within 3 tokens of the predicted floor of 196**. The two accounts are
therefore not separated by the Stage-1 sweep at all:

* **A — span capacity.** Below B\*, the budget cannot hold instructions (104) plus
  one record (63), so evidence or instructions get evicted.
* **B — query truncation.** Below B\*, the recent window is shorter than the
  question, so part of the question itself is evicted. The question region carries
  no BM25 weight (segmentation stops at `</FACTUAL_EVIDENCE>`), so nothing else
  protects it.

Both predict a knee at ~195. Confirming F1 would therefore confirm *nothing about
which mechanism is responsible*, and I would have reported a formula that happens to
coincide with an artefact of the recent-window policy. This was not noticed when §1
was written; it is recorded here rather than discovered afterwards.

**Stage 4 — the separating arm.** Re-run the two budgets furthest below the floor,
**B ∈ {128, 96}**, with a FIXED `--recent 32` instead of `--recent-frac 0.15`, so the
question is fully covered at every budget. Everything else identical.

* **A predicts** little or no recovery: 32 + 4 + 104 + 63 = 203 still exceeds 128, so
  the spans still do not fit and quality stays down.
* **B predicts** substantial recovery at both budgets, because the question is now
  intact and the only thing that changed is query coverage.

**Scoring.** A is supported if pooled quality at B = 128 with fixed recent stays
significantly below uncompressed; B is supported if it is no longer significantly
below. If both budgets recover fully, F1's arithmetic is the wrong account of the
floor and §1 must be retracted, however well the knee itself lands.

---

## 3b. AMENDMENT, 2026-09-04, written before any 16K budget was run: the sharpest
## consequence of the formula

If the floor is set by **absolute span lengths** rather than by a compression ratio,
it must not move when the prompt gets longer. The spans that have to fit are the
instruction prefix (~104 tokens) and one evidence record (~63); neither depends on
how many distractors surround them. On `rag_long16k` the prompt is 16,500 tokens
instead of 4,226, but the instruction prefix and the record are the same size.

**F3 — the floor is absolute, not proportional.** The 16K floor is the same
`B* ≈ 196` tokens as at 4K. Since the prompt is ~4× longer, that is an **84×**
cache reduction rather than 21.6×, and quality at B = 256 should still be at or
above the uncompressed model.

This is the strongest and most falsifiable claim the formula makes, and it is
cheap: budgets **B ∈ {512, 256, 196, 160}** on `rag_long16k`, n = 100, against the
existing `long16k-off` reference. Both segmentations (`-doc-` and `-line-`) at
B = 256 to check the boundary-awareness result transfers too.

* **F3 confirmed** if the 16K knee is in {160, 196, 256}, matching 4K.
* **F3 refuted** if the knee scales with prompt length — a proportional floor would
  put it near 196 × (16500/4226) ≈ 765, i.e. B = 512 would already be below it.

The competing account is that what matters is the *fraction* of context retained, in
which case B = 256 at 16K (1.6% kept) should fail where B = 256 at 4K (6.1% kept)
succeeded. The two accounts differ by a factor of four and cannot both be right.

**Power caveat, stated in advance.** n = 100 at 16K gives 72 answer and 28 abstain
cases against 432/168 at 4K, so a small degradation may not reach significance.
"Not significantly below uncompressed" at 16K is therefore weaker evidence than the
same phrase at 4K, and the knee is read from the pooled column with that in mind.

---

## 4. What this cannot settle

One model, one dataset, one depth, one seed, one recent_frac. The floor is a property
of *these* span lengths; the claim under test is the **formula**, not the number 196.
A different corpus with 300-token records should, by the same arithmetic, put the
floor near `(4 + 100 + 300)/0.85 = 475` — untested here and the obvious next step if
the formula survives.
