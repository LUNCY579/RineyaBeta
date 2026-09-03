# The triviality control: results

Preregistration: [`PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md`](PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md)
(v3 — read its §10 disclosure first). Analyser:
`scripts/design/analyse_triviality_control.py`, artifact
`experiments/kv_significance/results/triviality_control.json`.

Ran 2026-09-03 on a single RTX 4070 8 GB shared with another project's CUDA sweep,
at `--cap-mib 4500` (4K) and `5200` (16K) so an OOM would land inside this process's
budget rather than on the card.

---

## The headline

**On the retrieval task the score is replaceable. On the instruction task it is not,
and that is the finding worth keeping.**

The preregistered primary verdict, 4K answer stratum, n = 144, is **TRIVIAL**: a
document-level BM25 selector that needs no forward pass at all is statistically
indistinguishable from — and numerically above — accumulated-attention significance.
**The 16K cell returns TRIVIAL too**, with the two arms exactly equal (§7).

But the same selector scores **0.000** where the mechanism scores **0.857** (4K) and
**0.214** (16K), on the ~28% of cases that ask the model to abstain. It does not fail
to retrieve; it deletes the system prompt. That is measured, not inferred (§4).

Neither of the two things the paper argues for — that the pooling window must reach
backward, and that it must reach at all — produces a resolvable effect at either
length, and the point estimate for directionality is on the wrong side in both
cells (§3).

---

## 1. The 4K cell, complete

Nine arms, matched on model / keep_frac 0.25 / sinks 4 / recent_frac 0.15 / window 16
/ seed 20260826 / cap_mib, verified by manifest diff. Endpoint is
`contract_fulfillment_score >= 0.999` — the required phrase literally present —
with paired exact McNemar. The composite `quality_score` is not the endpoint and is
reported diagnostic-only.

**Provenance.** This campaign's arm A is **200/200 byte-identical** to the committed
`long4k-sig-k0.25-rf0.15` arm under `transformers 5.15.0` / `torch 2.13.0+cu130`.
The whole committed language line reproduces on this machine, so §6.1's reproduction
gate passes and the legacy random arm is licensed for reuse.

| arm | what it ranks by | answer (n=144) | abstain (n=56) |
|---|---|---:|---:|
| **G** | trivial: document-level BM25, no forward pass | **0.8472** | **0.0000** |
| **B** | significance, centred (3,3) | 0.8194 | 0.8571 |
| **A** | significance, causal (6,0) — *the mechanism* | **0.8056** | **0.8571** |
| **C** | significance, no pooling | 0.7847 | 0.8929 |
| **D** | trivial: per-position IDF | 0.7361 | 0.0000 |
| **E** | trivial: naive question-token match | 0.6250 | 0.0000 |
| **F** | floor: random, post-pool (committed) | 0.1875 | 0.0000 |
| **N** | floor: contiguity-matched random | 0.1806 | 0.0179 |
| **Ndoc** | floor: granularity-matched random | 0.1597 | 0.0000 |

Uncompressed reference on the same stratum: **0.8403**.

Paired contrasts on the answer stratum:

| contrast | diff | 95% CI | discordant | exact p |
|---|---:|---|---:|---:|
| A − G | **−0.0417** | [−0.0552, +0.0029] | 1/7 | 0.0703 |
| A − D | +0.0694 | [+0.0059, +0.1090] | 14/4 | 0.0309 |
| A − E | +0.1806 | [+0.1110, +0.2134] | 29/3 | 2.6e−06 |
| **A − B** | **−0.0139** | [−0.0274, +0.0170] | 1/3 | 0.625 |
| A − C | +0.0208 | [−0.0204, +0.0450] | 5/2 | 0.453 |
| G − Ndoc | +0.6875 | [+0.6257, +0.7010] | 100/1 | 8.1e−29 |
| D − N | +0.5556 | [+0.5055, +0.5556] | 80/0 | 1.7e−24 |
| Ndoc − N | −0.0208 | [−0.1054, +0.0672] | 17/20 | 0.743 |

Trivial-recovery fraction, each family against the floor of its own granularity:

* **R_structural = 1.0645**, bootstrap 95% CI **[1.0101, 1.1319]** — entirely above 1.
* R_positional = 0.8889, CI [0.8041, 0.9767].

---

## 2. Finding 1 — the score is not what makes retrieval work

R_structural > 1 with its whole interval above 1 means the trivial selector recovers
*more* than 100% of significance's advantage over a floor with matched contiguity.
A − G is −4.2 pp with an exact CI inside ±10 pp and p = 0.0703, which is the
preregistered TRIVIAL condition: not significantly different, and bounded.

This is not a weak-baseline artifact. The trivial arms span 62 pp
(E 0.625 → G 0.847) and the floors sit at 0.16–0.19, so the endpoint discriminates
perfectly well; what it does not do is separate accumulated attention from BM25.

The ordering **G > A > D > E ≫ floors** says the useful signal is *query–document
lexical match at the right granularity*, and the score is one way of computing an
approximation to it. O11 already found the score measures "what the question reaches
for". This puts a number on how much that costs to compute: a full extra scoring pass
per layer, for something a BM25 over the prompt's own delimiters does better.

---

## 3. Finding 2 — the paper's central claim does not transfer to language at length

Cornerstone 2's contribution is that the pooling window's *direction* matters. On
passkey that is overwhelming: 1.000 causal vs 0.028 centred at 32K, 216 discordant
pairs, zero against. **The ablation had never been run on natural language at
length. It has now.**

> **A (causal 6,0) = 0.8056 against B (centred 3,3) = 0.8194.**
> Difference −1.4 pp, 1/3 discordant, exact p = 0.625.

By the preregistered symmetric rule this is **UNDERPOWERED / INDETERMINATE**, and
that wording is load-bearing: exact McNemar cannot reach p < 0.05 below 6/0
discordant, so at n = 144 nothing under ~6 pp can resolve. **This is not evidence
that centred is better.** But it is not support for causal either, and the point
estimate is on the wrong side.

The same holds for pooling as such: **A − C = +2.1 pp, p = 0.453**. Removing span
inheritance entirely costs nothing measurable here.

So on this benchmark, at this budget, neither of the two things the paper argues for —
that the window must reach backward, and that it must reach at all — produces a
detectable effect. Both effects on passkey are tens of points. Whatever the mechanism
is doing on natural language at 4K, it is not the span-repair story.

**One honest qualifier, and it is large.** At keep 0.25 compression costs the
mechanism only 5 of 144 answer cases (0.8403 uncompressed → 0.8056). There is almost
nothing for an ablation to move. A directionality effect could be real and invisible
here simply because the operating point is too easy. That is a limitation of the
design, recorded in the preregistration before these numbers existed (§10 v3), not a
conclusion drawn after them.

---

## 4. Finding 3 — the trivial selector wins by deleting the instructions

Every trivial arm scores exactly **0.000** on the abstain stratum where the mechanism
scores 0.857. The abstain cases require reproducing
`"I don't know based on the available information."`, whose text appears **only** in
the 71-token system prompt.

Measured with the shipped weight functions on all 200 4K prompts:

| arm | system-prompt positions kept | refusal sentence kept intact |
|---|---:|---:|
| G (doc-BM25) | 0.056 | **0/200** |
| Ndoc (record-random) | 0.056 | 0/200 |
| E (naive) | 0.108 | 0/200 |
| N (prepool-random) | 0.247 | 5/200 |
| D (per-position IDF) | 0.326 | 0/200 |

`doc_bm25_weights` writes 0.0 at every position outside the evidence block, so the
system prompt is protected only by the four attention sinks. The mechanism keeps it
because accumulated attention mass finds instruction text that the question never
lexically matches — and that is precisely the class of content a lexical index cannot
see.

**This is the constructive result.** The score's measurable advantage is not
retrieval. It is that it identifies load-bearing context with *no lexical overlap
with the query at all*. Stated that way it is a narrower claim than the paper makes,
and a more defensible one.

It also means arm G is not a deployable alternative, and any residual A-over-G margin
on the answer stratum is confounded with "G discarded the instructions" — which is
why the primary statistic is stratified and why these two strata are never pooled.

---

## 5. What this does not show

1. **Benchmark-scoped.** `rag_long*` prints its own record delimiters and carries
   `lexical_support_fraction = 1.0`. BM25 over those delimiters solves it. A TRIVIAL
   verdict here is a statement about benchmarks of this shape, not about the score in
   general. The withdrawal §7 of the preregistration mandates should be reported
   scoped until P4 runs.
2. **One operating point.** keep 0.25 leaves 5 cases of headroom. Nothing here speaks
   to budgets where eviction actually bites.
3. **P4 is still unrun** — whether the score survives when question and evidence share
   no content words. That is the experiment that would convert this into a statement
   about the score rather than about this dataset.
4. **Capability, not cost.** No timing claim is made or implied.

---

## 6. Consequences for the paper

* §4.10's natural-language section now has the causal-vs-centred ablation it was
  missing, and the ablation does not support the claim it was missing for. That
  belongs in the abstract's framing, not in limitations.
* The abstract's "accumulated importance" framing should be replaced by what O11 and
  this campaign jointly establish: the score is a query-similarity index computed in
  attention space, whose distinctive value is covering context with no lexical
  overlap with the query.
* Any claim that the score is what makes long-context retrieval work needs the
  document-level BM25 arm reported alongside it.

---

## 7. The 16K cell — same verdict, cleaner

The 16K cell is the same 100 source questions as the first 100 of the 4K cell
(preregistration §9.2), so it is a length replication on one item set, not an
independent sample, and §5 forbids reading an equivalence verdict from n = 72 —
direction only. Its verdict is also **TRIVIAL**, and the point estimate is exact:

| arm | answer (n=72) | abstain (n=28) |
|---|---:|---:|
| B — significance, centred | 0.9028 | 0.2143 |
| **A — significance, causal** | **0.8889** | **0.2143** |
| **G — trivial doc-BM25** | **0.8889** | **0.0000** |
| C — significance, no pooling | 0.8472 | 0.2143 |
| D — trivial per-position IDF | 0.7778 | 0.0000 |
| E — trivial naive match | 0.5417 | 0.0000 |
| N — contiguity floor | 0.1944 | 0.0357 |
| Ndoc — granularity floor | 0.1528 | 0.0000 |

* **A − G = 0.0000 exactly**, 2/2 discordant, p = 1, CI [−0.048, +0.048].
  **R_structural = 1.0000**, bootstrap CI [0.9259, 1.0769]. R_positional = 0.84,
  CI [0.717, 0.9545].
* **A − B = −1.4 pp**, 0/1 discordant, p = 1 — the same sign as 4K, and again
  underpowered. Two cells, both with the point estimate against the paper's claim.
* A − C = +4.2 pp, 3/0, p = 0.25 — pooling again not resolvable.
* A − D = +11.1 pp, p = 0.0215; A − E = +34.7 pp, p = 4.2e−07.
* **Abstain: A 0.2143 against 0.0000 for every trivial arm, 6/0, p = 0.0312** — the
  instruction finding replicates and is significant even at n = 28.

### The reproduction gate fired here, and it was right to

Arm A at 16K is **99/100** byte-identical to the committed
`long16k-sig-k0.25-rf0.15`, not 100/100. By the preregistered rule that is a FAIL:
the analyser dropped arm F from the 16K cell on its own and flagged that no 16K number
may be compared to a committed one from the earlier code state.

The single divergence is immaterial to the endpoint and is reported so a reader can
judge it:

```
v175_fact_225_pinerelay_1225_13_region_conflict__L16384_d0.5
  committed: 'PineRelay 1225 is launched in Australia.'
  now      : 'PineRelay 1225 launches in Australia.'
  contract_fulfillment: 0.3333 both
```

A paraphrase that fails the hard endpoint in both versions, so **no hard-endpoint
count in this document is affected**. The gate is deliberately stricter than the
endpoint; it fired, it cost one optional arm, and that is the behaviour that was
written down before the data.

### One arm failed and was re-run

`random-prepool` at 16K failed on its first attempt under GPU contention (another
project held ~3 GB throughout; the card sat at 86–88 °C and arms ran 550–720 s
against 390–530 s at 4K). Its partial artifacts were deleted rather than analysed,
per §8, and the campaign continued through the remaining arms — the arm order exists
so that a mid-campaign death costs the least valuable arm. It completed on the retry
in 623 s with no code change. The campaign log recorded `FAILED: 1 ['N/rag_long16k']`
and refused to treat the gap as a null.

---

## 8. Both cells, one sentence

At keep 0.25, on both lengths, a BM25 retriever over the delimiters the prompt already
prints is indistinguishable from accumulated-attention significance on the retrieval
task (R = 1.06 and 1.00, both CIs containing or above 1), and loses to it completely
on the instruction task (0.857/0.214 against 0.000/0.000). The mechanism's
demonstrated value on natural language at length is **not** finding the evidence; it
is keeping context the query does not lexically match.
