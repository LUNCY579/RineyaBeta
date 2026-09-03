# The triviality control: results

Preregistration: [`PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md`](PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md)
(read its §10 disclosures first). Analyser
`scripts/design/analyse_triviality_control.py` → artifact
`experiments/kv_significance/results/triviality_control.json`. Instruction-retention
generator `scripts/design/instruction_survival.py` → artifact
`experiments/kv_significance/results/instruction_survival.json`. Both re-verify with
`--check`.

Ran 2026-09-03 on one RTX 4070 8 GB, for most of the campaign shared with another
project's CUDA sweep holding ~3 GB.

**This document was revised after a six-lens correctness audit that independently
recomputed every number in it.** The statistics all reproduced exactly; several
prose claims did not and are corrected here. What changed is listed in §9.

---

## The headline

**Where compression actually bites, a retriever with no forward pass does not merely
match the score — it beats it, decisively.**

| keep | significance | doc-BM25 (trivial) | A − G | discordant | exact p | verdict |
|---|---:|---:|---:|---:|---:|---|
| 0.25 | 0.8056 | 0.8472 | −0.0417 | 1/7 | 0.0703 | TRIVIAL |
| **0.05** | 0.5972 | **0.8681** | −0.2708 | **0/39** | 3.6e−12 | **BEATEN** |
| **0.02** | 0.4861 | **0.8611** | −0.3750 | **0/54** | 1.1e−16 | **BEATEN** |

4K answer stratum, n = 144, paired exact McNemar. Uncompressed reference **0.8403**.

At keep 0.25 the two are statistically indistinguishable, which is the preregistered
TRIVIAL condition. Shrink the budget 12.5× and significance falls 0.8056 → 0.4861
while doc-BM25 *holds* at 0.8611 — above the uncompressed model. **Not one case at
either tight budget has significance succeeding where doc-BM25 fails** (0/39 and
0/54 discordant).

The one thing the score does that no trivial arm does is keep the **system prompt**,
and therefore abstention: 0.8571 against 0.0000 at 4K. That is measured, and its
cause is exact (§4).

---

## 1. The keep-0.25 cell

Nine arms, generated from one table so matching is structural, and checked by
manifest diff. Endpoint `contract_fulfillment_score >= 0.999` — the required phrase
literally present. `quality_score` is **not** the endpoint and is diagnostic only.

Held identical: model, keep_frac 0.25, sinks 4, recent_frac 0.15, window 16, seed
20260826, decoding. Arms A–E, G, N, Ndoc also share `cap_mib`; **arm F is the
committed legacy arm and is exempted from the diff, and it ran with `cap_mib` unset** —
stated because §1 previously claimed all nine were diff-verified.

| arm | ranks by | answer (n=144) | abstain (n=56) |
|---|---|---:|---:|
| **G** | trivial: document-level BM25, no forward pass | **0.8472** | **0.0000** |
| B | significance, centred (3,3) | 0.8194 | 0.8571 |
| **A** | significance, causal (6,0) — *the mechanism* | **0.8056** | **0.8571** |
| C | significance, no pooling | 0.7847 | 0.8929 |
| D | trivial: per-position IDF | 0.7361 | 0.0000 |
| E | trivial: naive question-token match | 0.6250 | 0.0000 |
| F | floor: random, post-pool (committed) | 0.1875 | 0.0000 |
| N | floor: contiguity-matched random | 0.1806 | 0.0179 |
| Ndoc | floor: granularity-matched random | 0.1597 | 0.0000 |

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

Recovery, each family against the floor of its own granularity:
**R_structural = 1.0645, CI [1.0101, 1.1319]** — entirely above 1.
R_positional = 0.8889, CI [0.8041, 0.9767].

**A caveat on the equivalence half of the TRIVIAL condition.** `paired_diff_ci` is an
interval *conditional on the observed discordant count*, so it is algebraically
bounded within ±n_disc/n. At 8/144 and 4/72 that is ±5.6 pp and ±5.6 pp, so the
preregistered "CI within ±10 pp" clause is **automatically satisfied and does no
work**. The TRIVIAL verdict therefore rests on R and the exact McNemar p, not on that
clause. An independent profile-likelihood interval is ~2× wider and still clears
±10 pp, so the verdict survives — but the clause should not be read as evidence.

**Provenance.** Arm A at 4K is **200/200** byte-identical to the committed
`long4k-sig-k0.25-rf0.15`, and 4K is fully deterministic: three independent runs of
the identical config agree 200/200 with zero endpoint flips.

---

## 2. Finding 1 — the score is not what makes retrieval work

R_structural's whole interval sits above 1: the trivial selector recovers *more* than
100% of significance's advantage over a contiguity-matched floor. A − G is −4.2 pp,
p = 0.0703.

This is not a weak-baseline artifact. The trivial arms span **22.2 pp** (E 0.6250 →
G 0.8472) and the floors sit at 0.16–0.19, so the endpoint discriminates; what it does
not do is separate accumulated attention from BM25. *(An earlier version of this
sentence said "62 pp" — that was 0.625 misread as a range. The correct span is
22.2 pp.)*

Independently confirmed by the audit: arm G's top-ranked record is a support document
in **144/144** (4K) and **72/72** (16K) cases, with no ties at the top, and the
result does not depend on the `[doc_…]` header — stripping it leaves 144/144.

The ordering **G > A > D > E ≫ floors** says the useful signal is query–document
lexical match at the right granularity. O11 already found the score measures "what
the question reaches for"; this prices that: an extra scoring pass per layer, for
something BM25 over the prompt's own delimiters does better.

---

## 3. Finding 2 — and where compression bites, the trivial arm wins outright

The keep-0.25 operating point has almost no headroom: uncompressed 0.8403 → A 0.8056
is **5 of 144 cases**. So the tight-budget cell was run (arms A and G only, 200 cases,
matched geometry, `heavy_zero_count` 0 throughout, differing only in `select`):

* **keep 0.05** — A 0.5972, G 0.8681, **0/39** discordant, p = 3.6e−12
* **keep 0.02** — A 0.4861, G 0.8611, **0/54** discordant, p = 1.1e−16

Zero discordant pairs in favour of the mechanism, at both budgets. Significance
degrades roughly linearly as slots are removed; doc-BM25 does not degrade at all,
because identifying one record out of ~187 does not get harder when the budget
shrinks — it only has to keep that record.

This is the preregistered **BEATEN** row, at budgets where compression is the point.
It also closes the escape hatch that keep 0.25 was "too easy to discriminate": the
harder operating point does discriminate, and it discriminates against the mechanism.

---

## 4. Finding 3 — the trivial selector wins by deleting the instructions

Every trivial arm scores exactly **0.000** on the abstain stratum against the
mechanism's 0.8571 (4K) and 0.2143 (16K, 6/0, p = 0.0312). Those cases require
reproducing `"I don't know based on the available information."`, whose text appears
**only** in the 71-token system prompt.

From `scripts/design/instruction_survival.py` — the real `select_keep`, the real
8 KV heads, the real multi-eviction path, over all 200 4K prompts (means over cases
and heads; "first" = after the first eviction, when the reply's opening tokens are
produced):

| arm | system prompt kept (1st / last) | refusal sentence intact (1st / last) |
|---|---:|---:|
| G (doc-BM25) | **0.0563** / 0.0563 | **0.0000** / 0.0000 |
| Ndoc (record-random) | 0.0563 / 0.0563 | 0.0000 / 0.0000 |
| E (naive) | 0.1037 / 0.1013 | 0.0000 / 0.0000 |
| N (prepool-random) | 0.2533 / 0.1639 | 0.0244 / 0.0163 |
| D (per-position IDF) | 0.3263 / 0.3263 | 0.0000 / 0.0000 |

**0.0563 = 4/71 exactly.** Both record-level arms retain the four attention sinks of
the system prompt and *nothing else*, because they write 0.0 at every position
outside the evidence block. The refusal sentence survives in 0 of 200 cases for
every trivial arm; the one arm that ever keeps it, N, does so in 2.4% of
(case, head) pairs. *(An earlier version of this table reported "N 5/200" — that was
a per-head fraction printed as a case count, from an uncommitted single-head
simulation. The generator is now in the repository and the units are fractions.)*

The mechanism keeps the system prompt because accumulated attention mass finds
instruction text the question never lexically matches — precisely the class of
content a lexical index cannot see.

**This is the constructive result.** The score's demonstrated advantage is not
retrieval; it is retaining load-bearing context with *no lexical overlap with the
query*. Narrower than the paper's framing, and more defensible. It also means arm G
is not a deployable alternative, and that any residual A-over-G margin on the answer
stratum is confounded with "G discarded the instructions" — which is why the two
strata are never pooled.

---

## 5. Finding 4 — the paper's central claim does not transfer

Cornerstone 2's contribution is that the pooling window's *direction* matters. On
passkey that is overwhelming — 1.000 causal against 0.028 centred at 32K. **The
ablation had never been run on natural language at length.**

> **A (causal 6,0) 0.8056 against B (centred 3,3) 0.8194** — −1.4 pp, 1/3
> discordant, p = 0.625 at 4K; −1.4 pp, 0/1, p = 1 at 16K.

By the preregistered symmetric rule this is **UNDERPOWERED / INDETERMINATE**: exact
McNemar cannot reach p < 0.05 below 6/0 discordant, so nothing under ~6 pp resolves
at these n. **It is not evidence that centred is better.** But it is not support for
causal either, and the point estimate is on the wrong side at both lengths.

Pooling as such is the same story: A − C = +2.1 pp (p = 0.453) at 4K and +4.2 pp
(p = 0.25) at 16K.

The 5-of-144 headroom is the honest qualifier and it was recorded before these
numbers existed. Note though that §3 removes part of its force: at keep 0.05 and 0.02
the endpoint has plenty of headroom, and there the mechanism loses badly to a
selector with no pooling geometry at all.

*(The passkey figure quoted here is the 32K cell. An earlier version of this section
said "216 discordant pairs at 32K"; that cell has 35 pairs, and 216 was a
seven-cell total. The seven-cell total recomputes to 226.)*

---

## 6. The 16K cell

Same 100 source questions as the first 100 of the 4K cell — verified 100/100 on id
stem and question text — so this is a **length replication on one item set**, not an
independent sample. Preregistration §5 forbids reading an equivalence verdict from
n = 72, so the 16K cell speaks to **direction only**.

| arm | answer (n=72) | abstain (n=28) |
|---|---:|---:|
| B — centred | 0.9028 | 0.2143 |
| **A — causal** | **0.8889** | **0.2143** |
| **G — trivial doc-BM25** | **0.8889** | **0.0000** |
| C — no pooling | 0.8472 | 0.2143 |
| D / E | 0.7778 / 0.5417 | 0.0000 / 0.0000 |
| N / Ndoc | 0.1944 / 0.1528 | 0.0357 / 0.0000 |

A − G = **0.0000 exactly** (2/2, p = 1). R_structural = 1.0000 [0.9259, 1.0769].
A − D = +11.1 pp (p = 0.0215); A − E = +34.7 pp (p = 4.2e−07). Abstain: A 0.2143
against 0.0000, 6/0, p = 0.0312.

The direction agrees with 4K on every contrast. The analyser labels this cell TRIVIAL
by the same rule, and the artifact carries the "direction only" caveat — but the
verdict that *counts* is the 4K one, and this document no longer claims "TRIVIAL at
both lengths" as a headline.

### The 16K reproduction gate fired — on nondeterminism, not code drift

Arm A at 16K is **99/100** byte-identical to the committed arm, and the analyser
dropped arm F from the cell on its own. **My original reading of that as a code-state
difference was wrong.** Four independent runs of the identical config:

| pair | replies | endpoint flips |
|---|---:|---:|
| committed vs campaign | 99/100 | 0 |
| committed vs replicate-a | **100/100** | 0 |
| committed vs replicate-b | **100/100** | 0 |
| campaign vs replicate-a / -b | 99/100 | 0 |
| replicate-a vs replicate-b | **100/100** | 0 |

Exactly **one case of 100 ever varies**, three of four runs agree, and the outlier is
the campaign arm A — the only one produced while another project held ~3 GB and the
card ran at 86–88 °C. The divergence is a paraphrase (`'is launched in'` →
`'launches in'`) scoring `contract_fulfillment = 0.3333` in both forms.

**Zero endpoint flips in all six pairwise comparisons**, and the hard endpoint is
0.7000 in all four runs — so no number in this document is affected. But two things
follow: byte-identity is the wrong instrument at 16K, and the arm-F drop was
spurious. 4K, by contrast, is deterministic across three runs.

### One arm failed and was re-run

`random-prepool` at 16K failed on its first attempt with `CUDA error: out of memory`
under contention. Its partials were deleted rather than analysed, the log recorded
`FAILED` rather than a null, the campaign continued, and it completed on retry in
623 s with no code change. A later 16K run OOM'd the same way at `--cap-mib 5200` and
succeeded at 6500 once the card was free.

---

## 7. What this does not show

1. **Benchmark-scoped.** `rag_long*` prints its own record delimiters and every
   answer case carries `lexical_support_fraction = 1.0` (verified 144/144 and 72/72).
   BM25 over those delimiters solves it. A trivial-wins verdict here is a statement
   about benchmarks of this shape.
2. **P4 is unrun** — whether the score survives when question and evidence share no
   content words. That is the experiment that would turn this into a statement about
   the score rather than about this dataset.
3. One model, one depth (0.5), one seed. The keep axis now has three points; the
   others have one.
4. **Capability, not cost.** No timing claim is made or implied.

---

## 8. Consequences for the paper

* §4.10 now has its missing causal-vs-centred ablation, and the ablation does not
  support the claim. That belongs in the abstract's framing.
* The "accumulated importance" framing should be replaced by what O11 and this
  campaign jointly establish: the score is a query-similarity index computed in
  attention space, whose distinctive value is covering context with no lexical
  overlap with the query.
* Any claim that the score is what makes long-context retrieval work must report the
  document-level BM25 arm beside it — including the keep-0.05 and keep-0.02 cells,
  where the mechanism loses at p < 1e−11.

---

## 9. Corrections made after the correctness audit

The audit independently recomputed every statistic here from the per-case records
with a from-scratch implementation and found **0 mismatches** — all arm rates, all 42
contrasts, all 8 recovery points, both bootstrap CI sets, the strata sizes, and the
reproduction counts. It also proved, by negative control, that the trivial arms'
selection is independent of the accumulated score (identical selection under three
wildly different score fields, while significance differs), so there is no score
leak; and it reproduced arm G's 1.000/1.000 gold-span retention exactly.

Corrected here: the "62 pp" span (→ 22.2 pp); "216 discordant pairs at 32K" (→ 35 in
that cell, 226 over seven); the §4 table's N refusal entry and its provenance (→ a
committed generator, fractions not counts); "142/144 and 65/72" support-document
ranking (→ 144/144 and 72/72); the claim that all nine arms were manifest-diff
verified (→ arm F is exempt and ran with `cap_mib` unset); the 16K reproduction
attribution (→ nondeterminism, §6); and the headline "TRIVIAL at both lengths", which
asserted an equivalence verdict the frozen design permits only from 4K.

Code bugs the audit proved, all fixed, none of which moved a published number —
confirmed by the analysis artifact being byte-identical after the fixes:

* `verdict_for` could pair one arm's name with another arm's p-value and CI, because
  `L` was a global argmax over rates while the recovery family elected its own best
  arm. Demonstrated to flip TRIVIAL↔BEATEN on constructed data.
* The PARTIAL branch was unbounded above, so cells with R wholly above 1 printed
  "recovers most of the advantage" where the frozen table says INCONCLUSIVE.
* The manifest gate whitelisted `select` and `pool`, so a directory holding one arm's
  run under another arm's name passed every gate. Both are now asserted against the
  arm's definition.
* The campaign runner emptied an arm's output directory *before* waiting for the GPU,
  so `--force` on a busy card destroyed a complete 200-case arm and ran nothing.
* `--gate` returned success while printing FAIL.

Test coverage was the weakest part and is now stronger: a mutation study showed 8 of
10 mutations surviving green, including deletion of the post-pool mask re-assertion
(which gives 30–32 of 32 KV heads duplicate slots while `mean_kept_ratio` still reads
0.25). Nothing had asserted that `select_keep` returns *distinct* positions. New
assertions kill all three worst mutations. One pre-existing assertion was also
**flaky** — its heavy budget exceeded the span a window can reach, so surplus slots
fell to the tie-break and could fail it about 1 run in 6; it is pinned and the suite
now passes 20/20 consecutively.
