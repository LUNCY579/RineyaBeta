# From passkey to hard reasoning: the next research programme

Status: **planning, not started.** To begin after cornerstone 2 closes (P2.4
disentanglement, the power run, and the paper). Written 2026-08-23.

This document exists to be revised. Every section is dated and every claim it makes
about the future is a *prediction* that later measurement is expected to overturn.

---

## 1. Why reasoning, and why now

Every long-context number in this program is passkey retrieval: a five-digit key in
homogeneous filler. That is not a small caveat. It is the single largest threat to
the significance matrix's external validity, and it is now the binding one, because
the mechanism has stopped failing on the task — 128K at B=16 scores 36/36.

The reason to move to reasoning is not that it is harder. It is that **reasoning is
the direct falsification test of cornerstone 2's central claim.**

Cornerstone 2 claims evidence lives in contiguous *spans*, and that independent
per-position ranking fragments them. Passkey contains exactly **one** span. Multi-hop
reasoning requires **several spans held simultaneously**, which is the regime where
the span account either holds or breaks. No result to date has entered it.

A second claim is equally exposed. P2.2 measured that the score prefers text
*resembling the query* — framing survives at 54%, payload at 35% across 224 streams.
Passkey's haystack contains nothing that resembles the query. Real prose contains a
great deal. The prediction is uncomfortable and specific: **distractors that look
like the question should be preferentially retained, crowding out the true answer.**

The user's framing — "retrieve **and guard** the correct answer" — is exactly right.
Passkey tests retrieval. Only a task with competitors tests guarding.

**Expected outcome, stated in advance so it is not read as a regression:** compression
will fall sharply from 7,864×. That figure depends on a haystack with one distinctive
region and on a policy with 8 heavy slots at B=16. A task with several plausible
distractors cannot behave that way; there is not room. A large drop is the *correct*
result and is itself a finding.

---

## 2. The comparison table

Everything that differs between what has been measured and what is proposed. This is
not background — §3 turns each row into an experiment.

### 2.1 Task and evidence structure

| axis | passkey (measured) | hard reasoning (proposed) | why it matters |
|---|---|---|---|
| operation | locate, copy | locate, **combine**, infer | copying needs one span intact; combining needs several |
| evidence spans | **1** | **2+** | the direct test of the span account |
| span length | 5 tokens, atomic | 20–40+ tokens, variable | pool `back` reach may need to scale with span length |
| span boundary | crisp (digit run) | **fuzzy** — where does evidence end? | the pooling window has no notion of a fuzzy edge |
| evidence contiguity | perfectly contiguous | often **discontiguous** (coref, bridging) | span-based protection may not apply at all |
| answer form | exact 5-digit string | phrase, entity, number, or sentence | changes the metric and the failure taxonomy |

### 2.2 Haystack and competition

| axis | passkey | hard reasoning | why it matters |
|---|---|---|---|
| haystack | homogeneous filler, 1 distinctive region | real prose, **many** distinctive regions | localisation is trivial in the first case |
| distractors | **none** | many, some query-shaped | the P2.2 bias predicts these are *preferred* |
| query–evidence lexical overlap | framing appears **verbatim** in the needle | often **low**; question and answer share few words | query-aware scoring may lose its signal entirely |
| evidence position | controlled (depth 0.1/0.5/0.9) | natural, uncontrolled | depth can no longer be a clean factor |
| information density | ~1 bit per 25K tokens | high and uneven | budget must cover more genuinely-useful text |

### 2.3 Measurement

| axis | passkey | hard reasoning | why it matters |
|---|---|---|---|
| metric | exact match, binary | EM / F1 / judged, graded | graded metrics need different statistics |
| model ceiling uncompressed | ≈ 1.00 | **well below 1.00, possibly near 0** | a floor effect would make every number noise |
| reference available? | yes below 32K, **no above** | same hard limit | see §5 |
| failure mode | truncated prefix — **legible** | wrong answer / hallucination — **illegible** | the P2 diagnosis was only possible because failure was legible |
| trials per cell | 12 independent needles | must be far higher for graded metrics | see §6 |
| contamination risk | none (synthetic) | **real** for public benchmarks | must be controlled |

### 2.4 Mechanism exposure

| mechanism component | stressed by passkey? | stressed by reasoning? |
|---|---|---|
| accumulated attention score | yes | yes |
| count normalisation (`sum/count^e`) | yes | yes |
| attention sinks | yes | yes |
| recent window | weakly | strongly (multi-turn, longer answers) |
| **query-aware scoring** | trivially — query text is *in* the needle | **severely** — may share no words with evidence |
| **pool window `back` reach** | tuned on a 5-token span | must generalise to variable-length spans |
| **span integrity** | one span | **many spans, competing** |
| budget schedule | fixed | may need to grow with hop count |

---

## 3. Reading the table as an experiment matrix

The central design decision: **do not jump from passkey to LongBench.** Every row of
§2 would change at once and no result would be attributable to anything.

Instead, **each row becomes a rung.** Change one axis, keep the rest synthetic and
controlled, and measure what that single change costs. This preserves the property
that made P2 → P2.2 → P2.3 work: when failure is legible and one variable moved, the
mechanism can be diagnosed rather than merely scored.

| stage | what changes | everything else | isolates |
|---|---|---|---|
| **0** | — | — | baseline (done: 7,864× at 128K) |
| **1** | add **distractor keys** (k plausible competitors) | filler haystack, 1 real needle | the query-similarity bias |
| **2** | needle becomes a **sentence**, then a paragraph | filler haystack, no distractors | span length vs pool `back` reach |
| **3** | **multi-hop** chain (X=5, Y=X, Z=Y → Z?) | filler haystack | multi-span; budget vs hop count |
| **4** | haystack becomes **real prose** | synthetic needle preserved | haystack homogeneity |
| **5** | evidence becomes **natural** (real QA over real text) | length still controlled | query–evidence lexical gap |
| **6** | **public benchmarks** | nothing controlled | external validity |

Stages 1–3 are cheap: they are edits to `build_prompt` in
`experiments/kv_significance/src/passkey.py`, reusing the entire existing harness,
paired design, and `verify_claims` discipline. Stage 4 needs a prose corpus. Stages
5–6 need real datasets and a grader.

**The ordering is deliberate.** Stage 1 is first because it is the cheapest test of
the sharpest prediction, and because if the query-similarity bias is severe, it
changes the design of everything after it.

---

## 4. Pre-registered predictions

Declared before any of it runs. Each names the measured finding it follows from and
the observation that would falsify it.

| # | prediction | follows from | falsified by |
|---|---|---|---|
| P1 | Adding *k* query-shaped distractors degrades accuracy faster than adding *k* neutral distractors, at matched budget | P2.2: framing 54% vs payload 35% | no difference between query-shaped and neutral distractors |
| P2 | Required budget grows roughly **linearly in the number of hops**, not in context length | span account: each hop needs one intact span | budget requirement flat in hop count, or growing with length instead |
| P3 | The pool `back` reach needed for a span of length *L* scales **with L**, not with a constant | P2.3: back=6 suffices for a 5-token key | a fixed `back` works for all span lengths |
| P4 | When query and evidence share few words, query-aware scoring loses most of its advantage over query-blind | P1: future attention unpredictable from content; the query is the only signal | query-aware retains its full margin at zero lexical overlap |
| P5 | Compression at fixed accuracy falls by **at least an order of magnitude** from 7,864× on stage 4+ | B=16 leaves 8 heavy slots; real haystacks have many candidate regions | real-prose haystacks compress comparably to filler |
| P6 | Failure becomes illegible — wrong answers rather than truncations — as spans lengthen | P2: 4-of-5 prefix truncation is a property of atomic digit spans | truncation-style failures persist on sentence-length evidence |

**P4 is the one to fear.** If it holds strongly, the mechanism's largest single win
(0.33 → 0.78 at 32K) is an artefact of passkey's verbatim query overlap, and the
whole approach needs a different scoring signal for real tasks. That would be the
most important negative result the program could produce, and it is cheap to test at
stage 5.

---

## 5. The two constraints that gate the design

**The model's ceiling comes first.** If a 474M-parameter model cannot do multi-hop
reasoning *uncompressed*, every compression number is a floor effect measuring the
model, not the mechanism. **The first experiment is not a compression run.** It is:
can this model do the task at 4K with a full cache? If it scores near zero, the task
must be recalibrated downward until it does — or the programme moves to a larger
model and a different card.

**There is no uncompressed reference above ~32K on this hardware.**
`mem_64k_full.json` OOM'd at 6,425.6 MiB allocated with the cap at 8,100 of 8,187.5;
the highest successful peak anywhere in 75 result files is 4,972.9 MiB. Compressed
runs peak at ~1.5 GiB. So "compressed vs uncompressed at 128K" **cannot be run here.**

Three designs that work anyway:

1. **Anchor at 32K.** Establish the compressed/uncompressed gap where both fit, then
   carry it forward as a bound rather than a measurement.
2. **Compare mechanisms, not against truth.** Paired arms — significance vs
   StreamingLLM vs H2O vs random-at-equal-budget — all fit at 1.5 GiB. The random
   control has already caught two false results in this program.
3. **Budget-ladder saturation.** Raise the budget until accuracy plateaus and treat
   the plateau as the reference. Weaker than a true reference; state it as such.

---

## 6. Methodology standards

Carried forward from what already works here, plus what the graded-metric setting
newly requires.

| standard | rule |
|---|---|
| pre-registration | declare thresholds and predictions before running; this document is the first instalment |
| pairing | same items across arms; exact McNemar on discordant pairs |
| independence | **`n` is the number of distinct items, not item×condition cells.** The current harness reuses keys across depths, so 12×3 is 12, not 36 |
| power | a perfect cell at n=12 supports a 95% lower bound of only 0.779. Reaching 0.97 needs 99; 0.99 needs 299 |
| graded metrics | binary McNemar no longer applies; use paired bootstrap or Wilcoxon signed-rank on per-item scores |
| controls | always include random-at-equal-budget and at least one published baseline |
| provenance | every claim pinned in `scripts/design/verify_claims.py`, recomputed from committed data, with manifest assertions |
| contamination | for public benchmarks, report the risk and prefer datasets released after the base model's cutoff, or perturb items |
| negative reporting | negative and retracted results are documented with the number that closed them |

---

## 7. What would make the significance matrix fail, and how we would know

Stated plainly, because a programme that cannot fail is not a programme.

| failure | what it would look like | which stage catches it |
|---|---|---|
| query-similarity bias dominates | query-shaped distractors are retained over the answer | 1 |
| span protection does not generalise | fixed `back` cannot cover variable spans | 2 |
| multi-span is qualitatively different | budget explodes with hop count | 3 |
| filler was doing the work | large drop moving to real prose at matched budget | 4 |
| query-aware scoring was a passkey artefact | its margin vanishes at low lexical overlap | 5 |
| the whole thing is task-specific | no advantage over StreamingLLM on real benchmarks | 6 |

Any one of these is publishable. The programme is designed so that a failure at
stage *k* still leaves stages < *k* as standing results.

---

## 8. Revision protocol

This document is versioned in git and revised as results arrive. On each revision:
mark superseded predictions as **retracted** with the measurement that closed them
rather than deleting them; date every change; and keep the comparison table in §2 in
sync with what has actually been measured.

Precedent: the √L floor law (`B_half = 0.141·L^0.535`) was fitted from the centred
pool window and became inapplicable within hours when the causal window pushed every
length below the sampled range. It is marked retracted in `FINDINGS_P2.md` rather
than removed. The same treatment applies here.

---

## Appendix — reusable assets already in the repository

| asset | path | note |
|---|---|---|
| passkey harness | `experiments/kv_significance/src/passkey.py` | paired design, per-trial records, manifest |
| the mechanism | `scripts/kv/significance.py` | `Policy(pool, pool_back, pool_fwd, count_exponent, …)` |
| chunked prefill | `scripts/kv/chunked_prefill.py` | keeps peak memory flat in length; `score_with_query` |
| paired statistics | `scripts/design/paired_passkey.py` | exact McNemar; refuses unmatched comparisons |
| claim verification | `scripts/design/verify_claims.py` | 93 pinned numbers, manifest assertions, quarantine guard |
| span tracing | `experiments/stabilizer/src/trace_span_positions.py` | per-position survival through every gather |
| non-passkey eval set | `artifacts/eval_sets/rag_dev.jsonl` | with gate runs in `artifacts/gate_runs/` |
