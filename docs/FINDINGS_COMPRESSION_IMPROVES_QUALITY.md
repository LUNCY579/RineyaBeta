# Compression that improves quality: +5.2 pp at 20× KV cache reduction

**Claim.** On `rag_long4k`, evicting 95% of the KV cache with a selector that needs
no forward pass makes the model **better**, not merely no worse: +5.2 pp on the
repository's hard endpoint over the uncompressed model, 34 wins against 3 losses,
exact McNemar p = 1.2e-07 on 600 cases. It improves *both* sub-tasks — answering
and abstention — and it is not a conservatism shift.

Data: `experiments/kv_significance/results/c7-*`, `c8-*`. Endpoint
`contract_fulfillment_score >= 0.999`, paired exact McNemar, same protocol as
[`FINDINGS_TRIVIALITY_CONTROL.md`](FINDINGS_TRIVIALITY_CONTROL.md).

---

## 1. The result

`lexical-doc-bm25-sys` at keep 0.05 — BM25 over the record delimiters the prompt
already prints, **plus one rule: never evict the instructions** — against no
compression at all:

| stratum | n | uncompressed | compressed @0.05 | diff | win/lose | exact p |
|---|---:|---:|---:|---:|---:|---:|
| answer | 432 | 0.7708 | 0.8032 | **+3.2 pp** | 15/1 | 5.2e−04 |
| abstain | 168 | 0.8214 | **0.9226** | **+10.1 pp** | 19/2 | 2.2e−04 |
| **pooled** | **600** | **0.7850** | **0.8367** | **+5.2 pp** | **34/3** | **1.2e−07** |

`mean_kept_ratio` 0.0500, `heavy_zero_count` 0 — the arm really compressed.

**Out-of-sample.** Records 200–599 were never looked at while any of this was
designed; every design decision used records 0–199. On that disjoint slice, both
strata, n = 400: **0.7475 → 0.8050, +5.8 pp, 24 wins / 1 loss, p = 1.6e−06.**

---

## 2. Why it works, and why the first attempt did not

The plain document-BM25 selector improves answering and **annihilates** abstention:

| arm @ keep 0.05 | answer | abstain | pooled |
|---|---:|---:|---:|
| uncompressed | 0.7708 | 0.8214 | 0.7850 |
| doc-BM25 | 0.8056 (+3.5) | **0.0000** (−82) | 0.5800 (**−20.5**) |
| doc-BM25 **+ instructions kept** | 0.8032 (+3.2) | 0.9226 (**+10.1**) | **0.8367 (+5.2)** |
| oracle: gold record only | 0.8102 (+3.9) | **0.0000** (−82) | 0.5833 (−20.2) |

The cause was measured, not guessed. `doc_bm25_weights` writes 0.0 at every position
outside the evidence block, so the 71-token system prompt survives only through the
four attention sinks — **4/71 = 0.0563**, and the verbatim refusal sentence
`"I don't know based on the available information."` survives in **0 of 200** cases
(`scripts/design/instruction_survival.py`). The abstention answer lives *only* there,
so the arm could not abstain at all.

That is a **selector defect, not a limit of compression**. Protecting the instruction
prefix — a blind rule reading only the position of the prompt's own evidence
delimiter — repairs it and turns a −20.5 pp disaster into a +5.2 pp gain.

**Note the oracle row.** An arm that keeps *only the gold record*, using
`support_document_ids`, reaches 0.8102 on the answer stratum against the blind
selector's 0.8032. The blind selector is at **99% of the oracle's ceiling**: there is
almost nothing left for a cleverer retriever to win. What was left was the
instructions, and that is now taken.

---

## 3. It is not a conservatism shift

The obvious confound is that compression simply makes the model abstain more, trading
answer accuracy for abstention accuracy. It does not:

| | false refusals on answer cases | answer accuracy | abstain accuracy |
|---|---:|---:|---:|
| uncompressed | **62 / 432** | 0.7708 | 0.8214 |
| doc-BM25 + instructions | **62 / 432** | 0.8032 | 0.9226 |

The false-refusal rate is *identical*. The model abstains exactly as often on cases it
should answer, yet answers more of them correctly and abstains correctly more often
where it should. Both sub-tasks move up at once, which a threshold shift cannot do.

---

## 4. Where the gain comes from — and where it does not

Rescued cases by `evidence_clarity`, doc-BM25 at keep 0.05, n = 432 answer cases:
`clear_direct` 9, `answer_present_with_related_distractor` 2, `clear_procedural` 2,
`trusted_fact_plus_malicious_distractor` 1, `clear_procedural_with_distractor` 1.

The "compression removes harmful distractors" story is **only part of it**, and on the
first 200 records it was not the story at all: there, the strata *designed* to contain
adversarial distractors (`trusted_fact_plus_malicious_distractor`,
`stale_and_current_with_explicit_precedence`, `clear_corroborated_multi_document`) are
all at **1.000 uncompressed** — nothing to remove. Most rescues are in `clear_direct`,
where by construction there is no adversarial distractor, so the effect is better
described as **relief from irrelevant context in general** than as defeat of
adversarial distractors specifically.

Against that, the benefit did **not** grow with context length: at 16K, doc-BM25 at
keep 0.05 is +2.8 pp over uncompressed (2/0, p = 0.5) — the same magnitude as 4K, not
larger. A simple attention-dilution account predicts growth, and does not get it.

---

## 5. What this does not show

1. **This is not the Cornerstone 2 mechanism.** The accumulated-attention score does
   the opposite: at keep 0.05 it *loses* to uncompressed (−16.7 pp at 16K, 1/13) and
   collapses to 0.4861 at keep 0.02. The arm that wins here is BM25 plus a
   keep-the-instructions rule — no attention scores, no forward pass, no training.
2. **Benchmark-scoped.** `rag_long` prints `[doc_… | trust=… ]` delimiters, which is
   what makes record segmentation possible, and the instruction block is a
   syntactically identifiable prefix. A prompt without explicit structure needs a
   chunker, and that is untested.
3. **keep 0.05 was selected after seeing 0.25 / 0.05 / 0.02 on records 0–199.** The
   disjoint replication controls for the *selector* design but not for that budget
   choice; a fresh budget sweep on held-out data would.
4. One model (474M base core), one depth (0.5), one seed. No cost measurement.
5. The abstention gain is measured against a system prompt whose refusal text is a
   fixed string. A task with a less literal abstention contract may behave differently.

---

## 6. What follows

The programme's framing should change. The question is no longer "does the
accumulated-attention score justify itself" — the triviality control settled that, and
the tight-budget cells settled it harder. The question this result opens is:

> **What is the cheapest blind selector that preserves the two things that matter —
> the evidence and the instructions — and how far can the budget fall before either
> breaks?**

At keep 0.05 both hold and quality rises. At keep 0.02 the budget (~84 slots) can no
longer hold instructions (71) plus a record (~67), and the oracle itself degrades to
0.8333. That boundary is now the interesting quantity, and it is a *capacity* question
with a clean prediction: the floor is `|instructions| + |evidence record|`, and a
selector that knows those two spans should ride the budget down to it.
