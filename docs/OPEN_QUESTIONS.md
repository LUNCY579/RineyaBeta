# Open questions: results we cannot yet explain

Every anomaly in this programme that has an observation but not a cause. An
unexplained result is indistinguishable from a bug until it is explained, and this
programme has repeatedly found that it *was* a bug — `Policy.pool` defaulting to 1 so
every long-context result measured pooling disabled; `count_exponent` inert on the
self-compressing path; eviction running during prefill and reading as a 2.6× speedup;
`recent=512` consuming the whole budget so `heavy()==0` looked like a capability cliff.

Rule: prefer a **bug** explanation first, and concede a mechanism property only after
the bug is ruled out. Re-running an experiment to settle one of these is expected, not
a detour.

Status: **open** until the cause is named and demonstrated. Closing an entry requires a
measurement, not an argument.

**Register state, 2026-09-03: 14 entries (O1–O14), 13 closed, 1 OPEN (O14).** Counted by
distinct question, not by heading — several entries carry more than one heading because
they were re-opened and re-resolved, and counting headings inflates the total. *(An
earlier status line in this programme said "16 closed"; that was a heading count and is
corrected here.)*

Two of the closures are **supersessions rather than answers** (O1, O8): the phenomenon
they asked about was never established, because the task that produced it was withdrawn as
confounded. Saying a question is unanswerable on this model is not the same as explaining
it, and the distinction is preserved in each entry. The register is kept in full because a
resolved anomaly records how the mechanism actually behaves, and that reasoning outlives
the verdict.

---

## O14 — OPEN 2026-09-03: the score's only demonstrated advantage is instruction retention, and nobody knows how far that generalises

**Observed.** The preregistered triviality control
([`PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md`](PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md),
results in [`FINDINGS_TRIVIALITY_CONTROL.md`](FINDINGS_TRIVIALITY_CONTROL.md)) ran
nine matched arms at two lengths and returned **TRIVIAL** at both. On the
answer-bearing stratum a BM25 retriever over the record delimiters the prompt already
prints is indistinguishable from accumulated-attention significance — 0.8472 against
0.8056 at 4K (1/7 discordant, p = 0.0703) and **exactly equal** at 16K (0.8889 both,
2/2, p = 1). The trivial-recovery fraction against a granularity-matched floor is
1.0645 [1.0101, 1.1319] and 1.0000 [0.9259, 1.0769].

The one thing the score does that no trivial arm does is keep the **system prompt**.
Every trivial arm scores 0.0000 on the abstain stratum against the mechanism's 0.8571
(4K) and 0.2143 (16K, 6/0, p = 0.0312), and the cause is measured: `doc_bm25_weights`
writes 0.0 outside the evidence block, so doc-BM25 keeps 5.6% of the 71-token system
prompt and retains the refusal sentence intact in **0 of 200** cases.

**Why it matters.** This is a *reframing*, not a refutation, and the reframing is
narrower than anything the paper currently claims. The mechanism's demonstrated value
on natural language at length is not finding the evidence — it is retaining
load-bearing context that has **no lexical overlap with the query at all**. That is a
real property, it is exactly what a lexical index cannot have, and it is untested
beyond one 71-token system prompt on one benchmark.

**What is unexplained, and why this is open rather than closed.** Three things.

1. **Is instruction retention the whole residual, or just the part this benchmark
   exposes?** The abstain stratum is 28% of two slices that share one item set. A
   single system prompt is a thin basis for a mechanism claim.
2. **Why does the abstain rate collapse from 0.8571 at 4K to 0.2143 at 16K** while
   the answer rate *rises* (0.8056 → 0.8889)? The mechanism keeps the instruction at
   both lengths, so the abstention loss is not an eviction story. Prefer a bug
   explanation first, per this register's own rule.
3. **Directionality did not transfer, and the sign is wrong in both cells.** Causal
   (6,0) against centred (3,3) is −1.4 pp at 4K (1/3, p = 0.625) and −1.4 pp at 16K
   (0/1, p = 1); removing pooling entirely costs +2.1 pp and +4.2 pp, neither
   resolvable. On passkey the same contrast is 1.000 against 0.028. Underpowered is
   the correct verdict at these n — but the operating point is the likelier culprit:
   at keep 0.25 compression costs the mechanism only 5 of 144 answer cases, so there
   is almost nothing for an ablation to move.

**Decisive tests, in cost order.**

* **A tight-budget cell** — arms A and G at keep 0.05 and keep 0.02 on `rag_long4k`,
  ~10 GPU-min each and *faster* than the keep-0.25 arms because the caches are
  smaller. CPU simulation puts doc-BM25 at 0.993 and 0.931 phrase retention there
  while per-position IDF collapses to 0.056 and 0.000. If A collapses and G does not,
  the verdict becomes BEATEN at a budget that matters; if A holds and G collapses,
  the mechanism has an advantage this campaign's operating point could not see.
  **This is the highest-value unrun experiment in the programme.**
* **An instruction-retention cell** — vary the system prompt's length and position and
  measure whether the A-over-G margin tracks it. Converts finding 3 from one number
  into a law or kills it.
* **P4** (`PLAN_LONG_REASONING.md`) — question and evidence sharing no content words.
  Still unrun, and still the only test that turns a TRIVIAL verdict into a statement
  about the score rather than about benchmarks that print their own document
  boundaries.

**One measurement issue found on the way, recorded because it is the kind that hides.**
The 16K reproduction gate came out **99/100** rather than 100/100 against the
committed arm under `transformers 5.15.0` / `torch 2.13.0+cu130`. The single
divergence is a paraphrase (`'is launched in'` → `'launches in'`) scoring
`contract_fulfillment = 0.3333` in both versions, so no hard-endpoint count moved —
but the gate is stricter than the endpoint by design, it fired, and it cost the 16K
cell its optional legacy floor arm. The 4K cell reproduced 200/200.


## O1 — CLOSED 2026-08-27 as SUPERSEDED: the measurement behind it was withdrawn

> **Closed without an answer, and the distinction matters.** O1 asked why a 20-token span
> scored 0.528 where 5-token spans scored 1.000. That figure came from the **word-phrase
> task**, which was later withdrawn as confounded: its filler is also words, so the model
> can answer with filler and be scored wrong for reasons unrelated to the cache —
> recitation accounts for **100% of failures in the uncompressed control** (see O1/O8).
>
> So the phenomenon O1 was asking about was never established. It is closed as
> *superseded*, not as *answered*: nobody has shown that long spans are harder, and on
> this model nobody can — the digit axis hits a **~11-token verbatim recall ceiling with a
> full cache** (A4), and the word axis is confounded. The question is unanswerable here
> rather than resolved.

### The question as originally posed

**Observed.** Stage 2, 128K, B=256, derived reach `grow=2.0`, 20-word payload:
0.83 / 0.42 / 0.33 across depths, mean **0.528**. The same policy scores 1.000 at
W=3-equivalent spans and 0.889 at d=7.

**Why it matters.** Natural-language evidence runs 20–40 tokens. If the mechanism
degrades at 20, its whole external-validity case is in question.

**Candidates.**
1. *Harness bug.* `max_new = max(12, 4W + 6)` = 86 decode tokens for W=20. If the model
   emits any preamble the answer could be cut off and scored wrong while being right.
2. *Scoring bug.* `hit = str(key) in got` is exact substring; a correct phrase with
   different spacing or a newline would score as a failure.
3. *Mechanism.* Longer spans genuinely fragment more — each additional token is another
   chance to lose one, and P2's failure signature was always losing exactly one.

**Blocked on.** The trial record truncated replies to 32 characters, so these failures
could not be classified at all. Fixed (`keep = max(32, len(key) + 24)`) but the cells
predate the fix.

**Decisive test.** Re-run W=20 with the record fix, then classify: a correct prefix that
stops early implicates fragmentation; a complete-but-differently-formatted phrase
implicates the hit test; an empty or truncated-at-86-tokens reply implicates `max_new`.

---

## O2 — the span law is marginal at d=7

**Observed.** `back ≥ d − 1` holds in 5 of 6 cells, but at the threshold itself the
margin narrows with span length: **0.972** (d=4, back=3), **1.000** (d=5, back=4),
**0.778** (d=7, back=6).

**Candidates.** The effective span may be longer than *d* — the leading space token
before the digits, or the `.` after — so the true requirement is `back ≥ d` or `d + 1`
and d=7/back=6 is simply below it. Or the law is approximate and something else
degrades with length.

**Decisive test.** d=7 at back=7 and back=8. If back=7 restores 1.000, the requirement
is `d`, not `d − 1`, and the whole law shifts by one — cheap and it would sharpen the
claim rather than weaken it.

---

## O3 — CLOSED 2026-08-27: answered by the reach ladder, and it cost a published claim

> **Answered.** O3 asked why derived reach stops at 0.889 at d=7 instead of reaching
> 1.000. The reach ladder answers it directly: a **tuned `back=8` scores 1.000** at d=7
> (and `back=7` scores 0.944), so the 0.111 shortfall is not a limit of the task — it is
> derived reach losing to per-span tuning.
>
> That answer cost a published claim. *"Derived reach matches or beats the best hand-tuned
> fixed reach at every span length"* appeared in **four documents** and is false at d=7.
> It survived a checker at 136/136 because both falsifying arms were pinned **at their
> true values** — no number was stale; the false claim was a RELATION between numbers, and
> no pin had ever checked a relation. Relational pins were added as a result.
>
> What survives is the deployment-relevant form: derived reach beats any *single* fixed
> reach held constant across span lengths (1.000 / 1.000 / 0.889 at d = 4/5/7 against
> back=6's 0.972 / 1.000 / 0.778). Per-span oracle tuning wins at d=7 — it is simply not
> available at inference time.

### The question as originally posed

**Observed.** `grow=2.0` beats the best tuned fixed reach at every span length
(1.000 / 1.000 / 0.889 against 0.972 / 1.000 / 0.778) but does not reach 1.000 at d=7.

**Candidates.** Growth stops early because a payload token dips below the median gate;
or `grow_max` truncates propagation; or the same unexplained long-span effect as O2.

**Decisive test.** Dump the grown score tensor for a d=7 failure and check whether every
digit position cleared the gate. This is a CPU-only inspection of saved tensors and
needs no re-run if the trace tool is extended to record them.

---

## O4 — the multiplier is inert at short spans and decisive at long ones

**Observed.** `grow=2.0` vs `grow=5.0`: identical at d=4 and d=5 (p = 1.0000), but
0.889 vs 0.611 at d=7 (exact McNemar **p = 0.0020**).

**Candidate, not yet verified.** A stricter gate excludes weak payload positions, so
growth cannot traverse a long span. Plausible and mechanical — but *asserted*, not
demonstrated.

**Decisive test.** Same tensor dump as O3, at both multipliers, counting how many
payload positions clear each gate.

---

## O5 — B=16 scores 36/36 with only 8 heavy slots

**Observed.** 128K, B=16 → 4 sinks + 4 recent + **8 heavy**. 36/36 at every depth,
7,864× compression.

**Why it matters.** This is the headline number. Eight positions out of 125,826 is
extraordinary, and extraordinary is exactly when to look for a free lunch.

**Candidates.**
1. The **final chunk may never be compressed**, so a late needle survives without the
   mechanism at all. Depth 0.9 being lossless everywhere is consistent with this.
2. The needle sits inside the recent window or the sinks for some depths.
3. The task is trivially separable — the filler contains **no digits at all**, so the
   payload is the only object of its kind and a regex scores 1.000 with no model.

**Decisive test.** Count, for chunk=4096 at 128K, how many compression rounds a needle
at each depth actually survives, and whether the last chunk is compressed before
decoding. If a depth-0.9 needle faces zero rounds, that cell measures nothing and must
be excluded from the headline. Currently under review.

---

## O6 — RESOLVED 2026-08-27: a third are a DECODE pathology, not a retrieval one

> **Answered by the decisive test O6 itself specified.** The decoys are generated from a
> seeded RNG (`random.Random(f"{key}:decoys")`), so for every failure the exact decoy set
> that was in the prompt can be reconstructed and the emitted number checked against it.
>
> Thirty stage-1 failures, classified by **where the emitted number came from**:
>
> | origin of the emitted number | n | share | what it is |
> |---|---:|---:|---|
> | a decoy actually present in the prompt | 12 | 40% | retrieval failure — crowded out |
> | a prefix of the true key | 9 | 30% | retrieval failure — partial span |
> | **not in the prompt at all** | **9** | **30%** | **decode pathology** |
>
> The third category is named. Those nine are **invented digit strings**, and they are not
> arbitrary — they are degenerate runs: `555222`, `77777`, `999999999`, `222222`,
> `12345678`, `112`. The model is not retrieving a wrong number; it is failing to decode a
> number at all and emitting a repetition or a counting pattern.
>
> **So the "other" third is not a missing retrieval category.** It is the decoder
> collapsing under severe cache damage, which is a different failure from eviction picking
> the wrong positions. Two of the nine (`65667` for 85667, `23649` for 42649) are
> near-misses on the true key rather than degenerate runs, so the boundary is not perfectly
> sharp — but seven of nine are unambiguous repetition or counting artefacts.
>
> **Consequence for how failures should be counted.** A third of stage-1 "failures" are
> not evidence about the eviction policy. Any future comparison of policies on this task
> should separate decode pathologies from retrieval errors, or it will attribute decoder
> collapse to whichever policy was running.

### The question as originally posed

**Observed.** Classifying 21 stage-1 failures: 52% fragmented true span, 14% crowded out
by a decoy, **33% "other"** — degenerate output such as `75526` → `555222`.

**Candidates.** Severe cache damage producing repetition loops; or a classifier gap
where a real category is being missed.

**Decisive test.** Read the full replies now that records are no longer truncated, and
either name the category or show it is a decode pathology rather than a retrieval one.

---

## O7 — depth 0.9 is immune, and the explanation has never been measured

**Observed.** Depth 0.9 is lossless in nearly every cell, including where 0.1 and 0.5
collapse — stage 1 at k=16 framed: 0.42 / 0.42 / **1.00**.

**Stated explanation.** A late needle enters in a late chunk and survives few
compression rounds. This is repeated throughout the findings and has **never been
verified by counting rounds**. It is also the same mechanism that would make O5 an
artefact, so the two are linked.

**Decisive test.** Instrument the compression loop to report, per depth, how many rounds
the needle's positions actually pass through. Cheap, and it either grounds a claim the
programme leans on constantly or overturns it.

---

## O8 — stage 2 has no capability gate, and the depth pattern says it needs one

**Observed.** At W=12 (128K, B=256, grow=2.0) accuracy by depth is
**1.00 / 0.92 / 0.67** — depth 0.9 is the *worst*. That inverts every other cell in
this programme, where depth 0.9 is easiest because it faces 3 compression rounds
against 27 at depth 0.1 (resolved as O7).

**Why an inversion is alarming.** The depth ordering is a property of the compression
schedule. If it reverses, the thing being measured is probably not compression.

**The failures are not retrieval failures.** At W=12, depth 0.9:

| emitted | reading |
|---|---|
| `"bottles and packets" and "lantern" and "signal"` | paraphrasing, not reciting |
| `"The pass phrase is mountain hammer station bridge` | restating the frame with wrong words |
| `"The grass is green. The sky is blue.` | **reciting the filler** |

The model produces narrative rather than verbatim recall. A 12- or 20-word phrase is a
recitation task; a 5-digit key is not.

**The design gap is mine.** Stage 3 was gated — 2-hop and 3-hop were measured
uncompressed first, scoring 0.833 and 0.722, before any compressed number was believed.
**Stage 2 was never gated.** So the W=12 and W=20 numbers have no uncompressed
reference, and the drop from 1.000 at 5 tokens to 0.528 at 20 could be the mechanism
fragmenting a longer span, the model failing to recite a longer phrase, or both.

**Decisive test.** Run W=3, W=12 and W=20 with a FULL CACHE (`--budgets 0`) at a length
where it fits. If uncompressed accuracy also falls with span length, that fraction of
the drop is the model and not the mechanism, and O1's numbers must be restated against
that reference rather than against 1.000.

**This also puts O1 in question**, since O1 concluded "mechanism, not bug" from the
fact that failures were legible truncations. That reasoning ruled out a *harness* bug.
It did not rule out a *model capability* ceiling, which this entry is about.

---

## O9 — every long-context result uses ONE chunk size, and the schedule depends on it

**Observed.** 124 of 132 committed result files use `chunk=4096`. At 128K that is 31
chunks and **30 compressions**. At `chunk=1024` it would be 123 chunks and **122
compressions — four times the eviction pressure** on the same evidence.

**Why it matters.** O7 established that survival is governed by how many compression
rounds a position faces: 27 at depth 0.1 against 3 at depth 0.9 produces a 9× difference
in exposure. Chunk size is the knob that sets that number globally. If the headline
(36/36 at B=16) holds only at 30 rounds and collapses at 122, the result is a property
of the *schedule* rather than of the mechanism, and the compression figure is not
transferable to a deployment that prefills differently.

**This is the most consequential unasked question in the register.** It is also cheap.

**Decisive test.** 128K, B=16, backward reach, at `chunk` ∈ {1024, 2048, 4096}. If
accuracy holds at 1024, the mechanism is robust to eviction pressure and the claim
strengthens. If it falls, the headline needs the chunk size attached to it.

---

## O10 — RESOLVED 2026-08-25 by A2: the sinks are load-bearing

> **Answered.** The ablation ran. `sinks=0` scores **0.000** while holding *more*
> score-chosen slots (12 against 8); `sinks=2` scores 1.000. So the sinks are not
> optional overhead, two suffice, and the headline reads "four sinks **plus** eight
> score-chosen positions". See the A2 entry below. The original text is kept for the
> reasoning that motivated the ablation.

### The question as originally posed

**Observed.** Every cell uses `sinks=4`, hardcoded rather than swept. At **B=16 that is
25% of the entire budget**, leaving 8 heavy slots after the recent window.

**Why it matters.** Cornerstone 1 measured that R2 places 46.3% of its attention mass on
the first four positions, which is why sinks are pinned. But the headline result is
"eight significance-chosen positions retrieve a needle from 125,826 tokens". If the four
sinks are doing part of that work, the honest count is not eight.

**Decisive test.** 128K, B=16, backward reach, at `sinks` ∈ {0, 2, 4, 8}. Sinks=0 with
accuracy preserved would strengthen the claim considerably — the budget would be entirely
significance-chosen. A collapse at sinks=0 bounds what the score alone achieves.

---

## O11 — RESOLVED 2026-08-27: it is SIMILARITY TO THE QUERY

> **Answered, and it changes the mechanism's description.** A `--score-query` control
> substituted a different question for the scoring pass while the model was still asked
> the real one. At 8K/B=16/back=6, chunk 1024 (every depth compressed):
>
> | scoring signal | mean |
> |---|---:|
> | the real question | **1.0000** |
> | wrong question (weather) | 0.0556 |
> | wrong question (sky) | 0.0000 |
> | no query pass | 0.0000 |
>
> The score does not measure intrinsic importance. It measures what the question reaches
> for. Replicated on two independent wrong queries. Consequences for the name, for
> multi-turn (B4), and for separating the passkey path from the language path are in
> FINDINGS_P2. The original framing is kept below.

### The question as originally posed

**Observed.** P2.2 measured that the score retains query-shaped framing at 54% and the
payload it introduces at 35%, and §5.1 showed query-shaped distractors cost 0.361 where
neutral ones cost nothing. Both say the score responds strongly to *lexical resemblance
to the question*.

**Why it matters conceptually.** If the significance score is largely a query-similarity
index rather than an attention-importance measure, the contribution should be framed
differently — closer to a retrieval index computed in attention space than to
"accumulated importance". That changes the related-work positioning and what the method
should be compared against.

**It also predicts a failure mode not yet tested.** In real question-answering the
question and the answer-bearing span often share few words. If the score depends on
lexical overlap, it should degrade sharply there — which is pre-registered prediction P4
in `PLAN_LONG_REASONING.md` and remains unrun.

**Decisive test.** Rewrite the needle so it shares no vocabulary with the question while
still containing the answer, and compare against the current framing at matched budget.

---

## O12 — RESOLVED 2026-08-29: all three costs measured, and the docstring was wrong

> **Two of the three parts are measured.**
>
> **Decode-time scoring cost — answered from committed data.** `measure_decode`'s
> `evict` against `evict_noscore` arms are the same policy with the score off:
> **+20.2% / +20.3% / +22.9% / +22.6% / +22.5%** at 4K / 16K / 32K / 64K / 128K.
> **Scoring costs ~22% of decode throughput, flat in context length.** The same data
> shows a crossover: compression is **0.83× (slower)** at 4K, 1.78× at 16K, **5.41×** at
> 32K.
>
> **Per-compression selection cost — measured without a model** (`measure_select.py`,
> because the machine had 1.04 GB free and could not load one). Directional pooling is
> **free** against centred (0.95–1.02× at four lengths); derived reach costs **1.12–2.14×**,
> shrinking with length; pooling itself is **8.5×** the no-pooling cost at 32K and is the
> dominant term.
>
> **The query-aware prefill pass — measured 2026-08-29, CPU only, GPU untouched.** The
> `score_with_query` docstring predicted the pass would cost `len(query)/chunk` of a chunk
> and be "negligible". That prediction assumes forward cost is proportional to token count.
> **It is not.** Fitting t(T) = a + bT over T = 8…512 gives **164.82 ms + 2.1138 ms/token**
> (R² = 0.9972), so **82.97%** of a 16-token forward is the fixed intercept — the cost of
> streaming every weight once, paid to move sixteen tokens. Measured end to end the pass is
> **15.49% / 8.46% / 5.54%** of prefill at chunk 512 / 1024 / 2048, which is **7.9× / 8.0× /
> 11.3×** the token-ratio prediction. A second term the docstring never mentioned:
> `_restore` copies every retained key and value on every layer, **12.4 → 99.5 ms** across
> cache 512 → 4096, linear in bytes at ~18 GB/s and independent of question length.
>
> **And the direction of the headline is the opposite of what "cost" implies.** Against
> vanilla chunked prefill with no mechanism at all, the complete query-aware mechanism runs
> at **0.84–0.96×** the wall-clock while holding 4× less cache: every chunk after the first
> attends to a capped cache instead of a growing one, and that saving exceeds what the query
> pass costs. **At prefill, compression pays for itself.**
>
> One more null worth recording: score accumulation costs **nothing measurable at prefill**
> (1.073× and 1.011×, with an independent run at 0.983× and 0.994× — the two straddle
> unity). That does not contradict the ~22% decode figure, it explains it: at decode the
> forward is one token, so accumulating over the whole cache is comparable work; at prefill
> the forward is hundreds of tokens and the accumulation covers a 16-row window.
>
> **A defect in this measurement, caught by its own implausibility.** The first version of
> the `_restore` timing built a stub cache at length n and restored it to n, so the slice
> was the whole tensor, `.contiguous()` returned `self`, and it timed a no-op — reading
> **0.15 ms for a 939 MB copy**, i.e. 6 TB/s. The tell was the implied bandwidth. Fixed by
> growing the stub by q first so the truncation is real. Same class as `grow_spans` and the
> B1 floor: a measurement that does not reproduce its caller's preconditions measures the
> wrong function.
>
> **O12 is closed.** The docstring is corrected in place, and a fusion optimisation is
> registered in `CORNERSTONE_LEDGER.md` §8 with a bit-equality kill test.
>
> One caveat carried forward: the decode rows record **no temperature** — they predate the
> thermal instrumentation, so they are interleaved but not thermally attributed, on a host
> with a 2.17× thermal band.

### The question as originally posed

**Observed.** Every number in this programme is a benefit: accuracy retained, memory
saved. `grow_spans` iterates up to `grow_max` times per compression, and query-aware
scoring runs an extra forward pass per chunk boundary.

**Why it matters.** A practicability claim needs both sides. Compression that costs 30%
of decode throughput is a different proposition from one that costs 2%, and neither has
been measured.

**Decisive test.** Wall-clock per compression round for centred pooling, fixed backward
reach, and derived reach at matched budget; plus prefill time with and without
query-aware scoring. The repository already has `measure_decode.py` for the decode side.

---

## Closing an entry

State the cause, cite the measurement that demonstrates it, and move the entry to a
**Resolved** section with the date. Do not delete it — a resolved anomaly is a record
of how the mechanism actually behaves, and the reasoning is worth more than the verdict.

---

# Resolved

## O5 — RESOLVED 2026-08-23: the exemption is real, the headline is not a free lunch

**Cause found, in code.** `scripts/kv/chunked_prefill.py:120`:

```python
if compress_fn is not None and ci < n_chunks - 1:
```

The **final chunk is never compressed**. That is a genuine exemption and it was worth
suspecting. But it does not reach the needles.

At 128K the prompt is 125,826 real tokens in chunk=4096 → **31 chunks, 30
compressions**, with the exempt chunk covering positions [122,880, 125,826). Needle
positions, taken from the committed span trace rather than estimated:

| depth | needle span | chunk | in the exempt chunk? | compressions faced |
|---:|---|---:|---|---:|
| 0.1 | [12,606, 12,632) | 3 | no | **27** |
| 0.5 | [62,910, 62,936) | 15 | no | **15** |
| 0.9 | [113,214, 113,240) | 27 | no | **3** |

No needle lands in the exempt chunk at any depth, so no cell survives without the
mechanism. The exempt region holds the question, which is where it belongs.

**And the headline holds on the hardest depth alone.** 128K/B=16 causal scores 1.00 at
depth 0.1 — 12/12 needles that each survived 27 compression rounds with 8 heavy slots.
The claim does not depend on averaging in the easy depth.

## O7 — RESOLVED 2026-08-23: the depth asymmetry, finally measured

The findings repeated throughout that "a late needle enters in a late chunk and
survives few compression rounds". It was never measured. It is now, in the table
above: **27 rounds at depth 0.1 against 3 at depth 0.9 — a 9× difference in exposure.**

**A methodological consequence that applies to every cell in this programme.** The
three depths are *not* equally difficult, so the pooled three-depth mean overweights
the easy one. A cell reported as 0.972 may be 12/12 at depth 0.9 and 11/12 at depth
0.1, and those are not the same result. Depth-wise numbers are reported throughout and
should be read in preference to the mean wherever the two disagree — and the depth-0.1
column is the one that carries the claim.

## O4 — RESOLVED 2026-08-24: the asserted explanation is WRONG

`FINDINGS_P2.md` explained the multiplier's effect at d=7 as "a stricter gate excludes
weak payload positions, so growth cannot traverse a long span". That was asserted from
the design and never demonstrated. `scripts/kv/probe_grow_gate.py` demonstrates it is
**false**, on CPU, with no GPU needed — `grow_spans` is a pure function of a score
tensor, so the question is fully determined by feeding it the score shape P2.2 measured.

Payload positions carrying the head's score after growth, head 1.0 / payload 0.15 /
filler 0.01:

| span | mult 2.0 | mult 3.0 | mult 5.0 | mult 8.0 |
|---:|---:|---:|---:|---:|
| 4 | 100% | 100% | 100% | 100% |
| 7 | 100% | 100% | 100% | 100% |
| 20 | 100% | 100% | 100% | 100% |

**No effect at any span, at any multiplier.** The gate sits at median × mult, the median
*is* the filler level, and a payload at 15× filler clears even mult=8. Gate coverage
cannot explain a difference that appears only at d=7.

**And the gate is a cliff, not a gradient**, which sharpens what the real explanation
must look like. Sweeping the payload level at span 7:

| payload / filler | mult 2.0 | mult 3.0 | mult 5.0 | mult 8.0 |
|---:|---:|---:|---:|---:|
| 20× | 7/7 | 7/7 | 7/7 | 7/7 |
| 5× | 7/7 | 7/7 | 7/7 | **0/7** |
| 3× | 7/7 | **0/7** | 0/7 | 0/7 |
| 2× | **0/7** | 0/7 | 0/7 | 0/7 |

Coverage is all-or-nothing per stream: full until the payload drops below mult × filler,
then zero. So if gate coverage *is* the mechanism, the real cache must contain streams
whose payload sits between 2× and 5× the stream median, and raising the multiplier flips
those streams from full coverage to none. Survival is measured over 224 streams, so a
fraction flipping would produce exactly the observed 0.889 → 0.611.

**That is a testable prediction about the real score distribution**, and a much sharper
one than the claim it replaces. It needs a dump of real scores at d=7, which is O3's
decisive test — the two are now the same experiment.

## Stage 3 capability gate — PASSED 2026-08-24

The reasoning plan requires the model's uncompressed ceiling to be established before
any compression number on a task is meaningful. Multi-hop chains, 8K, no compression:

| chain | accuracy | by depth |
|---|---:|---|
| 2 hops | **0.833** | 0.67 / 1.00 / 0.83 |
| 3 hops | **0.722** | 0.75 / 0.58 / 0.83 |

Both clear a floor comfortably. This model *can* chain hops, unlike its performance on
`Rineya_Long_Context_Demo_v1` (0.2031 against a 0.2500 constant-abstention baseline),
so stage 3's compression numbers will measure the mechanism rather than the model's
incapacity. Stage 3 is viable and the hop ladder can proceed.

## O8 — ~~PARTLY RESOLVED~~ **WITHDRAWN 2026-08-24**: the gate below is confounded

> **Read the O1/O8 entry further down before using anything in this section.** The gate
> used the word-phrase task, whose filler is also words, so the model can answer with
> filler and be scored wrong for reasons unrelated to the cache. Filler recitation
> accounts for **100% of failures in the uncompressed control**. The table below is
> retained as the record of what was measured, not as a result. The digit-span axis was
> then tried instead and hit a model ceiling — see A4.

### The withdrawn gate, for the record: "the model does not degrade with span length"

Full cache, no compression, 16K:

| span | d0.1 | d0.5 | d0.9 | accuracy |
|---:|---:|---:|---:|---:|
| 3 | 0.83 | 0.92 | 0.92 | 0.889 |
| 12 | 0.67 | 0.92 | 0.83 | 0.806 |
| 20 | 1.00 | 0.92 | 0.92 | **0.944** |

Roughly **flat at ~0.88**, not monotonic, and W=20 is the *highest* of the three. At
n=12 needles that spread is noise. There is no recitation ceiling falling with span
length, and the suggestion that one might explain the stage 2 curve is **withdrawn**.

**What changes.** At W=12 the compressed arm scores 0.861 against an uncompressed
0.806 — no measurable compression cost — so "stage 2 declines monotonically with span
length" is **wrong at its middle point and retracted**.

**What does not.** At W=20 the compressed arm scores 0.528 against an uncompressed
0.944. That is a real cost of −0.417, so **O1 stands** rather than dissolving.

**The W=20 comparison is not matched and must not be quoted as if it were.** The gate
is 16K uncompressed; the compressed arm is 128K, facing eight times the context and 27
compression rounds. The gap mixes span length with context length. The matched cell —
compressed W=20 at 16K — is queued and settles it: near 0.944 means the collapse is
context-driven and the span account is wrong; near 0.528 means span length is the cause.

**And the depth inversion is not a compression signal.** At W=12 *uncompressed* the
depths run 0.67 / 0.92 / 0.83 with no compression at all, so the ordering that triggered
O8 is a property of the word-phrase task. The alarm was right and its reason was wrong:
the gate was necessary, but depth ordering carries no diagnostic information here,
unlike on digit needles where it tracks compression rounds (O7).

## Halo 16K full cache — OOM, and the cap was mine

Recorded rather than hidden: the first Halo reference arm lost all 36 trials to OOM at
a 3,500 MiB cap, reaching 3,147.8 MiB. My core's 16K full cache peaks at 3,013 MiB and
Halo's weights are 233 MiB larger, so ~3,246 MiB was expected and 3,500 left only 11%
headroom for transient allocations. Re-queued at 4,400.

The harness behaved correctly throughout: trials excluded not scored, `ooms=12` per
cell, `acc=None`, and the warning that an empty cell is not a failure. That guard exists
because a 64K full-cache reference once became a "result" this way.

**A properly sized 32K Halo reference would need ~6,164 MiB against a demonstrated-safe
allocated ceiling of 4,972.9 MiB on this card.** If that arm also OOMs, the honest
statement is not that compression is cheaper at 32K but that on this hardware a
full-cache Halo run at 32K does not exist to be compared against — which is the
practicability claim arriving as data.

## O2 — RESOLVED 2026-08-24: the law is not `back ≥ d − 1`, and it is retracted

Extending d=7 past the threshold settles it:

| digits | back | accuracy |
|---:|---:|---:|
| 7 | 4 | 0.000 |
| 7 | 6 (= d−1) | 0.778 |
| 7 | 7 (= d) | 0.944 |
| 7 | **8** (= d+1) | **1.000** |

Monotonic, reaching 1.000 only at **back = d + 1**. So `back ≥ d − 1` was fitted on
d=4 and d=5 and does not extrapolate — **retracted**.

**And the requirement grows faster than the span.** Full accuracy needs back=4 at d=5
but back=8 at d=7: two extra tokens nearly doubled the required reach. Two points do
not make a curve, which is why d=9 at back=8 and back=10 is queued.

**The `d+1` shape has a natural reading that does not survive scrutiny.** The payload's
last token must reach the query-shaped framing, which sits one token before the digits
begin (` is`, then the space, then *d* digits) — that predicts back ≥ d+1 everywhere.
But back=4 already gives 1.000 at d=5, where the same reading demands 6. So the
mechanism is not simply "reach the label", and no replacement law is stated here.

**What survives is the qualitative claim**, which is what the mechanism rests on: the
required backward reach *tracks the evidence span* and is not a constant. d=7 at back=4
scores 0.000 where d=5 at back=4 scores 1.000. That is why deriving the reach beats
tuning it, and it does not depend on the exact form of the law.

## O1 / O8 — RESOLVED 2026-08-24: the word-phrase task is confounded, and the flaw is mine

**The span-length results of stage 2 do not measure span length.** They are contaminated
by a task-design error, and the evidence is direct.

Failures where the model recites the **filler** instead of the payload:

| cell | depth | failures | filler-recitations |
|---|---:|---:|---:|
| W=3 compressed | 0.9 | 9 | **9 (100%)** |
| W=20 compressed | 0.9 | 8 | 6 (75%) |
| W=12 compressed | 0.5 | 1 | 1 (100%) |
| **W=3 uncompressed** | all | 4 | **4 (100%)** |
| **W=20 uncompressed** | 0.5, 0.9 | 2 | **2 (100%)** |

Emitted text: `"The grass is green. The sky is blue. The su…"`, `"green. blue. yellow."`

**It occurs in the uncompressed cells too**, so it is a property of the task rather than
of compression.

**The cause.** Passkey's filler contains **no digits at all**, so a filler recitation can
never be mistaken for an answer — the answer is the only object of its kind in the
context. Building the word-phrase task for stage 2, I made the payload out of *words*,
and the filler is also words (`grass`, `green`, `sky`, `blue`). That removed the property
that made the task scoreable, and the model answers with filler at depths where filler
dominates the recent window.

**Consequences, stated plainly.**

1. The span-length curve **0.750 (W=3) / 0.861 (W=12) / 0.528 (W=20)** is **withdrawn as
   a span-length measurement**. It is non-monotonic, and the shortest span scores worse
   than the middle one, because confusability rather than span length drives it.
2. **O1 is resolved**: the W=20 collapse is substantially this confound, not evidence
   that the mechanism fails on long spans.
3. **O8 is fully resolved**: the depth inversion that triggered it is filler dominating
   the recent window at late depths, which has nothing to do with the compression
   schedule.
4. **§4.5's span law is unaffected** and stands. It is measured on *digit* spans
   (d = 4, 5, 7, 9), where the filler contains no digits and the confound cannot occur.
   The `d=7 back=4 → 0.000` cell against `d=5 back=4 → 1.000` remains clean evidence
   that required reach tracks the span.

**What a corrected stage 2 needs.** A payload drawn from a vocabulary the filler cannot
supply — rare tokens, proper nouns, or a symbol class absent from the haystack —
restoring the "only object of its kind" property that makes an answer checkable. Until
that runs, this programme has **no valid measurement of span length beyond 9 tokens.**

## O9 — RESOLVED 2026-08-24 **with one qualification added 2026-08-25**: the headline survives 4x the eviction pressure

> **Qualification.** This entry compares chunk 4096 against 1024 only, and the honest
> reading of those two cells is "eviction *pressure* does not degrade the headline". It
> is **not** "the chunk axis is safe": chunk **2048** — between the two rows below —
> scores 0.806/0.833 at two seeds. See O13, which remains open.

| chunk | compressions at 128K | d0.1 | d0.5 | d0.9 | accuracy | peak |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 30 | 1.00 | 1.00 | 1.00 | **36/36** | 1,501 MiB |
| 1024 | **122** | 1.00 | 1.00 | 0.92 | 35/36 | **1,057 MiB** |

Exact McNemar on the discordant pair: **p = 1.0000, indistinguishable**. The headline
survives a 4× increase in eviction pressure.

> **CORRECTED the same day, when chunk=2048 landed.** That cell scores **0.806** —
> 1.00 / **0.42** / 1.00 — which is worse than *both* 4096 and 1024. **The chunk axis is
> non-monotonic**, so the endpoints agreeing does not make it flat, and the claim
> "compression figures do not need a chunk size attached" was written one cell too early.
> It is withdrawn pending O13. What survives is narrower and still useful: the mechanism
> does not degrade with eviction pressure per se, since the highest-pressure setting
> (122 rounds) is the second-best of the three.

**Two further observations.**

*Peak memory falls 30%* at the smaller chunk (1,501 → 1,057 MiB), because the in-flight
prefill chunk is the dominant variable term rather than the compressed cache. A
deployment can chunk smaller for less memory and keep the accuracy — a lever nobody had
checked.

*The single failure lands where O7 predicts.* At chunk 1024 the depth-0.9 needle sits in
chunk 110 of 123 and therefore faces **12** compressions instead of 3 — a 4× increase.
The one lost needle appears at exactly the depth that gained the most rounds, so the
failure is consistent with the measured mechanism rather than arbitrary.

## O13 — the chunk axis is non-monotonic, and the round-count account does not explain it

**Observed.** 128K, B=16, backward reach, identical in every other respect:

| chunk | compressions | rounds faced at depth 0.5 | d0.1 | d0.5 | d0.9 | accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 30 | 15 | 1.00 | 1.00 | 1.00 | **1.000** |
| 2048 | 61 | 31 | 1.00 | **0.42** | 1.00 | 0.806 |
| 1024 | 122 | 61 | 1.00 | 1.00 | 0.92 | 0.972 |

Depth 0.5 faces 15 / 31 / 61 rounds and scores 1.00 / 0.42 / 1.00. **More rounds is not
monotonically worse**, so O7's round-count account does not explain this on its own.

**Ruled out.** The needle does not straddle a chunk boundary at any of the three sizes —
it sits inside chunk 15, 30 and 61 respectively.

**The failure signature is precise.** Seven of twelve failures at depth 0.5 are the
4-of-5 truncation: `90190` → `9019`, `66445` → `6644`, `70176` → `7017`. The final token
of the key span is lost, which is exactly the P2 fragmentation mode.

**Leading candidate, untested.** Pooling operates on the *compressed cache*, not on the
text. After a compression the cache holds B entries drawn from scattered original
positions, so a `back=6` window spans six *cache entries*, not six tokens. The window's
reach in text therefore depends on how compacted the cache is when the needle's
protecting compression occurs — which is a function of chunk size and of where in the
chunk the needle falls. If so, the effective reach is not a fixed quantity at all and
§4.4's "backward reach" needs restating in cache coordinates.

**Decisive test.** Log, at each compression, the original-position gap between adjacent
cache entries in the pooling window. If the gap is ~1 at chunk 4096 and larger at 2048,
the account holds. This is an instrumentation change to the existing trace tool and needs
no new sweep.

**Why this matters more than one cell.** If confirmed, it affects every reach result in
§4.4 and §4.5, because those were all measured at chunk 4096 where the cache is least
compacted at the critical moment.

## O13 — CONFIRMED 2026-08-25: the chunk-2048 anomaly reproduces at a fresh seed

| cell | d0.1 | d0.5 | d0.9 | accuracy |
|---|---:|---:|---:|---:|
| chunk 2048, seed A | 1.00 | 0.42 | 1.00 | 0.806 |
| chunk 2048, **seed B** | 1.00 | **0.50** | 1.00 | **0.833** |
| chunk 4096, seed A | 1.00 | 1.00 | 1.00 | 1.000 |
| chunk 4096, **seed B** | 1.00 | 1.00 | 1.00 | **1.000** |

**It is real, not a bad run.** Two independent seeds put chunk 2048 at 0.42/0.50 on
depth 0.5 while chunk 4096 is perfect at both. The chunk axis is genuinely
non-monotonic and the cause remains open — the cache-coordinate hypothesis is the
leading candidate and its instrumentation is written (`trace_cache_gaps.py`).

## A2 — RESOLVED 2026-08-25: the sinks are NOT optional, and the headline was wrong

| sinks | heavy slots | d0.1 | d0.5 | d0.9 | accuracy |
|---:|---:|---:|---:|---:|---:|
| **0** | **12** | 0.00 | 0.00 | 0.00 | **0.000** |
| 2 | 10 | 1.00 | 1.00 | 1.00 | **1.000** |
| 4 | 8 | 1.00 | 1.00 | 1.00 | **1.000** |

**With no sinks the mechanism scores zero — while having MORE score-chosen slots (12
against 8).** So attention sinks are load-bearing, not budget overhead, which is
consistent with cornerstone 1's measurement that this model places 46.3% of its
attention mass on the first four positions. Remove them and the attention distribution
is broken; nothing the score does can recover it.

**The headline must be restated.** "Eight significance-chosen positions retrieve a
needle from 125,826 tokens" is wrong. It is **four sinks plus eight score-chosen
positions**, and the sinks are required. What the ablation *does* establish is that
**two sinks suffice** — sinks=2 with 10 heavy slots is also perfect — so the
requirement is "some sinks", not "four".

## A4 — KILLED BY ITS OWN GATE 2026-08-25: the model cannot recite past ~11 digits

| span (digits) | uncompressed 16K | compressed 128K |
|---:|---:|---:|
| 12 | **0.000** | 0.000 |
| 20 | **0.000** | 0.000 |
| 30 | **0.000** | 0.000 |

The **uncompressed** column is the finding. With a full cache and no eviction at all,
the model scores zero at 12 digits. It emits **11 correct digits then diverges** — the
same ceiling at d=20 and d=30 (`11/20`, `11/30`).

This is the kill test stated in advance: *"accuracy falls with span length in the gate
as well → the model cannot recite long payloads and the axis is unmeasurable on this
model, which is itself a finding worth stating."*

**Consequences.**

1. **This programme cannot measure evidence spans beyond ~9–11 tokens on this model**,
   and no experimental design fixes that — it is a property of the model, not of the
   task or the cache. The word-phrase attempt failed for a different reason
   (confounded filler); this attempt fails for a reason no design can route around.
2. The span law of §4.5 is therefore measured over d = 4, 5, 7, 9 and **cannot be
   extended on this hardware**. Testing 20–40 token evidence needs a model with a
   larger verbatim-recall capacity.
3. The gate earned its place. Without it, 0.000 at d=20 compressed would have been
   reported as a catastrophic compression failure. It is a model ceiling.

## O13 — the cache-coordinate hypothesis is REFUTED (2026-08-25); the cause stays open

`trace_cache_gaps.py` carried original position identity through every gather and
measured what a `back=6` window actually spans at the compression where the needle is
present:

| chunk | round | stored before | entries in needle | reach in TOKENS | max gap | contiguous |
|---:|---:|---:|---:|---:|---:|---|
| 4096 | 15 | 4,112 | 26 | **6** | 1 | **yes** |
| 2048 | 30 | 2,064 | 26 | **6** | 1 | **yes** |

`back=6` spans exactly six tokens at both chunk sizes, with adjacent cache entries one
original position apart. **The window is not fragmented, so §4.4 and §4.5 do not need
restating in cache coordinates.** The leading explanation for O13 is dead.

**A limitation of this probe, stated so the record is not over-read.** It inspects
**layer 0, head 0 only**, and records a row only while that stream holds needle entries.
Both runs show a single round, which looks like immediate eviction — but P2.1 measured
the needle surviving in 79–90% of the 224 (layer, KV-head) streams, so one stream losing
it is expected and says nothing about retrieval. Aggregating across streams would be
needed to read survival from this tool, and the single-round observation should not be
cited as evidence of anything.

**O13 therefore remains open**: chunk 2048 reproducibly scores 0.42/0.50 at depth 0.5
where 4096 scores 1.000, the failures are the 4-of-5 truncation signature, the data and
needles are identical, chunked prefill is bit-identical to single-shot at every chunk
size, and the window is contiguous. No candidate explanation currently survives.

## O13 — a NEW hypothesis, pre-registered 2026-08-26: marginal capacity, not chunk schedule

O13 has had no surviving explanation since the cache-coordinate account was refuted.
Four candidates were eliminated: differing data, a chunked-prefill bug, boundary
straddling, and window fragmentation. This is a fifth, registered **before** the test so
it cannot be fitted afterwards.

### What the failure records actually say

| cell | depth 0.1 | depth 0.5 | depth 0.9 |
|---|---|---|---|
| chunk 2048 seed A | 12/12 | **5/12** | 12/12 |
| chunk 2048 seed B | 12/12 | **6/12** | 12/12 |
| chunk 1024 seed A | 12/12 | 12/12 | 11/12 |

**All 14 failures across every cell emit exactly the first four digits of a five-digit
key, then diverge.** Not a wrong answer, not a refusal — the span's *final token* is
missing, every time.

### An intermediate hypothesis, and its refutation

Seed A's depth-0.5 failures are trials **5–11: a contiguous tail**. That pattern points
hard at state leaking across trials — `reset_scores()` failing, later trials degrading.
It would have been a serious defect.

**Seed B refutes it.** Its failures are scattered (trials 0, 2, 3, 5, 6, 8). The
contiguous tail was coincidence in a single draw, and a one-seed reading would have
reported a state-leakage bug that does not exist. Recorded because the near-miss is the
point: the second seed existed only because O13 demanded a replication.

### The hypothesis

With a **causal** window the last digit of the key inherits `max` over digits 1–5 plus
the framing tokens — the same maximum the 4th digit inherits. Their pooled scores
**tie**, so the last digit is the *marginal* position in the top-k, and which member of a
tied pair survives depends on the `S / C**e` ranking. The compression-round count changes
`C`. So the chunk size does not damage retrieval directly; it perturbs a tie-break at the
span's trailing edge, and only where the margin is already thin.

This is the same shape as the competition result, which is already established: sixteen
query-shaped decoys took B=64 from 1.000 to 0.611 and **budget alone restored it**
(B=512 → 1.000), diagnosed as a capacity cost rather than a scoring failure. Depth 0.5
being the only affected depth fits — it is the position with the most competing context
on both sides.

### Why it has never been tested

**Every chunk-2048 cell ever run is at B=16.** The budget axis and the chunk axis have
never been crossed. The whole anomaly lives at a single budget.

### Pre-registered predictions and kill conditions

1. **B=32 at chunk 2048 scores 1.000.** If capacity at the margin is the cause, doubling
   the budget removes the margin.
   *Kill:* if B=32 at chunk 2048 still falls below chunk 4096, **capacity is not the
   explanation and this hypothesis is dead.** Say so and return O13 to unexplained.
2. **back=8 at chunk 2048, B=16, improves on back=6.** A longer reach lifts more of the
   span above the competition, so fewer trailing tokens sit at the margin.
   *Kill:* no improvement means the tie-break account of the trailing edge is wrong, even
   if prediction 1 holds for some other reason.
3. **Depth 0.1 and 0.9 stay at 1.000 in both arms.** If either falls, the arms are not
   measuring what this hypothesis describes.

A secondary consequence if it holds: the chunk axis is not a free parameter *at thin
budgets*, and the deployment rule becomes "size the budget to the span with headroom",
not "avoid chunk 2048". The current documents say only that the axis is non-monotonic and
unexplained.


---

## O13 — RESOLVED 2026-08-26: marginal capacity, confirmed; the mechanism story refuted

Both pre-registered predictions were tested. **The hypothesis survived; the reasoning
behind it did not**, and that split is the whole value of having registered them
separately.

| arm | d0.1 | d0.5 | d0.9 | overall |
|---|---:|---:|---:|---:|
| B=16 chunk 4096 (baseline) | 1.000 | 1.000 | 1.000 | 1.000 |
| B=16 chunk 2048 (the anomaly) | 1.000 | 0.417 | 1.000 | **0.806** |
| **B=32 chunk 2048** | 1.000 | **1.000** | 1.000 | **1.000** |
| B=32 chunk 4096 (control) | 1.000 | 1.000 | 1.000 | 1.000 |
| B=16 back=8 chunk 2048 | 1.000 | 0.000 | 0.667 | **0.556** |

**Prediction 1 confirmed.** Doubling the budget removes the chunk-2048 deficit entirely,
and the matched B=32 / chunk-4096 control also scores 1.000 — so the budget is not merely
lifting everything, and the effect is specific. **The chunk axis is not a property of the
schedule. It is a marginal-capacity effect that exists only where the budget is thin.**
O13 is closed after five hypotheses, four of them dead.

**Prediction 2 refuted, and it cost the explanation rather than the finding.** `back=8`
was predicted to help, on the account that a longer reach lifts the span's trailing edge
above the competition. It made things much worse — 0.556 against back=6's 0.806, depth 0.5
falling to 0.000. The registered kill condition said that refutes the trailing-edge
tie-break story, and it does.

**What survives both results.** A longer backward reach **inflates the candidate set** —
more positions inherit the span's high score — and at a thin budget that is strictly
harmful, because more high-scoring positions compete for the same eight slots. Backward
reach therefore **trades against budget** rather than being monotonically good. This is
the same shape as the competition result (§5.1 of the paper): sixteen query-shaped decoys
took B=64 from 1.000 to 0.611 and budget alone restored it.

**Deployment rule.** Size the budget to the span with headroom. Do not tune the chunk
schedule, and do not assume more reach is safer.

*Recorded 2026-08-29: this resolution was established on 2026-08-26 and written into
`FINDINGS_P2.md` the same day, but neither this register nor §5.3 of the paper was
updated, so both read "no candidate explanation currently survives" for three days. The
finding was never wrong; the documents were. Both are now corrected, and this note stays
because a resolution that only lives in one file is one refactor away from being lost.*
