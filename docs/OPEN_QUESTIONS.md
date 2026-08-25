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

---

## O1 — a 20-token span scores 0.528 where 5-token spans score 1.000

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

## O3 — derived reach leaves 0.889 at d=7, not 1.000

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

## O6 — a third of stage-1 failures are neither truncation nor substitution

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

## O10 — sinks are fixed at 4 and were never ablated with backward reach

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

## O11 — is the score measuring importance, or similarity to the query?

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

## O12 — the mechanism's own runtime cost is unmeasured

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

## O8 — PARTLY RESOLVED 2026-08-24: the model does not degrade with span length

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
