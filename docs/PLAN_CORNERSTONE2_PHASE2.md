# Cornerstone 2, phase 2: validity, capability, practicability

**Written 2026-08-24.** A programme for the significance matrix after the first
cornerstone-2 results. Three tracks run in parallel because they answer different
questions and gate different claims.

> **Standing principle.** This mechanism is early. Every entry below states the
> measured evidence it follows from, the experiment, **the observation that would kill
> it**, and its cost. An entry without a kill test is a hope, not a plan.

---

## Where we actually are

**Established, with statistics:**

| finding | evidence |
|---|---|
| the score is what works, not the budget | random 0/36 and StreamingLLM 0/36 against 36/36, at both B=16 and B=64, *p* = 2.9 × 10⁻¹¹ |
| backward reach alone decides | forward reach inert (spread 0.000); width-5 backward beats width-7 centred |
| required reach tracks the span | d=7 at back=4 → 0.000 where d=5 at back=4 → 1.000; thresholds 3, 4, 8, 10 for d = 4, 5, 7, 9 |
| deriving the reach beats **holding one reach fixed** | 1.000/1.000/0.889 at d=4/5/7 vs back=6's 0.972/1.000/0.778. *(NOT "beats the best tuned reach": a tuned back=8 scores 1.000 at d=7. Corrected 2026-08-25.)* |
| competition costs capacity, not correctness | 1.000 → 0.611 with 16 query-shaped decoys, restored to 1.000 by budget alone |
| transfers across checkpoints | Halo: 8 cells, **287/288** needles, no code change. *(Corrected 2026-08-26: 36 of those needles originally faced ZERO compressions — the depth-0.9 cells at 16K sat inside the never-compressed final chunk. Re-run at chunk 1024 they return the same 1.000, and one real failure appears at B=32 depth 0.1.)* |
| inference-time only | 0 parameters, 1.5625% cache overhead, no retraining |

**Known-invalid or unmeasured:**

- **No valid measurement of evidence spans > 9 tokens.** The word-phrase attempt was
  confounded: filler made of words, payload made of words, model answers with filler.
- **One task family.** Passkey filler contains no digits, so the answer is the only
  object of its kind — a regex scores 1.000 with no model.
- **n = 12** independent needles per cell; 95% lower bound on a perfect cell is 0.779.
- **The mechanism's own runtime cost is unmeasured.**
- **One prefill chunk size** in 124 of 132 result files.

---

# Track A — validity

*What must be true for the current claims to survive review. Nothing here is optional.*

### A1. Chunk-size robustness — **the biggest single threat** *(DONE 2026-08-25 — and it found something: 4096/1024 are fine, 2048 is not. See O13, open)*

**Follows from.** O7: survival is governed by how many compression rounds a position
faces (27 at depth 0.1 vs 3 at depth 0.9). Chunk size sets that count globally: 128K at
chunk 4096 is 30 compressions; at chunk 1024 it is 122.

**Experiment.** 128K, B=16, backward reach, chunk ∈ {1024, 2048, 4096}.

**Kill test.** Accuracy falls materially at chunk=1024 → **7,864× is a property of the
prefill schedule, not the mechanism**, and every compression figure must carry its chunk
size. That would be a serious limitation, not a footnote.

**Cost.** 2 cells, ~35 min.

### A2. Sink ablation *(DONE 2026-08-25 — sinks are load-bearing; sinks=0 scores 0.000 with MORE score-chosen slots. Forced a headline restatement)*

**Follows from.** At B=16 the budget is 4 sinks + 4 recent + 8 heavy. Sinks are 25% of
the budget and were hardcoded, never swept. The headline claims *eight* score-chosen
positions retrieve a needle from 125,826 tokens.

**Experiment.** 128K, B=16, sinks ∈ {0, 2, 4}.

**Kill test.** Collapse at sinks=0 → the honest count is not eight, and the claim must be
restated as "eight score-chosen plus four pinned".

**Cost.** 2 cells, ~35 min.

### A3. Statistical power — the last gate on the word "lossless" *(running: n=99 per depth)*

**Follows from.** Each cell draws the same 12 keys at all three depths, so a 36-trial row
is 12 independent needles. A perfect cell supports a 95% Clopper–Pearson lower bound of
**0.779** — far weaker than "lossless" implies.

**Experiment.** The headline cell (128K, B=16, backward reach) at n = 99 and n = 299 per
depth. Bounds: 0.970 and 0.990.

**Design note.** State the claim as a conjunction over depths ("every depth ≥ 0.99"), so
intersection–union applies and no Bonferroni correction is needed — the difference
between n=299 and n=517.

**Cost.** ~5 GPU-hours.

### A4. A corrected long-span experiment — ~~**the highest-value single item**~~ *(DONE 2026-08-25 — KILLED BY ITS OWN GATE: this model cannot recite past ~11 digits, so the span axis is unmeasurable here at any budget)*

**Follows from.** The withdrawal. Real evidence spans run 20–40 tokens and we have no
valid measurement past 9.

**The design flaw to avoid.** The payload must be an object the filler *cannot supply*.
Passkey works because the filler contains no digits. Candidate payloads that preserve
that property while extending span length:

| payload | span | filler can supply it? |
|---|---:|---|
| *k*-digit key (current) | ≤ 9 tokens | no — filler has no digits |
| **digit groups**: `4821 7735 9160 2284` | 20–40 tokens | **no** — same property, arbitrary length |
| hex or base-32 strings | tunable | no |
| rare proper nouns | 20–40 | partly — risky |

**Digit groups are the clean fix**: they extend the span axis arbitrarily while keeping
the answer the only object of its kind, and they stay exactly-checkable.

> **SIMPLIFIED 2026-08-25, no new code needed.** Verified on the tokeniser that a long
> digit string stays *one token per digit* — 12, 20 and 30 digits give spans of exactly
> 12, 20 and 30 tokens. So `--key-digits 20` already **is** a 20-token span, in filler
> that contains no digits. Digit groups were unnecessary; the existing flag covers the
> whole 20–40 token regime.

**Experiment.** Groups of 4 digits, 1–8 groups (5–40 tokens), at 128K with derived reach,
plus an uncompressed 16K gate at every span length.

**Kill test.** Accuracy falls with span length in the *gate* as well → the model cannot
recite long payloads and the axis is unmeasurable on this model, which is itself a
finding worth stating.

**Cost.** ~8 cells + 4 gate cells, ~3 h.

### A5. Runtime cost

**Follows from.** Every number in this programme is a benefit. `grow_spans` iterates per
compression; query-aware scoring adds a forward pass per chunk boundary.

**Experiment.** Wall-clock per compression for centred / fixed-backward / derived reach at
matched budget; prefill time with and without query-aware scoring; decode throughput
with and without the mechanism.

**Kill test.** Derived reach costs more than it saves at realistic budgets.

**Cost.** ~1 h.

---

# Track B — capability

*What the mechanism might do that we have never tried. Each follows from something
already measured; none is speculation for its own sake.*

### B1. Self-sizing budget — **the most valuable untested idea**

**Follows from.** Stage 1's arithmetic: 17 spans × ~22 tokens ≈ 374 tokens, and B=512
(380 heavy slots) restored accuracy exactly. Budget requirement tracks **candidate count
× span length**, not context length.

**The idea.** The mechanism already computes, at every compression, a score distribution
over positions. The number of *distinguishable candidate spans* is visible in that
distribution — count the connected components above the noise gate, which `grow_spans`
already produces. So the policy could **size its own budget**: keep `k × (components) ×
(mean component length)` rather than a fixed B.

**Why this matters more than a better ratio.** Every deployment currently guesses a
budget. A mechanism that reports "this context needs 400 slots, that one needs 40" is
qualitatively more useful than one that compresses harder.

**Experiment.** Instrument `grow_spans` to emit component count and mean length per
compression. Correlate against the measured budget requirement across the distractor
ladder (k = 0, 1, 4, 16 at B = 16…512). If component count predicts the knee, implement
self-sizing and test against fixed budgets at matched memory.

**Kill test.** Component count does not correlate with the measured knee → the score
distribution does not carry candidate structure and self-sizing is unfounded.

**Cost.** Instrumentation is CPU-only; validation reuses committed cells. ~2 h.

### B2. Iterative query re-scoring for multi-hop

**Follows from.** P1 measured that future attention is not predictable from content —
the query is the *only* signal. In a chain, **hop 1's answer is hop 2's query**, and hop
2's evidence shares no tokens with the original question by construction.

**The idea.** Re-score against the *generated prefix* as decoding proceeds, not only
against the original question at chunk boundaries.

**Why it is the sharp test.** Stage 1 showed budget can rescue competition. If budget
*cannot* rescue later hops but one extra scoring pass can, the binding resource is
**query-shaped access**, not capacity — a different mechanism, and a stronger claim.

**Experiment.** Stage 3 hop ladder (2, 3, 4 hops) at 128K, with and without re-scoring,
across budgets.

**Kill test.** Budget rescues later hops as well as it rescued distractors → access is
not the binding resource and this adds nothing.

**Cost.** ~3 h including implementation.

### B3. All-or-nothing span retention

**Follows from.** The whole span thesis. Pooling *approximates* span retention by score
contagion; it does not guarantee it. P2.2 showed a payload whose neighbours are all
payload receives nothing.

**The idea.** After selection, expand each kept position to its full connected component
and drop components that do not fit whole. **Never keep a fragment** — a half-retained
key is not half-useful.

**Why it may beat derived reach.** It is the logical endpoint of the argument, and it
converts wasted budget (fragments) into whole spans.

**Experiment.** 128K across budgets, against derived reach at matched memory.

**Kill test.** No improvement over derived reach → contagion already achieves what
explicit span retention would, and the simpler mechanism wins.

**Cost.** ~2 h.

### B4. Multi-turn persistence — the biggest untested application surface

**Follows from.** Nothing yet — and that is the point. Every result is single-turn. Chat
and agent workloads reuse a cache across many turns, with the *question changing every
turn*.

**The question.** Query-aware scoring optimises for the current question. What happens to
evidence that mattered two turns ago? Does accumulated significance protect it, or does
each turn's scoring erode it?

**Why it matters commercially.** This is the dominant real deployment pattern, and the
mechanism has never been run in it.

**Experiment.** A multi-turn passkey: three needles, three questions asked in sequence
over one persistent compressed cache. Measure retrieval of needle *i* at turn *j*.

**Kill test.** Later turns destroy earlier evidence → the mechanism is single-query and
must be documented as such.

**Cost.** Harness work ~3 h, experiments ~2 h.

### B5. Compression as attention denoising

**Follows from.** Two hints, neither significant: W=12 compressed 0.861 vs uncompressed
0.806; 32K full cache 35/36 vs compressed 36/36.

**The hypothesis.** Removing distractor positions may *improve* focus, so compression
could beat full cache at long context where attention dilutes.

**Why to test it properly rather than quote the hints.** Both are within noise at n=12.
Either it is real and it is a *quality* argument rather than a memory one — a much
stronger claim — or it is noise and should stop being mentioned.

**Experiment.** Powered comparison (n ≥ 99) of compressed against full cache at 16K and
32K, where a reference fits.

**Kill test.** No significant difference → say so plainly and drop the hint from the paper.

**Cost.** ~4 h.

### B6. Heterogeneous per-layer budget

**Follows from.** The trace showed retention varies across the 28 × 8 = 224 streams;
budget is currently uniform across layers.

**The idea.** Allocate budget by measured per-layer utility rather than uniformly.

**Kill test.** Uniform allocation matches tuned allocation → layers are interchangeable
for this purpose and the complexity is unjustified.

**Cost.** ~3 h. **Lower priority** — an optimisation, not a claim.

---

# Track C — practicability

### C1. Real prose haystack

Replace synthetic filler with real text while keeping a synthetic, checkable payload
(digit groups from A4). Isolates *haystack realism* from *payload realism*. **Kill test:**
a large drop against synthetic filler at matched budget means the compression figures do
not survive real text.

### C2. Deployment sizing rule

Formalise and test: `budget ≈ candidates × span_length × safety`. Validate against the
distractor ladder and A4. **This is the deliverable a practitioner actually wants.**

### C3. Quantisation interaction

KV quantisation and eviction are orthogonal axes and compose multiplicatively in
principle. Untested here. Lower priority but high ceiling.

---

## REVISION 2026-08-25 — the plan's top-priority item is dead, and what replaces it

A1, A2 and A4 all ran. A1 and A2 did what a validity check is supposed to do: A2 forced
a restatement of the headline, A1 turned up the non-monotonic chunk axis (O13, open).

**A4 did not fail — it was killed by its own pre-stated gate, and that changes the
plan's strategy section, not just its status table.** With a *full uncompressed cache*
this model scores 0.000 on a 12-digit key, emitting 11 correct digits then diverging.
No budget, window, or task redesign routes around a model's recall ceiling. So item 1
of "what would make this mechanism genuinely strong" below — *a valid long-span
measurement* — **cannot be obtained on this model through a generation-scored task.**

### A4′ — span preservation measured in the CACHE, not through generation

The replacement is not a different task; it is a different **instrument**. A4 tried to
read span preservation through the model's mouth, so it inherited the model's recall
limit. The quantity the mechanism actually controls is *how much of the evidence span
survives eviction*, and that is directly observable.

- **Metric.** Carry original-position identity through every gather (the machinery
  already exists in `experiments/stabilizer/src/trace_cache_gaps.py`) and report, per
  (layer, KV-head) stream, the fraction of the needle's positions still resident after
  the final compression. P2.1 already did this at one span length and measured 79–90%
  survival across the 224 streams.
- **Why it dodges the ceiling.** Survival is a property of the cache, not of decoding.
  A 40-token span can be measured for retention even when the model could never recite
  it — which is exactly the regime real evidence occupies.
- **Why it is worth doing.** §4.5's reach law is currently measured over d = 4, 5, 7, 9
  by generation. If cache-level survival reproduces the *same* reach law over that
  range, the law can then be extended to d = 20–40 on the instrument alone, with the
  generation results serving as the calibration that licenses it.
- **The pre-stated kill condition.** If cache-level survival does **not** reproduce the
  known reach law over d = 4, 5, 7, 9, the instrument is not measuring what generation
  measures and the extension is not licensed. Say so and stop.
- **Known limitation to fix first.** `trace_cache_gaps.py` inspects **layer 0, head 0
  only** and must be aggregated across all 224 streams before anything is read from it.
  The current single-stream output must not be cited (see O13's stated limitation).
- **Cost.** CPU/GPU-cheap relative to a generation sweep: no sampling, no scoring, one
  forward pass per cell.

**A4′ does not restore item 1 below.** It converts an unmeasurable quantity into a
measurable proxy, and the proxy needs its own calibration before it carries weight. The
honest position remains: *this programme has no generation-validated measurement of
evidence spans beyond 9 tokens, and cannot get one on this model.*

### C1′ — natural language, and why it is now FIRST rather than seventh

**The finding that reorders the plan.** Every result cornerstone 2 has produced is
passkey-shaped. I audited all 40 committed result files that set `pool_back` — 31 `pk_*`
and 9 `halo_*` — and **not one carries `rag_dev` or `chat_dev`.** The scorer in that
harness is a single line, `hit = str(key) in got`: a substring test for a digit string,
on filler containing no digits. So every cornerstone-2 number is a *span-preservation*
measurement and **none of it is evidence about response quality.**

Worse, the mechanism could not have been evaluated on language even if we had tried:
`run_eval_sig.py`, the only natural-language runner, had **no `--pool-back`,
`--pool-fwd`, `--grow` or `--grow-max` flags**. Cornerstone 2's directional pooling was
unreachable from the language harness. (Fixed 2026-08-25; the flags are wired, the
effective window is recorded in the manifest, and `Policy.pool_window()` now raises if
only one of back/fwd is set — `--pool 7 --pool-back 6` silently gave a *forward-reaching*
(6, 3) window, which would have measured a different mechanism and looked fine.)

**What cornerstone 1 already measured on language, which is the reason this is urgent.**
On 752 RAG cases with a centred window and a heuristic rubric scorer:

| keep | quality | vs uncompressed | byte-identical replies |
|---:|---:|---:|---:|
| off | 0.66939 | — | — |
| 0.75 | 0.65677 | −0.013 | 711/752 |
| 0.50 | 0.61401 | −0.055 | 605/752 |
| **0.25** | **0.33863** | **−0.331** | **159/752** |

At **4× compression on real prose, half the quality is gone.** Against the passkey
headline of 7,864× these are different universes, and the honest reading is that passkey
describes a haystack with exactly one distinctive region, not a document. Cornerstone 1
also published where its score *loses* to a matched positional control: +0.181 at keep
0.75, +0.137 at keep 0.50, and **−0.083 at keep 0.25**.

**The question C1′ answers.** Does cornerstone 2's directional pooling — which on passkey
was the difference between 0.000 and 1.000 — move any of those language numbers, and in
particular does it repair the keep-0.25 loss?

**Design.** Eight cells on `rag_dev`, 752 cases each, ~8–18 min per cell, against the
committed `off` reference. Every arm matches the committed cornerstone-1 manifests on
`sinks=4, recent=32, window=16` and one protocol, so a directional arm differs in exactly
one variable:
- **centred pool=7 controls at keep 0.25 and 0.75** — these did not exist, and without
  them a directional arm would be compared against a `pool=1` arm, confounding *pooling*
  with *direction*, which is the entire variable of interest. (keep 0.5 already has one.)
- **directional (back=6, fwd=0) at keep 0.25 / 0.50 / 0.75**
- **derived reach (`grow=2.0`) at keep 0.25 / 0.50**
- **directional at keep 0.25 with `--recent-frac 0.25`** — the plain keep-0.25 arm spent
  so much on a fixed `recent=32` that only 37% of the budget was score-chosen, making it
  StreamingLLM in disguise; this raises that to ~67%, which is where the score can
  actually be judged.

**Pre-stated reading of the outcome, so it cannot be rationalised afterwards.**
- If directional pooling closes the keep-0.25 gap: cornerstone 2 transfers to language
  and there is a deployable claim.
- If it moves the language numbers by less than the run-to-run spread: **the passkey
  result does not transfer**, and cornerstone 2 must be described as a retrieval
  mechanism rather than a general cache-compression method. That is a publishable
  negative and must be stated as plainly as the positive would be.
- Either way, `quality` here is a **heuristic rubric**, not a model judge — the code
  itself flags `quality_pass_is_diagnostic_only: True` — and no statistical test has ever
  been run on these deltas. A quality difference smaller than the rubric's resolution
  will not be claimed.

---

### Operational rule added 2026-08-25: this GPU is not exclusively ours

A3's shard queue ran for over an hour alongside **the user's own job** —
`run_stability192_validation.py`, from the Chat-Bubble repository, started 21:33:57 —
on the same 8 GB card. My cooldown gate saw 5,182 MiB and then 6,734 MiB already in use
and *proceeded anyway* on its fallback, because the gate was written to wait out **my
own** previous cell, not somebody else's job.

The three completed shards are unaffected (60/60 each, peak 1,501 MiB, matching the
committed headline exactly), but the card was at **100% utilisation and 93 °C** with both
jobs on it, and my run was consuming headroom the other job might have needed. I stopped
my queue and killed only my own PIDs.

**Correction to rule 1, added after it cost an hour.** The first version blocked
whenever a process whose command line matched another project was *alive*. It then
yielded for a full hour to a **launcher shim holding 1 MB and 0 CPU seconds** while the
card sat at baseline — confusing "a process exists" with "the card is in use" — and the
A3 shard and the whole allocation sweep were skipped. The guard now requires **both** a
foreign process holding real memory (>200 MB working set) **and** the card actually
consuming (≥1500 MiB). Being a good neighbour means yielding to work, not to a name in a
process table.

**Rules for every future queue here:**
1. Before starting a cell, check for GPU processes that are **not ours** — match on the
   command line, not on "is python running". If a foreign job holds the device, wait or
   yield; do not fall through and start anyway.
2. A cooldown fallback that proceeds after a timeout is correct when waiting on our own
   teardown and *wrong* when another job is simply using the machine. Distinguish them.
3. Never kill a process we did not start. Target our own PIDs explicitly by command line.
4. Long GPU campaigns should be resumable per-cell (the shard design already is), so
   yielding costs one cell rather than the run.

---

### Revised ordering from here

1. **C1′ — natural language.** *(running)* Promoted from seventh to first. Nothing else
   on this list matters if the mechanism does not transfer off passkey, and until
   2026-08-25 it could not even be run on language.
2. **A3 — SHARDED after three failures.** Three attempts, three *different* CUDA errors
   (`CUBLAS_STATUS_EXECUTION_FAILED`, death on load, `cudaErrorUnknown`), the last on a
   verified-drained GPU at 31 minutes in. The drain fix cured the language cells but not
   this, so the remaining variable is sustained 128K work: ~2.5 h of continuous
   full-budget prefill on an 8 GB card that reached 91 °C. Re-cast as **5 shards of 20
   trials on 5 distinct seeds** (verified: 100 keys, zero cross-seed collisions), each
   ~25–30 min with a drain and thermal cooldown between. n = 100 pooled gives a
   one-sided 95% Clopper–Pearson lower bound of **0.9705** against 0.7791 at n = 12,
   which is the entire purpose of A3.
   *(Original note, kept:)* Crashed on its first attempt with
   `CUBLAS_STATUS_EXECUTION_FAILED` after ~21 min at 91 °C; no partial artifact was
   written, so nothing is contaminated. Re-queued behind C1′ with full logging. (The
   wrapper reported *exit code 0* for that crash — it piped python through `grep | tail`,
   so the pipeline returned tail's status and the traceback was discarded. Fixed.)
3. **B1 — self-sizing budget. PROMOTED, and no longer speculative.** C1′ produced the
   largest language effect on record and it is an *allocation* effect, not a scoring one:
   holding the causal window fixed and changing `recent` from a fixed 32 to 25% of the
   budget moves keep-0.25 quality from **0.41286 to 0.60617** (+0.19331, p = 3.08e-60),
   and drops `heavy_zero_count` from **17,360 to zero**. That counter records events where
   the budget left the score no slots at all, so the policy had been silently degenerating
   to StreamingLLM — the mechanism was not underperforming at aggressive compression, it
   was *switched off* for much of the work. `--recent-frac` is a crude proxy for what B1
   would do properly, since it still fixes a ratio rather than deriving the split. The
   headroom above 0.60617 is what B1 is for.
4. **A4′** — the cache-level span instrument, with its calibration gate.
5. **B4, B2** — multi-turn persistence, then iterative re-scoring.
6. **A5** — runtime cost. *(Partially answered 2026-08-25: derived reach costs
   **1.71x** at keep 0.50 and **2.48x** at keep 0.25 against the causal window on
   rag_dev, while also scoring lower. Not thermally controlled, so A5 proper still
   stands — but the programme is no longer in the position of having zero measured
   costs.)* **Every number in this programme is still a benefit with no
   measured cost.** Note the thermal band: A3 is currently running at 91 °C, and the
   programme's own rule is that no throughput claim below 2× is defensible against a
   2.17× thermal band, so A5 must control temperature or it will measure the fan.
7. **O13** — the open anomaly. Not scheduled, because no candidate explanation survives;
   it needs a new hypothesis before it needs GPU time.
8. **C2, B3, B5, B6, C3** — as below. *(C1 is promoted to slot 1 as C1′.)*

---

## Ordering, and why *(superseded by the revision above; kept as the record)*

1. **A1, A2** *(running)* — cheap, and either could force a restatement of the headline.
2. **A4** — the withdrawal is the largest hole; digit groups fix the design flaw cleanly.
3. **B1** — highest value per hour; instrumentation is CPU-only and reuses committed cells.
4. **A3** — the power run, once the claims it certifies are stable. Running it before A1/A4
   risks certifying a number that later changes.
5. **B2, B4** — the two capability directions with real application surface.
6. **A5, B3, B5** — cost, span retention, denoising.
7. **C1, C2** — practicability, once the mechanism is characterised.
8. **B6, C3** — optimisations.

**A3 deliberately comes after A1 and A4.** Spending five GPU-hours certifying a headline
that a chunk-size result might invalidate is the wrong order, and the temptation to lock
in a good number early is exactly what this programme has repeatedly had to undo.

---

## What would make this mechanism genuinely strong

Not a bigger compression ratio. Three things:

1. ~~**A valid long-span measurement** (A4)~~ — **UNOBTAINABLE ON THIS MODEL.** The gate
   fired: 0.000 at 12 digits with a full uncompressed cache. The method remains unproven
   in the regime that matters for real documents, and on this hardware it will stay that
   way. A4′ above offers a cache-level proxy, which needs its own calibration before it
   counts. This is now a stated limitation of the programme rather than a task on it.
2. **A self-sizing budget** (B1) — turns a tuned hyperparameter into a derived quantity,
   which is the same move that made derived reach beat fixed reach.
3. **Multi-turn evidence** (B4) — the dominant deployment pattern, entirely untested.
4. **Any natural-language evidence at all** (C1′) — added 2026-08-25, and it belongs at
   the top of this list rather than the bottom. The mechanism has never been measured on
   a natural-language token; cornerstone 1's language numbers show quality falling by
   0.331 at 4× compression, where passkey reports 7,864× losslessly. Until C1′ runs, the
   practical claim rests on a task a regex can solve.

Any one of these is worth more than another order of magnitude on passkey.
