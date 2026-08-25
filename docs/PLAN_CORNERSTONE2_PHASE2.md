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
| deriving the reach beats tuning it | matches or beats the best hand-tuned reach at every span length |
| competition costs capacity, not correctness | 1.000 → 0.611 with 16 query-shaped decoys, restored to 1.000 by budget alone |
| transfers across checkpoints | Halo: 8 cells, 288 needles, zero failures, no code change |
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

### A1. Chunk-size robustness — **the biggest single threat** *(running)*

**Follows from.** O7: survival is governed by how many compression rounds a position
faces (27 at depth 0.1 vs 3 at depth 0.9). Chunk size sets that count globally: 128K at
chunk 4096 is 30 compressions; at chunk 1024 it is 122.

**Experiment.** 128K, B=16, backward reach, chunk ∈ {1024, 2048, 4096}.

**Kill test.** Accuracy falls materially at chunk=1024 → **7,864× is a property of the
prefill schedule, not the mechanism**, and every compression figure must carry its chunk
size. That would be a serious limitation, not a footnote.

**Cost.** 2 cells, ~35 min.

### A2. Sink ablation *(running)*

**Follows from.** At B=16 the budget is 4 sinks + 4 recent + 8 heavy. Sinks are 25% of
the budget and were hardcoded, never swept. The headline claims *eight* score-chosen
positions retrieve a needle from 125,826 tokens.

**Experiment.** 128K, B=16, sinks ∈ {0, 2, 4}.

**Kill test.** Collapse at sinks=0 → the honest count is not eight, and the claim must be
restated as "eight score-chosen plus four pinned".

**Cost.** 2 cells, ~35 min.

### A3. Statistical power — the last gate on the word "lossless"

**Follows from.** Each cell draws the same 12 keys at all three depths, so a 36-trial row
is 12 independent needles. A perfect cell supports a 95% Clopper–Pearson lower bound of
**0.779** — far weaker than "lossless" implies.

**Experiment.** The headline cell (128K, B=16, backward reach) at n = 99 and n = 299 per
depth. Bounds: 0.970 and 0.990.

**Design note.** State the claim as a conjunction over depths ("every depth ≥ 0.99"), so
intersection–union applies and no Bonferroni correction is needed — the difference
between n=299 and n=517.

**Cost.** ~5 GPU-hours.

### A4. A corrected long-span experiment — **the highest-value single item**

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

## Ordering, and why

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

1. **A valid long-span measurement** (A4) — without it the method is unproven in the
   regime that matters for real documents.
2. **A self-sizing budget** (B1) — turns a tuned hyperparameter into a derived quantity,
   which is the same move that made derived reach beat fixed reach.
3. **Multi-turn evidence** (B4) — the dominant deployment pattern, entirely untested.

Any one of these is worth more than another order of magnitude on passkey.
