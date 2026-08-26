# Evidence lives in spans: the unit of KV-cache eviction, and why its reach must be derived

**Draft, 2026-08-24.** Cornerstone 2 of the Rineya significance-matrix programme.
Companion to `PAPER_DRAFT_SIGNIFICANCE.md` (cornerstone 1, branch `main`), which
introduces the significance matrix itself. This paper is about how that matrix's
scores must be *read*, not about the scores.

> **Status of every number below.** All figures reproduce from committed JSON via
> `scripts/design/verify_claims.py` (124 pinned claims). Where a claim has been
> retracted during the work it appears in Appendix A rather than being deleted.
> References in §2 are recalled from the literature and **must be verified by hand
> before submission**.

---

## Abstract

KV-cache eviction methods rank cached positions by an importance score and keep the
top-*k*. We show that this per-position framing is the wrong unit, and that fixing the
unit is worth more than improving the score.

Working from a 0.6B-parameter model on a single 8 GB GPU, we first establish by tracing
position identity through every compression round that retrieval failures are **not
evictions**: the target survives in 79–90% of (layer, KV-head) streams while accuracy
sits at 0.42/0.33/0.83, and only 57–65% of its *tokens* survive. Failure is
fragmentation, not deletion.

We then measure *why*. Query-aware scoring — which produced the largest single quality
gain in the parent programme — preferentially retains text that **resembles the query**:
across 224 streams, the query-shaped framing of an evidence span survives at 54% while
the payload it introduces survives at 35%. The score keeps the label and discards the
answer.

This diagnosis makes a prediction, and the prediction holds. Letting a position inherit
score from what *precedes* it takes 128K/B=64 from 0.278 to **1.000** (216 discordant
pairs across seven cells, zero regressions, exact McNemar *p* < 10⁻⁵ each). Isolating
the variable shows the operative quantity is **backward reach alone**: forward reach is
inert (matched-pair spread 0.000), and window *width* is not the variable — a width-5
backward window scores 1.000 where a width-7 centred window scores 0.278.

Critically, the required reach **tracks the evidence span and is not a constant**: a
7-digit key at back=4 scores 0.000 where a 5-digit key at the same reach scores 1.000.
We therefore derive the reach from the score's own structure by geodesic dilation, which
beats any *single* fixed reach held constant across span lengths — the comparison a
deployed system faces, since the span is not known in advance — without being told the
span length. Against a *per-span oracle* it is not uniformly better: at d=7 derived
reach scores 0.889 where a hand-tuned back=8 scores 1.000 (§4.6).

Against controls the mechanism has never previously faced, the score is what works: at
128K with a 16-slot budget — four attention sinks, four recent, **eight score-chosen** —
significance scores **36/36** where equal-budget random and StreamingLLM both score
**0/36** (*p* = 2.9 × 10⁻¹¹). The result transfers unchanged to a separately published
core.

An ablation shows the sinks are **not optional overhead but load-bearing**: with
`sinks=0` and *more* score-chosen slots (12 against 8), accuracy is **0.000**. The claim
is therefore four sinks *plus* eight score-chosen positions, not eight positions alone.

On **natural language** the mechanism transfers but the headline does not (§4.10). At
~232-token RAG prompts, causal pooling retains **90.6%** of uncompressed quality at **4×**
compression and beats a matched positional control by **+0.184** — but only once the
budget is allocated proportionally. Under the fixed `recent` inherited from prior work,
the score was left **zero** slots on 17,360 compressions and the policy silently
degenerated to StreamingLLM; repairing that allocation is worth **+0.193** (*p* = 3 ×
10⁻⁶⁰), the largest language effect we measure. We report the 7,864× figure as a property
of passkey and not of prose, and note that our language evidence sits **543× shorter**
than our retrieval evidence with nothing in between.

We report where it fails. Sixteen query-shaped distractors take accuracy from 1.000 to
0.611, and budget alone restores it — competition is a capacity cost, not a scoring
failure. We also withdraw a word-phrase variant built to extend the span axis past nine
tokens: its filler was made of words, like the payload, so the model could answer with
filler and be scored wrong for reasons unrelated to the cache. Consequently this work
has **no valid measurement of evidence spans longer than nine tokens**.

---

## 1. The problem, and why the unit matters

A KV cache grows linearly with context. Eviction methods (§2) assign each cached
position an importance score and retain the top-*k*. Every such method inherits an
assumption that is rarely stated: **that positions are independently rankable**.

They are not. Evidence in text is contiguous. A five-digit key occupies five consecutive
token positions; a fact occupies a clause. Ranking positions independently can retain a
majority of a span's tokens while destroying the span — and a partially retained key is
not partially useful, it is useless.

The parent programme (cornerstone 1) built a persistent per-position, per-KV-head
accumulator of received attention mass, corrected its ratchet failure mode by ranking on
`S/C^e`, and demonstrated lossless 128K retrieval at 12.5% keep. This paper starts from
the observation that its remaining failures had a signature nobody had looked at.

### 1.1 Contributions

1. **Failure is fragmentation, not eviction** (§4.1), established by carrying original
   position identity through every gather.
2. **The score prefers query-shaped text over the payload it introduces** (§4.2):
   framing 54%, payload 35%, six cells of six.
3. **A repair predicted by that diagnosis and confirmed** (§4.3): 0.278 → 1.000.
4. **The operative variable is backward reach alone** (§4.4); forward reach is inert and
   width is not the variable.
5. **The required reach tracks the span** (§4.5), so it cannot be a tuned constant.
6. **Deriving the reach beats tuning it** (§4.6).
7. **Controls**: equal-budget random and StreamingLLM both score 0/36 where the
   mechanism scores 36/36 (§4.7).
8. **Transfer** to a separately published core with no code change (§4.8). Eviction
   pressure per se does not degrade it — at 122 compressions instead of 30 the result
   is statistically indistinguishable (*p* = 1.0000) while peak memory falls 30% — but
   the chunk axis is **non-monotonic** and we cannot yet explain it (§5.3).
9. **The attention sinks are load-bearing** (§4.9): removing them collapses accuracy to
   0.000 despite a larger score-chosen budget.
10. **Two measured failure conditions, one withdrawn experiment, and one unexplained
    anomaly** (§5), reported rather than omitted.

---

## 2. Related work

*Recalled from the literature; not resolved against a bibliographic database.*
**Every citation here must be verified by hand before submission.**

- **H2O / Heavy-Hitter Oracle** — accumulated attention as an eviction score. The
  per-position framing this paper argues against.
- **Scissorhands** — persistence of importance hypothesis; similar accumulator.
- **SnapKV** — introduces max-pooling over scores so that a high-scoring token pulls
  neighbours with it. **This is the closest prior work**: it recognises spans matter and
  supplies a *centred, fixed-width* pooling window. We show the window's shape and reach
  are not free parameters — direction matters (forward reach is inert) and the required
  reach is a function of the evidence span, which a fixed window cannot express.
- **StreamingLLM** — attention sinks plus a recent window. Used here as a baseline.
- **YaRN** — training-free RoPE scaling; used to reach 128K from a 40,960 native window.
- **Needle-in-a-haystack / passkey retrieval** — the evaluation family used throughout,
  whose limits we state plainly in §6.

---

## 3. Method

### 3.1 The score being read

Per layer, two fp32 tensors of shape `[b, n_kv, n]`: accumulated attention mass `S` and
observation count `C`. Scores are taken **pre-`repeat_kv`**, at KV-head granularity,
because eviction decides about stored tensors. Ranking key is `S / max(C,1)^e`, `e=1`.
Both tensors are carried through every eviction gather. Memory overhead: **1.5625%** of
the cache; **zero added parameters**; no training; no weight modification.

### 3.2 The selection rule, and the window

Keep set = `sinks ∪ top-k(pooled) ∪ recent`. Masking of sinks and recent happens
**before** pooling, so a sink's mass cannot leak into its neighbours through the pool.

The contribution is the pooling window. SnapKV uses a centred max-pool of fixed width.
We generalise it to an arbitrary `(back, fwd)` reach:

```
pooled[i] = max( s[i-back] … s[i+fwd] )
```

with the centred window as the default, so every prior result reproduces exactly
(verified bit-identical for widths 3, 7, 15, 31 across three sequence lengths).

### 3.3 Derived reach: geodesic dilation

Because the required reach tracks the span (§4.5), we replace the fixed window with
growth gated on the score itself:

```
gate  = s > median(s) · m
repeat: s ← max(s, shift_forward(s)) masked by gate, until convergence or max_reach
```

The score enters a span at its **head** (§4.2) and is grown forward through positions
that are themselves above the stream's noise floor, so it reproduces the span and stops
where the score collapses to filler.

The gate is relative to each stream's **median**, which tracks the noise floor because
filler dominates the distribution. Two alternatives were implemented and rejected by CPU
tests before reaching the GPU, and both failures are instructive:

- *Fraction of the maximum* presumes a known head-to-payload ratio — the exact prior
  knowledge this exists to avoid. With head 1.0 and payload 0.15, any threshold above
  0.15 makes growth a silent no-op.
- *A quantile* fails oppositely: when signal is sparse the k-th largest score **is** the
  filler level, the gate opens on everything, and growth floods the sequence. On a
  200-position stream holding one 5-token span, a 0.95 quantile admitted all 200.

A third error survived to the GPU and is worth recording: requiring a position to be in
the top-*k* **to be grown into** inverts hysteresis, which needs a selective seed and a
*permissive* gate. A payload position is individually weak — that is the entire problem
— so the top-*k* requirement excluded exactly what the mechanism exists to rescue. It
scored 0.00 and emitted degenerate output.

---

## 4. Experiments and results

Model: 28 layers, 16 query / 8 KV heads, head_dim 128, 474M parameters, bf16, single
8 GB RTX 4070. Context to 128K via YaRN ×4. Chunked prefill, chunk 4096, so peak memory
is independent of context length. All cells: 12 paired needles × 3 depths, count
normalisation `e=1`, query-aware scoring. Statistics: exact McNemar on discordant pairs.

### 4.1 Failure is fragmentation, not eviction

Carrying original position identity through every gather at 128K/B=2048:

| depth | streams retaining the target | accuracy | tokens retained |
|---:|---:|---:|---:|
| 0.1 | 178 / 224 (79%) | 0.42 | 57–65% |
| 0.5 | 181 / 224 (81%) | 0.33 | |
| 0.9 | 201 / 224 (90%) | 0.83 | |

Retention moves ~10 points while accuracy moves 0.50. The target is not being evicted.
A planned "demotion tier" was cancelled: its premise was false.

### 4.2 The score keeps the label and discards the payload

Per-token survival inside the evidence span, 224 streams, six cells:

| | framing tokens | digit tokens | gap |
|---|---:|---:|---:|
| mean over 6 cells | **54%** | **35%** | **+19 pts** |

Six of six in the same direction. The cause is the query itself: query-aware scoring runs
the question against the cache, the framing tokens *are* the query lexically, and the
digits are what the query asks for and look nothing like it.

Within the digit run survival is U-shaped in all six cells — the digit adjacent to the
framing does best, the interior worst — confirming that pooling propagates score to
*neighbours of high scorers* rather than protecting spans as such.

### 4.3 The predicted repair

| length | B | centred (3,3) | backward (6,0) | 0→1 | 1→0 | *p* |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 16 | 0.028 | **1.000** | 35 | 0 | <10⁻⁵ |
| 32K | 32 | 0.389 | **1.000** | 22 | 0 | <10⁻⁵ |
| 64K | 16 | 0.000 | 0.972 | 35 | 0 | <10⁻⁵ |
| 64K | 32 | 0.000 | **1.000** | 36 | 0 | <10⁻⁵ |
| 128K | 16 | 0.000 | **1.000** | 36 | 0 | <10⁻⁵ |
| 128K | 32 | 0.000 | **1.000** | 36 | 0 | <10⁻⁵ |
| 128K | 64 | 0.278 | **1.000** | 26 | 0 | <10⁻⁵ |

**216 discordant pairs, every one in the same direction, none against.** Peak memory
1,500–1,507 MiB throughout: the gain costs nothing. At 128K/B=16 this is **7,864×**
compression on 125,826 real tokens at 1,501 MiB.

The failure text confirms the account: the centred window emits `755`, `856`, `4264` for
keys `75526`, `85667`, `42649` — the truncation signature of §4.1 — and the backward
window emits the keys.

### 4.4 Backward reach alone; forward reach is inert

At 128K/B=64, matched-backward-reach pairs:

| back | fwd | width | accuracy |
|---:|---:|---:|---:|
| 3 | 3 | 7 | 0.278 |
| **4** | **0** | **5** | **1.000** |
| **4** | **4** | **9** | **1.000** |
| 7 | 7 | 15 | 1.000 |

Spread across forward reach at back=4 is **0.000**. And a width-**5** backward window
beats a width-**7** centred one. Width is not the variable; backward reach is.

*Mechanistically*: the score enters a span at its head, so payload tokens must look
**back** to reach it. Forward reach points away from the score.

### 4.5 The required reach tracks the span

| digits | back=2 | back=3 | back=4 | back=6 | back=7 | back=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.444 | 0.972 | — | — | — | — |
| 5 | — | 0.278 | **1.000** | — | — | — |
| 7 | — | — | **0.000** | 0.778 | 0.944 | **1.000** |

The discriminating cell is **d=7 at back=4 → 0.000**, where d=5 at the same reach scores
1.000. Backward reach is not a constant that happened to suit one key length.

We state no closed-form law. `back ≥ d−1` fits d=4 and d=5 and fails at d=7; `back ≥ d+1`
fits d=7 and over-predicts at d=5. What is established, and is what the method rests on,
is the **qualitative** claim: the requirement is a function of the evidence span.

### 4.6 Deriving the reach beats a *constant* reach — on passkey only

> **Audited 2026-08-26 and unchanged.** `grow_spans` was found to compute its noise floor
> over masked slots, flooding the growth gate. Every arm in this section was re-run with
> the corrected floor and **every `grow=2.0` figure is identical to four decimals**. Only
> the `grow=5.0` arm moved (0.611 → 0.722), so the multiplier contrast at d=7 weakens
> from *p* = 0.0020 to *p* = 0.0312 — still one-sided, with no needle preferring the
> larger multiplier under either gate.

One policy, never told the key length, against the best *hand-tuned* fixed reach:

| digits | derived (`grow=2.0`) | best tuned fixed | |
|---:|---:|---:|---|
| 4 | **1.000** | back=3 → 0.972 | +0.028 |
| 5 | **1.000** | back=4 → 1.000 | +0.000 |
| 7 | **0.889** | back=6 → 0.778 | +0.111 |

The multiplier is **not** inert: at d=7, `grow=2.0` gives 0.889 against `grow=5.0`'s
0.611 (*p* = 0.0020), though the two are identical at d=4 and d=5 (*p* = 1.0000). The
honest claim is that deriving the reach **reduces** the tuning problem — one setting
spans every tested length where no fixed reach does — not that it eliminates it.

### 4.7 Controls: the score is what works

Until these arms ran, every comparison in this programme was against an earlier version
of itself. At 128K/B=16 — four sinks, four recent, **eight heavy slots** — with identical
budget, seed and needles:

| arm | d0.1 | d0.5 | d0.9 | accuracy |
|---|---:|---:|---:|---:|
| significance + backward reach | 1.00 | 1.00 | 1.00 | **36/36** |
| random, same number of slots | 0.00 | 0.00 | 0.00 | **0/36** |
| StreamingLLM (sinks + recent) | 0.00 | 0.00 | 0.00 | **0/36** |

Both contrasts: 36 discordant pairs, none against, **exact McNemar *p* = 2.9 × 10⁻¹¹**.

Neither baseline degrades — both retrieve *nothing*, emitting a paraphrase of the
question with no digits at all. **7,864× is a property of the score, not of the budget.**

### 4.8 Transfer to a separately published core

Rineya-Halo-1.2.0, a published core from a separate repository with different weights
and a different vocabulary (151,936 against 32,768), 16K, derived reach:

| budget | compression | accuracy | peak |
|---:|---:|---:|---:|
| 32 | **491×** | **36/36** | 1,735 MiB |
| 128 | 123× | 36/36 | 1,748 MiB |
| 512 | 31× | 36/36 | 1,795 MiB |

**No code change** — only a model path. The mechanism keys off the attention shape, and
Halo matches it exactly.

---

**Correction (2026-08-26).** Three of these cells measured nothing. `chunked_prefill`
never compresses the final chunk, and at 16K with chunk 4096 the depth-0.9 needle sits
inside it — facing **zero** compressions, where a policy keeping nothing would also score
1.000. Re-run at chunk 1024, where that needle faces two compressions, the depth-0.9
cells give **the same 1.000**: the claim was true, it had simply not been tested. One
real failure appears at B=32 depth 0.1 (0.917), because fifteen compression rounds
instead of three is five times the eviction pressure. **The corrected total is 287/288**,
and 36 of those needles only became evidence with the re-run.

---

### 4.9 The attention sinks are load-bearing

Sinks were fixed at four in every result and never ablated. At B=16 they are 25% of the
budget, so the natural reading of §4.7 is "eight score-chosen positions retrieve a needle
from 125,826 tokens". That reading is wrong.

| sinks | score-chosen slots | d0.1 | d0.5 | d0.9 | accuracy |
|---:|---:|---:|---:|---:|---:|
| **0** | **12** | 0.00 | 0.00 | 0.00 | **0.000** |
| 2 | 10 | 1.00 | 1.00 | 1.00 | **1.000** |
| 4 | 8 | 1.00 | 1.00 | 1.00 | **1.000** |

With no sinks the mechanism scores zero *while holding more score-chosen slots than the
configuration that scores 1.000*. The parent programme measured that this model places
46.3% of its attention mass on the first four positions; removing them breaks the
attention distribution past anything the score can recover.

Two sinks suffice, so the requirement is "some sinks", not four — but it is a
requirement, and the correct statement of the headline is **four sinks plus eight
score-chosen positions**.

---

### 4.10 Natural language: the mechanism transfers, but the headline does not

Every result above is passkey-shaped, scored by `hit = str(key) in got` on filler that
contains no digits — a task a regular expression solves without a model. This section is
the first natural-language evidence the mechanism has produced. 752 RAG cases, paired on
case id, a heuristic rubric scorer that flags itself `quality_pass_is_diagnostic_only`.
Uncompressed reference: **0.66939**.

**Window direction matters, and the sign flips with compression pressure.**

| keep | compression | centred (3,3) | causal (6,0) | paired Δ | exact sign test |
|---:|---:|---:|---:|---:|---|
| 0.25 | 4× | 0.26096 | **0.41286** | **+0.15190** | *p* = 5.7 × 10⁻³⁷ |
| 0.75 | 1.33× | **0.65999** | 0.65136 | **−0.00863** | *p* = 3.0 × 10⁻⁶ |

Causal pooling wins under pressure and loses, slightly but significantly, without it —
with an ample budget a centred window's forward reach buys context; with a tight budget
the same reach spends slots on positions the query has not reached. Note also that at
keep 0.25 the centred window (0.26096) is *worse than not pooling at all* (0.33863): the
causal window is repairing damage that pooling causes, not merely adding a gain.

**The dominant variable is budget allocation, not scoring.** The arms above inherit
`recent` pinned at a fixed 32 positions. On a 232-token prompt at keep 0.25 the whole
budget is ~58, so sinks + recent consume 61% of it before the score is consulted:

| keep-0.25 arm | score-chosen share | `heavy_zero_count` | quality |
|---|---:|---:|---:|
| fixed `recent = 32` | 39.0% | **17,360** | 0.41286 |
| proportional `recent = 0.25·B` | **68.2%** | **0** | **0.60617** |

Paired difference **+0.19331**, *p* = 3.1 × 10⁻⁶⁰ — the largest effect we measure on
language. `heavy_zero_count` records compressions where the budget left the score *zero*
slots, so the policy degenerated to StreamingLLM: under the fixed allocation it fired
17,360 times, under proportional allocation zero. **The mechanism was not
underperforming at aggressive compression; it was not running for much of the work.**
Any evaluation of a budgeted eviction policy should report this counter, and we had it in
our manifests unread.

**What this supports, against a control that isolates the score.** Every earlier
language comparison used an arm differing in three things at once — no score, no pooling,
and a recent window of 53.88 against 14.38. A 3.7× larger recent window is a second
treatment, not a control. Passkey has had `--select random` since §4.7; language did not
until we added it.

At keep 0.25, all arms CPU, causal window, `sinks=4`, `recent-frac 0.25`, paired on 752
cases — `random` differing from `significance` in **nothing but how the heavy slots are
chosen**:

| arm | quality | paired vs significance |
|---|---:|---|
| **significance** | **0.59488** | — |
| streaming, matched budget | 0.42740 | **+0.16747** (*p* = 3.2 × 10⁻³²) |
| **random**, matched slots | 0.34304 | **+0.25184** (*p* = 2.0 × 10⁻⁶⁶) |

**The score is worth +0.252 at 4× compression** against an arm holding the same number of
positions by lottery. This supersedes the previously reported +0.184 — not because that
figure was too large, but because it could not be attributed to the score. With
proportional allocation, causal pooling retains **90.6%** of uncompressed quality at 4×
and **96.9%** at 2×. That is the deployable claim. It is not 7,864×, which is a passkey
property and should never be quoted beside a language number.

**Derived reach does not transfer — one leg retracted, one standing.** Geodesic
dilation, the best passkey arm of §4.6, scored −0.03513 (*p* = 1.5 × 10⁻⁵) below the
causal window at keep 0.50.

*The keep-0.25 leg (−0.07452) is **withdrawn**.* `grow_spans` computed its noise floor
as the median of a score vector with `0.0` substituted for the sink and recent masks;
on that arm's geometry those masks are 59% of the vector, so the floor was exactly zero
and the growth gate admitted **every** eligible position (24 of 24, against a true median
admitting 2). That arm measured a degenerate positional policy, not derived reach, and
is quarantined. The defect is fixed and the arm will be re-run.

The keep-0.50 leg sat at 30% masked — floor depressed to roughly the 27th percentile
rather than collapsed to zero — so it is **degraded, not void**, and is reported with
that caveat. The runtime figures (**1.71×** and **2.48×**) are likewise inflated: an
over-open gate runs the dilation loop further before converging, which is the most
likely reason the *smaller* budget carried the *larger* cost.

**The scale gap, which bounds everything in this section.** These prompts have a median
of 238 tokens and a maximum of 330; not one exceeds 1K. The passkey results run at
125,826 tokens — **543× longer** — and we have nothing in between. Every figure here
should be read as *"at ~232 tokens"*, and no long-context language claim follows from
any of it.

---

## 5. Where it fails

### 5.1 Competition is a capacity cost, not a scoring failure

Sixteen decoys at 128K/B=64, paired on decoy identity:

| decoy style | accuracy |
|---|---:|
| neutral (no frame overlap) | 0.972 |
| **query-shaped** (share the question's frame) | **0.611** |

Difference **−0.361**, *p* = 0.0010 — query similarity, not numeric competition, is the
operative variable, as §4.2 predicts.

But budget alone restores it: B=128 → 0.806, B=256 → 0.944, **B=512 → 1.000**. The score
still finds the right span; it cannot afford it. Competition costs **32×** of
compression — 7,864× down to 246×.

### 5.2 A word-phrase span task we had to withdraw

We built a word-phrase variant to extend the span axis past what digits allow, and
**withdraw its results**. The payload was made of words and the haystack filler is also
words, so the model can answer with filler and be scored wrong for reasons unrelated to
the cache. Filler recitation accounts for 100% of failures at W=3 depth 0.9, 75% at
W=20 depth 0.9, and **100% of failures in the uncompressed control** — so it is a
property of the task, not of compression.

Passkey's filler contains no digits at all, which is what makes the answer the only
object of its kind and therefore checkable. The word variant removed that property.

The consequence is a stated limit rather than a result: **this work has no valid
measurement of evidence spans longer than 9 tokens.** §4.5's span law is measured on
digit spans, where the confound cannot occur, and is unaffected.

We then asked whether the digit axis could be pushed instead, and it cannot: a 12-digit
key fails **with the cache uncompressed** (§6, limitation 3). So the 9-token limit is not an artefact
of the withdrawn task — it is where this model's verbatim recall ends.

---

### 5.3 The chunk axis is non-monotonic and we cannot explain it

| chunk | compressions at 128K | d0.5 | accuracy |
|---:|---:|---:|---:|
| 4096 | 30 | 1.00 | **1.000** |
| **2048** | 61 | **0.42 / 0.50** | **0.806 / 0.833** |
| 1024 | 122 | 1.00 | 0.972 |

The two chunk-2048 figures are independent seeds; it reproduces. Eviction pressure alone
does not explain it, since the highest-pressure setting is the second best. We have
eliminated: differing data (the prompt is byte-identical and the failing needles pass at
both other chunk sizes), a prefill bug (chunked prefill is bit-identical to single-shot
at all three sizes with compression off), boundary straddling (the needle lies inside one
chunk at every size), and pooling-window fragmentation (a `back=6` window spans exactly
six tokens with no gaps at both 4096 and 2048).

The failures carry the fragmentation signature — seven of twelve are the 4-of-5
truncation. **No candidate explanation currently survives**, and we report it rather than
omit it.

---

## 6. Limitations

1. **n = 12 independent needles per cell.** Each cell draws the same 12 keys at all
   three depths, so a 36-trial row is 12 needles observed three times. A perfect cell
   supports a 95% Clopper–Pearson lower bound of **0.779**. Every *contrast* here is
   overwhelming; the word "lossless" is weaker than it sounds.
2. **Two task families, 543× apart, with nothing between them.** The retrieval evidence
   is at 125,826 tokens; the language evidence (§4.10) is at a median of 238 tokens with
   no prompt above 1K. The regime this mechanism is *for* — a long natural-language
   document — is unmeasured, and neither body of evidence speaks for it.
3. **The language scorer is a heuristic rubric, not a model judge**, and flags itself
   `quality_pass_is_diagnostic_only`. It is adequate for a paired within-task contrast
   and should not be read as an absolute quality scale.
4. **One task family for the retrieval results.** Everything there is passkey-shaped. The filler contains **no digits at
   all**, so the answer is the only object of its kind and a regex scores 1.000 with no
   model. These are span-*preservation* measurements, not retrieval-difficulty ones.
5. **The span axis cannot be extended on this model, and we measured why.** With a *full
   cache and no eviction*, the model scores **0.000** on a 12-digit key, emitting 11
   correct digits then diverging — the same ceiling at 20 and 30 digits. So the reach law
   of §4.5 is measured over d = 4, 5, 7, 9 and **no experimental design can extend it
   here**; testing the 20–40 token spans that real evidence occupies requires a model
   with greater verbatim-recall capacity. Unlike the withdrawal in §5.2, this is a
   property of the model rather than a flaw we could fix.
6. **No uncompressed reference above ~32K.** A 64K full cache OOM'd at 6,425.6 MiB
   against a highest-survived 4,972.9 MiB. Above 32K the claim is not "cheaper than
   full" but "runs at all".
7. **The mechanism's own runtime cost is unmeasured.** Derived reach iterates; every
   number here is a benefit.
8. **One model family.** Halo shares the architecture; nothing is shown across scales or
   head configurations.

---

## Appendix A — retracted during this work

Kept rather than deleted, because the reasoning is worth more than the verdict.

| claim | status | what closed it |
|---|---|---|
| `B_half = 0.141·L^0.535`, a √L floor law | **retracted** | fitted from the centred window; with backward reach every length is lossless at or below B=16, so no exponent can be fitted |
| `back ≥ d − 1` | **retracted** | fitted on d=4/d=5; d=7 needs back=8 |
| "the floor is a fixed absolute budget of 256 tokens" | superseded | backward reach moved it to ≤16 |
| "the grow multiplier is inert" | **withdrawn** | inert at d=4/d=5, decisive at d=7 (*p* = 0.0020); the claim came from a cell where both settings saturate |
| "the multiplier acts through gate coverage" | **refuted on CPU** | coverage is 100% at every span and multiplier on the measured score profile |
| "stage 2 declines monotonically with span length" | retracted at its middle point | W=12 compressed 0.861 against uncompressed 0.806 — no measurable cost |
| n=36 giving a 0.903 bound | corrected | depths share keys; n=12, bound 0.779 |

---

## Appendix B — engineering defects that produced false results

Recorded because each one initially looked like a finding.

| defect | what it looked like |
|---|---|
| `Policy.pool` defaulted to 1 and the harness never set it | every long-context result measured the mechanism with span protection **disabled** |
| `count_exponent` inert on the self-compressing path | a ranking-key ablation that changed nothing |
| eviction ran during prefill instead of after | a 2.6× speedup |
| `recent=512` consumed the whole budget at B=512 | a capability cliff (`heavy()==0`, i.e. plain StreamingLLM) |
| trial records truncated to 32 characters | 20-word failures unclassifiable |
| manifest recorded the CLI flag, not the effective value | a run with `heavy=0` recorded as `heavy=8` |
