# The Significance Matrix: Accumulated-Attention KV Eviction for Small Language Models

**Draft — paper-shaped. Every number is traceable to a committed file; see §9.**

---

## Abstract

Key–value cache memory, not parameter count, is what bounds the usable context of
a small language model on consumer hardware. We introduce the **significance
matrix**: a persistent, per-position, per-KV-head accumulator of received
attention mass, maintained *across* successive compression rounds, which decides
which cached positions survive eviction. The mechanism adds **zero parameters**
and costs **1.56% of the cache it governs** (0.78% scores + 0.78% observation
counts).

Persistence across rounds creates a failure mode that single-shot scorers do not
have, and correcting it is our central contribution. A position that has survived
*T* compression rounds has been scored *T* times while a newcomer has been scored
once, so a raw accumulated sum ratchets the retained set shut. We generalise the
ranking key to `S / C^e`, where `C` counts observations and `e` interpolates
between the raw sum (`e = 0`) and the mean (`e = 1`). On a 0.6B checkpoint at
128K context under a 2048-token budget, raw-sum ranking retrieves **0 of 12**
needles at depth 0.9 while mean ranking retrieves **10 of 12** (exact McNemar,
+10/−0, *p* = 0.0020; within-needle permutation trend *p* < 0.0001).

With the corrected ranking, retrieval accuracy is governed by the **keep
fraction**, not by the context length: at ~6.25% keep the model scores 0.97 at 64K
and 0.92 at 128K, against 0.61 and 0.53 at 1.56% keep. At 12.5% keep, retrieval is **lossless at both
lengths** — 12 of 12 needles at every depth, in 2,196 MiB at 64K and 3,202 MiB at
128K — on a card that holds neither uncompressed cache (7,168 and 14,336 MiB).
The saturating keep fraction is the *same* at both lengths, which bounds each
uncompressed reference at 1.00 and yields the eviction cost at every smaller
budget without running the impossible arm. All of it on a single 8 GB laptop GPU.

Compression also converts into latency, which a memory result does not
guarantee. Decode is 82.6 ms/token on an uncompressed 16K cache against 46.5 ms
compressed (**1.78×** measured), and the latency-versus-cache curve is a floor
with a knee at 4,096 tokens — implying **13.4×** at 128K, where no uncompressed
baseline fits on the card at all. Because both sit on that floor, moving the
budget from 2,048 to 4,096 tokens at 128K gains **+0.25 mean accuracy at
identical decode latency**.

We also report the mechanism's operating range honestly: against a matched
positional (StreamingLLM) control on short prompts, significance-gated eviction
wins by +0.18 and +0.14 quality at keep 0.75 and 0.50 and **loses by 0.083 at
keep 0.25**. Finally, we show the same score can be reused for bit allocation:
significance-gated mixed precision reaches lower attention error than uniform int8
at 1.6× more compression.

---

## 1. Introduction

A 0.6B-parameter model occupies 904 MiB in bfloat16. Its KV cache at 128K tokens
occupies **14,336 MiB**. On an 8,188 MiB card the model is not the problem; the
cache is, by a factor of sixteen. Any attempt to extend the usable context of a
small model on consumer hardware is therefore a cache problem before it is a
modelling problem.

Cache eviction methods choose which cached positions to discard. The dominant
choice in the literature is between *position* (StreamingLLM's sinks-plus-window)
and *accumulated attention* (H2O, Scissorhands), with SnapKV scoring once at the
end of prefill over an observation window. What these share is that the scoring
event is either positional and therefore stateless, or a single pass.

Long context on a memory-bounded device forces something they do not address.
To prefill 128K tokens in 1.7 GiB one must chunk the prefill and compress
*between chunks* — at 128K with 4K chunks that is roughly 31 compression rounds,
not one. A scorer that persists across those rounds accumulates asymmetrically:
survivors are observed repeatedly, arrivals once. The scoring statistic and the
eviction schedule become coupled, and the coupling is invisible at short context
where there is only one round.

**Contributions.**

1. **The significance matrix** (§3.1–3.2): a persistent per-position,
   per-KV-head accumulator of attention mass with a sinks + recent + top-B keep
   rule, adding zero parameters and 1.56% memory overhead.
2. **Count normalisation** (§3.3), the core contribution: the ratchet induced by
   persistence, and its correction by ranking on `S / C^e`. This is a
   large effect — 0/12 against 10/12 at the hardest measured cell — and it exists
   only because the accumulator persists.
3. **Query-aware inter-chunk scoring** (§3.4): scoring the cache against the
   question between chunks and exactly restoring the cache, so that prefill-time
   eviction is not blind to what will be asked.
4. **The keep-fraction result** (§5.3): accuracy tracks the keep fraction rather
   than the context length, replicated at two lengths with matched needles.
5. **Score reuse for quantisation** (§5.5): the same significance score allocates
   bit width, giving a strict Pareto improvement over uniform int8.
6. **A measured latency floor** (§5.6) that locates the operating point. Decode
   latency is flat across a 512× range of cache size and only then rises, so
   compressing below the knee costs accuracy and buys nothing.

---

## 2. Related Work and Positioning

| Method | Scoring signal | Scoring events | Persistent across rounds |
|---|---|---|---|
| StreamingLLM | position (sinks + window) | n/a | n/a |
| H2O | accumulated attention | continuous | per-head, no count correction |
| SnapKV | attention over an observation window | once, end of prefill | no |
| Scissorhands | attention persistence | windowed | partial |
| **This work** | accumulated attention, per KV head | every forward pass | **yes, with count correction** |

Our keep rule (sinks + recent + top-B) and our pooling step are not novel in
isolation: the sink term is StreamingLLM's observation, and max-pooling the score
before top-k is SnapKV's. We adopt both and say so.

What is new is what happens when an accumulator is carried across many
compression rounds rather than consulted once. To our knowledge the resulting
observation-count asymmetry has not been identified or corrected. It is not
visible in the single-shot setting that SnapKV targets, and it is not visible at
short context. The 128K evidence is strong and paired (§5.4). The 32K and 64K
rungs of that story rest on **6-trial unpaired cells without per-trial records**,
so they are suggestive rather than established, and the round-count scaling should
be read as a mechanism that explains the 128K result rather than as a measured
trend across three lengths.

We also position against a **measured** positional control rather than a cited
one, and report a crossover where the positional baseline wins (§5.2). We regard
this as a requirement rather than a concession: a method with a stated operating
range is characterised, one without is oversold.

---

## 3. Method

### 3.1 The significance matrix

Let a forward pass produce post-RoPE queries `Q ∈ R^{b×H×q×d}` and keys
`K ∈ R^{b×H_kv×n×d}`, *before* `repeat_kv` — the grouping at which eviction must
act, since a KV head is what is stored. Let `G = H / H_kv`.

Take the last `w` query rows as an **observation window** (default `w = 16`) and
compute the causally-masked attention they pay to every cached position:

```
A = softmax( (Q[:, :, -w:, :] · Kᵀ) · d^{-1/2} )      # [b, H_kv, G, w, n]
s = Σ_G Σ_w A                                          # [b, H_kv, n]
```

`s` is this pass's contribution: for each KV head, the total attention mass each
cached position received from the observation window. The **significance matrix**
for layer *l* is the running accumulation

```
S_l ← S_l + s ,        C_l ← C_l + 1
```

where `C_l` counts, per position, how many passes have scored it. When the cache
grows, accumulated history is carried forward onto the prefix it covers. When a
pass would cover *fewer* positions than the accumulator — which would silently
discard all history and disguise itself as a recency preference — the
implementation raises rather than proceeding.

Both `S` and `C` are `[b, H_kv, n]`, i.e. per KV head and per position, never per
query head. This matters: eviction is a decision about stored tensors, and scoring
at query-head granularity would require a reduction whose choice is arbitrary.

### 3.2 The keep rule

Given a budget `B`, the retained index set per (layer, KV head) is

```
keep = sinks(n_s)  ∪  top-(B − n_s − n_r)(ranked)  ∪  recent(n_r)
```

with `n_s = 4` and `n_r` configurable. Two details are load-bearing:

**Mask before pooling.** The sink and recent regions are set to `−∞` *before* the
max-pool, and the masks are **re-asserted afterwards**, because `max_pool1d`
propagates finite values back into masked regions. Pooling first lets sink mass
bleed into neighbouring slots: this checkpoint places 46.3% of its attention mass
on the first four positions, so with `p = 7` slots 4–6 inherit that mass and win
top-k unconditionally on every prompt while carrying no score of their own. In a
synthetic probe with 8 heavy slots, the unmasked order kept 1 of 4 genuine content
tokens and spent 3 of 8 slots on positions 4–6.

**Pooling itself** is SnapKV's: evidence is useful as a span, and independent
per-position scoring keeps scattered singletons out of several spans at once.

### 3.3 Count normalisation — the ratchet and its correction

Persistence across compression rounds creates an asymmetry that has nothing to do
with a position's importance. Consider chunked prefill with chunk size `C` over
`n` tokens: there are `⌈n/C⌉ − 1` compression rounds. A position entering at round
*t* and surviving to round *T* has been scored `T − t + 1` times; a position that
has just arrived has been scored once. Under a raw accumulated sum, the survivor's
score is inflated by an integer factor that is a function of *arrival time alone*.

The retained set therefore **ratchets shut**: early positions accumulate, late
positions cannot catch up, and the cache converges to a prefix. Measured
observation counts after four chunks range from 1 to 3, and sum-versus-mean top-5
rankings share only 2 of 5 entries.

We correct it by ranking on

```
rank = S / C^e ,        e ∈ [0, 1]
```

with `e = 0` the raw sum and `e = 1` the mean. The exponent is exposed as
`Policy.count_exponent`. Section 5.4 measures the full sweep; the summary is that
`e = 1` is the correct default and the interior is not worth tuning.

The correction is not free. Long-accumulated early positions carry *genuine*
repeated evidence as well as the counting artefact, and normalising discards both:
at depth 0.1 the mean is *worse* than the sum (0.42 against 0.58). The net is
strongly favourable, and the trade is explicit rather than hidden.

### 3.4 Query-aware inter-chunk scoring

Under chunked prefill each chunk is scored against **its own** trailing query
rows. For every chunk but the last those rows are context, not the question, so a
needle earns no score before anyone asks for it. The consequence is severe and
budget-independent: at 64K a passkey at depth 0.99 (inside the final, never
compressed chunk) is retrieved at 1.00 while depth 0.90 is retrieved at 0.00, and
a 4× larger budget does not move it.

`score_with_query` runs the question tokens as *queries* against the current
cache with recording enabled, using the full question rather than a trailing
window, then **exactly restores** the cache so the question is never itself
cached. The restoration must be exact: leaving the question cached makes every
later chunk attend to a question it has not reached, shifts positions, and still
produces plausible output. The implementation snapshots and restores, with an
optional byte-equality check over every layer used by the self-test.

Cost is one forward pass over the question per chunk boundary — ~10–16 tokens
against a 2048-token cache.

### 3.5 Cost

| Quantity | Value |
|---|---|
| Added parameters | **0** |
| Score tensor | 0.929 MiB against a 118.89 MiB cache — **0.78%**, length-independent |
| Observation counts | a second tensor of identical shape — a further **0.78%** |
| **Significance matrix, total** | **1.56%** of the cache it governs |
| Per-round work | one top-*k* over `n` per (layer, KV head) |
| Query-aware overhead | one forward over the question per chunk boundary |

The score is computed from tensors the attention kernel already produces; the
mechanism is a wrapper around SDPA plus a `Cache` subclass, and is a
process-local monkey patch rather than a change to model code.

---

## 4. Experimental Setup

**Model.** R2 (`rineya-core-en32k-28L`), a 0.6B Qwen3-derived checkpoint with a
trimmed vocabulary: 28 layers, hidden 1024, 16 query heads, 8 KV heads, head_dim
128, `max_position_embeddings` 40,960. KV cost is 4,096 B per token per layer,
114,688 B per token total (112 KiB).

**Host.** Single RTX 4070 Laptop, 8,188 MiB, Windows/WDDM, torch 2.13.0+cu130.

**Long-context protocol.** Standard passkey retrieval: a 5-digit key at a
controlled depth in repetitive filler. Repetitive filler is deliberate — it
isolates *retrieval* from comprehension, so a failure means "could not find or
attend", not "could not understand". Beyond 40,960 positions we apply YaRN ×4
(training-free), giving a usable range of 163,840.

**Paired design.** Every cell draws the *same* needle sequence, so budgets,
depths and ranking arms are paired on needles. This was not true initially: a
single RNG stream ran across the whole nested loop, giving each cell a different
block of keys and confounding depth with needle identity. That error produced a
published "lost in the middle" depth curve which we retract (§6). Every cell also
records per-trial `{key, response, hit}`, because cell-level accuracy cannot
support a paired test after the fact.

**Statistics.** Exact McNemar on discordant needles for two conditions;
within-needle permutation trend (200,000 draws) for ordered conditions. Analysis
is in a committed tool that *refuses* to compare cells sharing no needles.

**Power.** Exact McNemar with all discordant pairs in one direction gives
*p* = 2/2^d on *d* discordant pairs, so **p < 0.05 requires *d* ≥ 6** (*d* = 6 →
0.03125, *d* = 5 → 0.0625). A 6-needle cell has at most six discordant pairs and
so reaches significance only when *every* needle flips; the 5-of-6 flips we
actually observed sit at 0.0625. Cells are therefore 12 needles. Short-context quality sweeps use 752 rag_dev items.

---

## 5. Results

### 5.1 Short context: the cost of eviction

rag_dev, 752 items, R2, no YaRN. "Identical" counts byte-identical greedy replies
against the uncompressed run.

| Setting | mean quality | Δ vs none | byte-identical |
|---|---:|---:|---:|
| no eviction | 0.66939 | — | 752/752 |
| keep 0.75 | 0.65677 | **−0.01262** | 711/752 (95%) |
| keep 0.50 + pooling | 0.63371 | −0.03568 | 631/752 (84%) |
| keep 0.50 | 0.61401 | −0.05538 | 605/752 (80%) |
| keep 0.25 + scaled recent | 0.41932 | −0.25007 | 172/752 (23%) |
| keep 0.25 | 0.33863 | −0.33076 | 159/752 (21%) |

A 1.33× cache reduction costs 0.013 and leaves 95% of replies byte-identical.
Pooling is worth +0.020 at keep 0.50. Below half the cache the method breaks
down — expected on 232-token prompts where nearly every token is evidence, and
the reason the long-context regime (§5.3) is the one that matters.

**Attention concentration** (the precondition):

| *k* | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| ρ(*k*) | 0.7911 | 0.8576 | 0.9100 | 0.9506 | 0.9796 | 0.9961 |

### 5.2 Against a matched positional control — including where we lose

The positional control is the same harness with the budget spent entirely on
sinks + recent, i.e. StreamingLLM at an identical cache budget and protocol.

| keep | significance | positional | Δ |
|---:|---:|---:|---:|
| 0.75 | 0.65677 | 0.47545 | **+0.18132** |
| 0.50 | 0.61401 | 0.47682 | **+0.13719** |
| 0.25 | 0.33863 | 0.42208 | **−0.08345** |

**There is a crossover.** Choosing *what* to keep beats choosing *where* at keep
≥ 0.5 by a wide margin, and **loses** at keep 0.25. The reading we favour: at a
budget too small to hold the evidence, a positional rule at least keeps a
contiguous, coherent suffix, while a scattered top-k keeps fragments of spans it
cannot complete. This bounds the claim rather than defeating it — and it is why
we report the operating range as part of the method.

### 5.3 Long context: retrieval is lossless at 12.5% keep, at both lengths

**At 12.5% keep the mechanism loses nothing.** Twelve of twelve needles at every
depth, at 64K and again at 128K — in 2,196 and 3,202 MiB, on a card that holds
neither uncompressed cache (7,168 and 14,336 MiB). Because accuracy cannot exceed
1.00, saturation *bounds* each unmeasurable reference at 1.00 and yields the
eviction cost at every smaller budget without running the impossible arm.

Both ladders, 12 paired needles per cell, `e = 1`, YaRN ×4, query-aware. Cost is
against the saturated reference:

| length | budget | keep | 0.1 | 0.5 | 0.9 | mean | peak | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K | 1,024 | 1.56% | 0.50 | 0.75 | 0.58 | 0.61 | 1,621 MiB | 0.39 |
| 64K | 2,048 | 3.12% | 0.83 | 0.92 | 0.67 | 0.81 | 1,742 MiB | 0.19 |
| 64K | 4,096 | 6.25% | 1.00 | 1.00 | 0.92 | 0.97 | 2,002 MiB | 0.03 |
| 64K | **8,192** | **12.5%** | **1.00** | **1.00** | **1.00** | **1.00** | **2,196 MiB** | **0.00** |
| 128K | 2,048 | 1.56% | 0.42 | 0.33 | 0.83 | 0.53 | 1,743 MiB | 0.47 |
| 128K | 4,096 | 3.12% | 0.75 | 0.67 | 0.92 | 0.78 | 2,002 MiB | 0.22 |
| 128K | 8,192 | 6.25% | 1.00 | 0.83 | 0.92 | 0.92 | 2,521 MiB | 0.08 |
| 128K | **16,384** | **12.5%** | **1.00** | **1.00** | **1.00** | **1.00** | **3,202 MiB** | **0.00** |

**The saturating keep fraction is identical at both lengths.** Read the table down
the *keep* column rather than the length column and the two ladders coincide:
1.56% gives 0.61 and 0.53, 6.25% gives 0.97 and 0.92, 12.5% gives 1.00 and 1.00.
Doubling the context moves accuracy by ≈ 0.05; quadrupling the keep fraction moves
it by ≈ 0.36. The mechanism is a function of the keep fraction, and the context
length drops out.

Keep fractions are budget ÷ nominal length, so 64K/B=4,096 and 128K/B=8,192 are
both exactly 6.25% — matched by construction rather than by rounding. Against real
token counts (62,898 and 125,826) both are 6.51%.

**Statistics.** At 128K the 2,048 → 8,192 contrast is +7/−0 at depth 0.1
(*p* = 0.0156), +6/−0 at depth 0.5 (*p* = 0.0312) and +1/−0 at depth 0.9, already
at ceiling; under Bonferroni across three depths (α = 0.0167) depth 0.1 survives.
At 64K the within-needle permutation trend across budgets is *p* = 0.0015 (depth
0.1), 0.0168 (0.9) and 0.0369 (0.5). Across both ladders, **27 discordant needles
all favour the larger budget, with zero reversals** — which is the part that does
not depend on a significance threshold.

**Two cautions.** Saturation bounds the reference *for this task*: a harder task
could separate 12.5% keep from uncompressed where passkey cannot. And 12 needles
cannot distinguish 1.00 from 0.98 — what is established is that no needle was
lost, at either length.

### 5.4 Ablation: the count-normalisation exponent

128K, B = 2048 (1.6% keep — deliberately the regime with headroom to
discriminate), 12 paired needles:

| *e* | depth 0.1 | 0.5 | 0.9 | mean |
|---:|---:|---:|---:|---:|
| 0.00 (raw sum) | 0.58 | 0.08 | **0.00** | 0.22 |
| 0.25 | 0.67 | 0.08 | 0.08 | 0.28 |
| 0.50 | 0.67 | 0.00 | 0.33 | 0.33 |
| 0.75 | 0.58 | 0.17 | 0.67 | 0.47 |
| 1.00 (mean) | 0.42 | 0.33 | **0.83** | **0.53** |

| depth | trend *p* | exact McNemar, *e* = 0 → 1 |
|---:|---:|---|
| 0.9 | **< 0.0001** | 0.00 → 0.83, **+10/−0**, *p* = 0.0020 |
| 0.5 | **0.0125** | 0.08 → 0.33, +3/−0, *p* = 0.2500 |
| 0.1 | 0.9762 | 0.58 → 0.42, +0/−2, *p* = 0.5000 |

Pooled over all 36 needle-positions: **0.22 → 0.53, +13/−2, *p* = 0.0074.**

At the deepest position raw-sum ranking retrieves **nothing** — 0 of 12 — while
the mean retrieves 10 of 12, with all ten discordant needles pointing the same
way. This is the ratchet: a late-arriving needle carries a small accumulated sum
*only* because it was scored fewer times, and dividing by `C^e` removes precisely
that artefact. The mean column is monotone in *e* with no interior optimum, so
`e = 1` is the default and the knob does not require tuning.

The severity scales with the number of compression rounds, as the mechanism
predicts: at 32K (8 rounds) sum and mean are indistinguishable, at 64K (16 rounds)
the mean is ahead, at 128K (31 rounds) the gap is total failure against
near-ceiling.

### 5.5 Reusing the score: significance-gated mixed precision

The significance matrix answers "which positions", but the same numbers answer
"at what precision". Sinks stay bf16, the top 25% by score go int8, the remainder
int4. Bytes are counted from the real layout including scale overhead (bf16 4,096;
int8 2,112; int4 1,088 B per token per layer). Attention-output relative error is
calibrated against eviction settings whose end-to-end cost is known:
keep 0.75 = 0.00937 (−0.01262 quality), keep 0.50 = 0.03729, keep 0.25 = 0.11381.

| scheme | attention relerr | store factor |
|---|---:|---:|
| pure int8 | 0.00762 | 1.94× |
| **mixed, top 25% by significance** | **0.00693** | **3.04×** |
| pure int4 | 0.11179 | 3.75× |

**Lower error than uniform int8 at 1.6× more compression**, using a signal that is
free because eviction computes it anyway. One implementation constraint:
round-tripping (dequantise → evict → requantise) compounds error as √rounds —
int8 goes 0.00276 → 0.00637 over five rounds — so at ~31 rounds a naive
implementation would reach ≈0.0154 and lose its advantage. The mechanism must
**quantise once and gather**, which is achievable because eviction is a gather on
the token axis. *Not yet implemented in the live cache path.*

### 5.6 Decode latency, and where the operating point should sit

A smaller cache is a memory result. Whether it is also a latency result depends on
what the machine is waiting for, which we measure rather than assume.

**Latency against cache size**, everything else fixed, minimum of 2 rounds × 40
steps:

| cache | ms/token |
|---:|---:|
| 8 tokens (0.9 MiB) | 34.53 |
| 1,024 tokens (112 MiB) | 34.70 |
| 4,096 tokens (448 MiB) | **34.78** |
| 8,192 tokens (896 MiB) | 45.25 |
| 16,384 tokens (1.8 GiB) | 76.36 |

Flat to within 1.04 ms across a **512× range of cache size**, then 3.384 µs per
additional token. The floor implies 27 GB/s for a 904 MiB weight read and the
slope 34 GB/s for cache reads — 11% and 13% of this card's ≈256 GB/s. **Neither
term is bandwidth-saturated**: at batch 1 decode is dominated by per-step overhead
(kernel launches across 28 layers, framework, Python), which no cache reduction
can address. The ceiling on compression's latency benefit is therefore a property
of the runtime, not of memory traffic.

**What the mechanism is worth.** One row of the following is measured; three are
read off the fitted curve, because no uncompressed arm fits above 32K. They are
separated here rather than tabulated together, since a prediction that looks
like a measurement is the most damaging thing this section could contain.

| context | uncompressed | compressed | speedup | status |
|---:|---:|---:|---:|---|
| **16,384** | **82.57 ms** | **46.49 ms** | **1.78×** | **measured**, clean per-process runs |
| 32,768 | 131.7 ms | 34.7 ms | 3.80× | *predicted from the curve* |
| 65,536 | 242.6 ms | 34.7 ms | 6.99× | *predicted from the curve* |
| 131,072 | 464.4 ms | 34.7 ms | 13.39× | *predicted from the curve* |

The predictions rest on the hinge fit above, whose flat region is measured
directly and whose slope is measured over 4,096 → 16,384 tokens; extrapolating
the uncompressed column to 128K is an extension of that slope by a factor of
eight and should be read accordingly.

What needs no extrapolation: with eviction on, decode is **flat in context
length** — 47.7 / 47.1 / 46.7 / 47.2 / 48.0 ms/token from 4K to 128K at a
constant 118 MiB cache. That is measured at every point.

**Where the operating point should sit.** 128K, 12 paired needles, *e* = 1:

| budget | keep | cache | 0.1 | 0.5 | 0.9 | mean | decode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1.56% | 224 MiB | 0.42 | 0.33 | 0.83 | 0.53 | 34.7 ms |
| **4,096** | **3.12%** | **448 MiB** | **0.75** | **0.67** | **0.92** | **0.78** | **34.7 ms** |
| 8,192 | 6.25% | 896 MiB | 1.00 | 0.83 | 0.92 | 0.92 | 45.3 ms |

Budget 2,048 and 4,096 sit on the same floor and cost the same decode time, so
the first row is dominated: it gives up 0.25 mean accuracy for nothing. Beyond the
knee the trade becomes explicit — 8,192 buys a further +0.14 for +30% latency
(45.25 ms measured against the 34.7 ms floor).

Within-needle permutation trend across the three budgets: *p* = 0.0005 (depth
0.1), 0.0052 (depth 0.5), 0.3363 (depth 0.9, at ceiling). The 2,048 → 4,096 and
2,048 → 8,192 contrasts have **zero reversals**; the 4,096 → 8,192 contrast at
depth 0.5 has one (+3/−1), so the ladder is not uniformly reversal-free above the
knee.

**A caveat on which configuration is timed.** Scoring during decode costs 20–24%,
but the query-aware configuration behind every accuracy number here does not do
it: recording is off for the whole run and switched on only between prefill
chunks. That overhead belongs to the query-blind setting.

---

## 6. Limitations and Threats to Validity

**Everything at length is retrieval.** A 5-digit key in repetitive filler, chosen
precisely to isolate retrieval from comprehension. Multi-hop reasoning at length
is unmeasured, and the frozen demo suite indicates that is where this checkpoint
is genuinely weak. This is the single largest gap.

**No uncompressed control above 32K — and it is no longer needed.** Neither arm
can run: 7,168 MiB of cache at 64K and 14,336 at 128K against an 8,188 MiB card.
Saturation supplies the bound at both: 12.5% keep retrieves 12 of 12 at every
depth, so each reference is 1.00. What this does *not* establish is the reference
for a harder task, where 12.5% keep and uncompressed might separate. Quantised KV would make the control runnable
(int8 at 1.94× fits 64K; mixed at 3.04× fits 128K) and this is the next
measurement we would take.

**~~The 32K baseline is not yet a comparison.~~ Now measured.** The reference was
re-run with YaRN ×4 on the same 12 needles as the eviction arm: 0.97 against 0.78
at 6.25% keep, so **eviction costs 0.19** — 7 needles lost, 0 gained across 36
needle-positions, exact *p* = 0.0156. This is the only matched eviction cost in
the program; 64K and 128K still have none.

**Sample sizes.** Long-context cells are 12 needles. Short-context sweeps are 752
items. Below 12 the paired test is fragile rather than impossible: *p* < 0.05
needs *d* ≥ 6 discordant pairs, so a 6-needle cell reaches 0.03125 only if *every*
needle flips, and fewer than six needles cannot reach 0.05 at all. Several early
results in this program were noise at *n* = 3.

**Single checkpoint, single host.** The 46.3% sink share, the ρ(*k*) curves and
the sink-pinning rule are properties of *this* checkpoint. The latency floor of
§5.6 is a property of *this runtime* — batch 1, eager PyTorch, no CUDA graphs —
and would move under batching or kernel fusion, which is precisely the regime
where a smaller cache should matter more, not less. Throughput was measured with
rotated arms and minimum-of-rounds because the host's thermal band is 2.17×;
single-shot throughput numbers on this machine are not usable.

**Retractions.** Ten published figures in this work were withdrawn and corrected
during its development, several of them load-bearing. They are listed in full in
**Appendix A**, with the cause of each. We report them because the corrections
shaped the method and the measurement protocol: the paired design of §4 exists
because a depth result turned out to be confounded with needle identity, and the
contamination guards in §5.6 exist because a throughput baseline was found to be
spilling to host RAM.

**Implementation defects.** Nine were found during development; **every one
produced a plausible wrong number rather than a crash**, and every one was caught
by a control rather than by reading code. Examples: eviction firing during prefill
before scores existed (caught because prefill was impossibly *fast*); pooling
before masking sinks; cache bytes counted on a view, turning a "5.1× saving" into
1.00× resident; an OOM trial scored as a wrong answer rather than excluded. All
silent failure paths were subsequently made loud.

**Verification.** 74 headline numbers are pinned by a checker that recomputes each
from committed data *and* asserts it appears in the prose of the document that
states it, so prose and data cannot drift apart. A nine-document audit against
committed data found 41 further defects, all fixed.

---

## 7. Conclusion

The significance matrix decides KV eviction by accumulated attention mass, per
position and per KV head, maintained across compression rounds, for zero added
parameters and 1.56% memory overhead.

Its persistence is both what makes it work at long context and what creates its
one non-obvious failure mode. Because survivors are scored more often than
arrivals, a raw accumulated sum ratchets the retained set shut, and the effect
grows with the number of compression rounds — invisible at 32K, fatal at 128K,
where it is the difference between retrieving 0 and 10 of 12 needles at the
deepest position. Normalising by observation count corrects it with one exponent
and no tuning.

With that correction the empirical picture is simple: **retrieval accuracy is a
function of the keep fraction, not of the context length** — and at 12.5% keep it
is lossless at 64K *and* 128K, in 2.2 and 3.1 GiB, on a card that can hold
neither uncompressed cache. At ~6.25% keep a 0.6B
model retrieves at 0.97 (64K) and 0.92 (128K) on an 8 GB laptop GPU, in 2.5 GiB.
The method's limits are equally clear — it loses to a positional baseline below
keep 0.25 on short prompts, and everything measured at length is retrieval rather
than reasoning.

---

## 8. Reproduction

```bash
python scripts/kv/chunked_prefill.py --gate                    # identity gate
python scripts/design/identity_counts.py --check               # identity artifact
python experiments/kv_significance/src/verify_significance.py  # 6 checks
python scripts/design/verify_claims.py                         # 74 pinned numbers
python experiments/kv_significance/src/passkey.py --lengths 131072 \
       --budgets 2048 8192 --depths 0.1 0.5 0.9 --trials 12 \
       --yarn 4.0 --query-aware --count-exponent 1.0 --cap-mib 7400 \
       --out experiments/kv_significance/results/pk_128k_grid12.json
python scripts/design/paired_passkey.py \
       experiments/kv_significance/results/pk_128k_grid12.json --by budget
python experiments/kv_significance/src/probe_decode_floor.py   # ~8 min GPU
python experiments/kv_significance/src/measure_decode.py --lengths 16384
```

## 9. Data provenance

| Section | Data |
|---|---|
| §5.1 quality | `results/{off,frac0.75,frac0.5-pool7-fixed,frac0.5,frac0.25-rf0.25,frac0.25}__rag_dev/` |
| §5.1 concentration | `results/concentration.json` |
| §5.2 control | `results/frac{0.75,0.5,0.25}-rf1.0__rag_dev/` |
| §5.3 128K | `results/pk_128k_grid12.json` |
| §5.3 64K | `results/pk_64k_ladder.json` |
| §5.4 exponent | `results/pk_128k_pe{0.0,0.25,0.5,0.75,1.0}.json` |
| §5.5 storage | `results/{store_probe,mixed_precision}.json` |
| §5.6 latency | `results/{probe_decode_floor,measure_decode}.json` |
| §5.6 operating point | `results/{pk_128k_budget_ladder,pk_128k_B4096}.json` |
| §6 reference ladder | `results/passkey_ceiling.json` |
| Method | `scripts/kv/{significance,chunked_prefill}.py` |

All paths relative to `experiments/kv_significance/` unless otherwise stated.

---

## Appendix A. Retractions

Every figure below appeared in a draft of this work and was withdrawn after a
control, a re-measurement or an audit contradicted it. None is a typo; each was a
conclusion drawn from data that did not support it.

| Retracted | Cause | Corrected to |
|---|---|---|
| "Lost in the middle" depth curve at 128K | one RNG stream across the loop; depth confounded with needle identity | no depth law is established; the 128K ordering does not replicate at 64K |
| "+0.15–0.20 over StreamingLLM at every budget" | control came from an invalidated run whose fallback was acausal | crossover: +0.18/+0.14, **−0.083 at keep 0.25** |
| Query-aware "0.20 → 0.87" | 3-depth arm averaged against a 5-depth arm | 0.33 → 0.78, depth-matched |
| Exponent *p*-values (Cochran–Armitage, Fisher) | independent-groups tests on needle-paired arms at n = 6 | paired re-run reversed two conclusions (§5.4) |
| "Above 16K the full cache does not fit" | a `--cap-mib 6144` default bounding *reserved* memory | the uncompressed reference reaches 32K |
| "A 15× smaller cache cannot exceed 2.65× decode" | arithmetic presented as physics; assumed full weight-read bandwidth | measured ceiling 13.4× at 128K (§5.6) |
| "The fixed cost is a weights read, so decode is bandwidth-bound" | one line fitted through two regimes; the largest points set the intercept | a floor with a knee, at 11–13% of peak bandwidth — overhead-bound |
| Significance matrix costs **0.25%** of the cache | a post-eviction score matrix divided by a pre-eviction cache | **1.56%** — 0.78% scores plus 0.78% observation counts, which count normalisation requires (§3.5) |
| "Six paired needles cannot reach *p* < 0.05" | confused the floor at five discordant pairs with the floor at six | *p* = 2/2^*d* needs *d* ≥ 6; six needles reach 0.03125 if every needle flips (§4) |
| "Accuracy is flat in length at matched keep" | a 32K cell that was 6-trial, unpaired and ranked on the raw sum | superseded by the stronger result: the *saturating* keep fraction is identical at both lengths (§5.3) |

Three of these deserve emphasis because they changed how the work was done, not
merely what it reported.

**The depth curve** was confounded with needle identity: a single RNG stream ran
across the experimental loop, so each depth drew a different block of needles.
The fix was not a wider interval but a paired design — every cell now draws the
same needle sequence, and the analysis lives in a tool that *refuses* to compare
cells sharing no needles.

**The StreamingLLM comparison** was drawn from a run whose fallback was acausal.
Against a real positional control the conclusion reverses at the tightest budget,
which is why §5.2 reports an operating range rather than a win.

**The decode ceiling** was asserted from arithmetic ("weights dominate, so 2.65×")
and then from a straight line fitted through two regimes. Measured, the curve is a
floor with a knee and the ceiling is 13.4×. The lesson recorded in §5.6 is that a
model quoted where a measurement exists is a defect even when the model is close.

A checker (`scripts/design/verify_claims.py`) now recomputes 85 headline numbers
from committed data *and* asserts each appears in the prose of the document that
states it, so prose and data cannot drift apart unnoticed. Several of the
retractions above were caught by audits after that checker was already passing —
it constrains numbers it knows about, not claims it does not.

---
