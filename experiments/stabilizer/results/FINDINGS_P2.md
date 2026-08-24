# P2 — the floor, and what the mechanism actually loses when it fails

Two questions the first cornerstone could not answer, because it measured only at
keep *fractions* and never below 1.56%: how far down does this go, and when it
finally breaks, what breaks?

All cells: `pool=7`, `count_exponent=1.0`, `recent=budget//4`, query-aware,
YaRN ×4, chunk 4096, 12 needles × 3 depths.

## The full curve

| length | B=16 | B=32 | B=64 | B=128 | B=256 | B=512 |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 0.028 | 0.389 | 0.833 | **1.000** | 1.000 | 1.000 |
| 64K | 0.000 | 0.000 | 0.806 | 0.972 | **1.000** | 1.000 |
| 128K | 0.000 | 0.000 | 0.278 | 0.972 | **1.000** | 1.000 |

Peak memory across the entire table spans 1,500–1,563 MiB. Every cell fits in
19% of an 8 GiB card, and the largest lossless compression is **512× at 128K**
(B=256, 0.195% keep).

## The budget required grows as roughly the SQUARE ROOT of context

Interpolating the 0.5-accuracy crossing in log₂(B):

| length | B_half |
|---:|---:|
| 32K | 38.1 |
| 64K | 49.2 |
| 128K | 79.9 |

A least-squares fit gives **B_half = 0.141 · L^0.535**.

> **RETRACTED by P2.3, below.** This law was fitted from the *centred* pool window.
> With a backward-reaching window every length is lossless at or below B=16, so
> B_half falls outside the sampled range at all three lengths and no exponent can be
> fitted at all. What survives is the negative half — the linear keep-fraction law is
> excluded. It is left here as the record of what was measured, not as a live claim.

The comparison that matters is against the alternative the first cornerstone
implied. A fixed keep fraction means B_half doubles when L doubles: from 38 at
32K it would predict 76 at 64K and 152 at 128K. Observed: 49 and 80. At 128K the
linear law is wrong by a factor of ~1.9, and B=128 — below even the linear law's
half-accuracy point — already scores 0.972.

So the compression ratio available is not constant and not linear in L; it grows
about as √L. **Doubling the context buys roughly √2 more compression.**

Stated honestly: three points, each interpolated between budgets spaced 2× apart,
12 needles per point. The exponent 0.535 carries wide uncertainty and 0.5 is well
inside it. What the data does exclude is the linear law.

## What breaks is one token, and it is always the same token

The tokeniser gives every digit its own token, so a five-digit key occupies
exactly five consecutive positions:

```
'The',' pass',' key',' is',' ','7','5','5','2','6','.',' Remember',' it','.'
```

Among the failures at B=32–128, across all three lengths:

| length | B | failures | mean digits recovered | zero digits |
|---:|---:|---:|---:|---:|
| 32K | 32 | 22 | 4.00 / 5 | 0% |
| 32K | 64 | 6 | 4.00 / 5 | 0% |
| 64K | 32 | 36 | 3.89 / 5 | 0% |
| 64K | 64 | 7 | 4.00 / 5 | 0% |
| 64K | 128 | 1 | 4.00 / 5 | 0% |
| 128K | 64 | 26 | 4.00 / 5 | 0% |
| 128K | 128 | 1 | 4.00 / 5 | 0% |

**Exactly four of five digits, and never zero.** The model emits `6644` for
`66445`, `4264` for `42649`, `9987` for `99873`. At 128K/B=64, 26 of 26 failures
are correct prefixes.

This is not "the needle was evicted" and not "the model could not find it". The
model localises the needle perfectly and reproduces its span verbatim — including
the following `. Remember it` — while missing precisely the **final token of the
key**. At the transition budget the mechanism is losing one position, and it is
the same position every time.

Only at B=16, far below the floor, does this degenerate: 1.9–2.7 digits recovered,
i.e. the span is genuinely being shredded rather than clipped.

## Why this matters, and the experiment it implies

The first cornerstone found that eviction ranks positions independently and
therefore fragments spans; pooling fixed most of it. This says the *residual*
failure after pooling is not diffuse fragmentation but a systematic loss of the
span's trailing edge.

That is a much sharper target than "add budget", and it is testable: a max-pool
window centred on position i draws support from i±k, so a span's interior is
covered from both sides while its final position is supported on one side only.
If that is the cause, a trailing-biased or asymmetric pool should recover the
last token at no extra budget. If it is not — if the last digit simply accumulates
less attention during prefill because fewer subsequent queries attend to it — then
the fix belongs in scoring, not pooling.

Neither explanation has been tested. Both are cheap.

## Reproduce

```bash
python experiments/kv_significance/src/passkey.py --lengths 131072 \
  --budgets 16 32 64 128 256 512 --pool 7 --count-exponent 1.0 \
  --query-aware --yarn 4.0 --trials 12 --cap-mib 7400
```

## A caveat that applies to every accuracy in this file

Each cell draws the SAME 12 keys at all three depths, so depths are paired on
needles. A cell is 12 independent needles observed three times, not 36. A perfect
cell supports a one-sided 95% Clopper–Pearson lower bound of **0.779**, not the
0.903 an n=36 reading would give. Reaching 0.97 requires 99 needles per depth and
0.99 requires 299 — about 100 minutes per cell at 128K on this card. The word
"lossless" in this file means "no failure in 12 needles", and should be read that
way until that run is done.

---

# P2.2 — the mechanism keeps the label and discards the payload

`src/trace_span_positions.py`. Original position identities carried through every
gather; after the final compression, per-token survival inside the needle across
all 28×8 = 224 (layer, KV head) streams. 128K, pool=7, e=1, recent=B/4,
query-aware, key 75526, six cells (depths 0.1/0.5/0.9 × B=64/128). The prompt is
built by importing `passkey.build_prompt`, so it cannot drift from the harness.

## Both pre-registered explanations are refuted

| prediction | expected | observed | |
|---|---|---|---|
| tie-break at the budget boundary | survival ≈ equal across the span | 19-point gap | **refuted** |
| scoring gradient along the span | survival falls toward the span's end | offsets 20–25 rise to 62–70% | **refuted** |

The shape is neither flat nor a ramp. It is a **trough**.

## What actually happens

| depth | B | framing tokens | digit tokens | gap |
|---:|---:|---:|---:|---:|
| 0.1 | 64 | 41% | 25% | +16 |
| 0.1 | 128 | 55% | 41% | +14 |
| 0.5 | 64 | 45% | 23% | +22 |
| 0.5 | 128 | 60% | 42% | +18 |
| 0.9 | 64 | 57% | 31% | +26 |
| 0.9 | 128 | 66% | 50% | +16 |
| | | **54%** | **35%** | **+19** |

Six cells out of six, in the same direction. The words `The pass key is` survive
at 54%; the digits they introduce survive at 35%.

**The reason is the query.** Query-aware scoring runs `What is the pass key? The
pass key is` against the cache before compressing. The framing tokens are
lexically the query. The digits are what the query is *asking for*, and they look
nothing like it. So the score rewards the label and not the answer.

This is worth stating plainly: query-aware scoring produced the largest single
quality gain in this program (0.33 → 0.78 at 32K), and it carries a systematic
bias toward evicting the very thing being retrieved.

## Pooling works by contagion, which is why payload interiors starve

Within the five-digit run, survival is U-shaped in **all six cells**:

| depth | B | d1 | d2 | d3 | d4 | d5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 64 | 34% | **22%** | 31% | 30% | 28% |
| 0.5 | 128 | 52% | **44%** | 46% | 48% | 49% |
| 0.9 | 128 | 59% | **49%** | 56% | 55% | 54% |

d1 is adjacent to the framing and does best; d2 is interior and does worst. The
same effect appears at offset 19 — the last digit of the *second* key copy, sitting
next to ` is the pass key` — which jumps back to 41–56% while its interior
neighbours sit at 15–29%.

So max-pooling does not protect spans as such. It propagates high scores to
*neighbours of high scorers*. A payload whose neighbours are all payload receives
nothing. That is why pooling rescued the 128K result (the span gets lifted by its
framing) and why it still leaves a hole in the middle of the payload.

## The fix this points at

Not a wider pool — width does not help a token whose entire neighbourhood is low.
Two candidates, both cheap and both untested:

1. **All-or-nothing spans.** If any token of a contiguous run is kept, keep the
   run. Turns eviction's unit from the position into the span, which is what the
   whole cornerstone argues it should be.
2. **Forward-biased query credit.** The query matches the *label*; the answer sits
   immediately after it. Crediting positions i+1…i+k when i matches the query is a
   one-line change to the pooling direction and targets the observed bias exactly.

## Limitations

One key (75526), one length, one pool width, six cells. Survival is measured per
(layer, KV head) stream, and a digit present in half the streams may still be
readable — so this trace establishes the *eviction preference* cleanly but does
not by itself explain the 4-of-5 prefix pattern in P2. The single-token decode in
this tool reports the space preceding the digits, so it cannot cross-check the
harness's accuracy; use `passkey.py` for that.

## Reproduce

```bash
python experiments/stabilizer/src/trace_span_positions.py \
  --length 131072 --budgets 64 128 --depths 0.1 0.5 0.9
```

---

# P2.3 — the fix P2.2 predicted, and it holds at every length

`--pool-back` / `--pool-fwd` on `passkey.py`, backed by `Policy.pool_window()`. The
score pool was a centred window; it is now an arbitrary `(back, fwd)` reach, with
the centred window as the default. The refactor is behaviour-preserving: the new
padding path is bit-identical to `max_pool1d(kernel=p, padding=p//2)` for p in
{3,7,15,31} across n in {64,257,1000}, and the `(3,3)` control arm reproduced the
committed baseline's per-trial hit vector **exactly** (10/36 at 128K/B=64).

## The result

Paired on the same 12 keys, exact McNemar on discordant needles:

| length | B | compression | centred (3,3) | causal (6,0) | gain | 0→1 | 1→0 | p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 16 | 1,966× | 0.028 | **1.000** | +0.972 | 35 | 0 | <10⁻⁵ |
| 32K | 32 | 983× | 0.389 | **1.000** | +0.611 | 22 | 0 | <10⁻⁵ |
| 64K | 16 | 3,931× | 0.000 | 0.972 | +0.972 | 35 | 0 | <10⁻⁵ |
| 64K | 32 | 1,966× | 0.000 | **1.000** | +1.000 | 36 | 0 | <10⁻⁵ |
| 128K | 16 | **7,864×** | 0.000 | **1.000** | +1.000 | 36 | 0 | <10⁻⁵ |
| 128K | 32 | 3,932× | 0.000 | **1.000** | +1.000 | 36 | 0 | <10⁻⁵ |
| 128K | 64 | 1,966× | 0.278 | **1.000** | +0.722 | 26 | 0 | <10⁻⁵ |

**216 discordant pairs, every one in the same direction; not a single needle that
the centred window retrieved and the causal window lost.** Pooled 251/252. Peak
memory 1,500–1,507 MiB across the whole table — the gain costs nothing.

The failure text confirms the mechanism end to end. At 128K/B=32 the centred window
emits `755`, `856`, `998`, `4264` for keys `75526`, `85667`, `99873`, `42649` — the
4-of-5 truncation P2 documented. The causal window emits the keys.

**P2.2 diagnosed the failure, predicted this repair, and the repair worked.** That
chain is stronger evidence for the span account than any single accuracy number.

## Two things this does NOT establish

**It is not shown to be the causal shape.** A centred `(7,7)` window — width 15, no
causal bias — also scores 1.000 at 128K/B=64. So "reach backward to the label" and
"reach further in any direction" are confounded. P2.4 tests matched backward reach
with different forward reach to separate them. Until it reports, the honest claim is
*a wider or backward-reaching pool window*, not *a causal one*.

**The floor is now the policy's, not the task's.** At B=16 the budget is 4 sinks +
4 recent + **8 heavy slots**. At B=8 the recent window and sinks consume everything,
`heavy()` reaches 0, and the policy degenerates to StreamingLLM with no significance
ranking at all. So B=16 is close to the smallest budget at which this mechanism can
operate, and the ladder cannot be pushed much lower without changing the policy's
own overhead. The 7,864× figure is a floor of the *implementation*, not a measured
information limit.

## And the √L law from P2 no longer applies

P2 fitted B_half = 0.141·L^0.535 from the centred window. With the causal window
every length is lossless at or below B=16, so B_half is below the sampled range at
32K, 64K and 128K alike and no exponent can be fitted. The sub-linear law described
the *centred* mechanism. It should not be carried forward as a property of the
significance matrix.

## Limitations

Twelve independent needles per cell (the three depths share keys), so a perfect cell
supports a 95% Clopper–Pearson lower bound of 0.779. The *comparisons* above are
solid — 216 discordant pairs is overwhelming — but "lossless" remains "no failure in
12". One task, one span type, one key format, homogeneous filler. The B=16 result in
particular leans on a haystack with exactly one distinctive region; with 8 heavy
slots, a task carrying several plausible distractors cannot possibly behave this way.

## Reproduce

```bash
python experiments/kv_significance/src/passkey.py --lengths 131072 --budgets 16 \
  --recent 4 --pool 7 --pool-back 6 --pool-fwd 0 --trials 12 \
  --count-exponent 1.0 --yarn 4.0 --query-aware --cap-mib 7400
```

---

# P2.4 — it is not the shape and not the width: it is backward reach alone

P2.3 left a confound. Two arms both rescued 128K/B=64: causal `(6,0)` at width 7 and
centred `(7,7)` at width 15. "Reach backward" and "be wider" could not be separated.

Matched-backward-reach pairs, 128K, B=64, 12 paired needles per cell:

| back | fwd | width | d0.1 | d0.5 | d0.9 | acc |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 3 | 0.08 | 0.00 | 0.00 | 0.028 |
| 2 | 2 | 5 | 0.08 | 0.17 | 0.17 | 0.139 |
| 3 | 3 | 7 | 0.42 | 0.00 | 0.42 | 0.278 |
| **4** | **0** | **5** | 1.00 | 1.00 | 1.00 | **1.000** |
| **4** | **4** | **9** | 1.00 | 1.00 | 1.00 | **1.000** |
| 5 | 0 | 6 | 1.00 | 0.92 | 1.00 | 0.972 |
| 5 | 1 | 7 | 1.00 | 1.00 | 1.00 | 1.000 |
| 5 | 5 | 11 | 1.00 | 1.00 | 1.00 | 1.000 |
| 6 | 0 | 7 | 1.00 | 1.00 | 1.00 | 1.000 |
| 7 | 7 | 15 | 1.00 | 1.00 | 1.00 | 1.000 |
| 12 | 2 | 15 | 1.00 | 1.00 | 1.00 | 1.000 |

**Forward reach is inert.** At back=4 the spread across fwd ∈ {0,4} is **0.000**; at
back=5 across fwd ∈ {0,1,5} it is 0.028 (one needle, p = 1.0); at back=2 it is 0.111
(p = 0.125). No matched pair differs.

**Width is not the variable.** `(4,0)` is width 5 and scores 1.000. `(3,3)` is width 7
and scores 0.278. The wider window is worse.

**The threshold is sharp and it is in backward reach:** 3 → 0.278, 4 → 1.000.

So the centred `(7,7)` window did not win by being wide. It won because its backward
reach happened to be 7.

## Why backward, mechanistically

P2.2 measured that survival within the digit run is U-shaped, highest at the digit
*adjacent to the framing*. The score enters an evidence span at its **head**, because
query-aware scoring rewards the query-shaped label and the label precedes the payload.
Max-pooling propagates a score to neighbours. Therefore every payload token must be
able to look **back** to the head. Forward reach points away from the score and buys
nothing — which is exactly what the table shows.

## The law this implies, and the test now running

Every digit is one token (verified: `75526` → `'7','5','5','2','6'`), so a *d*-digit
key is a *d*-token span. If the payload must reach its head, the requirement is

> **back ≥ d − 1**  
> **RETRACTED 2026-08-24 — see O2 in `docs/OPEN_QUESTIONS.md`.** Extending d=7
> past the threshold gives 0.778 at back=6, 0.944 at back=7 and **1.000 only at
> back=8**. This form was fitted on d=4 and d=5 and does not extrapolate. What
> survives is that the requirement TRACKS THE SPAN and is not a constant, which
> is what the mechanism rests on.

which predicts a threshold at back=3 for a 4-digit key, back=4 for 5 digits (measured:
3 fails, 4 works), and back=6 for 7 digits. P2.5 tests it with `--key-digits`. The
discriminating cell is **a 7-digit key at back=4**: the law says it must fail. If it
passes, the law is wrong and 4 is merely a constant that suited a 5-digit key.

This matters well beyond passkey. If the requirement tracks span length, then the pool
window is not a hyperparameter to tune but a **quantity to derive from the evidence
structure of the task** — and for natural-language evidence, spans of 20–40 tokens
would need a backward reach an order of magnitude larger than anything measured here.

---

# P2.5 — the required reach tracks the span, so it cannot be a constant

`--key-digits` on `passkey.py`. Every digit is one token (verified:
`75526` → `'7','5','5','2','6'`), so a *d*-digit key is a *d*-token evidence span.
P2.4's threshold sat at back=4 for d=5; the law `back ≥ d − 1` predicts it moves
with *d*. 128K, B=64, 12 paired needles per cell.

| digits | back | d−1 | predicted | measured | agrees |
|---:|---:|---:|---|---:|---|
| 4 | 2 | 3 | fail | 0.444 | yes |
| 4 | 3 | 3 | pass | 0.972 | yes |
| 5 | 3 | 4 | fail | 0.278 | yes |
| 5 | 4 | 4 | pass | **1.000** | yes |
| **7** | **4** | 6 | **fail** | **0.000** | **yes** |
| 7 | 6 | 6 | pass | 0.778 | no |

**The discriminating cell settles it.** A 7-digit key at back=4 scores **0.000**
where a 5-digit key at the same reach scores **1.000**. Backward reach is not a
constant that happened to suit passkey; the requirement moves with the span.

**But `d − 1` is approximate, not exact.** At the threshold itself the margin
narrows as spans lengthen: 0.972 at d=4, 1.000 at d=5, **0.778** at d=7. The d=7
failures are again one-token truncations (`9387353` → `938735`), the same signature
P2 recorded. So reach ≥ d−1 is necessary and, for short spans, sufficient; for
longer spans something further degrades it. Not yet explained.

## Why this matters more than the accuracy numbers

A fixed reach is a hyperparameter that can only be tuned when the evidence span
length is known in advance. Passkey grants that. **No real task does.** Natural
language evidence runs 20–40 tokens and varies per instance, so any single fixed
reach is wrong for most of the distribution — and being wrong is not graceful: d=7
at back=4 is 0.000, not degraded.

---

# P2.6 — deriving the reach instead of tuning it (in progress)

`grow_spans()` in `scripts/kv/significance.py`, opt-in via `Policy.grow`, exposed as
`--grow`. Geodesic dilation: the score enters a span at its head and is grown forward
through positions that are themselves above the stream's noise floor, so it stops at
the span boundary rather than at a fixed distance. `grow=None` leaves every committed
result on the unchanged fixed-window path, and the 93 pinned numbers still reproduce.

**Two design errors, both caught before the card.** `test_grow_spans.py` runs on CPU
with no model.

*Gating at a fraction of the stream maximum* presumes a known head-to-payload ratio —
the exact prior knowledge the function exists to avoid. Head 1.0, payload 0.15, any
threshold above 0.15 makes growth a silent no-op.

*Gating on a quantile* fails the other way. A quantile bounds how many positions may
be grown into, but when signal is sparse the k-th largest score **is** the filler
level: on a 200-position stream holding one 5-token span, a 0.95 quantile admitted
all 200 positions and growth flooded the sequence.

**And one error the CPU tests did not catch.** Requiring a position to be in the
top-k *to be grown into* inverts hysteresis, which needs a selective seed and a
**permissive** gate. A payload position is individually weak — that is the entire
problem — so the top-k requirement excluded precisely the positions the mechanism
exists to rescue. It scored 0.00 at 16K/B=64 and emitted degenerate `77777777777`.
Caught by a 2-trial 16K smoke run costing about a minute, not by a 128K sweep.

The gate is now relative to each stream's **median**, which tracks the noise floor
because filler dominates. After the fix, 16K/B=64 scores 1.00 at both `grow=2.0` and
`grow=5.0`, matching the fixed `back=4` reference — and the insensitivity to the
multiplier is the point: the knob is not doing the work the span structure is.

## The result: deriving beats tuning, but does not remove the knob

One policy, never told the key length, against the best *tuned* fixed reach:

| digits | grow=2.0 | grow=5.0 | best fixed reach | grow=2.0 vs fixed |
|---:|---:|---:|---:|---:|
| 4 | **1.000** | 1.000 | back=3 → 0.972 | **+0.028** |
| 5 | **1.000** | 1.000 | back=4 → 1.000 | +0.000 |
| 7 | **0.889** | 0.611 | back=6 → 0.778 | **+0.111** |

At `grow=2.0` the derived reach **matches or beats the best hand-tuned fixed reach at
every span length**, without being told the span length. That is the claim this
mechanism was built for.

**But the multiplier is not inert, and the earlier claim that it was is withdrawn.**
Exact McNemar between the two multipliers:

| digits | 2.0 vs 5.0 | p | |
|---:|---:|---:|---|
| 4 | +0.000 | 1.0000 | inert |
| 5 | +0.000 | 1.0000 | inert |
| **7** | **+0.278** | **0.0020** | **the multiplier matters** |

The insensitivity claim came from a 16K smoke run at d=5 — a cell where both settings
saturate, and therefore the wrong cell to test it on. At d=7 they differ by 0.278.

Mechanically this is what the design predicts: a stricter gate excludes weak payload
positions, so growth cannot traverse a long span. The failure direction is therefore
known — too strict, not too loose — and the setting should err low.

So the honest claim is weaker than *hyperparameter-free*. Growth **reduces** the
tuning problem: one setting spans d ∈ {4, 5, 7} where no fixed reach does, and it is
never worse than the best fixed reach. It does not eliminate it.

The residual long-span degradation from P2.5 also survives: d=7 reaches 0.889, not
1.000, and remains unexplained.

---

# Stage 1 — retrieval under competition: P1 confirmed, its mechanism refuted

The first measurement outside passkey's comfort zone. `--distractors k` adds *k* decoy
keys; `--distractor-style` sets whether they are shaped like the question. Both styles
draw the SAME decoy keys from one seed, so the contrast is paired on decoy identity and
isolates query similarity from mere numeric competition. 128K, B=64, `grow=2.0`,
5-digit keys, 12 paired needles per cell.

| k | bare | framed | difference | exact McNemar |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | — | — |
| 1 | 1.000 | 1.000 | +0.000 | 1.0000 |
| 4 | 0.944 | 0.889 | −0.056 | 0.6875 |
| **16** | **0.972** | **0.611** | **−0.361** | **0.0010** |

**Prediction P1 is confirmed.** It was pre-registered in `docs/PLAN_LONG_REASONING.md`
before any of this ran, and it follows from P2.2's measurement that the score keeps text
resembling the query. Sixteen *neutral* distractors cost almost nothing (0.972). Sixteen
*query-shaped* distractors cost 0.389. Same count, same keys, same budget — the only
difference is whether the decoy wears the question's sentence frame.

Query similarity, not numeric competition, is the operative variable.

## But the mechanism P1 assumed is wrong

P1 predicted **crowding out**: decoys preferentially retained, so the model returns a
decoy's key. Classifying all 21 failures by what was actually emitted
(`scripts/design/distractor_failures.py`, which regenerates decoy keys from the seed):

| failure mode | count | share |
|---|---:|---:|
| fragmented true span (truncated or corrupted) | 11 | **52%** |
| crowded out by a decoy, returned whole | 3 | 14% |
| other (degenerate repetition, e.g. `75526` → `555222`) | 7 | 33% |

The model does not answer with a decoy. It answers with a **damaged version of the true
key** — `70176` → `7017`, `85667` → `65667` — the same signature P2 recorded.

So the causal path is not substitution but **budget consumption**: query-shaped decoys
are preserved *because* they look like the query, which leaves too few of the 44 heavy
slots to hold the true span intact, and it returns fragmented. The score still finds the
right span; it can no longer afford to keep all of it.

That distinction decides where the fix belongs. Substitution would be a **scoring**
problem. This is a **capacity** problem created by a scoring bias — so the repair is
either to stop rewarding query-shaped text that is not evidence, or to make span
retention all-or-nothing so a partially affordable span is not kept at all.

## Depth is untouched at 0.9

At k=16 framed: 0.42 / 0.42 / **1.00** at depths 0.1 / 0.5 / 0.9. Late needles enter in
a late chunk and survive few compression rounds, so they never face the competition. The
same asymmetry the program measured for query-blind scoring.

## What this costs the headline

At B=64 with 16 query-shaped competitors, accuracy is 0.611 — against 1.000 with none.
`docs/PLAN_LONG_REASONING.md` stated in advance that compression would fall sharply once
distractors were present, and why: at these budgets there is no room for many spans. This
is that prediction landing, and it is the correct result rather than a regression.

The 7,864× figure remains true of passkey and should never be quoted without its
condition: **a haystack containing no competitors at all.**

## The budget control: capacity confirmed, and what competition costs

Stage 1's collapse was read as capacity — decoys consuming slots — rather than the
score being fooled. That reading predicts budget alone should undo it. It does.

128K, 16 query-shaped decoys, `grow=2.0`, everything else held:

| B | heavy slots | d0.1 | d0.5 | d0.9 | accuracy | peak |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 44 | 0.42 | 0.42 | 1.00 | 0.611 | 1,507 MiB |
| 128 | 92 | 0.42 | 1.00 | 1.00 | 0.806 | 1,516 MiB |
| 256 | 188 | 0.83 | 1.00 | 1.00 | 0.944 | 1,537 MiB |
| **512** | **380** | **1.00** | **1.00** | **1.00** | **1.000** | 1,563 MiB |

Monotonic, and fully restored. The score was never choosing the wrong span — it could
not afford the right one. **Competition is a capacity cost, not a scoring failure.**

**Early positions pay first.** Depth 0.9 is lossless at every budget; 0.5 recovers at
B=128; 0.1 needs B=512. A late needle enters in a late chunk and survives few
compression rounds, so it never faces the competition. The deficit is ordered by how
many rounds a position must survive, which is the same asymmetry the program measured
for query-blind scoring.

**What competition costs, as a number.** Lossless at 128K needs B=16 with no
competitors and **B=512 with sixteen** — a **32× loss of compression**, from 7,864×
down to **246×**. Seventeen spans at ~22 tokens each is ~374 tokens, against the 380
heavy slots B=512 provides, so the requirement tracks *total evidence mass* rather
than anything subtler. One point on that curve is not a law, but the arithmetic is
close enough to be worth testing directly.

**This is the honest external-validity figure.** `docs/PLAN_LONG_REASONING.md`
predicted in advance that compression would fall sharply once competitors existed, and
by roughly an order of magnitude. Measured: 32×. The 7,864× headline is a property of
a haystack with **no competitors at all**, and 246× is what the same mechanism delivers
when sixteen plausible ones are present — still a large number, and a very different
one.

---

# The baselines: the score is doing the work

Every comparison in this document until now put the mechanism against an earlier
version of itself — centred against causal, fixed reach against derived, one multiplier
against another. That establishes which *variant* wins. It says nothing about whether
the significance score does any work at all, which is the question a practicability
claim actually rests on. Thirty-eight causal and derived-reach result files existed
before either of these arms ran.

128K, B=16 — eight heavy slots after sinks and recent — identical budget, seed, and the
same twelve needles at every depth:

| arm | d0.1 | d0.5 | d0.9 | accuracy |
|---|---:|---:|---:|---:|
| significance + causal reach | 1.00 | 1.00 | 1.00 | **36/36** |
| **random**, same number of slots | 0.00 | 0.00 | 0.00 | **0/36** |
| **streaming**, sinks + recent only | 0.00 | 0.00 | 0.00 | **0/36** |

Both contrasts: **36 discordant pairs, all in favour, none against, exact McNemar
p = 2.9 × 10⁻¹¹.**

Neither baseline degrades. Both retrieve *nothing*, emitting `the key that is used to
pa` — a paraphrase of the question containing no digits at all. With eight slots and no
ranking, the needle is simply not in the cache.

So **7,864× is a property of the score, not of the budget**, and not of "any eviction
policy works at this scale". That was genuinely open until these two cells ran.

A note on the streaming arm's provenance. Its policy overrides `recent` to `budget − 4`
so `heavy()` reaches 0, but the manifest recorded `args.recent`, so a run that genuinely
had `heavy=0` was recorded as `heavy=8`. The policy was right and the record was wrong,
which is the harder failure to catch — the result is valid while its provenance lies
about it, and `verify_claims.py` asserts manifest fields. The manifest now records the
effective value with `recent_flag` alongside.

# Stage 3 is viable: the model can chain hops

The reasoning plan requires an uncompressed ceiling before any compression number on a
task means anything. Multi-hop chains, 8K, no compression:

| chain | accuracy | by depth |
|---|---:|---|
| 2 hops | **0.833** | 0.67 / 1.00 / 0.83 |
| 3 hops | **0.722** | 0.75 / 0.58 / 0.83 |

Both clear a floor comfortably, unlike this model on `Rineya_Long_Context_Demo_v1`
where it scores 0.2031 against a 0.2500 constant-abstention baseline. Stage 3 will
measure the mechanism rather than the model's incapacity.

The failure mode is the informative one: `want 42649, got 99873` — a *different chain
value*, so the model stops mid-chain rather than hallucinating.

# Span length, and a gate this stage should have had

| span | policy | d0.1 | d0.5 | d0.9 | accuracy |
|---:|---|---:|---:|---:|---:|
| 12 tokens | grow=2.0 | 1.00 | 0.92 | 0.67 | 0.861 |
| 20 tokens | grow=2.0 | 0.83 | 0.42 | 0.33 | 0.528 |
| 20 tokens | fixed back=4 | 0.00 | 0.00 | 0.00 | 0.000 |

Accuracy falls monotonically with span length, which is what the fragmentation account
predicts. But **depth 0.9 is the worst** at both W=12 and W=20, inverting every other
cell in this programme, where depth 0.9 is easiest because it faces 3 compression rounds
against 27 at depth 0.1.

Depth ordering is a property of the compression schedule, so an inversion means the
thing being measured may not be compression. The failure text agrees: at W=12 depth 0.9
the model paraphrases, restates the frame with wrong words, and in one case recites the
*filler* back verbatim. That is a recitation failure, not a retrieval failure.

Stage 2 was never given the capability gate stage 3 got. See **O8** in
`docs/OPEN_QUESTIONS.md`; the uncompressed reference at W=3/12/20 is running, and until
it reports, the fall from 1.000 at five tokens to 0.528 at twenty cannot be attributed
to the mechanism.
