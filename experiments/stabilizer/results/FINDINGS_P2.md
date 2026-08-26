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

**CORRECTION 2026-08-25 — the "best hand-tuned" column above is not the best.** At d=7
the ladder also contains back=7 → **0.944** and back=8 → **1.000**, both measured, both
beating derived reach's 0.889. The row compares against back=6 only. Restated against
the true per-span best:

| span (digits) | derived `grow=2.0` | true best tuned fixed | Δ |
|---:|---:|---|---:|
| 4 | **1.000** | back=3 → 0.972 | +0.028 *(only back=2,3 were measured at d=4)* |
| 5 | **1.000** | back=4 → 1.000 | 0.000 |
| 7 | 0.889 | **back=8 → 1.000** | **−0.111** |
| 9 | *no derived arm exists* | back=10 → 1.000 | *untested* |

**So "matches or beats the best hand-tuned fixed reach at every span length" is
WITHDRAWN.** It is false at d=7 by four needles, it was never tested at d=9, and the
d=4 win is against an incomplete ladder that stops at back=3.

**What survives, and it is the claim that actually matters for deployment.** Derived
reach beats any *single* fixed reach held constant across span lengths. A deployed
system does not know the span in advance, so it must pick one reach and keep it: back=6
held across d = 4/5/7 gives 0.972 / 1.000 / 0.778, where derived gives 1.000 / 1.000 /
0.889 with no span argument. Per-span oracle tuning still wins at d=7 — but per-span
oracle tuning is not available at inference time, which is the whole point.

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
same twelve needles at every depth. (Read this table with the sink ablation below: the
eight heavy slots do **not** work without the four sinks under them.)

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

---

## 2026-08-25: three results that change how the earlier sections must be read

### The sinks are load-bearing, so the headline decomposition matters

Sinks sat at four in every arm above and were never ablated, so "eight heavy slots
retrieve a needle from 125,826 tokens" was the natural reading. It is wrong.

| sinks | heavy (score-chosen) slots | d0.1 | d0.5 | d0.9 | accuracy |
|---:|---:|---:|---:|---:|---:|
| **0** | **12** | 0.00 | 0.00 | 0.00 | **0.000** |
| 2 | 10 | 1.00 | 1.00 | 1.00 | **1.000** |
| 4 | 8 | 1.00 | 1.00 | 1.00 | **1.000** |

`sinks=0` gets the freed slots — twelve score-chosen against eight — and still scores
zero. P1 measured this model placing 46.3% of its attention mass on the first four
positions; removing them breaks the attention distribution past what any ranking can
recover. **Two sinks suffice**, so the requirement is "some sinks", not "four", but it
is a requirement. Every headline above should be read as *four sinks plus eight
score-chosen positions*.

### The span axis is capped by the model, and the cap is measured

With a **full cache and no eviction at all** at 16K:

| span (digits) | accuracy | what the model emits |
|---:|---:|---|
| 12 | **0.000** | 11 correct digits, then diverges |
| 20 | **0.000** | 11 of 20 |
| 30 | **0.000** | 11 of 30 |

This was the pre-stated kill condition for the corrected span experiment, and it fired.
The reach law is therefore measured over d = 4, 5, 7, 9 and **cannot be extended on this
model** — not by any budget, window, or task redesign. Testing the 20–40 token spans
real evidence occupies needs a model with more verbatim-recall capacity. Without this
gate, 0.000 at d=20 compressed would have been written up as a compression failure.

### The chunk axis is non-monotonic, and no explanation survives

| chunk | compressions at 128K | d0.5 | accuracy |
|---:|---:|---:|---:|
| 4096 | 30 | 1.00 | **1.000** |
| **2048** | 61 | **0.42 / 0.50** | **0.806 / 0.833** |
| 1024 | 122 | 1.00 | 0.972 |

Two independent seeds for chunk 2048; it reproduces. Eviction pressure alone cannot
explain it — the highest-pressure setting is the second best. Eliminated so far:
differing data (byte-identical prompts; the failing needles pass at both other chunk
sizes), a chunked-prefill bug (bit-identical to single-shot at all three sizes with
compression off), boundary straddling (the needle lies inside one chunk at every size),
and window fragmentation (`trace_cache_gaps.py` measured a `back=6` window spanning
exactly six tokens with gap 1 at both 4096 and 2048).

Seven of twelve failures carry the 4-of-5 truncation signature, which is the P2
fragmentation mode. **The cause is open**, and it is recorded here rather than dropped
because the earlier "survives 4× the eviction pressure" result compares only 4096
against 1024 and steps over the cell between them.

---

## 2026-08-25, later: the FIRST natural-language evidence, and it is not the passkey story

Until today the mechanism had never been run on a natural-language token. All 40
committed `pool_back` result files were passkey; `run_eval_sig.py`, the only rag_dev /
chat_dev runner, had no `--pool-back` flag at all. These are the first cells.

752 RAG cases, paired on case id, every arm matching the committed cornerstone-1
manifests on `sinks=4, recent=32, window=16` and one generation protocol, so a
directional arm differs in exactly one field.

### Direction matters on language, and the sign FLIPS with compression pressure

| keep | compression | centred (3,3) | causal (6,0) | paired diff | discordant | exact sign p |
|---:|---:|---:|---:|---:|---|---|
| 0.25 | 4x | 0.26096 | **0.41286** | **+0.15190** | 235 up / 36 down | **5.66e-37** |
| 0.50 | 2x | 0.63371 | **0.64858** | +0.01487 | *unpaired* | *n/a* |
| 0.75 | 1.33x | **0.65999** | 0.65136 | **-0.00863** | 1 up / 23 down | **2.98e-06** |

The keep-0.50 row is means-only: the committed centred arm's per-case records are
gitignored and absent here, so it cannot be paired. It is reported as a mean difference
and nothing is claimed from it.

**The crossover is the finding.** Causal pooling wins hugely under pressure and loses,
slightly but significantly, when there is none. That is mechanically coherent: with an
ample budget a centred window's forward reach buys real context, and with a tight budget
that same forward reach spends slots on positions the query has not reached yet.

**Pooling itself is harmful under pressure; direction is what rescues it.** At keep 0.25
the centred window scores **0.26096**, *worse than not pooling at all* (`frac0.25`,
pool=1, **0.33863**). So the causal window is not adding a gain on top of pooling -- it
is repairing damage that pooling causes, and then some.

### Derived reach LOSES on language, where it was the star on passkey

| keep 0.50 arm | window | grow | quality |
|---|---|---|---:|
| `frac0.5-pool7-fixed` | centred (3,3) | -- | 0.63371 |
| `frac0.5-grow2` | centred (3,3) | 2.0 | **0.61346** |
| `frac0.5-b6f0` | **causal (6,0)** | -- | **0.64858** |

Paired, causal vs derived: **-0.03513**, 1.48e-05. Geodesic dilation was the mechanism's
best passkey result -- it beat every single fixed reach held constant across span
lengths. On prose it is the worst of the three. Stated plainly because the passkey
finding was stated plainly.

*Scope note:* the `grow` arm carries a centred (3,3) window, because `--grow` without
`--pool-back` defaults the window to `pool//2` -- the same convention the passkey `grow`
arms used. So this contrast is (centred+grow) against (causal), not a clean grow / no-grow
test. Against the matched centred arm without grow, adding grow costs **-0.020**
(0.63371 -> 0.61346), unpaired.

### What this supports, and what it does not

Against the uncompressed reference (**0.66939**): causal pooling retains **96.9%** of
full quality at **2x** compression and 97.3% at 1.33x, and collapses to 61.7% at 4x.
Against the published positional control it is **+0.172** at keep 0.50 and **+0.176** at
keep 0.75, but only **-0.009** at keep 0.25 -- where recency-only is just as good.

**The honest deployable claim is ~2x KV reduction at ~97% quality retention.** Not
7,864x. That figure is a passkey property and must never be quoted near a language
number.

### The scale gap, which is larger than any effect measured above

`rag_dev` prompts: **median 238 tokens, maximum 330. Not one exceeds 1K.** The passkey
headline runs at **125,826** real tokens -- **543x longer**. This programme has evidence
at ~232 tokens and evidence at ~126K tokens and **nothing between them**. The mechanism's
entire value proposition is long-context memory, and the only natural-language benchmark
available exercises it where compression saves on the order of 170 cache entries.

Every language number above should be read as *"at ~232 tokens"*, and no long-context
language claim is supported by any of it.

### Two cells did not finish, and one of them is a defect worth chasing

- `frac0.25-grow2` **timed out** at the 2400 s limit, having reached 200/752 in 481 s
  and slowed from there.
- `frac0.25-rf0.25-b6f0` reached 400/752 and died with **CUDA illegal memory access**,
  after slowing progressively: 233 s, 245 s, 571 s, 528 s per 100 cases. A monotone
  slowdown ending in an illegal address is a leak signature, not a thermal one, and both
  affected arms are the aggressive keep-0.25 configurations.
- The A3 power run, launched immediately after that crash, died right after loading
  weights with no further output -- consistent with inheriting a poisoned context. The
  GPU tests clean afterwards (6,990 MiB free, bf16 matmul fine), so it is retryable.

**No result above depends on the failed cells.** They are recorded because an
unexplained slowdown-then-illegal-access in the eviction path is a stability risk for
exactly the deployment case this mechanism is for.

### RESOLVED same day: it was the queue, not the mechanism, and the leak reading was wrong

The slowdown-then-illegal-access looked like a leak in the eviction path. It is not.

A bounded repro of the exact crashing configuration, run alone on an idle GPU with an
allocator trace added to the progress line:

| cases | wall | allocated | reserved |
|---:|---:|---:|---:|
| 100 | 93 s | 918 MiB | 968 MiB |
| 200 | 194 s | **918 MiB** | **968 MiB** |

**Memory is flat to the megabyte, and the run is 2.4x faster than the one that died**
(97 s per 100 cases against 233 / 245 / 571 / 528). There is no leak.

The cause is in the queue. `timeout -k` SIGKILLed the preceding `grow` cell and the next
cell started **in the same second**, before the killed process had released the device.
The follow-on ran starved, degraded monotonically, and died with an illegal address; the
A3 run launched immediately after inherited the same state and died on load.

Fixed by draining the device between cells — poll until GPU memory falls below a
threshold before starting the next run. The lesson generalises past this harness: a
SIGKILL of a CUDA process is not synchronous with the device being free, and a queue
that assumes it is will manufacture failures that look like bugs in whatever runs next.

Also corrected in the wrapper: a run whose summary was written but whose optional
`--compare-to` step failed was being reported as **FAILED**. Six good cells were labelled
that way. Artifact presence now decides success, and a non-zero code with an artifact is
reported as what it is.

### The first runtime cost this programme has measured

A5 has stood open with the note that *every number here is a benefit with no measured
cost*. The language runs supply one for free, since the arms differ only in policy:

| arm | keep 0.50 | keep 0.25 |
|---|---:|---:|
| causal (6,0) | 67.7 s / 100 cases | 89.8–97.0 s / 100 cases |
| derived (`grow=2.0`) | 115.6 s / 100 cases | 240.5 s / 100 cases |
| **derived / causal** | **1.71x** | **2.48x** |

So on natural language **derived reach is worse on both axes at once** — it costs 1.7–2.5x
the runtime *and* scores lower (−0.035 paired against the causal window, p = 1.48e-05).
Whatever case exists for geodesic dilation rests on passkey, where it was the best arm;
on prose there is currently no argument for it.

This is a same-hardware, same-harness, back-to-back comparison rather than a controlled
timing study — it is not thermally controlled, and A5 proper still needs to be. But the
ratio is large enough, and consistent enough across two keep fractions, that the
direction is not in doubt.

---

## 2026-08-25, final: the aggressive-compression result was an artifact of BUDGET ALLOCATION

Everything above about keep 0.25 -- including "recency-only is just as good at 4x" --
was measured with `recent` pinned at a **fixed 32 positions**, inherited from the
cornerstone-1 arms. On a 232-token prompt at keep 0.25 the whole budget is ~58, so
sinks(4) + recent(32) consumed **61% of it before the score saw anything**.

Setting `--recent-frac 0.25` instead, with the causal window unchanged:

| keep-0.25 arm | window | recent | score-chosen | `heavy_zero_count` | quality |
|---|---|---:|---:|---:|---:|
| `frac0.25-b6f0` | causal (6,0) | 31.33 | 22.6 (39.0%) | **17,360** | 0.41286 |
| `frac0.25-rf0.25-b6f0` | causal (6,0) | **14.38** | **39.5 (68.2%)** | **0** | **0.60617** |

Paired on 752 cases: **+0.19331**, exact sign test **p = 3.08e-60**. This is the largest
effect this programme has measured on natural language, and it is larger than the
direction effect that preceded it.

**The mechanistic explanation is in the `heavy_zero_count` column.** That counter records
(layer, head, compression) events where the budget left **zero** slots for the score, so
the policy silently degenerated to plain StreamingLLM. Under the fixed `recent=32` it
fired **17,360 times**; under proportional allocation it fires **zero** times. The
mechanism was not underperforming at aggressive compression. **It was switched off for a
large share of the work, and we were measuring its absence.**

### What this changes

| claim | before | after |
|---|---|---|
| quality at 4x compression | 0.41286, **61.7%** of uncompressed | **0.60617, 90.6%** |
| vs positional control at keep 0.25 | **-0.009** (recency ties it) | **+0.184** (`rf1.0` = 0.42208) |
| deployable operating point | ~2x at ~97% | **~4x at ~91%**, or 2x at ~97% |

The earlier reading -- *"at keep 0.25 the score merely ties recency-only"* -- is
**withdrawn**. It was true of a configuration that had disabled the score, not of the
mechanism.

### Derived reach loses at this keep fraction too

| keep 0.25 | quality | vs causal |
|---|---:|---:|
| causal (6,0) | 0.41286 | -- |
| derived (`grow=2.0`, centred) | 0.33833 | **-0.07452**, p = 5.52e-12 |

Consistent with keep 0.50 (-0.03513). Derived reach is worse on natural language at both
keep fractions tested, on top of costing 1.7-2.5x the runtime. Its case rests entirely on
passkey.

### Consequence for the plan

**B1 (self-sizing budget) is no longer speculative.** It was ranked highest-value-per-hour
on the argument that turning a tuned constant into a derived quantity is the move that
made derived reach beat fixed reach. That argument is now backed by the single largest
language effect on record -- and `recent-frac` is a crude proxy for what B1 would do
properly, since it still fixes a *ratio* rather than deriving the split from the score's
own distribution. The headroom above 0.60617 is what B1 is for.

**A methodological note worth keeping.** Every keep-0.25 conclusion in this programme,
including cornerstone 1's published `-0.083` loss to the positional control, was drawn
from arms whose `heavy_zero_count` was never inspected. The counter existed and was
recorded in the manifests the whole time. A metric that reports *the mechanism did not
run* is worth more than one that reports how well it scored, and it should be read first.

---

## 2026-08-26: a defect in the mechanism itself, found by adversarial audit

`grow_spans` derives the reach by growing each score forward through positions above a
noise floor, and its docstring is explicit that the gate must be **"relative to each
stream's MEDIAN"** — it rejects a quantile by name because "the gate opens on everything,
and growth floods the sequence so that the top-k after it is arbitrary."

The implementation computed that median over a vector with **`0.0` substituted for every
masked slot**. `select_keep` masks sinks and the recent window to `-inf` before calling
it, so once those masks pass half the vector the floor is exactly `0.0`, and since scores
are non-negative softmax sums the gate opens on everything. The implementation was
producing the precise failure its own docstring exists to prevent, by another route.

| geometry | masked | floor as-was | gate admitted | true floor admits |
|---|---:|---:|---:|---:|
| rag_dev keep 0.25, recent 31 | **59%** | **0.00000** | **24 / 24** | 2 |
| rag_dev keep 0.25, recent 14 | 31% | 0.01410 | 22 / 41 | 1 |
| rag_dev keep 0.50, recent 32 | 31% | 0.01257 | 33 / 81 | 9 |
| passkey 128K B=64, recent 16 | 31% | 0.01183 | 22 / 45 | 5 |

**All eleven existing `grow_spans` tests passed before and after the fix.** None of them
ever fed the function a masked tensor — they tested it in isolation, never in the
composition `select_keep` actually uses. That is the general lesson: a unit test that
does not reproduce the caller's preconditions can pin the wrong function entirely.

### What this invalidates, stated by severity rather than as a blanket retraction

**Collapsed — 1 arm.** `frac0.25-grow2__rag_dev` (59% masked, gate fully open).
Quarantined. Its 0.33833 and the −0.07452 keep-0.25 contrast are withdrawn, and the two
pins that read them are deleted rather than adjusted — there is no correct value until
the arm is re-run, and a pin that silently tracks a moved number is worse than none.

**Degraded — about thirty arms at 26–36% masked.** Every passkey grow arm, every Halo
transfer arm, the distractor arms behind the competition result, and `frac0.5-grow2`.
Their floors sat near the 27th percentile instead of the 50th.

**What survives, and why the distinction matters.** A contrast between two arms sharing
the same geometry keeps its internal validity: both faced the same mis-specified gate.
The competition result compares `dist16_framed` against `dist16_bare` at identical budget
and recent, so **−0.361 stands as a contrast**. What it no longer describes is the policy
we documented. Absolute claims, and comparisons across geometries, are what need
re-measurement.

### A second defect from the same audit: cells that faced zero compressions

`chunked_prefill` never compresses the final chunk (`ci < n_chunks - 1`), by design. But
a needle inside that final chunk therefore faces **no compression at all**, and the result
row records no chunk index or round count — so a vacuous cell is byte-indistinguishable
from one that survived thirty rounds.

Scanning all 424 committed compressed cells: **8 faced zero compressions.** Three of them
are inside a published claim — the Halo 16K depth-0.9 cells at B=32/128/512, where
`real_tokens = 15714`, chunk 4096 puts the exempt region at ≥ 12288, and the depth-0.9
needle sits at ~14142. **A policy keeping nothing would also score 1.00 there.**

O5 previously ran this check and resolved it — but only for the 128K geometry, where the
exempt fraction is 0.977 and a depth-0.9 needle still faces three rounds. At 16K the
exempt fraction is 0.782 against a needle at 0.897. O5's own pre-declared test reads: *"If
a depth-0.9 needle faces zero rounds, that cell measures nothing and must be excluded from
the headline."* It does, and it must.

So §4.8's "8 cells, 288 needles, zero failures" is **252 valid needle-observations**, not
288: the depth-0.9 column of the three 16K cells contributes nothing. The 32K and 128K
Halo cells are unaffected. Independent corroboration sits in the committed data:
`passkey_ratio.json` at 16K/B=2048 scores 0.0 / 0.0 / **1.0** across depths — the only
cell that succeeds is the only one never compressed.

---

## 2026-08-26: the language claim finally has a matched control

Every language number this programme published compared the mechanism against
`frac0.25-rf1.0`, an arm that differs from it in **three** things at once: no score, no
pooling, and a recent window of **53.88 against 14.38**. A 3.7x larger recent window is
not a control; it is a second treatment. The passkey results have had the clean control
since cornerstone 2 began — `--select random` keeps the same *number* of positions,
chosen uniformly, at matched everything. The language harness had no such flag until
today.

All four arms below are CPU, keep 0.25, causal (6,0) window, `sinks=4`,
`recent-frac 0.25`, same 752 cases, paired on case id. `random` differs from
`significance` in **nothing except how the heavy slots are chosen**.

| arm | select | recent | `heavy_zero` | quality |
|---|---|---:|---:|---:|
| **significance** | significance | 14.38 | 0 | **0.59488** |
| streaming (matched budget) | streaming | 53.88 | 317,604 | 0.42740 |
| **random** (matched slots) | random | 14.38 | 0 | **0.34304** |

| contrast | paired difference | exact sign test |
|---|---:|---|
| **score vs matched RANDOM** | **+0.25184** | ***p* = 1.97 × 10⁻⁶⁶** |
| score vs matched-budget streaming | +0.16747 | *p* = 3.24 × 10⁻³² |

**This is the first language number attributable to the score itself.** The score is
worth **+0.252** at 4x compression against an arm holding the same number of positions
by lottery. The previously published "+0.184 over the positional control" is superseded —
not because it was too large, but because it could not be attributed.

Two honest notes attached.

*The streaming arm cannot be matched on `recent`.* Spending the budget down to
`heavy = 0` is what makes it streaming, so its recent window is necessarily larger. Its
`heavy_zero_count` of **317,604** is the counter doing its job: it confirms the arm ran
with the score fully disabled, which is exactly what the control requires. Only the
`random` contrast is matched in the strict sense.

*Device.* These are CPU numbers. The GPU measurement of the same significance
configuration is **0.60617** against CPU's **0.59488** — a gap of 0.011. Devices are
comparable but not identical, so the contrasts above are computed CPU-against-CPU and no
CPU number is differenced against a GPU one anywhere in this section.

### A note on what the controls do NOT rescue

The matched control strengthens the attribution of the *score*. It says nothing about the
two defects found the same day: the `grow_spans` flood and the zero-compression cells. A
clean control over a mis-specified policy still measures the mis-specified policy.

---

## 2026-08-26: O13 RESOLVED, and the long-context result is not what either side expected

### O13 — capacity confirmed, my mechanism story refuted

Both predictions were registered with kill conditions before the runs.

| arm | d0.1 | d0.5 | d0.9 | overall |
|---|---:|---:|---:|---:|
| B=16 chunk 4096 (baseline) | 1.000 | 1.000 | 1.000 | 1.000 |
| B=16 chunk 2048 (the anomaly) | 1.000 | 0.417 | 1.000 | **0.806** |
| **B=32 chunk 2048** | 1.000 | **1.000** | 1.000 | **1.000** |
| B=32 chunk 4096 (control) | 1.000 | 1.000 | 1.000 | 1.000 |
| **B=16 back=8 chunk 2048** | 1.000 | **0.000** | 0.667 | **0.556** |

**Prediction 1 confirmed.** Doubling the budget removes the chunk-2048 deficit entirely,
and the matched B=32 chunk-4096 control also scores 1.000, so the budget is not merely
lifting everything. **The chunk axis is not a property of the schedule; it is a marginal-
capacity effect that only appears when the budget is thin.** O13 is resolved after four
failed explanations.

**Prediction 2 refuted, and informatively.** I predicted `back=8` would *help*, on the
account that a longer reach lifts the span's trailing edge above the competition. It made
it **much worse** — 0.556 against back=6's 0.806, with depth 0.5 falling to 0.000. My
kill condition said this refutes the tie-break account of the trailing edge, and it does.

The reading that survives both results: a longer backward reach **inflates the candidate
set**, because more positions inherit the span's high score. At a thin budget that is
strictly harmful — more high-scoring positions compete for the same eight slots. So
backward reach is not monotonically better; it trades against budget. That is consistent
with reach deciding retrieval (established) while adding *more* of it hurts when there is
nowhere to put the extra candidates.

**Deployment rule this yields:** size the budget to the span with headroom. Do not tune
the chunk schedule, and do not assume more reach is safer.

### Long context: compression is free, and the model is the constraint

The 543x gap, at last measured. `off` is the uncompressed arm at each length — included
precisely so a fall could be attributed to the model rather than the policy.

| length | uncompressed | keep 0.50 | keep 0.25 | share of native |
|---|---:|---:|---:|---:|
| native ~232 tok | **0.66939** | — | — | 100% |
| 4K | **0.39102** | 0.39126 | 0.38730 | 58% |
| 16K | **0.18571** | 0.19559 | 0.19559 | 28% |
| 32K | **0.12827** | — | **0.10851** | 19% |

**Compression costs nothing at length.** Every paired contrast against the uncompressed
arm is indistinguishable from zero:

| contrast | paired diff | discordant | p |
|---|---:|---|---|
| 4K, uncompressed → keep 0.50 | +0.00024 | 1/2 | 1.000 |
| 4K, uncompressed → keep 0.25 | −0.00372 | 1/4 | 0.375 |
| 16K, uncompressed → keep 0.50 | +0.00988 | 2/0 | 0.500 |
| 16K, uncompressed → keep 0.25 | +0.00988 | 3/1 | 0.625 |
| 32K, uncompressed → keep 0.25 | −0.01976 | 0/2 | 0.500 |

The 32K arm is the strongest form of the point: a **4.0x** cache reduction (32,922 → 8,226
stored positions) with only two discordant cases in fifty.

**But the model collapses with length on its own.** Uncompressed quality falls to 58% of
native at 4K, 28% at 16K, 19% at 32K — with the *entire* cache present.

The inference that follows is clean despite the collapse, and it is the important one:
**a full cache and a quarter cache perform identically at these lengths, so the cache is
not the binding constraint. The model is.** Had this been competition-as-capacity — the
§5.1 mechanism, where budget restores accuracy — more budget would have helped. It does
not; `off` holds 100% of the cache and scores the same as keep 0.25.

**This is the same shape as A4, and the `off` arm is why we can say so.** Without it,
"4x compression retains quality at 16K" would have read as a triumph. It is instead a
statement about a regime where the model has already lost the thread.

### A confound I built into the benchmark, stated plainly

Lengthening the haystack means adding distractor documents, so length and competitor
count rise together and cannot be separated in this design:

| dataset | mean prompt | mean distractor documents |
|---|---:|---:|
| rag_long4k | 4,227 tok | 46.4 |
| rag_long16k | 16,520 tok | 189.3 |
| rag_long32k | 32,902 tok | 380.0 |

Passkey has the same property but its filler is inert; these distractors are *plausible
evidence documents*, which makes the task harder in a way passkey's never is. So "the
model collapses with length" should be read as **"with length and with hundreds of
plausible competing documents"**, and this benchmark cannot apportion between them.

### What this means for the next step

The mechanism is free at 4-16K on this core, and the core cannot use 16K. That makes a
**long-context-trained checkpoint the single highest-value thing to test next** — the
policy comparison only becomes informative once the model can do the task at all. A
QLoRA long-context fine-tune of Halo is in progress on this machine, and it targets
exactly the capability whose absence is the binding constraint above.

---

## 2026-08-26: the allocation curve has an INTERIOR optimum, and it validates itself

The allocation finding said proportional beats fixed. It did not say what proportion.
`recent_frac = 0.25` was a first guess, not a searched value. Five arms, all CPU, all
752 cases, keep 0.25, causal (6,0), `sinks=4` — only the split moves.

| `recent_frac` | recent | score's share of budget | quality |
|---:|---:|---:|---:|
| 0.10 | 5.82 | 83.1% | 0.59788 |
| **0.15** | **8.70** | **78.1%** | **0.60834** |
| 0.25 | 14.38 | 68.3% | 0.59488 |
| 0.35 | 20.32 | 58.1% | 0.53656 |
| 0.50 | 28.94 | 43.2% | 0.43732 |

**The optimum is interior**, at roughly 78% of the budget under the score. That is a
finding in itself: pushing the score's share *higher* (83.1% at rf=0.10) is slightly
worse, so the recent window is not pure overhead — it earns its ~15%, and an allocation
rule that simply maximised the score's share would be wrong.

**The curve validates itself at the far end.** As the score's share falls, the policy
must approach StreamingLLM, and it does: rf=0.50 leaves 43.2% under the score and scores
**0.43732**, within 0.010 of the matched-budget streaming control's **0.42740**. Nothing
forced that agreement — the sweep and the control are separate runs — so it is a real
consistency check on both.

Against the matched random control (0.34304), the best allocation is worth **+0.26530**,
up from +0.25184 at the guessed rf=0.25.

**What this sets up for B1.** The self-sizing budget now has a number to beat: **0.60834**,
from a *tuned* ratio. B1 derives the split instead of fixing it, and the honest bar is
that deriving must match or beat tuning — the same bar derived reach was held to on
passkey, and failed at d=7. If B1 merely ties a tuned constant, that is worth saying
plainly rather than presenting a derived quantity as a win because it is derived.

---

## 2026-08-26: the re-run says the flooded gate did not matter here, and the retraction is lifted

The `grow_spans` fix was re-run on the quarantined arm. Same configuration, same 752
cases, only the noise floor changed.

| | quality |
|---:|---:|
| flooded gate (quarantined) | 0.33833 |
| **fixed gate** | **0.33820** |
| difference | **−0.00014** |

The retracted contrast returns essentially unchanged: derived reach loses to the causal
window at keep 0.25 by **−0.07466**, against the withdrawn −0.07452.

**So the claim was right and I retracted it on a defect that did not move it.** The
retraction was still the correct action — the arm *was* produced by a mechanism that did
not match its specification, and there was no way to know the magnitude without
re-running. But the outcome deserves stating as plainly as the retraction was: the
number survived.

**Why the flood did not matter here, which is the interesting part.** This arm carries
`recent = 31.33` — the fixed window, not the proportional one — and its
`heavy_zero_count` is **17,360**. The score had *zero* slots on seventeen thousand
compressions. A growth gate that admits too many candidates cannot matter where there is
no top-k for them to enter. **The two defects overlapped, and the allocation defect
dominated.** That is why the arm scores 0.338 rather than something near the 0.608 the
tuned allocation reaches.

This bounds what the re-run establishes: it shows the flood was immaterial **in an arm
already crippled by allocation**. It says nothing yet about arms where the score
genuinely runs — the passkey derived-reach cells have `heavy_zero_count = 0`, so the gate
is the only thing that was wrong there, and those re-runs are the real test of the
defect's magnitude.

---

## 2026-08-26: the audit closes — both defects bounded, both core claims survive

### The grow_spans flood mattered in exactly one place

The passkey derived-reach cells are the real test: their `heavy_zero_count` is **0**, so
the gate was the only thing wrong with them.

| arm | flooded gate | **fixed gate** | delta |
|---|---:|---:|---:|
| d=4, `grow=2.0` | 1.0000 | **1.0000** | 0.0000 |
| d=5, `grow=2.0` | 1.0000 | **1.0000** | 0.0000 |
| d=7, `grow=2.0` | 0.8889 | **0.8889** | 0.0000 |
| d=7, **`grow=5.0`** | 0.6111 | **0.7222** | **+0.1111** |

**Every `grow=2.0` result is unchanged to four decimals.** §4.6 rests entirely on
`grow=2.0`, so its numbers — 1.000 / 1.000 / 0.889 at d = 4 / 5 / 7 — and the corrected
claim built on them (derived reach beats a *constant* reach, loses to per-span tuning at
d=7, where `back=8` scores 1.000) are **untouched by the defect**.

The one arm that moved is `grow=5.0`, and it moved in the direction the mechanism
predicts: a *stricter* multiplier depends more on the floor being right, so a depressed
floor hurt it most. At `mult=2.0` the gate is permissive enough that halving the floor
changes little.

**The multiplier finding survives, weaker.** Exact McNemar on the paired needles at d=7:

| gate | 2.0-wins | 5.0-wins | exact *p* |
|---|---:|---:|---:|
| flooded (published) | 10 | 0 | 0.0020 |
| **fixed** | **6** | **0** | **0.0312** |

Still significant, and still one-sided — no needle prefers `mult=5.0` under either gate —
but the effect is about half what was reported. The published 0.0020 should be read as
0.0312.

### The zero-compression cells re-run correctly, and the transfer claim survives

Halo 16K re-run at chunk 1024, which moves the depth-0.9 needle from **0 compressions
faced to 2**:

| budget | published (chunk 4096) | **re-run (chunk 1024)** |
|---|---|---|
| B=32 | 1.000 / 1.000 / **1.000 (vacuous)** | **0.917** / 1.000 / **1.000 (real)** |
| B=128 | 1.000 / 1.000 / **1.000 (vacuous)** | 1.000 / 1.000 / **1.000 (real)** |
| B=512 | 1.000 / 1.000 / **1.000 (vacuous)** | 1.000 / 1.000 / **1.000 (real)** |

**The depth-0.9 cells give the same answer once they are genuinely compressed.** The
claim was true; it just had not been tested. That is the second time today a retraction
survived its own re-run, and it is worth noticing that neither could have been known
without doing the work.

One real change: **B=32 depth-0.1 falls to 0.917** (11/12). Chunk 1024 means fifteen
compression rounds instead of three, so the shallowest needle faces five times the
eviction pressure and loses one. That is consistent with everything else established
about round count, and it is the honest cost of measuring the cell properly.

**Corrected transfer total: 287/288.** The count of needles is unchanged — 108 at 16K
plus 180 at 32K/128K — but 36 of them (the three depth-0.9 cells) previously measured
nothing, and the one failure is real. So the published *"288 needles, zero failures"*
becomes **"288 needles, one failure, and 36 of them only became evidence today"**.

*(I first wrote 323/324 here from a miscount and corrected it before committing; the
arithmetic is 107/108 at 16K plus 180/180 at 32K and 128K.)*

---

## 2026-08-26: which model is each claim actually about?

A direct question exposed a provenance gap. Counting every committed result by the model
its manifest names:

| model | files | what rests on it |
|---|---:|---|
| `artifacts/rineya-core-en32k-28L` | 155 | **almost everything** — the passkey headline, the reach law, O13, the span law |
| `artifacts/Rineya-Halo-1.2.0-devfreeze` | 12 | the §4.8 transfer test, and nothing else |
| **not recorded** | **37** | **every natural-language arm** |

The language harness had its checkpoint **hardcoded** as a module constant and wrote it
into no manifest, so all 37 rag_dev arms are silently on the base core and no reader
could tell from the artifact. It also could not be pointed at another checkpoint at all —
which is precisely what evaluating a long-context fine-tune requires. `passkey.py` has
had `--model` for weeks; the language harness did not.

Fixed: `--model` added, honoured at load time (adding the flag while still loading the
constant would be the manifest-records-the-flag defect in reverse — the artifact naming a
checkpoint the run never opened), and recorded in every manifest.

**The 37 existing arms stay as they are.** They were all produced by the same base core
under the same hardcoded constant, so they are internally consistent and their contrasts
hold. What changes is that this is now *stated* rather than inferable only from reading
the source.

**The honest scope line for the paper:** every language result in this work is on the
474M base core `rineya-core-en32k-28L`. The only Halo evidence is the passkey transfer
test. There is no natural-language evidence on any published core, and none on any
long-context-trained checkpoint.

