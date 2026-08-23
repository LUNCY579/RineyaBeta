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
