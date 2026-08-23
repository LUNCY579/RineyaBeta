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
