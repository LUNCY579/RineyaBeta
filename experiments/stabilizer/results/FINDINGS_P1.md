# P1 — can a global predictor help the significance matrix?

The reversal of P0.3. That probe asked whether the significance matrix helps a
global predictor pick FFN neurons; it does not (+0.0002). This asks whether a
global predictor helps the significance matrix — which is the better question on
this hardware, because the KV cache is the measured bottleneck and the
significance matrix is what addresses it.

## The prize is large and its location is known

Against an uncompressed reference, at a 2048-token budget:

| length | keep | current | reference | gap |
|---:|---:|---:|---:|---:|
| 32K | 6.25% | 0.78 | 0.97 | 0.19 |
| 128K | 1.56% | 0.53 | 1.00 | **0.47** |

And at 128K the deficit is concentrated where a backward-looking score is blind:

| depth | accuracy | deficit |
|---:|---:|---:|
| 0.1 | 0.42 | 0.58 |
| 0.5 | 0.33 | **0.67** |
| 0.9 | 0.83 | 0.17 |

Depth 0.9 costs almost nothing because it enters in a late chunk and survives few
eviction rounds. Early and mid positions must survive many, with no score to
protect them until they have already been attended.

## P1.1 — the key does not predict future attention

`src/predict_future_attention.py`. A 3,923-token passkey prompt split at position
2,353; for every cached position, attention received *before* the split (`past` —
what the mechanism ranks by today) and *after* it (`future` — what it would want
to rank by). One global ridge fitted from the key to `future`, pooled over all
527,072 (layer, head, position) triples, **held out by (layer, head) group** so
the matrix must generalise across streams rather than memorise one.

Recall of the true top-147 positions by future attention:

| ranking | recall |
|---|---:|
| random | 0.0647 |
| **past-only** (current mechanism) | **0.3944** |
| predicted, key only | 0.1564 |
| past + predicted | 0.3811 |
| oracle (future itself) | 1.0000 |

**The key carries almost no signal about future attention.** Alone it reaches
0.1564 against a random floor of 0.0647 and the accumulated score's 0.3944.
Combined with the accumulated score it is *worse* than the score alone
(−0.0133) — the extra features add noise, not information.

Headroom to the oracle is **+0.6056**, so there is a great deal to win. It just
cannot be won from the key.

## What this means, and it is not only a negative

Put together with an earlier result, this explains something the program had
observed without understanding.

Future attention is not predictable from the past (the accumulated score plateaus
at 0.39 against an oracle 1.00) and not predictable from content (the key adds
nothing). **The only way to know what will be attended is to look at it** — and
that is exactly what query-aware scoring does. It runs the question against the
cache before compressing, and it produced the largest single quality effect in the
program, 0.33 → 0.78 at 32K.

That was previously an empirical fact with no account of why it was so much better
than anything else. It is now explained: the question is the *only* available
information about the future, and every attempt to substitute a prediction for it
has failed by measurement.

## The refinement this implies

Stop trying to predict the future and get more of it, earlier:

1. **Score against the question more often than once per chunk.** Compression
   currently happens between chunks; the question is scored at those boundaries
   only. Nothing prevents scoring more frequently, at a cost of one short forward
   pass each time.
2. **Score against a *proxy* for the question when it is not yet available.** In
   generation the question exists up front; in a streaming setting it does not.
   A cheap proxy — the running summary of what has been attended, or the decode
   prefix — could stand in.
3. **Delay irreversible decisions.** Eviction is permanent, which is why an early
   mistake at depth 0.5 is unrecoverable. A two-tier cache that demotes rather
   than deletes would give a second chance, at a memory cost that the 12.5%-keep
   result suggests is affordable.

## Limitations

One prompt, 3,923 tokens, one depth, one split point. The effect is stark (0.156
against 0.394) so it is unlikely to reverse, but it is not a sweep. The finding
also concerns *this* feature set — a position's key — and does not rule out that
some other cheap content feature carries the signal.

## Reproduce

```bash
python experiments/stabilizer/src/predict_future_attention.py --length 4096
```
