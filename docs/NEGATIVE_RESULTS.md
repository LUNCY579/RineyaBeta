# Things that do not work, and the measurement that killed each

Collected because this is the expensive knowledge. Each entry states what was
tried, the number that ended it, and — where it is known — *why* it failed, since
a mechanism generalises and a score does not.

Ordered by how likely someone is to propose the idea again.

---

## 1. Sliding-window attention as a KV-memory strategy

**Idea.** Give most layers a sliding window and keep a few global; the cache then
saturates instead of growing, so its advantage compounds with length.

**Killed by three independent measurements.**

- **Peak memory goes UP.** `w=256` and `w=1024` have byte-identical peaks
  (3832.5 MiB) despite different caches. A window is built as a dense
  `[1,1,n,n]` boolean mask that scales with *context*, not window.
- **Reachable context goes DOWN.** At 32K under a 6 GiB cap every SWA variant
  OOMs while `gqa-2KV` (2573 MiB) and `gqa-1KV` (2111 MiB) both fit.
- **The cache saving is not resident.** Measured with de-duplicated storage
  pointers, `swa-w1024-g4` at 16K reports 351.9 MiB logical against **1792.0 MiB
  resident** — identical to the full-attention baseline. `DynamicSlidingWindowLayer`
  keeps a *view* onto the full prefill allocation, so the saving only exists after
  the first decode step. Peak is set at end of prefill, which is exactly when it
  does not.

**Quality is a cliff with no usable middle.** At `w=512`, wider than every
chat_dev prompt, it is provably the identity — 510/510 byte-identical, Δ 0.00000.
At `w=128` the model emits empty strings, and one token past the window edge
costs as much as two hundred.

**Why.** R2 sends **46.3%** of its attention mass to the first 4 tokens (14 of 28
layers over 50%, layer 27 at 80.5%). A positional window evicts exactly those.
The only rag_dev category left untouched at `w=128` is `social`, whose 20 cases
are 41–45 tokens — precisely the 20 prompts in the set that fit inside the window.

**What survives.** Sliding window wins decode throughput and persistent cache. It
is a steady-state optimisation, not a way to reach longer contexts.

---

## 2. Geometry-aware eviction (diversity / non-maximum suppression)

**Idea.** Top-k selection is redundancy-blind: near-duplicate keys waste slots. Pick
local maxima over a neighbourhood graph instead — representatives of distinct
modes rather than raw top scorers. A natural generalisation of SnapKV pooling from
the position axis to the key-embedding geometry.

**Killed by two probes, four minutes of GPU.**

*Key space.* Against a **random** B-subset control, top-k already keeps 60.4 of 64
slots distinct versus 61.9 — 2% worse. The eye-catching "effective rank 4.79 of
64" is **anisotropy**: mean pairwise cosine across all keys is 0.661, so any two
keys look similar. About 91% of the apparent redundancy is there regardless of
selection.

*Attention-profile space*, where the similarity field genuinely is heavy-tailed
(median cosine 0.383, P99 0.933), suppression makes attention error monotonically
**worse**, and worse than a matched-random shuffle at every threshold:

| τ | slots moved | vs top-k | vs matched-random |
|---:|---:|---:|---:|
| 0.90 | 29.4 | **+33.7% worse** | +17.2% |
| 0.99 | 3.8 | +3.2% worse | +0.2% |

**Why.** Positions with similar attention profiles are not redundant, they are
**corroborating** — jointly participating in one computation, so deleting either
damages it. Profile similarity measures functional co-participation, not
informational duplication. That is why suppression loses to a *random* reshuffle
of equal size.

**Generalises to:** any similarity-suppression policy on attention profiles will
have this sign, at any length.

---

## 3. Ranking cached positions by raw accumulated attention alone

**Idea.** Rank by the running sum of attention mass received.

**Fails at long context, and the severity scales with the number of compression
rounds.** A position that survived T rounds was scored T times and a newcomer
once, so the heavy set ratchets shut. Observation counts after 4 chunks were reported as min 1, max 3, with sum-vs-mean
top-5 rankings sharing 2 of 5 entries; that probe was never committed, so treat
those three figures as unsourced. The mechanism does not rest on them — the
paired 128K sweep below is the evidence. The
round count is `n/C` — 8 at 32K, 16 at 64K, 32 at 128K.

Passkey accuracy, budget 2048, YaRN x4, query-aware, **6 trials per cell**:

| length | rank | depth 0.1 | 0.5 | 0.9 |
|---:|---|---:|---:|---:|
| 32K | sum | 1.00 | 0.67 | 1.00 |
| 32K | mean | 0.83 | 0.67 | 1.00 |
| 64K | sum | 0.83 | 0.50 | 0.33 |
| 64K | mean | 0.83 | **1.00** | **0.83** |

At 32K the two are indistinguishable — one trial in 18 separates them, which is
why a 3-trial run earlier produced "normalization hurts" and had to be withdrawn.
By 64K the mean wins at the two deeper positions and the sum has started to
collapse toward the end of the context, which is exactly where a frozen heavy set
predicts it should.

**At 128K the exponent is a large effect, and e = 1 is the right default.**
`Policy.count_exponent` ranks on `sum / count**e`, so e=0 is the raw sum and e=1
the mean. Measured with **12 needles identical in every cell** — 128K, B=2048
(1.6% keep), YaRN ×4, query-aware:

| e | depth 0.1 | 0.5 | 0.9 | mean |
|---:|---:|---:|---:|---:|
| 0.00 (sum) | 0.58 | 0.08 | **0.00** | 0.22 |
| 0.25 | 0.67 | 0.08 | 0.08 | 0.28 |
| 0.50 | 0.67 | 0.00 | 0.33 | 0.33 |
| 0.75 | 0.58 | 0.17 | 0.67 | 0.47 |
| 1.00 (mean) | 0.42 | 0.33 | **0.83** | **0.53** |

At depth 0.9 raw-sum ranking scores **zero of twelve** and mean ranking ten of
twelve — exact McNemar +10/−0, p = 0.0020, within-needle trend p < 0.0001. Pooled
over all 36 needle-positions, 0.22 → 0.53 at p = 0.0074.

**An earlier 6-trial version of this table is withdrawn along with its four
p-values** (Cochran–Armitage z = +2.840 / p = 0.0045 at depth 0.9, p = 1.0000 at
depth 0.5, p = 0.4216 at depth 0.1, Fisher p = 0.4705 overall). Those are
independent-groups tests and the arms are paired on needles, so the model was
wrong; and at six trials significance needs all six needles to be discordant
(p = 2/2^d < 0.05 requires d ≥ 6), which the data did not give. Two of its
conclusions reversed under the paired re-run: mean ranking *does* beat sum
ranking overall (0.31, p = 0.0074), and depth 0.5 *does* have a trend
(p = 0.0125). Full table and tests in
[`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) §4c.

The mechanism is the one the ratchet predicts. A late-arriving needle carries a
small accumulated sum only because it was scored fewer times; dividing by
`count**e` removes exactly that artefact. The cost is at the other end — at depth
0.1 the mean is worse (0.58 → 0.42), because long-accumulated early positions do
carry real evidence that normalising discards.
### RETRACTED — the depth curve at 128K was confounded with needle identity

A pooled depth curve of 0.367 / 0.133 / 0.500 over "n = 30 per depth" was
published from these arms and read as a lost-in-the-middle trough. **Withdrawn.**
One RNG stream ran across the whole nested loop, so depth 0.1 drew keys 1–6,
depth 0.5 keys 7–12 and depth 0.9 keys 13–18 — depth and needle identity were the
same column, and the "30 trials" per depth were five repetitions of six needles.
Running depth 0.5 alone, so that it draws keys 1–6, moves the same configuration
from 0.17 to 0.50.

**And the shape was wrong, not just the interval.** Re-measured with 12 needles
identical in every cell (`results/pk_128k_grid12.json`), 128K at B=2048 gives
0.42 / 0.33 / 0.83. The only contrast that separates is **late versus middle**
(depth 0.9 over depth 0.5, +6/−0, p = 0.0312); late versus early falls short
(+6/−1, p = 0.1250) and early versus middle is a coin flip (+2/−3, p = 1.0000).
There is no trough. And the ordering does not replicate at 64K, so no depth law is
established — see [`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) §4c.

**And the premise was wrong too.** At B=8192 the same 12 needles give
1.00 / 0.83 / 0.92 — mean 0.92, peak 2521 MiB — with no depth dependence left
(permutation p = 0.8879). 128K was a budget choice, not a length wall. See
[`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) §4c.

*Method fixed rather than worked around:* the RNG resets per cell so every cell
sees the same needles, cells record per-trial `{key, got, hit}`, and the analysis
lives in `scripts/design/paired_passkey.py`, which refuses to compare cells that
share no needles.
---
## 4. Query-blind eviction during prefill

**Idea.** Score each chunk against its own trailing query rows, then compress.

**Fails whenever the question comes last**, which is almost always. Those trailing
rows are filler for every chunk but the final one, so a needle earns no score
before anyone asks for it. At 64K with YaRN, budget 2048:

| depth | accuracy |
|---:|---:|
| 0.99 (last chunk, never compressed) | 1.00 |
| 0.90 | 0.00 |

**A 4× larger budget does not help** (B=8192 gives the same 0.00), so it is not a
budget limit. **Fixed** by `chunked_prefill.score_with_query`, which moved 32K
retrieval from **0.33 to 0.78** on the depth-matched pair
(`pk_blind.json` vs `pk_aware.json`, three depths each). An earlier statement of
this as `0.20 → 0.87` averaged a 3-depth arm against a 5-depth one and is
withdrawn; see [`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) §4c.

---

## 5. Architecture changes ranked and declined

Every attention-level relative error below is measured against the same three
calibration points, in the same pass, on the same rag_dev prompts
(`src/store_probe.py` -> `results/store_probe.json`). Relerr means nothing on its
own; these are what give it a scale:

| reference | attention relerr | end-to-end rag_dev quality |
|---|---:|---:|
| significance eviction, keep 0.75 | 0.00937 | −0.01262 |
| significance eviction, keep 0.50 | 0.03729 | −0.05538 |
| significance eviction, keep 0.25 | 0.11381 | **−0.33076** |

### Uniform int4 KV — dead

3.75× store at attention relerr **0.11179** (4K haystack, `pure_int4@4K`). That is
the keep-0.25 point, which cost a third of the benchmark. Protecting the 4 sinks
in bf16 is what makes it even that good: without it, 0.12804.

### Per-token grouping for int8 K — dead, and it is the grouping that matters

Same bit width, same 1.9× store, 5.3× the error:

| int8 scheme | attention relerr |
|---|---:|
| K per-channel (g=128), V per-token | **0.00774** |
| K and V both per-token | 0.04134 |

The good variant is *below* the keep-0.75 reference. The naive one is between
keep-0.50 and keep-0.25. Nobody would have guessed which from the bit width.

### Cross-layer KV sharing (CLA) — dead

Ridge regression from layer *l*'s [K_preRoPE ; V] to layer *l+1*'s, fitted on
3072 tokens and scored on a held-out 1024 (`results/store_probe.json`, `cla`):

| | test relerr |
|---|---:|
| mean over 27 adjacent pairs | **0.1720** |
| best pair (4→5) | 0.0550 |
| worst pair (15→16) | 0.2796 |
| pairs below 0.10 | 6 of 27 |

The mean sits past the keep-0.25 point — and that is with a *dense* adapter,
which does not pay for itself: predicting one layer's KV costs 2048×2048 =
**4.19M** parameters to remove a `k_proj`+`v_proj` pair worth 2×1024×1024 =
**2.10M**. A net parameter loss for a 0.17 relerr. The train/test gap is the
other half of the answer: 0.1288 train against 0.2190 test on pair 19→20, so
even the fit that looks acceptable is substantially memorisation.

### MLA — not now, and for a different reason than usually given

The *information* in the cache genuinely is low-rank. Truncating
[K_preRoPE ; V] with a per-prompt SVD:

| basis | attention relerr |
|---|---:|
| rank 512 | 0.00147 |
| rank 256 | 0.01236 |
| rank 128, K only | 0.08326 |
| rank 128, V only | **0.19383** |

Rank 512 beats int8. But that basis is fitted to the prompt it is applied to —
an oracle, not a deployable projection, and the gap between an oracle basis and a
fixed learned one is the entire engineering problem. **The fixed-basis number is
not measured here**, so no MLA verdict in this repo rests on one; see §7 of
`FINDINGS_KV_PROGRAM.md`.

What is settled is where the damage lives: at equal rank, truncating V costs
**2.3×** what truncating K costs. Any latent scheme that treats the two
symmetrically is spending its budget in the wrong place.

The separate structural argument against MLA — that absorbed MLA is MQA against
16 query heads, so `repeat_kv` still applies, and that it collapses eviction from
per-head to per-token — is reasoning, not measurement, and is recorded as such.

### Unhealed GQA head merging — not a quality point

Mean-merging 8 KV heads to 2 without a healing run gives attention relerr
**1.0206** — the output is uncorrelated with the original, not a degraded version
of it. End to end that is chat_dev 0.83498 → **0.11384**. The KV-head ladder is
only a quality axis with a healing run per rung.

### SSM / linear-attention hybrid — do not start

Zero measurements exist. It needs full retraining rather than healing, on a host
with no retraining capacity. Listing it as an option was scope, not a finding.

---

## 5a. What survived from the same measurements

Not everything in the store probe failed, and the survivors are the reason the
probe was worth committing.

**int8, K per-channel (g=128) and V per-token.** 1.94× at relerr 0.00774 — below
the keep-0.75 reference point that cost −0.01262. Composes with eviction.

**Significance-gated mixed precision** (`src/mixed_precision.py`). Sinks bf16,
top-25% by the score the eviction mechanism already computes int8, the rest int4.
On the 4K haystack:

| scheme | attention relerr | store factor |
|---|---:|---:|
| pure int8 | 0.00762 | 1.94× |
| **mixed, top 25%** | **0.00693** | **3.04×** |
| pure int4 | 0.11179 | 3.75× |

Lower error than pure int8 at 1.6× more compression, using a score that costs
nothing because eviction computes it anyway. This is a Pareto improvement over
both uniform schemes and it is the strongest unexploited result in the program.

**One constraint on how it must be implemented.** The same probe measures what
happens when a quantised cache is round-tripped — dequantise, evict half,
requantise — once per compression round. Mean K relative error, per round:

| rounds | 1 | 2 | 3 | 4 | 5 | growth |
|---|---:|---:|---:|---:|---:|---:|
| int8, K per-channel g=128 | 0.00276 | 0.00391 | 0.00484 | 0.00565 | 0.00637 | 2.30× |
| int4, K per-channel g=128 | 0.04583 | 0.06409 | 0.07911 | 0.09217 | 0.10396 | 2.27× |

Both track √rounds closely (√5 = 2.236). At 128K with chunk 4096 there are ~31
rounds, so a naive dequantise-evict-requantise loop lands int8 near 0.0154 on
**this table's metric**. Note that is a *mean K-tensor* relative error, while the
0.00937 keep-0.75 reference is an *attention-output* relative error — different
quantities on different inputs, so the two are not directly comparable and the
comparison above should be read as an order-of-magnitude warning, not a threshold
crossing.

So the mechanism has to **quantise once and gather**, never round-trip per
compression. Eviction is a gather on the token axis and quantisation groups are
per-channel over 128 tokens or per-token, so a gather can be done on the quantised
representation directly — the round trip is avoidable, not inherent. Worth knowing
before implementation rather than after.

A related worry is settled rather than open: the smallest int8 group scale at 4K
is 1.156e-04, comfortably above fp16's minimum normal 6.104e-05, so storing scales
in fp16 does not hit subnormals at any length measured.


---

## 6. Measurement approaches that produced wrong numbers


Not ideas, but methods — each cost real time.

- **chat_dev aggregates cannot detect attention-level damage and report it
  backwards.** Its score *rose* (+0.00247) while only 361/510 replies stayed
  identical. Damage lengthens replies and `prompt_function_alignment` — a
  response-mode heuristic in 502/510 cases, weight 0.28 with `task_adequacy` —
  rewards length, while `brevity` (0.01) and `completeness` (0.02) cannot outvote
  it.
- **Retained attention mass ρ is necessary but not sufficient.** ρ > 0.996 still
  cost 0.199 end-to-end — *but that 0.199 is reproducible only from
  `_invalidated_prefill_bug/frac0.75__rag_dev` (0.47007 against off 0.66939), a
  run this repo has quarantined as invalid. The point stands on the valid runs
  (keep 0.75 has ρ > 0.99 and still costs 0.01262); the specific 0.199 is
  withdrawn.* The bound controls one attention output; errors compose
  across 28 layers and iterate across decode steps.
- **Counting cache bytes with `numel()`** credits a saving the allocator has not
  made whenever a layer holds a view. Sum de-duplicated `untyped_storage()`.
- **Any throughput claim below 2× is indefensible here.** The host's own thermal
  band is 2.17× (33.8 ms cool against 71.6–73.2 ms at 84 °C) — *an observation from
run logs; no artefact records those particular timings. The tree does now carry
timing instruments (`src/measure_decode.py`, `src/probe_decode_floor.py`) and the
decode harness records GPU temperature per arm, which is how the 2.17× band is
managed rather than merely noted.*
- **A memory cap is not a memory measurement, and neither is a peak.**
  `set_per_process_memory_fraction` bounds *reserved* bytes while
  `max_memory_allocated` reports *allocated* ones. A 32K uncompressed run dies at
  a reported peak of 4571 MiB against its own 6144 MiB cap, because the caching
  allocator is holding fragmentation the peak never shows — `torch.cat` on a
  growing KV cache keeps two buffers live. Raising the cap to 7800 makes the same
  run succeed at 1.00. The harness default of 6144 was therefore the ceiling on
  the whole program's uncompressed references, and it read as "the card is too
  small".
- **Changing two variables and attributing the result to one.** The same finding
  was first read as a chunk-size effect, because the failing run used chunk 4096
  at cap 6144 and the succeeding one chunk 1024 at cap 7800. Filling in the other
  two cells settles it: at cap 6144 both chunk sizes OOM, at cap 7800 both reach
  1.00, and chunk moves the allocated peak by 5.5% and the verdict not at all.
  Two runs on a diagonal are not a comparison (P6).
- **Pooling repeated measurements of the same subjects as independent trials.**
  Five ranking arms sharing one key sequence were pooled as "n = 30 per depth".
  They were five repetitions of six needles, so every interval and between-depth
  contrast built on them was invalid — and because each depth drew a *different*
  six, the axis under test was confounded with subject difficulty. The fix is a
  paired design plus per-trial records, not a wider interval.
- **Miscounting the power floor of a paired test.** This document previously said
  six paired subjects "cannot reach p < 0.05". They can: exact McNemar with all
  *d* discordant pairs one way gives p = 2/2^d, and d = 6 gives 0.03125. What is
  true is that d ≥ 6 is required (d = 5 gives 0.0625), so a 6-needle cell
  succeeds only if *every* needle flips. Fragile, not impossible — and the
  observed 5-of-6 flip did sit at 0.0625. Correct conclusion, wrong reason.
- **Trial counts below ~6 mislead.** A 3-trial run produced "normalization hurts,
  0.87 → 0.60"; at 6 trials the arms were tied within one trial in 18 and the
  claim was withdrawn.


---

## Where these numbers live

Every figure above is recomputable from a committed file. If one of them ever
disagrees with its source, the source wins and the disagreement gets stated
rather than overwritten.

| Section | Data | Code |
|---|---|---|
| 1. sliding window | `experiments/attention_shape/results/measure_fixed.json`, `.../swa-*__{chat,rag}_dev/` | `experiments/attention_shape/src/measure.py`, `src/run_eval.py` (driven by `run_quality_sweep.sh`) |
| 2. geometry-aware eviction | `experiments/geometric_significance/results/{redundancy,rss_probe*}.json` *(branch `exp/geometric-significance`)* | `.../src/{redundancy,rss_probe}.py` |
| 3. sum vs mean ranking | `experiments/kv_significance/results/pk_{32k,64k}_{sum,mean}.json`, `pk_128k_pe*.json`, `pk_128k_grid12.json`; the withdrawn 3-trial pair sits in `.../results/_superseded_unmatched/` with a `NOTE.md` | `.../src/passkey.py --count-exponent` |
| 4. query-blind eviction | `experiments/kv_significance/results/{pk_blind,pk_aware,pk_long_aware}.json` | `scripts/kv/chunked_prefill.py` |
| 5, 5a. store and precision | `experiments/kv_significance/results/{store_probe,mixed_precision}.json` | `.../src/{store_probe,mixed_precision}.py` |
| calibration references | `experiments/kv_significance/results/{off,frac0.75,frac0.5,frac0.25}__rag_dev/summary.json` | `.../src/run_eval_sig.py` (driven by `run_sweep.sh`) |
| 6. measurement failures | see §5 and §6 of [`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) | `scripts/design/verify_claims.py` |

```bash
python scripts/design/verify_claims.py
```

---

## 7. RGE resume logs without checkpoint-tail reconciliation

**Idea.** Because model/optimizer checkpoints are atomic, append-only training
logs can be left untouched across an asynchronous interruption.

**Killed by the M2 pair audit.** Muon was forcibly paused after logging step 31
but its last durable checkpoint was step 30. Exact resume replayed step 31 and
produced the correct tensors, but the log contained two step-31 rows. The pair
matcher therefore rejected the learning-curve token axis even though final
positions and effective token counts matched.

**Fix.** On resume, truncate records newer than the checkpoint and keep the last
record for every completed global step. The historical pilot log was repaired by
removing one duplicate row; before/after hashes are committed, and no model or
optimizer tensor was touched. An injected stale-tail regression now passes.

---

## 8. Muon-LoRA at matched effective-step scale

**Idea.** Apply a pinned FP32 Muon update to every rank-16 LoRA factor, with raw
LR selected only by AdamW-relative effective-delta geometry.

**Training objective improved.** At 104,571 matched supervised tokens, Muon
reduced token-weighted loss by 0.1553 and final-step loss by 0.3633; it had lower
loss on 75/101 paired plan steps.

**Killed for the general model by frozen retention.** `chat_dev` fell 0.01524
(paired 95% CI entirely below zero), beyond the 0.01 non-inferiority margin.
`rag_dev` fell 0.00542 on mean and introduced four critical decoding-corruption
cases where AdamW had zero; contract violations also doubled. Muon is rejected
for the general Rineya v1 recipe. No retention-driven LR rescue is permitted.

**Interpretation.** Better optimization of the supervised objective did not
translate into a better retained endpoint. Timing from the pair is unusable due
to external GPU contention and is not part of the rejection.

---

## 9. DeltaBall radius extremes

**0.5x `d_ref`: overconstrained.** The radius activated on 37.14% of post-warm-in
matrix checks and 83.61% of steps, but worsened token-weighted and final training
loss. It is rejected.

**2x `d_ref`: inactive.** It produced zero hits through a fully warmed step-40
checkpoint. Because the radius also exceeded the source AdamW run's final maximum
relative delta, it was stopped early and receives no DeltaBall causal credit.

The 1x radius barely cleared the preregistered activity floor (1.054%) and is the
only radius retained for the next exploratory tier.

---

## 10. Component-local curriculum and constant tail alone

On an exact same-membership 12,875-row subset, curriculum ordering changed
token-weighted loss by only `-0.00137`, worsened final-step loss by `+0.145`, and
produced mixed retention (`chat -0.00368`, `RAG +0.00081`; both CIs cross zero).
No standalone ordering benefit is established.

A 12-step constant-LR continuation then changed chat by `-0.000884` and left all
752 RAG cases identical. The tail is neutral as an endpoint and is retained only
to test the distinct checkpoint-averaging hypothesis.
