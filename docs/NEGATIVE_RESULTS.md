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
once, so the heavy set ratchets shut. Measured observation counts after 4 chunks:
min 1, max 3, with sum-vs-mean top-5 rankings sharing only 2 of 5 entries. The
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

**At 128K neither endpoint is good, and the interior does not rescue them.**
`Policy.count_exponent` ranks on `sum / count**e`, so e=0 is the sum, e=1 the
mean, and the interior was measured because the two endpoints fail differently:

| e | depth 0.1 | 0.5 | 0.9 | mean |
|---:|---:|---:|---:|---:|
| 0.25 | 0.50 | 0.17 | 0.17 | 0.28 |
| 0.50 | 0.50 | 0.00 | 0.67 | 0.39 |

Depth 0.5 is at or near zero for every exponent tried. Whatever breaks the middle
of a 128K context is **not** the accumulation bias, because the knob that controls
that bias moves depth 0.1 and depth 0.9 around without touching it.

*Matched endpoint runs at 6 trials and these three depths are in flight; the
existing 128K endpoint files are 3-trial and two-depth, so they are not
comparable with the rows above and are deliberately not tabulated here.*

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
retrieval from 0.20 to 0.87.

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
  cost 0.199 end-to-end. The bound controls one attention output; errors compose
  across 28 layers and iterate across decode steps.
- **Counting cache bytes with `numel()`** credits a saving the allocator has not
  made whenever a layer holds a view. Sum de-duplicated `untyped_storage()`.
- **Any throughput claim below 2× is indefensible here.** The host's own thermal
  band is 2.17× (33.8 ms cool against 71.6–73.2 ms at 84 °C).
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
| 1. sliding window | `experiments/attention_shape/results/measure_fixed.json`, `.../swa-*__{chat,rag}_dev/` | `experiments/attention_shape/src/measure.py`, `sweep.py` |
| 2. geometry-aware eviction | `experiments/geometric_significance/results/{redundancy,rss_probe*}.json` *(branch `exp/geometric-significance`)* | `.../src/{redundancy,rss_probe}.py` |
| 3. sum vs mean ranking | `experiments/kv_significance/results/pk_128k_{sum,mean}.json`, `pk_128k_e*.json` | `.../src/passkey.py --count-exponent` |
| 4. query-blind eviction | `experiments/kv_significance/results/{pk_blind,pk_aware,pk_long_aware}.json` | `scripts/kv/chunked_prefill.py` |
| 5, 5a. store and precision | `experiments/kv_significance/results/{store_probe,mixed_precision}.json` | `.../src/{store_probe,mixed_precision}.py` |
| calibration references | `experiments/kv_significance/results/{off,frac0.75,frac0.5,frac0.25}__rag_dev/summary.json` | `.../src/sweep.py` |
| 6. measurement failures | see §5 and §6 of [`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) | `scripts/design/verify_claims.py` |

```bash
python scripts/design/verify_claims.py
```
