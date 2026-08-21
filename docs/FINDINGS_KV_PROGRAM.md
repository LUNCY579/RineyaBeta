# KV-compression program — consolidated findings

Single authoritative record of the long-context / KV-compression work on R2.
Every number here is reproducible from a committed file by the procedure stated
beside it. Where a figure was published and later withdrawn, both the wrong and
the right value appear, because a record that quietly overwrites its own mistakes
is not a record.

**Model under test.** `artifacts/rineya-core-en32k-28L` ("R2"): Qwen3-0.6B
derivative, 28 layers, hidden 1024, 16 query heads / 8 KV heads, head_dim 128,
vocab 32,768, tied embeddings, `rope_theta` 1e6, `max_position_embeddings`
40,960, `rope_scaling` null, 474,021,888 parameters. KV cache costs
`2 × 28 × 8 × 128 × 2 = 114,688` bytes/token = **112 KiB/token**.

**Host.** RTX 4070 Laptop, 8,188 MiB VRAM (~6 GiB usable), 15.2 GB system RAM
with 2–3 GB free, torch 2.13.0+cu130, transformers 5.15.0. Windows/WDDM does not
OOM when VRAM runs out — it spills to host RAM and the run goes silent at 0% GPU.
A per-process allocator cap converts that into a catchable error.

---

## 1. The working mechanism

`scripts/kv/significance.py` — significance-gated KV eviction. Promoted to the
main tree; self-contained (imports only `torch` and `transformers.cache_utils`).

**How it works.** Each cached position carries one scalar per KV head per layer:
the accumulated softmax attention mass it has received. Scores come from an
observation window of the last `w = 16` query rows during prefill, and from the
single query row during decode (free — that product is computed anyway). At
eviction each KV head independently keeps

```
keep = sinks(first 4)  ∪  recent(last r)  ∪  top-B by accumulated score
```

Each head keeps its *own* top-B, so the cache stays rectangular
`[batch, n_kv, B, head_dim]` and no ragged bookkeeping is needed. Cached keys
already carry their RoPE rotation, so dropping and reordering positions is safe.
`get_seq_length()` returns the **true** position count rather than the stored
length — reporting the stored length would silently re-index RoPE for every token
generated after a compression.

**Why it never asks the kernel for attention weights.** SDPA and FlashAttention
never materialise the `n × n` score matrix; forcing eager to recover it costs
`n_q × n² × 2` bytes per layer — **8.6 GB per layer at 16K**, measured. Scores are
instead computed by an explicit small matmul alongside the fast kernel:
`[heads, w, n]` at prefill instead of `[heads, n, n]` (a 1024× reduction at 16K
with w=16), and `[heads, 1, n]` at decode.

**Ordering contract.** `update()` runs *before* the attention call that produces
the scores, so eviction only fires once real scores exist. Full prefill attention
→ scores recorded → the first decode step compresses. Violating this is the bug
in §5.1.

**Cost.** One fp32 scalar per position, per KV head, per layer:
`28 × 8 × n × 4` bytes against a cache of `114,688 × n` bytes = **0.78%**,
length-independent, **measured at 0.78% at every length and budget tested**.
0.39% in fp16. **Zero added parameters** — a runtime statistic, never multiplied
into an output, so the parameter frontier in `RESEARCH_PROGRAM.md` is untouched.

---

## 2. Quality — measured

**Protocol.** rag_dev, 752 paired cases, greedy, `max_new_tokens=32`,
`eos_only`, v4.5 scorer, chat template with `enable_thinking=False`. Identical to
`scripts/eval/run_gates.py`; the harness reproduces the stored gate baseline
**exactly (|Δ| = 0.00e+00 over 510 chat_dev cases)** in `--no-compat` mode.

*Procedure:* `python src/run_eval_sig.py --keep-frac F --dataset rag_dev --compare-to off`

| arm | cache | mean quality | vs no eviction | byte-identical |
|---|---:|---:|---:|---:|
| no eviction | 1.00× | 0.66939 | — | — |
| keep 0.75 | 1.33× | 0.65677 | **−0.01262** | 711/752 (94.5%) |
| keep 0.50 + pool 7 | 2.00× | 0.63371 | −0.03568 | 631/752 |
| keep 0.50 | 2.00× | 0.61401 | −0.05538 | 605/752 |
| keep 0.25, recent 25% | 4.00× | 0.41932 | −0.25007 | 172/752 |
| keep 0.25 | 4.00× | 0.33863 | −0.33076 | 159/752 |

**A 1.33× cache reduction costs 0.013 and leaves 94.5% of replies byte-identical.**

### 2.1 Against a real positional control — and a crossover

*Procedure:* `--recent-frac 1.0` gives `recent = B − 4`, so `heavy() == 0` and the
policy is exactly sinks+recent (StreamingLLM). Identical budget and protocol.

| keep | StreamingLLM | Significance | advantage | paired p | sig wins |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.47545 | **0.65677** | **+0.18132** | 1.3e-34 | 231/263 |
| 0.50 | 0.47682 | **0.61401** | **+0.13719** | 3.9e-08 | 208/318 |
| 0.25 | **0.42208** | 0.33863 | **−0.08345** | 1.5e-02 | 178/405 |

**The sign reverses at keep 0.25.** Significance selection is worth +0.18 and
+0.14 at keep 0.75 and 0.50, and is *worse than doing nothing clever* at keep
0.25. Mechanism: at keep 0.25 on 232-token prompts the budget is ~58, so after 4
sinks and a recent window the heavy set is small; top-32 importance churns 42%
between consecutive query positions and eviction is irreversible, so a small
heavy set chosen from stale scores loses to simply keeping recent tokens.

**Operating rule:** significance selection at keep ≥ 0.5; plain sinks+recent
below it.

### 2.2 Two policy refinements, both measured

- **SnapKV pooling** (max-pool width 7 over scores before top-k): +0.01970 at
  keep 0.50 on rag_dev (0.61401 → 0.63371); −0.005 on chat_dev. Evidence spans
  are worth keeping whole; conversation is not.
- **`recent` scaled with the budget** instead of fixed at 32: +0.08069 at keep
  0.25 at *identical* memory (`mean_kept_ratio` 0.2499 in both), 0.33863 →
  0.41932. This is the largest single-hyperparameter effect measured.

---

## 3. Attention structure — measured

*Procedure:* `experiments/kv_significance/src/concentration.py --cases 8`,
8 rag_dev prompts, 192–265 tokens, last 16 query rows, 28 layers × 16 heads.

| budget k | ρ(k) | ρ(k), sinks removed | error bound 2(1−ρ)·V |
|---:|---:|---:|---:|
| 1 | 0.591 | 0.362 | 0.817 |
| 32 | 0.951 | 0.898 | 0.099 |
| 64 | 0.980 | 0.957 | 0.041 |
| 128 | 0.996 | 0.992 | 0.008 |

**The bound.** Attention output is a convex combination `o = Σ aᵢvᵢ`. Evict and
renormalise with retained mass ρ:

```
‖o − ô‖  ≤  2(1 − ρ)·max‖vᵢ‖
```

**Measured to be necessary but NOT sufficient:** ρ > 0.996 still cost 0.199
end-to-end under a positional policy. The bound controls one attention output;
errors compose across 28 layers and iterate across decode steps.

**Attention sinks.** R2 sends **46.3%** of its attention mass to the first 4
tokens; 14 of 28 layers exceed 50%; layer 27 sends 80.5%; layers 0–2 send under
3%. *Procedure:* `experiments/attention_shape/src/sink_audit.py`.

**Persistence.** Top-32 overlap between consecutive query positions is **0.584**
(chance 0.138 at these lengths). Importance is 4.2× more persistent than random,
but 42% churns every step, and eviction is irreversible.

---

## 4. Efficiency — measured

*Procedure:* `experiments/kv_significance/src/measure_sig.py`, 6 GiB allocator cap.

| length | budget | cache | reduction | peak | decode |
|---:|---:|---:|---:|---:|---:|
| 16,384 | off | 1799 MiB | — | 3128 MiB | 9.8 tok/s |
| 16,384 | 1024 | 118.9 MiB | **15.1×** | 3142 MiB (1.00×) | 17.1 tok/s (**1.74×**) |

- **Memory: strong.** 15.1× smaller cache; the significance matrix is 0.78% of it.
- **Peak: unchanged (1.00×).** The full prompt cache is resident before anything
  can be compressed. A peak win requires chunked prefill (§7).
- **Speed: modest.** 15× less cache buys 1.74× decode, because per-step top-k and
  gather across 28 layers eat most of the bandwidth saving. At 4K, eviction is a
  net **slowdown** (0.68×) — the overhead is fixed while the saving scales.
- **Thermal caveat:** this host's own band is 2.17× (33.8 ms cool vs 71.6–73.2 ms
  at 84 °C). **No throughput claim below 2× is defensible here.**

---

## 4a. Chunked prefill — the length variable eliminated, measured

*Procedure:* `python scripts/kv/chunked_prefill.py --gate` (correctness), then
`experiments/kv_significance/src/measure_chunked.py` (memory), 6 GiB cap,
B=2048, C=4096, recent=512, window=16.

**Correctness first.** With compression disabled, chunked prefill must reproduce
single-shot prefill. Compared at **every position**, fp32:

| chunk | max \|Δlogit\| | mean \|Δlogit\| | argmax agreement |
|---:|---:|---:|---:|
| 32 | 1.116e-04 | 5.268e-06 | 1.0000 |
| 64 | 1.185e-04 | 5.225e-06 | 1.0000 |
| 256 | 9.465e-05 | 2.342e-06 | 1.0000 |

Tolerance 1e-3. fp32 is the honest gate: in bf16 the final-position logit differs
by mean 0.481 on logits of scale ~39 from summation order alone, so a bf16
threshold would either pass everything or reject a correct implementation.

**Then the identity.**

| n | stored KV | resident cache | peak | prefill |
|---:|---:|---:|---:|---:|
| 32,768 | 6,144 | 672.0 MiB | 1744.1 MiB | 7.1 s |
| 65,536 | 6,144 | 672.0 MiB | 1744.4 MiB | 13.3 s |
| 131,072 | 6,144 | 672.0 MiB | **1744.9 MiB** | 27.8 s |

**Peak spread across a 4× length range: 0.0%.** Resident cache is exactly
`(B + C) × 112 KiB = 672 MiB` at every length, matching the identity to the byte.
Un-chunked, 128K needs 14,336 MiB of cache alone against 8,188 MiB of VRAM, and
`base-8KV` OOMs at 32K under this same cap.

**Prefill is linear, not quadratic.** Attention work per chunk is `O(C·(B+C))`
over `n/C` chunks, so total is `O(n·(B+C))`. Measured: n×4 → prefill ×3.9, where
quadratic would be ×16.

### What this does and does not establish

It establishes a **memory and compute** result: peak VRAM is now independent of
prompt length, and 128K is reachable on this card with 4.4 GiB to spare. That
converts the >32K rungs from arithmetic into something measurable, which was the
single largest blocker in the program.

It establishes **nothing about quality at those lengths.** R2's
`max_position_embeddings` is 40,960 with `rope_scaling` null, so positions beyond
~41K are outside the model's trained range and the text produced at 128K is not
meaningful. And no quality number anywhere in this program comes from a prompt
longer than 330 tokens. Chunked prefill removes the memory obstacle to measuring
long-context quality; it does not measure it.


---

## 4b. Passkey retrieval at length — the first long-context behaviour measured

*Procedure:* `experiments/kv_significance/src/passkey.py`. A 5-digit key hidden
at a controlled depth in repetitive filler; the model is asked to repeat it.
Repetitive filler isolates **retrieval** from comprehension. Only runnable
because chunked prefill removed the memory obstacle.

### The model is good at this, contrary to the prior we inherited

Full cache, no eviction, 3 trials x 3 depths:

| n | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

100% at every length and depth up to 32K. This corrects a pessimism carried
through the whole program from the frozen demo suite's 0.2031: that suite is
*reasoning*-hard (multi-hop, `count_with_exclusions` at 0.031), not retrieval.
On pure retrieval this 0.6B checkpoint is reliable.

#### The 32K column is new, and the reason it was missing is a defect

This section previously stopped at 16K and said "above 16K the full cache no
longer fits". That was wrong, and wrong in the way this program keeps having to
catch: **the binding constraint was a default in the harness, not the card.**

`passkey.py --cap-mib` defaults to 6144, and
`torch.cuda.set_per_process_memory_fraction` bounds **reserved** memory while
`max_memory_allocated` reports **allocated**. The gap between them is the caching
allocator's fragmentation, which is large here because `DynamicLayer.update`
grows the cache with `torch.cat` — old and new buffers are live at once. So a run
can die with a reported peak far below its own cap:

| | cap 6144 | cap 7800 |
|---|---|---|
| chunk 1024 | **OOM** at peak 4571 MiB | 1.00, peak 4699 MiB |
| chunk 4096 | **OOM** at peak 4897 MiB | 1.00, peak 4973 MiB |

One variable at a time (P6), and the answer is unambiguous: **the cap decides
feasibility and the chunk size does not.** Chunk moves the allocated peak by 274
MiB — 5.5% — and moves the verdict not at all. An earlier reading of this table
that credited chunk size was made from the diagonal only, changing both variables
at once, and is withdrawn.

Two consequences.

**There is now a measured baseline at 32K.** Eviction to a 2048-token budget
scores 0.89 there, so its cost is 0.11 *against a measurement* rather than against
an assumption. Every earlier statement of eviction cost at 32K was uncontrolled.

**64K remains out of reach, and now for a stated reason.** Its cache alone is
7,168 MiB against 8,188 MiB total with ~430 MiB already resident, before any
allocator fragmentation. A run at cap 8100 OOMs at peak 6,426 MiB. So eviction is
still **required** to measure at 64K and beyond — which is the correct version of
the claim this paragraph used to make about 16K.

*Data:* `results/passkey_ceiling.json` (chunk 1024, cap 7800),
`results/passkey_long.json` (chunk 4096, cap 6144 — the OOM record).

### Depth dominates, not the keep ratio

B=2048 fixed, sweeping n (this is the ratio-vs-absolute test):

| n | depth 0.1 | depth 0.5 | depth 0.9 |
|---:|---:|---:|---:|
| 8,192 | 0.00 | **1.00** | **1.00** |
| 16,384 | 0.00 | 0.00 | **1.00** |
| 32,768 | 0.00 | 0.00 | **1.00** |
| 65,536 | 0.00 | 0.00 | 0.00 |

At 8K, B=2048 *is* keep 0.25 — the ratio the rag_dev sweep called catastrophic
(−0.331) — yet mid and late needles retrieve perfectly. **Damage does not track
the keep ratio.** It tracks how many compression rounds the needle survived.

### Position and eviction, separated

With YaRN ×4 (training-free, `rope_parameters` → usable range 163,840):

| n=65,536, YaRN | B=2048 | B=8192 |
|---|---:|---:|
| depth 0.99 (last chunk, never compressed) | **1.00** | **1.00** |
| depth 0.90 | 0.00 | 0.00 |

**Position is not the limit at 64K.** With YaRN the model retrieves perfectly
when the needle escapes compression, at both budgets. And a 4× larger budget does
not help at depth 0.9 — so it is not a budget limit either.

YaRN also changed the *failure mode* rather than only the score: without it, 64K
output was digit-garbage (`'643210.

The pass '`); with it, coherent text
(`'the key that is us'`). Position was broken; now it is not, and the needle is
simply gone.

### The real limit: prefill-time eviction is query-blind

The question appears only at the end of the prompt. Each chunk is scored against
**its own last 16 query rows**, which for every chunk but the last are filler. A
needle therefore earns no score before the question is ever seen, and is evicted
regardless of budget. That explains the depth pattern exactly: depth 0.99 is in
the final chunk and never compressed; depth 0.9 survives one or two rounds; depth
0.1 survives none.

A first probe of this mechanism — repeating the question at the *start* of the
prompt — moved the outputs in the predicted direction without fixing them
(key 75526 → `'7526'` rather than `'75'`; 85667 → `'8567'` rather than
`'85686578568'`, and the degenerate repetition disappeared). It is a weak test:
the query hint sits at position 0 but the observation window is each chunk's
*last* rows, so the question never becomes a scoring query. **The proper test is
to append the question to each chunk's scoring pass without caching it**, which
is the obvious next step and is not yet done.

**This is the cornerstone finding for long context here.** Memory is solved;
position is solved to 64K by YaRN; what remains is that you cannot know which
tokens matter until you have seen the question.


---

## 4c. Query-aware scoring — the fix for the cornerstone limit

§4b established that prefill-time eviction is query-blind: each chunk is scored
against its own trailing rows, which are filler, so a needle earns no score
before the question is seen. `scripts/kv/chunked_prefill.py:score_with_query`
runs the question through the model as *queries* against the current cache
between chunks, records the attention mass, then truncates the cache back so the
question is never itself cached.

**Exactness is verified, not assumed.** `verify=True` checks byte-equality of
every layer's keys and values before and after the pass. Leaving the question
cached would make every later chunk attend to a question it has not reached,
shift positions, and still produce plausible output.

Cost: one forward over ~10 question tokens per chunk boundary.

### The effect, 32K, budget 2048 (keep 0.0625), YaRN ×4, 3 trials

| depth | blind | query-aware |
|---:|---:|---:|
| 0.1 | 0.00 | **1.00** |
| 0.3 | 0.00 | 0.67 |
| 0.5 | 0.00 | 0.67 |
| 0.7 | 0.00 | **1.00** |
| 0.9 | 1.00 | 1.00 |
| **mean** | **0.20** | **0.87** |

**0.20 → 0.87 at a 16× compression ratio.** This is the single largest quality
effect measured anywhere in the program.

### Sum vs mean: a ranking bug, a wrong fix, and a corrected finding

Scores are a running **sum**, so a position that survived T compression rounds
was scored T times and a newcomer once — the heavy set ratchets shut. Measured
observation counts after 4 chunks: min 1, max 3, and sum-vs-mean top-5 rankings
share only 2 of 5 entries, so the effect is real.

`Policy.normalize` ranks on `sum / observation_count` instead.

| | 32K (6 trials) | 64K (6 trials) | 128K (6 trials) |
|---|---:|---:|---:|
| rank on **sum** (e=0) | 0.89 | 0.55 | 0.22 |
| rank on **mean** (e=1) | 0.83 | **0.89** | **0.39** |

**A correction to an earlier claim in this program.** An initial 3-trial run read
"normalization hurts, 0.87 → 0.60". At 6 trials the 32K arms are tied within one
trial out of 18 (0.89 vs 0.83) and 64K clearly favours the mean (0.89 vs 0.55).
The original claim was noise at n=3 and is withdrawn. The coherent reading is
that the ratchet's severity scales with the number of compression rounds `n/C` —
8 rounds at 32K, 16 at 64K, 32 at 128K — so normalization is a wash where the
ratchet barely bites and a clear win where it does.

**A second correction, to a figure that appeared in review.** The 128K column
previously read `0.44 (n=3)` against `0.39 (n=6)` — a 3-trial two-depth run
compared with a 6-trial three-depth one, which is not a comparison. Both arms have
since been re-run matched at 6 trials × 3 depths and the sum arm is **0.22**, not
0.44. The per-depth figures that circulated with it (sum 1.00 / 0.33 / 0.00) were
noise at n=3 and are withdrawn; the matched values are 0.33 / 0.17 / 0.17.

### The decay exponent: one real effect, and one thing it cannot touch

`Policy.count_exponent` generalises the boolean to `sum / count**e`, so e=0 and
e=1 are the two endpoints above and the interior is reachable. All five arms
matched — 128K, budget 2048 (1.6% keep), YaRN ×4, query-aware, 6 trials × 3
depths, same committed code. Hits out of 6:

| e | depth 0.1 | 0.5 | 0.9 | pooled |
|---:|---:|---:|---:|---:|
| 0.00 (sum) | 2/6 | 1/6 | 1/6 | 4/18 |
| 0.25 | 3/6 | 1/6 | 1/6 | 5/18 |
| 0.50 | 3/6 | 0/6 | 4/6 | 7/18 |
| 0.75 | 2/6 | 1/6 | 4/6 | 7/18 |
| 1.00 (mean) | 1/6 | 1/6 | **5/6** | 7/18 |

Bonferroni over the three depth-wise trend tests, α = 0.0167:

| claim | test | result |
|---|---|---|
| accuracy rises with e at **depth 0.9** | Cochran–Armitage | z = **+2.840**, p = **0.0045** — survives |
| accuracy rises with e at depth 0.5 | Cochran–Armitage | z = +0.000, p = 1.0000 |
| accuracy rises with e at depth 0.1 | Cochran–Armitage | z = −0.804, p = 0.4216 |
| best arm beats worst overall | Fisher, 7/18 vs 4/18 | p = 0.4705 — **not established** |

**The exponent is a real control on late-context retrieval and does exactly
nothing for mid-context.** Overall accuracy does not improve significantly; the
knob redistributes it from early depths to late ones.

That is the mechanism, not a coincidence. A depth-0.9 needle sits in a late chunk,
has survived few compression rounds, and carries a small accumulated sum purely
because it was scored fewer times — dividing by `count**e` restores it. A
depth-0.5 needle has been through roughly fifteen rounds, and **if it was dropped
in an early one it is gone permanently**: eviction is not reversible, so no
ranking statistic can recover a position that is no longer in the cache.

So the mid-context failure is not a ranking problem. That is the sweep's real
result, and it is worth more than a tuned exponent would have been: it removes
an entire class of fix from consideration.

### The depth curve at 128K, pooled over all five arms (n = 30 per depth)

| depth | accuracy | 95% CI (Clopper–Pearson) |
|---:|---:|---|
| 0.1 | 0.367 | [0.199, 0.561] |
| 0.5 | **0.133** | [0.038, 0.307] |
| 0.9 | 0.500 | [0.313, 0.687] |

Depth 0.9 against depth 0.5 is p = **0.0048** (Fisher), which survives correction;
depth 0.1 against 0.5 is p = 0.0716, which does not. A U with a real trough — the
lost-in-the-middle shape, at 1.6% cache retention on a 0.6B model.

Failures are near-misses rather than noise (75526 → `'7552'`, 85667 → `'8567'`,
96659 → `'96969696969'`), so the information is partially retained even where the
answer is wrong.

**Unresolved, and unresolvable on this host as it stands.** Whether the trough
belongs to eviction or to the model cannot be decided, because the uncompressed
128K control does not fit: 14,336 MiB of cache against 8,188 MiB total. Every 64K
and 128K number in this program is measured against no baseline. That is what
makes §4d's mixed precision load-bearing rather than merely tidy — at 3.04× the
full 128K cache is 4,716 MiB, and the control becomes runnable.

### Where this leaves long context

| n | budget | keep | best accuracy | peak |
|---:|---:|---:|---:|---:|
| 32,768 | 2048 | 0.0625 | **0.89** | 1744 MiB |
| 65,536 | 2048 | 0.031 | **0.89** | 1742 MiB |
| 131,072 | 2048 | 0.0156 | ~0.4, depth-biased | 1743 MiB |

**64K retrieval works at a 32× compression ratio on an 8 GiB card.** 128K is
partial and depth-dependent. All of it is *retrieval*; multi-hop reasoning at
length remains untested, and the frozen demo suite indicates that is where this
checkpoint is genuinely weak.

Trial counts are small (3–6 per cell). The 32K sum/mean comparison moved by one
trial in 18 between runs, which is the noise floor to keep in mind for every
number in this section.


## 4d. What to STORE, not just what to keep

Eviction answers *which* positions survive. This answers *in what form*, and the
two costs compose on the same cache, so they have to be measured on the same
scale. Code `experiments/kv_significance/src/{store_probe,mixed_precision}.py`,
data `results/{store_probe,mixed_precision}.json`.

### The scale

Attention-output relative error is uninterpretable on its own. So the probe
computes, in one pass on the same 24 rag_dev prompts across all 28 layers, three
perturbations whose end-to-end cost is already on record in §2:

| reference | attention relerr | rag_dev quality |
|---|---:|---:|
| keep 0.75 | 0.00937 | −0.01262 |
| keep 0.50 | 0.03729 | −0.05538 |
| keep 0.25 | 0.11381 | **−0.33076** |

A candidate below 0.00937 is cheaper than the eviction setting already in use;
one near 0.11381 costs a third of the benchmark.

### The result that matters: the grouping, not the bit width

| scheme | attention relerr | store factor |
|---|---:|---:|
| int8, K per-channel g=128, V per-token | **0.00774** | 1.94× |
| int8, K and V both per-token | 0.04134 | 1.94× |

Same bit width, same bytes, **5.3× the error**. The good variant sits below the
keep-0.75 reference; the naive one lands between keep-0.50 and keep-0.25. Nothing
about "int8" predicts which you get.

### The survivor: significance-gated mixed precision

The cache is not uniform — 46.3% of attention mass is on the first 4 tokens — and
eviction already computes a per-position, per-KV-head score saying where the rest
goes. That score is free here. Sinks stay bf16, the top 25% by score go int8, the
rest int4. Bytes are counted from the real layout including scale overhead
(bf16 4096, int8 2112, int4 1088 B per token per layer), 4K haystack context:

| scheme | attention relerr | store factor |
|---|---:|---:|
| pure int8 | 0.00762 | 1.94× |
| **mixed, top 25% by significance** | **0.00693** | **3.04×** |
| pure int4 | 0.11179 | 3.75× |

**Lower error than pure int8 at 1.6× more compression.** It is a strict Pareto
improvement over both uniform schemes, it reuses a signal that has to be computed
anyway, and it composes with eviction rather than competing with it. Nothing in
this program is implemented in the live cache path yet — this is the strongest
unexploited result here.

### What the same probe killed

int4 uniformly, per-token int8 grouping, cross-layer KV sharing (held-out relerr
0.1720 with a dense adapter that costs 4.19M parameters to remove 2.10M), and any
latent scheme that treats K and V symmetrically (at equal rank, truncating V
costs 2.3× what truncating K costs). Full tables and reasoning in
[`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) §5.

**Not established:** none of these numbers is an end-to-end quality measurement.
The calibration makes them interpretable; it does not make them equivalent. See
§7.


---

---

## 5. Every bug found, and how it was caught

Nine defects, all in code written during this program. **Every one produced a
plausible wrong number rather than a crash, and every one was caught by a control
rather than by reading the code.**

| # | Defect | Detection | Wrong number it produced |
|---|---|---|---|
| 1 | Eviction fired during prefill before scores existed | prefill timing was impossibly **fast** (2.6×) | an entire quality sweep measured a positional policy |
| 2 | `select_keep` pooled before masking sinks | synthetic probe with dominant sinks | 3 of 8 heavy slots spent on positions 4–6; 1 of 4 content tokens kept |
| 3 | `_evict` fell back to zeros with no scores | code audit + contract check | silent sliding window |
| 4 | `_evict` `+inf` lag padding unbounded | audit; safe at 1-token lag, fatal at chunk lag | sinks + newest-chunk |
| 5 | `compress()` skipped layers on shape mismatch | audit | layers left uncompressed while reporting success |
| 6 | `Policy.heavy()` returned 0 silently | audit; live on 20/752 prompts | plain StreamingLLM reported as significance |
| 7 | Score accumulation discarded history | audit | ranking restarted from one forward pass |
| 8 | `t_crit` scaled a t value by a normal-quantile ratio | 400k null simulations | FWER 0.0985 vs 0.05; every training MDE 7–43% too small |
| 9 | Cache bytes summed over a **view** | audit + storage-pointer check | SWA cache saving credited before it happens |

Two harness defects with the same character: `measure.py` held two caches during
timing (producing a false OOM verdict), and `run_eval_sig.py` wrote the unused
`recent` flag into the manifest (reporting every `--recent-frac` run as if the
flag had never been passed — 13%/37% significance-selected where the truth is
63.7%/67.4%).

### The systemic weakness

These are not nine unrelated mistakes. They share one property: **the code has
many paths that produce a valid-looking output from an invalid state.** A missing
score becomes a zero vector; a shape mismatch becomes a skipped layer; an
exhausted budget becomes an empty heavy set; a view becomes a byte count. In each
case the failure is representable as a normal result, so nothing downstream can
tell.

The structural fix is the one applied in §5 rows 3–7: **make the invalid state
unrepresentable, or make it raise.** Where raising would truncate a sweep and
leave partial data (row 6), warn once and export a count into the run manifest so
the condition is visible in the artefact rather than only in a log.

---

## 6. Retractions

| Claim | Published | Correct | Why |
|---|---|---|---|
| Allocation gain | 7.4% | **4.2%**, and 0.00% at mean 128/layer | fabricated ρ(256) column; floor below `heavy()>0` |
| SWA prefill speedup | 3.2× faster | ~1.9× **slower** | kernel dispatch, not attention shape |
| Significance vs StreamingLLM | +0.187/+0.146/+0.204 | **+0.181/+0.137/−0.083** | baseline was a corrupted forward pass; sign wrong at keep 0.25 |
| MDE at k=3 | 9.6pp | **13.9pp** | `t_crit` multiplicity bug |
| SWA decode win | 18.1 vs 11.4 tok/s | gqa-1KV does **24.1** at 224 MiB resident vs SWA's 1792 | compared against the wrong baseline |
| SWA cache saving | 5.1× at 16K | **1.00×** resident at end of prefill | counted on a view |

---

## 7. What is not established

**Two entries here were retired rather than deleted.** "Nothing above 32K" and
"nothing at long context, full stop" were true when written and are false now —
chunked prefill (§4a) and passkey retrieval (§4b, §4c) reach 128K on this host.
They are recorded as superseded because a list of open questions that quietly
drops the ones that got answered is not auditable.

What is genuinely not established:

- **Reasoning at length.** Every long-context result in this program is
  *retrieval* — a 5-digit key in repetitive filler, chosen precisely because it
  isolates retrieval from comprehension. Multi-hop reasoning over a long context
  has never been measured here, and the frozen demo suite indicates that is where
  this checkpoint is genuinely weak. This is the largest open gap.
- **Perplexity at length.** Never measured, at any length, under any policy.
- **Short-context regression under long-context settings.** The quality sweep
  (§2) ran without YaRN; the long-context runs (§4b, §4c) ran with YaRN ×4.
  YaRN rescales RoPE globally, so the two are not the same model and the cost of
  that scaling to ordinary short prompts is unmeasured.
- **End-to-end cost of any storage format.** §4d measures attention-output
  relative error against calibrated reference points; it does not run the
  benchmark. The calibration makes the number interpretable, not equivalent.
- **Any low-rank or MLA scheme with a FIXED basis.** The rank-512 result in §4d
  uses a per-prompt SVD — an oracle fitted to the prompt it is applied to. The
  gap between that and a learned projection is the entire engineering problem and
  it is not measured.
- **Whether an interior decay exponent helps at 128K.** Under measurement; the
  two endpoints are known and neither is uniformly good (§4c).
- **No throughput claim below 2×**, per the thermal band.
- **Generality.** The 46.3% sink share, the per-layer ρ curves and the
  sink-pinning rule are properties of *this* checkpoint, measured on 8 prompts.

---

## 8. Stack-level findings that transfer

**`enable_gqa=True` silently falls back to the SDPA math kernel** on this stack
(sm_89, torch 2.13.0+cu130). At 16 Q / 8 KV / head_dim 128 / seq 16384, bf16:

```
enable_gqa=True, is_causal=True   ->  allocates 16.00 GiB  ->  OOM
repeat_kv to 16 heads, is_causal  ->  peak 224 MiB
```

16 GiB is exactly `16 × 16384² × 4` bytes — the score matrix in fp32.
`transformers` opts in whenever `attention_mask is None`. It surfaces as an OOM
at a plausible length, so it reads as "the card is too small" rather than "the
wrong kernel ran". **Any long-context work on this host is affected.** Workaround:
`experiments/attention_shape/src/compat.py`.

**Switching SDPA kernels changes 23% of greedy replies** while moving mean
quality by +0.00138. Mathematically equivalent, numerically not; one flipped
token cascades through greedy decoding.

**chat_dev cannot detect attention-level damage and reports it backwards.** Its
aggregate *rose* (+0.00247 at keep 0.75) while only 361/510 replies stayed
identical. Damage lengthens replies, and `prompt_function_alignment` (a
response-mode heuristic in 502/510 cases, weight 0.28 combined with
`task_adequacy`) rewards length, while `brevity` (0.01) and `completeness` (0.02)
cannot outvote it. **Never conclude from a chat_dev aggregate that an
attention-level intervention is harmless.**

### 9.1 The sliding-window cache saving was counted on a view

Re-measured with de-duplicated storage pointers, immediately after prefill at
16K (`experiments/attention_shape/results/measure_fixed.json`):

| variant | logical | **resident** | ratio |
|---|---:|---:|---:|
| `base-8KV` | 1792.0 MiB | 1792.0 MiB | 1.00 |
| `swa-w1024-g4` | 351.9 | **1792.0** | **5.09** |
| `swa-w256-g4` | 279.9 | **1792.0** | **6.40** |
| `gqa-2KV` | 448.0 | 448.0 | 1.00 |
| `gqa-1KV` | 224.0 | 224.0 | 1.00 |

**At end of prefill a sliding-window model holds exactly as much KV as the
baseline.** The saving appears only after the first decode step replaces the
view. Peak memory is set at end of prefill — which is precisely why SWA never
improved it. GQA's saving is resident from the moment it exists.

Decode at 16K, same run: `gqa-1KV` **24.1 tok/s at 224 MiB resident** against
`swa-w256-g4` 20.7 at 1792 MiB and `base-8KV` 7.7. The earlier claim that
sliding window "wins decode, 18.1 vs 11.4" compared it to the baseline rather
than to the axis under test, and is withdrawn: **GQA beats every SWA variant on
decode at a smaller resident cache.**

### 9.2 One audit prediction refuted

The audit held that `base-8KV @ 32768 → OOM` was purely a harness artefact and
would fit at 5,345 MiB against a 6,144 MiB cap. The harness defect was real and
is fixed (`peak_alloc_mib` and `peak_timed_mib` now agree in all 12 cells), **but
the verdict stands** — base-8KV and every SWA variant still OOM at 32K while both
GQA variants measure. The reconstruction was arithmetic; the measurement
disagrees. Recorded because an audit finding that does not survive its own test
should be marked, not quietly dropped.


---

## 10. Killed, with reasons

- **Sliding-window schedules.** Raise peak memory, *shorten* reachable context
  (every SWA variant OOMs at 32K where GQA fits), and collapse to empty strings
  below the window because they evict the sinks. Damage is a cliff: one token
  past the edge costs as much as two hundred.
- **Geometry-aware eviction.** In centred key space top-k already keeps 60.4 of
  64 slots distinct against 61.9 for random. In attention-profile space —
  where the similarity field *is* heavy-tailed — suppressing near-duplicates
  makes attention error monotonically worse (+33.7% at τ=0.90 to +3.2% at
  τ=0.99) and worse than a matched-random shuffle at every τ. Similar attention
  profiles mean **corroboration**, not duplication.
- **Unhealed GQA head merging** as a quality point: 0.83498 → 0.11384 at 2KV.
  Needs a healing run per rung.
