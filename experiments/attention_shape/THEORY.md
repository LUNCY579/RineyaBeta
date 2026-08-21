# Where attention wastes memory, and what can be done about it

Theory notes for the attention-shape experiment. Written before the quality
sweep, so the predictions in §6 are predictions rather than descriptions.

Companion figures (`figures/`):

| | |
|---|---|
| `01_attention_variants.svg` | MHA / GQA / MQA / MLA — what each one stores |
| `02_mla_vs_qwen_dataflow.svg` | Qwen3's GQA layer vs an MLA layer, with the absorption identity |
| `03_cache_growth.svg` | cache vs context length — which curves bend |
| `04_iso_cache.svg` | at one fixed budget, how much context each shape buys |

---

## 1. The redundancy is real, but it is not where it looks

A natural first instinct is that multi-head attention wastes parameters on the
query side. It does carry weight there — `q_proj` and `o_proj` are 4.2M of R2's
15.7M parameters per layer — but **queries are never cached**. Only K and V
persist across decode steps. Compressing `W_Q` buys parameters and a little
prefill compute; it does nothing whatsoever for the memory wall.

The genuine redundancy is one level down, and it is structural.

`W_Q` and `W_K` never appear separately in the function a layer computes. They
appear only through the bilinear form

```
q · k  =  h_mᵀ (W_Qᵀ W_K) h_n
```

That product is a `d × d` matrix of rank at most `d_head`. The same holds on the
output side: `W_V` and `W_O` only ever act through the composite `W_V W_O`. So
each head carries a `d_head × d_head` **gauge freedom** — replace `W_Q → W_Q R`
and `W_K → W_K R⁻ᵀ` for any invertible `R` and the model computes exactly the
same function. The two-matrix parameterisation is a choice, not a necessity.

This is what makes low-rank restructuring of attention *possible in principle*.
It is not free money — representing a rank-`d_head` form still costs
`2·d·d_head` parameters — but it means the cached object does not have to be
"K and V, per head". It can be anything from which K and V are recoverable.

---

## 2. RoPE is what stops you exploiting it naively

The gauge freedom above holds for vanilla attention. Qwen3 uses rotary position
embeddings, and RoPE inserts a position-dependent rotation *between* Q and K:

```
q_m · k_n  =  h_mᵀ W_Qᵀ R(m−n) W_K h_n
```

`R(m−n)` depends on the *pair* of positions. You can no longer fold `W_Qᵀ W_K`
into one precomputable matrix, because whatever you precompute would have to
commute with every rotation. Only transforms commuting with `R` survive, and
that is a much smaller set.

This single fact is the reason the best-known exploitation of the redundancy
needed an extra trick, and it is worth stating plainly because it is the part
most often skipped: **RoPE and low-rank KV compression are in tension, and any
scheme that claims both must say how it resolves that.**

---

## 3. Multi-head latent attention, formally

MLA (DeepSeek-V2/V3) caches a low-rank latent instead of per-head K and V.

**Compression.** For each token, project the hidden state down once:

```
c_t = W_DKV h_t                       c_t ∈ ℝ^{d_c},  d_c ≪ 2·n_kv·d_head
```

Per-head keys and values are *defined* as up-projections of that latent:

```
k_t^C = W_UK c_t          v_t^C = W_UV c_t          (per head)
```

**Absorption — why K and V never need to exist.** Substituting the definitions
into the attention score and the output:

```
q_t · k_s^C  =  (W_UQ c_t^Q)ᵀ (W_UK c_s)  =  c_t^Qᵀ (W_UQᵀ W_UK) c_s
o_t          =  Σ_s a_s W_UV c_s          =  W_UV ( Σ_s a_s c_s )
```

In the first line `W_UQᵀ W_UK` is a single position-independent matrix that can
be folded into the query projection once, offline. In the second, `W_UV` moves
outside the sum and composes with `W_O`. So at inference **neither `k^C` nor
`v^C` is ever materialised** — the only thing stored per token is `c_t`.

**Decoupled RoPE.** Absorption breaks the moment RoPE enters, for exactly the
reason in §2: `c_tᵀ W_UQᵀ R(t−s) W_UK c_s` has a position-dependent matrix in
the middle. MLA's resolution is to carve off a small subspace that carries
position and is shared across heads:

```
k_t^R = RoPE(W_KR h_t)                k_t^R ∈ ℝ^{d_r},  shared by all heads
key_t = [ k_t^C ‖ k_t^R ]             concatenation
```

Only `k^R` gets RoPE; the compressed part stays absorbable. The cache holds
`c_t` and `k_t^R`, nothing else.

**Cost, for R2's shape** (28 layers, bf16):

```
GQA-8KV :  2 × 8 × 128            = 2048 dims/token/layer  = 112.0 KiB/token
MLA     :  d_c + d_r = 256 + 64   =  320 dims/token/layer  =  17.5 KiB/token
```

A 6.4× reduction. The comparison that matters is not against 8KV but against
**MQA at 14.0 KiB/token**: MLA costs 25% more memory than MQA while keeping all
16 heads distinct, where MQA has collapsed them onto one shared key subspace.
Capacity is reduced by *rank*, not by *head count* — and rank is a much finer
dial than an integer head count with a hard floor at 1.

---

## 4. Four axes, one budget

Every scheme here spends the same resource. What differs is which dimension it
compresses, and — crucially — what else it changes.

| Axis | Cache | Prefill attention FLOPs | Decode bandwidth | Reachable how |
|---|---|---|---|---|
| **GQA** heads `g` | `÷ 8/g` | **unchanged** | `÷ 8/g` | surgery + healing |
| **SWA** window `w` | `→ min(n, w)`, **saturates** | `÷ ~n/2w` on sliding layers | `→ w` | **config only** |
| **CLA** share `s` | `÷ s` | unchanged | `÷ s` | weight tying + healing |
| **MLA** rank `d_c` | `(d_c+d_r) / 2·g·d_head` | ~unchanged | same ratio | new modelling code |

Two entries in that table deserve emphasis.

**GQA does not reduce attention FLOPs.** The number of query–key products is set
by the *query* head count, which GQA leaves at 16. Fewer KV heads means fewer
bytes to store and to stream during decode, but prefill does the same arithmetic.
So GQA helps memory and decode; it does not help prefill.

**Sliding window is the only axis whose advantage grows with length.** Every
other scheme is a constant factor: a straight line on the log-log plot of
`03_cache_growth.svg`, shifted down but never bent. Sliding-window curves bend,
because a sliding layer simply stops growing past its window. From `shapes.py`,
for `w=1024` with 4 global layers:

```
  4K   2.8×      16K   5.1×      32K   5.9×      64K   6.4×      128K   6.7×
```

Taken to its limit this is a qualitative change, not a quantitative one: a model
whose layers *all* slide has an O(1) cache and therefore unbounded context at
constant memory. The price is that no information can travel further than the
window, which is why every practical design keeps a few global layers — those
layers are the only path for long-range information, and they alone make the
total cache grow linearly again.

---

### Correction: the SWA cache saving is logical, not resident

The cache figures above were counted with `numel() * element_size()`.
`DynamicSlidingWindowLayer.update` keeps `full[:, :, -w+1:, :]` — a **view** onto
the full prefill allocation — so that count credits a saving the allocator has
not made. Measured with de-duplicated storage pointers
(`measure.py:cache_bytes_pair`, `results/measure_fixed.json`), immediately after
prefill at 16K:

| variant | logical | **resident** | ratio |
|---|---:|---:|---:|
| `base-8KV` | 1792.0 MiB | 1792.0 MiB | 1.00 |
| `swa-w1024-g4` | 351.9 MiB | **1792.0 MiB** | **5.09** |
| `swa-w256-g4` | 279.9 MiB | **1792.0 MiB** | **6.40** |
| `gqa-2KV` | 448.0 MiB | 448.0 MiB | 1.00 |
| `gqa-1KV` | 224.0 MiB | 224.0 MiB | 1.00 |

**At the end of prefill a sliding-window model holds exactly as much KV as the
full-attention baseline.** The window saving only materialises once the first
decode step replaces the view with a fresh tensor. GQA's saving, by contrast, is
resident from the moment it exists.

This completes the §4a mechanism rather than replacing it. The byte-identical
peaks for `w=256` and `w=1024` (3832.5 MiB both) now have two causes, not one:
the `[1,1,n,n]` mask *and* a resident cache that is identical between them
because both are views onto the same full allocation. It also explains directly
why SWA never improved peak memory — peak is set at end of prefill, which is
exactly the moment the saving does not exist.

The reduction figures quoted elsewhere in this document as "cache" are
**logical** occupancy. They are the right quantity for checking the analytic
model in `shapes.py`, which predicts logical occupancy, and the wrong quantity
for any memory claim.

### One audit prediction refuted

The audit also held that `base-8KV @ 32768 -> OOM` was purely a harness artifact
— `measure.py` held two caches during timing — and that a single prefill would
fit (reconstructed at 5,345 MiB against a 6,144 MiB cap). The harness defect was
real and is fixed (`peak_alloc_mib` and `peak_timed_mib` now agree in all 12
cells), **but the verdict stands: base-8KV and every SWA variant still OOM at
32K, while both GQA variants measure.** The reconstruction was arithmetic, not a
measurement, and the measurement disagrees.


---

## 4a. Measured: the window saving is real, the prefill saving is not

The FLOP table above says a sliding layer should compute `n·w` instead of `n²/2`
attention scores, so prefill ought to get cheaper. **It does not, on this stack.**
Measured on R2 at 16K (`results/measure_main.json`):

| variant | cache | peak alloc | prefill | decode |
|---|---:|---:|---:|---:|
| `base-8KV` | 1792 MiB | 3128 MiB | 2368 ms | 11.4 tok/s |
| `swa-w1024-g4` | 352 MiB | **3832 MiB** | **4510 ms** | 18.1 tok/s |
| `swa-w256-g4` | 280 MiB | **3832 MiB** | 4670 ms | 18.1 tok/s |

Three things to read off this.

**The cache saving is exactly as predicted** — 5.1× at `w=1024`, matching
`shapes.py` to 0.03%.

**Prefill gets ~1.9× slower, not faster.** A sliding window here is expressed as
a dense additive mask, not as a windowed kernel. Passing an explicit mask
defeats SDPA's `is_causal` fast path, and the kernel still does the full `n²`
work and then throws most of it away. The theoretical `n/2w` saving is
unreachable without a kernel that takes a window parameter.

**Peak memory goes *up*, and the smoking gun is that `w=256` and `w=1024` have
byte-identical peaks (3832.5 MiB).** Peak cannot depend on the cache, which
differs between them. Instrumenting a real forward (`src/mask_audit.py`) shows
what does:

```
base-8KV      create_causal_mask                 -> None (is_causal fast path)   peak 3128.5 MiB
swa-w1024-g4  create_sliding_window_causal_mask  -> [1,1,16384,16384] bool 256 MiB  peak 3832.5 MiB
```

The baseline gets no mask at all and SDPA takes its fast path. A sliding window
is expressed as a dense `[1, 1, n, n]` boolean mask — 256 MiB at 16K, and it
scales with **context, not window**, which is exactly why shrinking `w` from
1024 to 256 shrinks the cache and leaves the peak untouched. Downstream, an
explicit mask also blocks the `is_causal` path, so total transient memory rises
by ~2.1 GiB, far more than the mask tensor alone.

**The practical consequence is the one that matters: sliding window does not
extend reachable context here — it shortens it.** At 32K under a 6 GiB cap,
every SWA variant OOMs while `gqa-2KV` (peak 2573 MiB) and `gqa-1KV`
(peak 2111 MiB) both succeed. A cache-only analysis predicts the opposite, and
would have been wrong.

So sliding window on this stack buys **persistent cache and decode throughput**
(1.6x at 16K) and costs **prefill latency, prefill peak memory, and the maximum
context that fits at all**. Whether that is a good trade depends entirely on
which resource binds: it is attractive for a long multi-turn session whose cache
accumulates and whose cost is decode, and actively counterproductive for the
goal this program actually has, which is reaching longer contexts on a small
card. It is emphatically not the free win the FLOP table implies.

The remedy is a kernel with native windowing (FlashAttention exposes
`window_size`, and does not materialise the mask). `flash_attn` is **not
installed** in this environment, so that is an unexplored branch, not a fix
already in hand.

---

## 5. The asymmetry in what each axis costs to reach

This is the part a pure quality comparison would miss, and it may be the
experiment's most useful output.

- **Sliding window changes no weights.** It is `use_sliding_window`,
  `sliding_window`, and `layer_types` in a config, applied at load time. Free,
  instant, reversible, and it composes with anything.
- **GQA head reduction is irreversible weight surgery** that produces a broken
  model until a healing run repairs it — one run per rung.
- **CLA** is weight tying: cheap to apply, still needs healing.
- **MLA** needs an attention module that does not exist for Qwen3 in this stack,
  plus initialisation (SVD of the existing K/V projections) plus healing.

So "which axis is better" is not only about quality per byte. **A free option
that merely ties a paid one still wins.** The experiment is designed to report
cost alongside quality for exactly this reason.

---

## 6. Predictions, recorded before the quality sweep

Stated so they can fail.

**P1 — Sliding window is exactly free below the window.** With `w ≥ 512` and
gate prompts of ≤330 tokens, every token lies inside every window, so the model
is *mathematically identical* to the baseline. Prediction: **byte-identical
replies on all 510 chat DEV and 752 RAG DEV cases**, and identical quality
scores to the last decimal.
*This is a correctness test before it is a science test.* Anything less than
bit-identical means the masking or cache path has a bug, and no later number
from this harness can be trusted.

**P2 — GQA surgery degrades immediately, at any context length.** Head merging
removes capacity that short prompts use just as much as long ones. Prediction:
unhealed `gqa-2KV` and `gqa-1KV` collapse on the existing gates even at ≤330
tokens, where sliding window is provably free. This is the sharpest contrast the
existing evals can draw, and it needs no new corpus.

**P3 — Small windows degrade gracefully, and by category.** With `w=128` against
RAG DEV prompts averaging 232 tokens, roughly the first half of the evidence
falls outside the window on later tokens. Prediction: degradation concentrates in
categories requiring integration across the whole document — `multi_fact_supported`,
`conflict`, `supported_with_distractor` — and spares categories where the answer
is local, such as `social` and `supported`. Per category, never aggregate (P2 of
the research program).

**P4 — The two axes fail differently, and that is the finding.** Sliding window
truncates along the *time* axis: full head capacity, no distant context. GQA
compresses along the *capacity* axis: global reach, degraded fidelity everywhere.
Prediction: at matched cache, SWA wins on tasks whose evidence is local and loses
on tasks needing long-range integration, while GQA does the reverse. If instead
one axis dominates at every category and every length, the multi-axis framing
does not pay and the experiment says so.

**P5 — MLA should dominate both, and is the reason to keep it in view.** Same
cache as MQA, capacity near MHA. Not implemented here; this prediction is
recorded so that if P4 shows both existing axes have hard limits, the case for
writing the module is already framed.

---

## 6a. Outcomes — what the sweep actually found

Recorded against the predictions above, including the one that was wrong.

| | prediction | outcome |
|---|---|---|
| **P1** | `w>=512` is a no-op below the window: 510/510 byte-identical | **CONFIRMED exactly.** 510/510 identical, quality delta `-0.00000` |
| **P2** | unhealed head merging breaks even at short context | **CONFIRMED.** `gqa-2KV` 0.83498 -> 0.11384; `gqa-1KV` -> 0.02360; 0/510 identical |
| **P3** | small windows degrade *gradually*, concentrated in integration-heavy categories | **REFUTED.** Not gradual, not category-specific — a cliff |
| **P4** | the two axes fail differently | partially: they fail at different *thresholds*, not in different *categories* |
| **P5** | MLA should dominate both | untested, and now better motivated — see below |

### P1 is the cleanest result in the experiment

With `w=512` against chat_dev prompts of at most 199 tokens, every token sits
inside every window, so sliding attention is mathematically the identity. It
came back **510/510 byte-identical with a quality delta of exactly zero**. That
is simultaneously a confirmation of the prediction and a proof that the mask,
cache and eval paths are wired correctly — a bug anywhere in them would have
shown up as drift.

### P3 was wrong, and the way it was wrong is the finding

Damage does not scale with how much context falls outside the window:

| prompt tokens past `w=128` | n | base | `swa-w128-g4` | delta |
|---|---:|---:|---:|---:|
| inside the window | 20 | 0.8886 | 0.8886 | **+0.0000** |
| 1–100 over | 307 | 0.6863 | 0.0197 | −0.6666 |
| 101–200 over | 422 | 0.6487 | 0.0068 | −0.6419 |
| 200+ over | 3 | 0.3877 | 0.0000 | −0.3877 |

One token past the edge costs as much as two hundred, and the model emits empty
strings. A natural experiment makes it sharper still: the only rag_dev category
left untouched is `social`, and its 20 cases are 41–45 tokens — **exactly** the
20 prompts in the whole set that fit inside `w=128`. Every other category runs
past the window and collapses.

**The mechanism, measured on R2 rather than cited** (`src/sink_audit.py`):
attention mass on the first 4 tokens averages **46.3%** across layers, with
**14 of 28 layers above 50%** and layer 27 at **80.5%**. The model routes almost
half its attention to positions a naive sliding window evicts first. This is the
attention-sink phenomenon, and it explains both the cliff and its
context-independence.

The remedy is known and cheap — pin the first few tokens in every window
(StreamingLLM-style sinks). `transformers`' `sliding_window` does not do this,
so it cannot be reached by config alone, which removes the single biggest
advantage the window axis had.

### What this does to the axis question

The verdict on this stack is now clear, and it is not the one the cache-only
analysis predicted:

- **Sliding window** does not extend reachable context (it *shortens* it, §4a),
  and below the window it is free while above it the model collapses. There is
  no usable middle. It buys decode throughput and persistent cache only.
- **KV-head reduction** extends reachable context and degrades smoothly enough
  to be repairable — but every rung costs a healing run, and unhealed it is
  catastrophic.

So the multi-axis framing did **not** pay in the direction expected. The window
axis is not a free alternative to head reduction; it is a different trade with a
hard discontinuity. That strengthens rather than weakens the case for **P5/MLA**,
which is the only option on the table that reduces cache without either evicting
positions or collapsing heads — it compresses what each position stores, leaving
every position present and every head distinct.

---

## 7. What the harness had to fix before any of this was measurable

Two findings from building it, both of which would have silently corrupted
results.

**`enable_gqa=True` falls back to the SDPA math kernel on this stack.** Measured
on RTX 4070 Laptop / sm_89 / torch 2.13.0+cu130, at 16 Q heads, 8 KV heads,
`head_dim=128`, `seq=16384`, bf16:

```
enable_gqa=True, is_causal=True   ->  tries to allocate 16.00 GiB  ->  OOM
repeat_kv to 16 heads, is_causal  ->  peak 224 MiB
```

16 GiB is exactly `16 × 16384² × 4 bytes` — the full score matrix in fp32.
`transformers` opts into `enable_gqa` whenever `attention_mask is None`, on the
stated assumption that this avoids the math kernel. On this build it does not.
The failure appears as an out-of-memory error at a plausible length, so it reads
as "the card is too small" when it is really "the wrong kernel was chosen".
`src/compat.py` disables it locally and documents the measurement so the claim
can be rechecked rather than trusted.

**This confounded the first pilot.** Sliding-window layers pass an explicit
attention mask, which makes `use_gqa_in_sdpa` return False — so SWA variants
took the fast `repeat_kv` path while the baseline took the broken one. The
resulting "3.2× prefill speedup for sliding window" was an artefact of kernel
dispatch, not a property of sliding window. It is recorded here rather than
quietly dropped, because it is precisely the kind of result that would have
survived into a conclusion if the two paths had not been compared directly.

Any prefill-latency claim in this experiment therefore comes only from runs with
the compat patch applied, and the analytic FLOP model in §4 is the check on it:
a measured speedup far exceeding `n/2w` is evidence of a kernel effect, not of
attention arithmetic.
