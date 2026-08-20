# Significance-gated KV eviction: what it is, why it can work, and what would kill it

Design notes for the significance-matrix experiment. Written before the
implementation, so §6 is a set of predictions rather than a description.

---

## 1. What this is, named honestly

The proposal — keep a running significance score per cached position, update it
every forward pass, and delete cache entries whose score falls below threshold —
is **KV-cache eviction by accumulated attention**. It is an active research family
and naming it is worth more than pretending otherwise, because the failure modes
are already charted:

| method | selection signal | what it adds |
|---|---|---|
| **StreamingLLM** | position only | keep first *k* tokens ("sinks") + recent window |
| **H2O** | accumulated attention ("heavy hitters") | heavy hitters + recent window, fixed budget |
| **Scissorhands** | persistence of importance | shows important tokens stay important |
| **SnapKV** | attention from an observation window at the end of the prompt | compress the prompt cache once, after prefill |
| **TOVA** | last query row only | simplest policy, one score per step |
| **PyramidKV** | per-layer budgets | deeper layers need less |

What is specific to this project is not the mechanism but the target: a **0.6B
model on an 8 GB card**, where we have already measured the two things that decide
whether eviction is safe here.

---

## 2. The bound: why eviction is controllable at all

Attention output is a convex combination of value vectors:

```
o = sum_i a_i v_i ,    a_i >= 0 ,    sum_i a_i = 1
```

Keep a subset `K`, drop the rest, renormalise over what survives:

```
o_hat = (1/rho) sum_{i in K} a_i v_i ,     rho = sum_{i in K} a_i
```

Then, writing `V = max_i ||v_i||`:

```
o - o_hat = (1 - 1/rho) sum_K a_i v_i  +  sum_{K^c} a_i v_i

||(1 - 1/rho) sum_K a_i v_i||  <=  ((1-rho)/rho) * rho * V  =  (1-rho) V
||sum_{K^c} a_i v_i||          <=  (1-rho) V

  ==>   ||o - o_hat||  <=  2 (1 - rho) V
```

**The entire question reduces to retained attention mass.** Not to sequence
length, not to model size, not to the eviction policy's cleverness — those only
matter through `rho`. This is an upper bound and a loose one (it assumes the
dropped mass and the renormalisation error point the same way, and it uses
`max||v||` rather than the local spread), so measured quality should be better
than it suggests. But it makes the design falsifiable: if a small budget cannot
hold high `rho`, no policy can be safe.

**Caveat that keeps this honest.** The bound is per layer, per head, per step.
Errors compose across 28 layers and accumulate across decode steps, and a
perturbed token feeds the next step. The bound controls one attention output; it
does not directly bound end-to-end generation drift. That is what the quality
eval measures.

---

## 3. Measured on R2 — the bound instantiated

`src/concentration.py`, 8 rag_dev prompts, scored on the final 16 query rows,
averaged over 28 layers and 16 heads.

| budget *k* | rho(k) | rho(k) **with sinks removed** | error bound |
|---:|---:|---:|---:|
| 1 | 0.591 | 0.362 | 0.817 V |
| 2 | 0.704 | 0.488 | 0.591 V |
| 4 | 0.791 | 0.610 | 0.418 V |
| 8 | 0.858 | 0.720 | 0.285 V |
| 16 | 0.910 | 0.817 | 0.180 V |
| 32 | 0.951 | 0.898 | 0.099 V |
| 64 | 0.980 | 0.957 | 0.041 V |
| 128 | 0.996 | 0.992 | 0.008 V |

Three readings.

**Attention is extremely concentrated.** A single key carries 59% of the mass on
average; 32 keys carry 95%. Against prompts of 41–330 tokens that is a large
majority of the mass in a small minority of positions — exactly the regime where
the bound in §2 is favourable.

**The concentration is not only the sink, and this is the control that mattered.**
We measured previously that R2 sends 46.3% of its attention to the first 4
tokens. If that were the whole story, a significance matrix would be pointless —
StreamingLLM already keeps sinks plus a recent window, for free, with no scores
to maintain. Zeroing the first 4 positions and renormalising, top-32 still holds
**89.8%** and top-64 holds **95.7%**. There is real content-dependent structure to
find, so tracking significance buys something a positional rule cannot.

**Persistence is the weak point.** Overlap of the top-32 set between consecutive
query positions is **0.584** (min 0.489 at layer 11, max 0.658). Chance overlap at
these lengths is ~0.138, so importance is 4.2x more persistent than random — real,
but 42% of the set still churns every step. Since eviction is irreversible, this rules
out an aggressive greedy policy on instantaneous scores and dictates three things:
accumulate rather than snapshot, keep a protected recent window so a token that is
about to matter has not already been deleted, and choose a budget comfortably
above the size of the "important" set.

---

## 4. The cost accounting

The significance matrix is one scalar per cached position, per KV head, per layer.
It does **not** scale with hidden size or vocabulary — a natural guess, and the
reality is much cheaper.

```
significance = n_layers * n_kv_heads * n            (scalars)
             = 28 * 8 * n
KV cache      = 2 * n_layers * n_kv_heads * head_dim * n * 2 bytes
              = 2 * 28 * 8 * 128 * n * 2
```

Ratio, with fp32 scores:

```
significance / cache = (28*8*n*4) / (114688*n) = 896n / 114688n = 0.78%
```

| context | KV cache (bf16) | significance (fp32) | overhead |
|---:|---:|---:|---:|
| 4K | 448 MiB | 3.5 MiB | 0.78% |
| 32K | 3584 MiB | 28 MiB | 0.78% |
| 128K | 14336 MiB | 112 MiB | 0.78% |

Length-independent at 0.78%, halving to 0.39% in fp16. **Zero added parameters** —
it is a runtime statistic, never multiplied into an output, so the parameter
frontier in `docs/RESEARCH_PROGRAM.md` is untouched. The user's instinct that this
"does not perform regular multiplication when into parameter counts" is exactly
right and is what makes it attractive against GQA surgery, which permanently
removes 29–51M parameters and needs a healing run per rung.

---

## 5. The obstacle, and the standard resolution

**Attention weights are not available for free.** SDPA and FlashAttention never
materialise the `n x n` score matrix — that is the entire point of them — so a
significance score computed from attention weights has no source. Forcing eager
attention to recover them costs `n_q * n^2 * 2` bytes per layer: 8.6 GB at 16K on
this model. We measured that ceiling directly; it is fatal.

The resolution has two halves, and neither needs the full matrix:

**Decode is free.** With a single query token, the score row is `q · K^T` of shape
`[heads, 1, n]`. That is one matrix-vector product per layer — the same work
attention already does, and it costs `n * d` rather than `n^2 * d`.

**Prefill uses an observation window.** Score only the final `w` query rows against
all keys: `[heads, w, n]` instead of `[heads, n, n]`. With `w = 16` at 16K that is
a 1024x reduction, and §3 shows a 16-row window already ranks positions well. This
is SnapKV's construction.

So significance is computed by an explicit small matmul *alongside* the fast
attention kernel, never by asking the kernel for its internals. The main attention
path keeps SDPA and its `is_causal` fast path — which matters, because we measured
that passing an explicit mask instead costs ~1.9x prefill and raises peak memory.

**Honest limitation on peak memory.** Compressing after prefill bounds the
*steady-state* cache, not the *peak*: the full prompt cache exists transiently
before it is compressed. That is the same shape of problem the sliding window had,
and the fix is the same — chunked prefill with compression between chunks. Phase 3.

---

## 6. Policy, and predictions

Policy, following from §3 rather than from taste:

```
keep = sinks(first 4)  ∪  recent(last r)  ∪  top-B by accumulated score
```

- **sinks** because 46.3% of the mass lives there and evicting them collapsed the
  model when the sliding window did it;
- **recent** because persistence is only 0.584, so a token that has not mattered
  yet may matter next step;
- **accumulated, not instantaneous**, for the same reason — accumulation smooths
  the 42% churn.

Scores are summed over the query heads sharing each KV head, since eviction
removes a position for the whole group.

**Predictions, recorded now:**

| | prediction | falsified by |
|---|---|---|
| **S1** | At budget 128 with sinks+recent, chat_dev quality is within 0.005 of base | a larger drop |
| **S2** | Evicting *without* sink protection collapses the model, as the sliding window did | no collapse |
| **S3** | Decode throughput improves roughly in proportion to cache reduction | flat throughput |
| **S4** | Peak prefill memory is **not** improved without chunked prefill | an improvement |
| **S5** | At matched cache, significance eviction beats both the window schedule and unhealed GQA | either wins |

S2 is the one that connects this experiment to the last one: it predicts that the
*same* mechanism which destroyed sliding window is present here, and that the
significance policy neutralises it by construction, because sinks score highest.

S4 is a prediction of failure, recorded so that a peak-memory win cannot later be
claimed without the chunked-prefill work that would actually earn it.

---

## 7. A bug that invalidated the first sweep, and how it was caught

Recorded in full because the failure is instructive and because the first set of
results looked *good* in the places that mattered least.

**The bug.** `SignificanceLayer.update()` runs **before** the attention call that
produces the scores. The first implementation evicted as soon as the stored
length exceeded the budget — which is true on the very first prefill call, when
no scores exist for that layer. The fallback path then selected `sinks + recent`.

So prefill attention was computed over a sinks-plus-recent window instead of the
full prompt: **precisely the sliding window the previous experiment showed
collapses this model.** The entire first quality sweep measured a sliding window
wearing a significance badge.

**How it surfaced.** Not from the quality numbers, which were plausible, but from
an efficiency measurement that was *too good*: prefill at 16K came out 2.6x
faster with eviction on. Prefill cannot get faster if attention is being computed
correctly — the same work has to happen. Chasing that impossible number found it.

A direct probe confirmed it: at a 512-token prompt with budget 64, the recorded
score length for layer 0 was **64, not 512**, with one eviction during prefill.

**The fix** is one condition — evict only once real scores exist — which restores
the intended SnapKV ordering: full prefill attention, then scores, then the first
decode step compresses.

**The guard** is a new regression check (`verify_significance.py` [4b]) asserting
that prefill sees every key and performs zero evictions. It now passes 6/6.

**Why the original checks missed it.** They exercised the *explicit* `compress()`
path, constructing the cache without a policy and compressing afterwards. The
eval used the *self-compressing* path. Both were "tested"; the one that shipped
was not. That is the lesson worth carrying: a verification suite has to exercise
the code path the experiment actually runs, not an equivalent-looking one.

Invalidated artefacts are kept under `results/_invalidated_prefill_bug/` rather
than deleted, so the corrected numbers can be compared against what the bug
produced.

---

## 8. What survived the bug

Three findings do not depend on the eviction path and still stand.

**The cost accounting is confirmed to two decimals.** Measured
`score_pct_of_cache` is **0.78%** at every length and budget tested, exactly the
figure derived in §4, with zero added parameters.

**S2 is refuted, in the way that supports the thesis.** Evicting with
`protect_sinks=False` gives essentially the same output as with protection,
because the accumulated scores rank the sink positions at the top by themselves
(the best of the first four has mean rank 3.81). Significance-based selection
protects the attention sink without being told to — the open question left by the
sliding-window result.

**chat_dev cannot measure this class of damage, and reports the opposite.** In the
buggy sweep, chat_dev quality *rose* (+0.01364 at `keep 0.25`) while only **3 of
510** replies stayed byte-identical. Replies grew 63% longer (15.49 -> 25.26
visible tokens) and the dimensions show why:

```
brevity_appropriateness    0.9677 -> 0.8111   -0.157   (weight 0.01)
response_completeness      0.9929 -> 0.8615   -0.132   (weight 0.02)
prompt_function_alignment  0.4169 -> 0.5370   +0.120   (weight 0.12)
task_adequacy              0.4169 -> 0.5370   +0.120   (weight 0.16)
```

On chat_dev `prompt_function_alignment` comes from `prompt_shape_fallback` in 502
of 510 cases — a response-mode heuristic, not a correctness signal — and longer
replies trip more modes. The damage was real (it was a sliding window); the
aggregate moved the wrong way. **This is the sharpest P8 case in the program:
never conclude from a chat_dev aggregate that an attention-level intervention is
harmless.**

A fourth observation from the buggy run is worth keeping as a methodological
note: SnapKV pooling appeared to be a no-op on rag_dev (752/752 identical
replies) while changing 299 of 510 on chat_dev. Pooling demonstrably alters
selection (overlap 75/116 on a controlled input); it looked inert because the
model had already collapsed to a single stereotyped output — **732 of 752 replies
were the identical abstention string**. A metric can be insensitive because the
system is broken rather than because the intervention does nothing.

---

## 9. Efficiency, measured after the fix

`results/measure_sig.json`, R2, 6 GiB cap, 16 decode tokens, recent 128.

| length | budget | cache after decode | significance | peak | prefill | decode |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | off | 455 MiB | — | 1466 MiB | 456 ms | 17.0 tok/s |
| 4K | 1024 | 112 MiB (4.1x) | 0.88 MiB (0.78%) | 1470 MiB (1.00x) | 472 ms | 11.5 (0.68x) |
| 16K | off | 1799 MiB | — | 3129 MiB | 4930 ms | 6.2 tok/s |
| 16K | 1024 | 112 MiB (16.1x) | 0.88 MiB (0.78%) | 3143 MiB (1.00x) | 5847 ms | 7.1 (1.15x) |
| 16K | 512 | 56 MiB (32.1x) | 0.44 MiB (0.78%) | 3143 MiB (1.00x) | 5954 ms | 6.8 (1.10x) |

**S4 confirmed.** Peak memory is 1.00x — unchanged. The full prompt cache is
resident before anything can be compressed, exactly as predicted. Improving peak
needs chunked prefill with compression between chunks; it cannot be claimed
without that work.

**S3 largely refuted.** A 16x smaller cache buys only **1.15x** decode, and at 4K
eviction is a net **0.68x slowdown**. Decode is bandwidth-bound on the cache, so
the saving is real, but the machinery costs a per-step scoring matmul plus a
top-k and a gather in every layer, and at these cache sizes that overhead eats
almost all of it. The natural remedy is `slack` — amortise eviction over many
steps instead of firing every step — which is measured separately.

This is the honest efficiency verdict at the lengths reachable here: **the
cache shrinks enormously, and almost none of that converts into speed.**

### `slack` recovers part of it — provisionally

Firing eviction on every decode step means a top-k and a gather in all 28 layers
per token. `slack` defers it: evict only once the stored length exceeds
`budget + slack`, so the cost is amortised over many steps. At 16K with
`budget=1024, slack=4096` decode measured **17.1 tok/s against a 9.8 tok/s
baseline — 1.74x**, versus 1.15x with `slack=0`. Directionally this confirms that
per-step eviction overhead, not the bandwidth model, was the limiter.

**Treat that number as provisional.** In the same run the *baseline* prefill
measured **362,185 ms** where an earlier run of the identical configuration gave
4,930 ms — a 73x discrepancy that can only be host-memory thrashing (free system
RAM was down to 2.87 GB, and Windows/WDDM spills rather than failing). A run whose
baseline is corrupted cannot be trusted for its other columns either. The clean
comparison needs re-measuring on an unloaded host, with both `slack` settings in
one process.

Note also that `results/measure_sig.json` currently holds only the two rows from
that thrashed run; the table above comes from the preceding clean run and should
be regenerated.

---

## 10. Quality, measured against the fixed implementation

`run_sweep.sh`, protocol-identical to the gate baselines.

### rag_dev — the test that can actually detect damage

| keep | mean quality | vs off | byte-identical |
|---:|---:|---:|---|
| off | 0.66939 | — | — |
| 0.75 | 0.65677 | **−0.01262** | 711/752 (95%) |
| 0.50 | 0.61401 | −0.05538 | 605/752 (80%) |
| 0.50 + pooling | 0.63228 | −0.03711 | 630/752 (84%) |
| 0.25 | 0.33863 | −0.33076 | 159/752 (21%) |
| 0.25, recent scaled | 0.41932 | −0.25007 | 172/752 (23%) |

**A 1.33x cache reduction costs 0.013 and leaves 95% of replies byte-identical.
A 2x reduction with pooling costs 0.037.** Below half the prompt it falls apart,
which is unsurprising for 232-token prompts in which nearly every token is
evidence.

Two policy details earn their place. **Pooling helps where it was designed to**
(+0.018 on rag_dev at keep 0.50) and slightly hurts on conversation (−0.005 on
chat_dev) — evidence spans are worth keeping whole. **Scaling the recent window
with the budget matters at aggressive settings**: at keep 0.25 it is worth
+0.081 on rag_dev, because a fixed `recent=32` consumes almost the entire
allowance and leaves nothing for scoring.

### The comparison that matters: significance vs position, at matched budget

The invalidated run turned out to be a gift. Because its fallback selected
`sinks + recent`, it is exactly **StreamingLLM at an identical cache budget**,
evaluated under an identical protocol. That is the baseline this method has to
beat, and it was produced by accident:

| keep | StreamingLLM | Significance | advantage |
|---:|---:|---:|---:|
| 0.75 | 0.47007 | 0.65677 | **+0.18670** |
| 0.50 | 0.46839 | 0.61401 | **+0.14562** |
| 0.25 | 0.13501 | 0.33863 | **+0.20362** |

**S5 confirmed against the window schedule.** At every budget, choosing *what* to
keep by accumulated attention beats choosing *where* to keep by position, by
0.15–0.20 quality. The idea works, and it works for the reason predicted in §1:
the sink is retained either way, but a positional rule throws away the evidence
along with the filler, while a significance rule keeps the evidence and throws
away the filler.

On chat_dev the same comparison appears to favour StreamingLLM by 0.003–0.034.
That is the §8 artefact, not a real preference: StreamingLLM does more damage,
damage makes replies longer, and chat_dev's alignment heuristic rewards length.
The chat_dev column carries no valid signal about eviction and should not be read
as one.

### Predictions, settled

| | prediction | outcome |
|---|---|---|
| **S1** | budget 128 keeps chat_dev within 0.005 | **vacuous** — chat_dev cannot detect this damage (§8) |
| **S2** | eviction without sink protection collapses the model | **REFUTED** — scores protect the sink unaided |
| **S3** | decode improves in proportion to cache reduction | **largely refuted** — 16x cache buys 1.15x, or ~1.74x with `slack` (provisional, §9) |
| **S4** | peak prefill memory is NOT improved | **CONFIRMED** — 1.00x |
| **S5** | beats the window schedule at matched cache | **CONFIRMED** — +0.15 to +0.20 on rag_dev |

---

## 11. Where this leaves the idea

**The selection mechanism is validated.** Significance beats position decisively
at matched budget, protects attention sinks with no special-casing, costs 0.78%
of the cache it manages, and adds zero parameters. The premise was right.

**The efficiency payoff is not yet there.** Peak memory is unchanged (S4), and a
16x smaller cache converts to only 1.15x decode because per-step top-k and gather
in 28 layers eat the bandwidth saving. `slack` looks like it recovers a large part
of that but needs re-measuring on an unloaded host.

**And the regime that would justify it is unreachable here.** Every quality number
above comes from prompts of 41–330 tokens, where compression has almost no
redundancy to exploit. The case for eviction is 32K of context with real filler,
and nothing in this repository exceeds ~2.7K tokens
(`docs/EXPERIMENT_DESIGN_KV_LONGCTX.md` §2a). That a 1.33x reduction is nearly
free *at the worst possible length* is encouraging, but it is an extrapolation,
not a measurement.

The honest ordering of remaining work:

1. **`slack` re-measured cleanly** — decides whether the decode win is real.
2. **Chunked prefill** — the only route to a peak-memory improvement, and the
   only way the method helps the 8 GB ceiling rather than just steady state.
3. **A long-context corpus** — without it, the central claim cannot be tested at
   the lengths it is designed for. This is the same blocker as everywhere else in
   the program, and it is now blocking three separate lines of work.
