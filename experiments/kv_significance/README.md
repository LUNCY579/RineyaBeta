# Significance-gated KV eviction

Keep a running significance score per cached position, update it on every forward
pass, and delete the entries that stop earning their space.

Self-contained. Nothing outside `experiments/kv_significance/` is written, and
**`scripts/train/` is not touched** — `code_fingerprint()` hashes every `.py`
there and an edit would invalidate every checkpoint's resume path. The two
behaviour changes needed (`scripts/kv/significance.py`'s attention wrapper and the
sibling experiment's `compat.py`) are process-local monkey patches.

---

## The question, and why it follows from the last experiment

The attention-shape experiment ended with a clear negative: a sliding window
raises peak memory, *shortens* reachable context, and collapses the model the
moment the window stops covering the prompt. The mechanism was measured — R2
sends **46.3% of its attention mass to the first 4 tokens**, and a positional
window evicts exactly those.

That failure points at the fix. A sliding window evicts by **position**. If we
evict by **importance** instead, the sink is the *most* important thing in the
cache and gets protected automatically, for free. That is this experiment.

Named honestly, this is **KV-cache eviction by accumulated attention** — the
H2O / SnapKV / Scissorhands / StreamingLLM family. Naming it is worth more than
claiming novelty, because the failure modes are already charted. See
[THEORY.md](THEORY.md) §1.

---

## The mathematics, in one line

Attention output is a convex combination `o = Σ aᵢvᵢ`. Evict a set, renormalise,
and with retained mass `ρ`:

```
‖o − ô‖  ≤  2(1 − ρ)·max‖vᵢ‖
```

**Everything reduces to retained attention mass.** Not sequence length, not model
size, not the policy's cleverness — those matter only through `ρ`. Derivation in
[THEORY.md](THEORY.md) §2, including the caveat that this bounds one attention
output and not end-to-end generation drift.

---

## What is here

```
README.md                  this file
THEORY.md                  the bound, the cost accounting, the predictions
run_sweep.sh               quality vs budget, on two contrasting datasets
src/
  concentration.py         Phase 0: is attention concentrated enough? (no GPU cost)
  (the implementation itself lives at scripts/kv/significance.py, promoted
   to the main tree once it worked; chunked prefill is beside it)
  verify_significance.py   six correctness checks — run this before believing anything
  run_eval_sig.py          quality sweep, protocol-identical to run_gates.py
  measure_sig.py           cache bytes, prefill latency, decode throughput
  measure_chunked.py       peak memory against prompt length, chunked
  passkey.py               retrieval at depth x length; --yarn, --query-aware,
                           --count-exponent
  allocation.py            per-layer budget allocation by water-filling
  store_probe.py           what to STORE each kept position as, on a calibrated
                           scale (quantisation, low-rank, cross-layer sharing)
  mixed_precision.py       significance-gated mixed precision: bf16 sinks,
                           int8 for the top-p by score, int4 for the rest
  summarize.py             collects results/ into FINDINGS.md
results/
  concentration.json       measured rho(k), with and without sinks
  measure_sig.json         efficiency after the fix
  measure_chunked_long.json  peak memory 16K-128K under chunked prefill
  pk_*.json                passkey runs; the file name states the arm
  store_probe.json         calibration, quantisation, low-rank, CLA
  mixed_precision.json     relerr and store factor per precision scheme
  FINDINGS.md              generated summary
  _invalidated_prefill_bug/  artefacts from the buggy run, kept for comparison
```

Run everything with `../../.venv/Scripts/python.exe`.

```bash
python src/concentration.py          # decisive, cheap, can kill the idea
python src/verify_significance.py    # 6 correctness checks
python src/measure_sig.py            # cache, prefill, decode
python src/store_probe.py            # what to store, on a calibrated scale
python src/mixed_precision.py        # significance-gated mixed precision
bash run_sweep.sh                    # quality vs budget, ~80 min
```

---

## Phase 0: the idea survives its own kill-test

`src/concentration.py`, 8 rag_dev prompts, final 16 query rows, 28 layers × 16 heads:

| budget *k* | ρ(k) | ρ(k) **sinks removed** | error bound |
|---:|---:|---:|---:|
| 1 | 0.591 | 0.362 | 0.817 V |
| 32 | 0.951 | 0.898 | 0.099 V |
| 64 | 0.980 | 0.957 | 0.041 V |
| 128 | 0.996 | 0.992 | 0.008 V |

**Attention is extremely concentrated** — one key carries 59% of the mass, 32 keys
carry 95%.

**And it is not only the sink.** This was the control that could have ended the
experiment: if all the concentration were the first 4 tokens, StreamingLLM
already keeps those for free and a significance matrix would buy nothing. With
the sinks zeroed and renormalised, top-32 still holds **89.8%**. There is real
content-dependent structure to track.

**Persistence is the weak point.** Top-32 overlap between consecutive queries is
**0.584** (chance ≈ 0.138). Real, but 42% churns every step — and eviction is
irreversible. That is why the policy accumulates rather than snapshots, and
protects a recent window.

## Cost accounting

The significance matrix is one scalar per position, per KV head, per layer. It
does **not** scale with hidden size or vocabulary:

```
significance / cache  =  (28·8·n·4) / (114688·n)  =  0.78%
```

Length-independent at 0.78% in fp32, 0.39% in fp16, and **zero added parameters** —
it is a runtime statistic, never multiplied into an output, so the parameter
frontier is untouched. Compare GQA surgery, which permanently removes 29–51M
parameters and needs a healing run per rung.

## The obstacle and how it is handled

SDPA and flash never materialise the `n×n` score matrix, so scores have no free
source, and forcing eager costs 8.6 GB per layer at 16K (measured). The
resolution needs no full matrix: **decode is free** (a single query row is
`q·Kᵀ`, shape `[heads, 1, n]`), and **prefill uses an observation window** of the
last `w` rows — `[heads, w, n]`, a 1024× reduction at 16K with `w=16`. The main
attention path keeps SDPA and its `is_causal` fast path.

---

## Correctness: 6/6

`src/verify_significance.py`, budget 64 on a 197-token prompt:

| check | result |
|---|---|
| score shapes `(1, 8, 197)` × 28 layers, no future leakage | PASS |
| **recording is a no-op — reply byte-identical to baseline** | PASS |
| compression to budget; `get_seq_length` reports true 197, not 64 | PASS |
| sinks rank highly **even with protection off** (mean rank 3.81) | PASS |
| **prefill sees the full context, 0 evictions during prefill** | PASS |
| retained mass ρ = 0.9423 → bound 0.115·V | PASS |

The no-op check is the analogue of the sliding-window P1 test that caught real
bugs last time: machinery that should change nothing must change nothing.

`get_seq_length` returning the *true* position count rather than the stored
length matters — reporting the stored length would silently re-index RoPE for
every token generated after a compression.

---

## A bug that invalidated the first sweep

Documented prominently because the first results looked plausible and were not.

`SignificanceLayer.update()` runs **before** the attention call that produces the
scores. The first implementation evicted as soon as stored length exceeded the
budget — true on the very first prefill call, when no scores exist yet. The
fallback path then selected `sinks + recent`, so **prefill attention was computed
over a sliding window instead of the full prompt** — precisely the thing the
previous experiment showed collapses this model.

It surfaced not from the quality numbers but from an efficiency number that was
*too good*: prefill at 16K appeared **2.6× faster** with eviction on. Prefill
cannot get faster if attention is computed correctly. A direct probe confirmed
it — at a 512-token prompt with budget 64, layer 0 recorded a score length of
**64, not 512**.

The fix is one condition (evict only once real scores exist), restoring SnapKV's
ordering: full prefill attention → scores → compress on the first decode step. A
regression check now asserts prefill sees every key with zero evictions.

**Why the original 5 checks missed it:** they exercised the explicit `compress()`
path, building the cache *without* a policy. The eval used the self-compressing
path. Both were tested; the one that shipped was not. Invalidated artefacts are
kept in `results/_invalidated_prefill_bug/` rather than deleted.

---

## What stands

**The cost accounting, confirmed to two decimals.** Measured share of cache is
**0.78%** at every length and budget, exactly as derived, with zero added
parameters.

**S2 refuted, in the way that supports the thesis.** With `protect_sinks=False`
the output is essentially unchanged, because the scores rank the sinks at the top
by themselves (best of the first four has mean rank 3.81). **Significance-based
eviction protects the attention sink automatically** — the open question left by
the sliding-window failure, answered.

**chat_dev cannot detect this class of damage and reports the opposite.** In the
buggy sweep its aggregate *rose* (+0.01364) while only 3 of 510 replies stayed
identical. Replies grew 63% longer; `brevity` (weight 0.01) and `completeness`
(0.02) fell while `prompt_function_alignment` and `task_adequacy` (0.28 combined)
rose — and on chat_dev that alignment score is a response-mode *heuristic* in 502
of 510 cases, not a correctness signal. The sharpest P8 case in the program:
**never conclude from a chat_dev aggregate that an attention-level intervention
is harmless.**

## Efficiency after the fix

| length | budget | cache | significance | peak | decode |
|---:|---:|---:|---:|---:|---:|
| 4K | off | 455 MiB | — | 1466 MiB | 17.0 tok/s |
| 4K | 1024 | 112 MiB (4.1×) | 0.78% | 1470 (1.00×) | 11.5 (**0.68×**) |
| 16K | off | 1799 MiB | — | 3129 MiB | 6.2 tok/s |
| 16K | 1024 | 112 MiB (**16.1×**) | 0.78% | 3143 (1.00×) | 7.1 (1.15×) |
| 16K | 512 | 56 MiB (**32.1×**) | 0.78% | 3143 (1.00×) | 6.8 (1.10×) |

**S4 confirmed** — peak is unchanged (1.00×). The full prompt cache exists before
anything can be compressed. Improving peak requires chunked prefill; it cannot be
claimed without that work.

**S3 largely refuted** — a 16× smaller cache buys only 1.15× decode, and at 4K
eviction is a net **slowdown** (0.68×). The per-step scoring matmul, top-k and
gather in every layer eat almost the entire bandwidth saving. `slack` (amortising
eviction over many steps) is the obvious remedy and is being measured.

**The honest verdict at reachable lengths: the cache shrinks enormously and
almost none of it converts into speed.**

## Quality, against the fixed implementation

### rag_dev — the metric that can detect damage

| keep | mean quality | vs off | byte-identical |
|---:|---:|---:|---|
| off | 0.66939 | — | — |
| 0.75 | 0.65677 | **−0.01262** | 711/752 (95%) |
| 0.50 + pooling | 0.63371 | −0.03568 | 631/752 (84%) |
| 0.50 | 0.61401 | −0.05538 | 605/752 (80%) |
| 0.25, recent scaled | 0.41932 | −0.25007 | 172/752 (23%) |
| 0.25 | 0.33863 | −0.33076 | 159/752 (21%) |

**A 1.33× cache reduction costs 0.013 and leaves 95% of replies byte-identical.**
2× with pooling costs 0.036. Below half the prompt it breaks down — expected for
232-token prompts where nearly every token is evidence.

Pooling helps where designed (+0.020 on rag_dev at keep 0.50, −0.005 on chat_dev).
Scaling the recent window with the budget is worth +0.081 at keep 0.25, confirming
that a fixed `recent=32` was starving the scoring component.

> ## ⚠ The earlier StreamingLLM comparison was retracted; this is the real one
>
> The original figures (+0.18670 / +0.14562 / +0.20362) were read from
> `results/_invalidated_prefill_bug/` and are **withdrawn**. That run was not a
> sliding window (replaying the pre-fix `select_keep` with zero scores keeps
> positions 0–141 plus 200–231 — a *prefix* plus recent), its forward pass was
> acausal (eviction fired inside prefill with `attention_mask=None`, so SDPA ran
> `is_causal=True` with L≠S and an upper-left mask), and it did not respond to
> its own variable (keep 0.75 vs 0.50 gave 731/752 identical replies, Δ 0.0017).
>
> A real control has now been run: `Policy(budget=B, sinks=4, recent=B−4)`, so
> `heavy()==0` and the policy is exactly sinks+recent. Identical budget,
> identical protocol, current code, full prefill.

### Significance vs a real positional control, matched budget

| keep | StreamingLLM | Significance | advantage | paired p | sig wins |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.47545 | **0.65677** | **+0.18132** | 1.3e-34 | 231/263 |
| 0.50 | 0.47682 | **0.61401** | **+0.13719** | 3.9e-08 | 208/318 |
| 0.25 | **0.42208** | 0.33863 | **−0.08345** | 1.5e-02 | 178/405 |

**There is a crossover, and the corrupted baseline hid it.** At keep 0.75 and
0.50 significance selection is worth +0.181 and +0.137 — close to the withdrawn
figures, and now supported by a control that actually responds to its budget. At
keep 0.25 the sign **reverses**: a plain sinks+recent policy beats significance
selection by 0.083 (p = 0.015). The withdrawn number there was +0.204, i.e. the
wrong sign on the most aggressive setting.

The mechanism is consistent with what was already measured. At keep 0.25 on
232-token prompts the budget is ~58, so after 4 sinks and a recent window the
heavy set is small; top-32 importance churns 42% between consecutive query
positions, and eviction is irreversible. A small heavy set chosen from stale
scores is worse than simply keeping recent tokens. Significance selection needs
enough budget to be worth the risk of committing to it.

Practical reading: **use significance selection at keep ≥ 0.5, and plain
sinks+recent below it.** At keep 0.25 the three configurations rank
StreamingLLM 0.42208, recent=25%-of-budget 0.41932, significance 0.33863 — the
first two are within noise of each other and both beat the third.

### WITHDRAWN — "significance vs position at matched budget", +0.15 to +0.20

A section here reported StreamingLLM at 0.47007 / 0.46839 / 0.13501 against
significance at 0.65677 / 0.61401 / 0.33863, concluded a +0.15 to +0.20 advantage
**at every budget**, and called it the result the experiment was built to obtain.
It is withdrawn, and it contradicted the corrected section immediately above it.

Those StreamingLLM numbers came from the *invalidated* prefill-bug run, whose
fallback was described as "exactly StreamingLLM, produced by accident". It was
not: the fallback kept positions 0–141 plus 200–231 and its forward pass was
acausal, and the control did not respond to its own variable (731/752 replies
identical across settings). A real positional control was then run.

Against that real control the conclusion **reverses at the tightest budget**:

| keep | StreamingLLM (real control) | Significance | advantage |
|---:|---:|---:|---:|
| 0.75 | 0.47545 | 0.65677 | **+0.18132** |
| 0.50 | 0.47682 | 0.61401 | **+0.13719** |
| 0.25 | **0.42208** | 0.33863 | **−0.08345** |

So significance beats position at keep ≥ 0.5 and **loses to it at keep 0.25**.
That crossover is the finding; "+0.15 to +0.20 at every budget" is not.
## Verdict

**The selection mechanism is validated.** It beats positional eviction decisively,
protects sinks with no special-casing, costs 0.78% of the cache, adds zero
parameters.

**The efficiency payoff is not yet there.** Peak memory unchanged; 16× smaller
cache buys only 1.15× decode (≈1.74× with `slack`, provisional).

**The regime that would justify it is unreachable here.** Every number above comes
from 41–330-token prompts, where there is almost no redundancy to exploit. That a
1.33× reduction is nearly free *at the worst possible length* is encouraging but
is an extrapolation, not a measurement.

## What happened next — this experiment is not the end of the story

Everything above is measured on prompts of 41-330 tokens, because at the time
nothing longer could be run. Two later pieces of work removed that limit, and
their results live in **[docs/FINDINGS_KV_PROGRAM.md](../../docs/FINDINGS_KV_PROGRAM.md)**
(§4a-4c). In short:

- **Chunked prefill** (`scripts/kv/chunked_prefill.py`) bounds the resident cache
  at `B + C` tokens with the prompt length eliminated. Measured peak is
  **1744 MiB, flat from 16K to 128K (0.0% spread)**, against 14,336 MiB of cache
  alone for an un-chunked 128K prompt. Gated as a proven no-op first: max
  |Δlogit| 1.19e-04 at every position, argmax agreement 1.0000.
- **Passkey retrieval** is measurable now, and the model is *good* at it — 1.00 at
  every depth and every length to **32K** on a full cache
  (`results/passkey_ceiling.json`). The old "16K" ceiling was `--cap-mib 6144`,
  a default in our own harness, not the card.
- **Query-aware scoring** (`score_with_query`) fixes the limitation this
  experiment could not see: prefill-time eviction scores each chunk against its
  own trailing rows, which are filler, so a needle earns no score before the
  question is asked. Making scoring query-aware moved 32K retrieval from
  **0.20 to 0.87** at keep 0.0625.
- **Retrieval works at ~15x compression at both long lengths**, and what controls
  it is the keep *fraction*, not the length: 6.5% keep gives **0.97 at 64K**
  (B=4096, peak 2002 MiB) and **0.92 at 128K** (B=8192, peak 2521 MiB), while
  1.6% keep gives 0.61 and 0.53. All 12 paired needles. No depth law is
  established — the ordering at 128K/B=2048 does not replicate at 64K.

The conclusion this README reaches — that eviction cannot be evaluated in the
regime it was designed for — was correct when written and is no longer true.

## Open

1. **`slack` re-measured on an unloaded host** — decides whether the decode win is
   real. The one measurement taken was contaminated (baseline prefill 362 s vs
   4.9 s clean, from host-memory thrashing).
2. ~~Chunked prefill~~ — **done**, see above and FINDINGS §4a.
3. ~~A long-context corpus~~ — **done** for retrieval; a passkey generator with
   depth × length control now exists (`src/passkey.py`). Position-bucketed
   perplexity and long-context multi-hop remain unbuilt.
4. ~~A decay term between sum and mean ranking~~ — **tested; direction real,
   magnitude not established.** `Policy.count_exponent` ranks on `sum / count**e`.
   Five arms at 128K showed a monotone gain at depth 0.9 and none at depth 0.5.
   The arms were paired with each other, so the comparison itself is sound, but
   the stated p-values came from tests that model them as independent groups. A
   re-run under the paired design is in flight.
5. ~~Is the 128K trough eviction's fault or the model's?~~ — **wrong question;
   there is no trough.** It was an artefact of each depth drawing different
   needles. Re-measured with 12 identical needles per cell, 128K at B=2048 gives
   0.42 / 0.33 / 0.83 — late beats early, early and middle indistinguishable — and
   at B=8192 the depth dependence disappears entirely (permutation p = 0.8879).
6. ~~Does 128K work?~~ — **yes, at the budget that works everywhere else.** 0.92
   across depths at B=8192 (6.3% keep), peak 2521 MiB. Accuracy is flat in length
   at constant keep fraction: at **6.5% keep**, 0.97 at 64K against 0.92 at 128K,
   both 12 paired needles. What was previously read as a 128K weakness was B=2048,
   which is 1.6% keep at that length. FINDINGS §4c.
7. ~~Is 6.5% keep a general operating point or a 128K coincidence?~~ — **general.**
   The 64K ladder (B ∈ {1024, 2048, 4096} = 1.6 / 3.3 / 6.5% keep, 12 paired
   needles) gives 0.61 / 0.81 / 0.97, with a within-needle trend of p = 0.0015 at
   depth 0.1 and 13 discordant needles across depths moving one way with zero
   reversals. `results/pk_64k_ladder.json`, FINDINGS §4c.
8. **A quality reference above 32K.** The uncompressed control fits to 32K only
   (7,168 MiB of cache at 64K against 8,188 MiB total). int8 KV at 1.94× makes 64K
   reachable and mixed precision at 3.04× makes 128K reachable — §4d — so the
   store work is the instrument, not a side quest.
9. **Multi-hop reasoning at length.** Everything measured at length so far is
   *retrieval*. The frozen demo suite indicates reasoning is where this
   checkpoint is genuinely weak. This is now the largest gap in the program.

### How to run a comparison here without repeating our mistakes

Every cell draws the same needle sequence (`--seed`), so cells are paired and
`scripts/design/paired_passkey.py` can test them properly. Two rules that cost us
real work to learn:

- **Never compare cells from runs with different `--depths` or `--budgets` lists.**
  Before the pairing fix one RNG stream ran across the whole nested loop, so the
  needle set depended on the shape of the run. It no longer does, but a result
  file predating the fix has no `trials` field and `paired_passkey.py` will refuse
  it rather than guess.
- **Use at least 12 trials.** Exact McNemar with every discordant pair pointing
  one way gives p = 2/2ⁿ, so 6 needles floor at p = 0.0625 and cannot reach
  significance however clean the effect.
