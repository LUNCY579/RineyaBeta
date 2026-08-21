# Significance-gated KV eviction

Keep a running significance score per cached position, update it on every forward
pass, and delete the entries that stop earning their space.

Self-contained. Nothing outside `experiments/kv_significance/` is written, and
**`scripts/train/` is not touched** — `code_fingerprint()` hashes every `.py`
there and an edit would invalidate every checkpoint's resume path. The two
behaviour changes needed (`src/significance.py`'s attention wrapper and the
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
  significance.py          the implementation: scoring, cache, eviction policy
  verify_significance.py   six correctness checks — run this before believing anything
  run_eval_sig.py          quality sweep, protocol-identical to run_gates.py
  measure_sig.py           cache bytes, prefill latency, decode throughput
  summarize.py             collects results/ into FINDINGS.md
results/
  concentration.json       measured rho(k), with and without sinks
  measure_sig.json         efficiency after the fix
  FINDINGS.md              generated summary
  _invalidated_prefill_bug/  artefacts from the buggy run, kept for comparison
```

Run everything with `../../.venv/Scripts/python.exe`.

```bash
python src/concentration.py          # decisive, cheap, can kill the idea
python src/verify_significance.py    # 6 correctness checks
python src/measure_sig.py            # cache, prefill, decode
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
| 0.50 + pooling | 0.63228 | −0.03711 | 630/752 (84%) |
| 0.50 | 0.61401 | −0.05538 | 605/752 (80%) |
| 0.25, recent scaled | 0.41932 | −0.25007 | 172/752 (23%) |
| 0.25 | 0.33863 | −0.33076 | 159/752 (21%) |

**A 1.33× cache reduction costs 0.013 and leaves 95% of replies byte-identical.**
2× with pooling costs 0.037. Below half the prompt it breaks down — expected for
232-token prompts where nearly every token is evidence.

Pooling helps where designed (+0.018 on rag_dev at keep 0.50, −0.005 on chat_dev).
Scaling the recent window with the budget is worth +0.081 at keep 0.25, confirming
that a fixed `recent=32` was starving the scoring component.

> ## ⚠ RETRACTED — the StreamingLLM comparison below is invalid
>
> The "StreamingLLM" column was read from `results/_invalidated_prefill_bug/`,
> the run both this document and the README declare invalid. Characterising it
> as "an exact StreamingLLM baseline produced by accident" was wrong on three
> counts, all verified:
>
> 1. **It is not a sliding window.** Replaying the pre-fix `select_keep` with
>    all-zero scores at n=232, budget=174 keeps positions **0–141 plus 200–231** —
>    a *prefix* plus recent, close to the opposite of a window.
> 2. **The forward pass was corrupted, not merely differently-policied.** Eviction
>    fired inside prefill with `attention_mask=None`, so SDPA ran `is_causal=True`
>    with L≠S and an upper-left mask: query row 0 saw one key, and rows past the
>    budget saw end-of-prompt RoPE positions. Acausal and position-misaligned.
> 3. **It does not respond to its own variable.** In that run, keep 0.75 vs keep
>    0.50 gives **731/752 byte-identical replies** (Δ 0.0017), where the fixed
>    policy moves 609/752 and Δ 0.0428. A control insensitive to the quantity it
>    controls is not a control.
>
> The advantage figures (+0.18670 / +0.14562 / +0.20362) and the S5 verdict are
> withdrawn. A real control — `Policy(budget=B, sinks=4, recent=B−4)`, heavy=0,
> full prefill, current code — is being run; until it lands, **no matched-budget
> comparison against a positional policy has been made.**
>
> The absolute numbers are unaffected: keep 0.75 costing 0.01262 with 711/752
> byte-identical replies is measured against no eviction and stands.

### The comparison that matters: significance vs position, matched budget

The invalidated run turned out to be a gift — its fallback selected `sinks +
recent`, making it **exactly StreamingLLM at an identical budget and protocol**:

| keep | StreamingLLM | Significance | advantage |
|---:|---:|---:|---:|
| 0.75 | 0.47007 | 0.65677 | **+0.18670** |
| 0.50 | 0.46839 | 0.61401 | **+0.14562** |
| 0.25 | 0.13501 | 0.33863 | **+0.20362** |

**Choosing *what* to keep by attention beats choosing *where* to keep by position,
by 0.15–0.20 at every budget.** Both retain the sink; the positional rule throws
out the evidence with the filler, the significance rule keeps the evidence. This
is the result the experiment was built to obtain.

(On chat_dev the comparison appears to favour StreamingLLM by 0.003–0.034. That is
the scorer artefact above — more damage makes replies longer and the alignment
heuristic rewards length. The chat_dev column carries no valid signal here.)

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

## Open

1. **`slack` re-measured on an unloaded host** — decides whether the decode win is
   real. The one measurement taken was contaminated (baseline prefill 362 s vs
   4.9 s clean, from host-memory thrashing).
2. **Chunked prefill** — the only route to a peak-memory win, and the only way
   this helps the 8 GB ceiling rather than just steady state.
3. **A long-context corpus** — without it the central claim cannot be tested at
   the lengths it is designed for. Same blocker as the rest of the program, now
   gating three separate lines of work.
