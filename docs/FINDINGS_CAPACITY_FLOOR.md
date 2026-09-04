# The capacity floor, and what compression is actually buying

Preregistration: [`PREREG_CAPACITY_FLOOR.md`](PREREG_CAPACITY_FLOOR.md) (read §3a —
I found a confound in my own prediction while the sweep was running and recorded the
separating arm before the numbers came in). Follows
[`FINDINGS_COMPRESSION_IMPROVES_QUALITY.md`](FINDINGS_COMPRESSION_IMPROVES_QUALITY.md).

Analyser `scripts/design/analyse_capacity_floor.py` → artifact
`experiments/kv_significance/results/capacity_floor.json`, re-derives with `--check`.
All runs: `rag_long4k`, n = 600, seed 20260826, sinks 4, recent-frac 0.15, pool 7,
window (6,0), against the uncompressed reference `c7-off-n600`.

---

## 1. There is an optimal compression ratio, and it is not "as little as possible"

`lexical-doc-bm25-sys` swept over absolute budgets, pooled over both strata:

| B | reduction | answer | abstain | **pooled** | win/lose |
|---:|---:|---:|---:|---:|---:|
| 512 | 8.3× | +2.6 pp | +3.0 pp | **+2.7 pp** | 21/5 |
| **256** | **16.5×** | +3.5 pp | +10.1 pp | **+5.3 pp** | **34/2** |
| 196 | 21.6× | +2.8 pp | +6.0 pp | **+3.7 pp** | 31/9 |
| 160 | 26.4× | −2.3 pp | −6.0 pp | −3.3 pp * | 22/42 |
| 128 | 33.0× | −19.7 pp | −29.8 pp | −22.5 pp * | 12/147 |
| 96 | 44.0× | −45.1 pp | −51.8 pp | −47.0 pp * | 2/284 |

`*` = significantly below uncompressed, paired exact McNemar.

**B = 256 is a resolvable peak, not a plateau.** It beats B = 512 by +2.7 pp
(19/3, p = 8.6e−04) and B = 196 by +1.7 pp (14/4, p = 0.031). Discarding *more*
context actively helps until the floor — which is the strongest form of the
"compression eliminates unnecessary information" claim, and it is measured rather
than asserted.

At B = 256 the effect replicates out-of-sample on records 200–599, which were never
used for any design decision: **0.7475 → 0.8075, +6.0 pp, 25 wins / 1 loss,
p = 8.0e−07.**

---

## 2. The floor is where the spans stop fitting — predicted to the token

Before the sweep, from measured span lengths alone:

```
4 sinks + 0.15·B recent + 104 instructions + 63 one record  ≤  B
⟹  B ≥ (4 + 104 + 63) / 0.85 = 196
```

**The observed knee is 196.** F1 confirmed. F2 also holds: at the first budget below
the floor, abstention falls faster than answering (−6.0 pp against −2.3 pp), which is
what the larger instruction span (104 vs 63) predicts.

### The confound, and why the formula survives it

The recent window is `0.15·B` and the question tail is 29 tokens, so query coverage
fails at B ≥ 193 — **three tokens from the predicted floor.** The sweep alone could
not distinguish "the spans no longer fit" from "the question is being truncated".
This was recorded as an amendment while the sweep was still running, with a
separating arm specified in advance.

Re-running the two lowest budgets with a **fixed recent of 32**, so the question is
fully covered at every budget:

| | answer, recent = 0.15·B | answer, recent = 32 fixed |
|---|---:|---:|
| B = 128 | 0.5741 | **0.5741** |
| B = 96 | 0.3194 | 0.3148 |

Covering the question completely changes answering by **nothing at all**. Query
truncation is refuted; the capacity account stands. Abstention actually gets *worse*
with the larger fixed window (0.5238 → 0.3869 at B = 128), which is the same capacity
story from the other side: more recent means fewer slots for the instructions.

---

## 3. It does not need the prompt's printed delimiters — but it does need boundaries

The result so far depended on `rag_long` printing `[doc_… | trust=…]` headers. Three
segmentations at B = 256:

| segmentation | answer | abstain | pooled | vs records |
|---|---:|---:|---:|---:|
| `[doc_…]` records | 0.8056 | 0.9226 | **+5.3 pp** | — |
| **newlines** (no schema) | 0.7940 | 0.8988 | **+3.8 pp** | −1.5 pp (11/2, p = 0.022) |
| fixed 64-token chunks | 0.7222 | 0.9226 | −0.7 pp | −8.3 pp |

**The preregistered S1 arm — fixed chunks — is REFUTED.** A 64-token window splits a
63-token record and cuts the answer in half; it lands *below* uncompressed on the
answer stratum (18/39) despite identical abstention.

**Newline segmentation, added after seeing that failure, recovers almost all of it.**
It is post-hoc and labelled so, which is why it was checked out-of-sample: on the
disjoint records 200–599, **0.7475 → 0.7900, +4.3 pp, 19/2, p = 2.2e−04.**

**And newlines have a *lower* floor.** At B = 160, where record segmentation has
already broken (−3.3 pp, 22/42, significantly below), newlines are still at parity
(−0.2 pp, 19/20, not significant). The two cross over: records win above the floor,
newlines survive further below it. That is what the capacity account predicts — the
floor scales with the size of the unit you are forced to keep whole, and a line is
smaller than a record.

So the requirement is not a bespoke schema — it is *boundary awareness*. Newlines are
universal in real prompts and suffice. That converts a benchmark-scoped result into a
method with a stated, checkable precondition: the text must have lines, and a
retrieval unit must not straddle them.

---

## 4. Instruction sinks: predicted, confirmed, and still not worth deploying

Holding the instruction span outside the ranked budget instead of tying it at
weight 1.0 does exactly what §3's S2 predicted — it changes the *failure mode*, not
the floor. At B = 160 (below the floor):

| | answer | abstain | pooled |
|---|---:|---:|---:|
| weight-tied (`-sys`) | 0.7477 | 0.7619 | 0.7517 |
| instruction **sinks** | **0.4745** | **0.8631** | 0.5833 |

Abstention is preserved — 0.8631 is not below uncompressed (9/2) where the weight-tied
arm's 0.7619 is — while answering collapses, because 104 sink tokens consume 65% of a
160 budget and leave 32 slots for a 63-token record. Graceful degradation of the
safety-critical behaviour, bought at a price not worth paying above the floor.

**And it has a hard feasibility limit.** At B = 128 the arm is undefined: prompts
whose instruction prefix is 140 tokens leave no budget at all, `n_recent()` returns 0,
and the score-lag guard refused the run rather than degrading it silently. Instruction
sinks require `B > |instructions| + recent`.

---

## 5. The floor is absolute, not proportional — so longer context buys more compression

`rag_long16k` has the same instruction prefix and the same 63-token records as
`rag_long4k`; only the number of distractors changes, and the prompt grows from
4,226 to 16,520 tokens. **F3, registered before these runs, predicted the floor
would not move.**

| B | reduction | answer | pooled | |
|---:|---:|---:|---:|---|
| 512 | 32× | −2.8 pp (0/2) | −2.0 pp (0/2, p = 0.50) | |
| **256** | **65×** | −2.8 pp (1/3) | **−2.0 pp (1/3, p = 0.63)** | knee |
| 196 | 84× | −15.3 pp (1/12) | −9.0 pp (3/12, p = 0.035) | * |
| 160 | 103× | −36.1 pp (0/26) | −27.0 pp (0/27) | * |
| 128 | 129× | −72.2 pp (0/52) | −53.0 pp (1/54) | * |

**The 16K knee is B = 256 against 196 at 4K** — inside the preregistered
confirmation band {160, 196, 256}. **F3 confirmed.**

The competing account is refuted with room to spare. A floor set by the *fraction*
of context retained would scale with prompt length to ≈ 765 tokens, which would put
B = 512 well below it; B = 512 is not below it at all (0/2 discordant). The floor
moved by a factor of 1.3, not 4.

The practical consequence is the useful one: **because the floor is a fixed number
of tokens and the prompt is not, the achievable compression ratio grows with
context length.** 16.5× at 4K where quality *rises* by +5.3 pp; **65× at 16K** with
no significant loss. The same selector, the same budget in tokens.

### The length series, one budget

Holding **B = 256 fixed** and varying only the context length:

| dataset | prompt | reduction | pooled vs uncompressed | n |
|---|---:|---:|---:|---:|
| rag_long4k | 4,226 | **16.5×** | **+5.3 pp** (34/2, p = 1.9e−08) | 600 |
| rag_long16k | 16,520 | **65×** | −2.0 pp (1/3, p = 0.63) | 100 |
| rag_long32k | 32,902 | **129×** | −2.0 pp (3/4, p = 1.0) | 50 |

Same selector, same 256-token budget. The reduction ratio grows 8× across the
series while the measured cost stays flat at about −2 pp and never reaches
significance. At the shortest length it is not a cost at all but a gain.

The 32K cell is a probe, not a result: n = 50 with 38 answer cases and 7 discordant
pairs has almost no power, and the uncompressed model never abstains at that length
(0.0000), so the pooled figure is carried by the answer stratum. It is reported
because its point estimate sits exactly where the 16K cell's does, which is what the
absolute-floor account predicts and what a proportional account cannot accommodate.

Two honest qualifications. The knee did drift upward slightly (196 → 256), so the
floor is *nearly* absolute rather than exactly so — at 16K there are 187 distractors
against 65, and BM25's graded weights spread over more records. And n = 100 at 16K
gives 72 answer cases and few discordant pairs, so "not significantly below" is
weaker here than the same phrase at 4K; that caveat was registered in advance.

---

## 6. What this does not show

1. One dataset family, one model, one depth, one seed. **The claim under test is
   the formula, not the number 196.** The length axis is now tested (§5) and the
   floor held; the *record-length* axis is not — a corpus with 300-token records
   should put the floor near (4 + 104 + 300)/0.85 = 480, and that remains the
   sharpest untested consequence.
2. B = 256 is the best of six swept points, not a fitted optimum; the true peak lies
   somewhere in [196, 512] and was not bracketed more finely.
3. Newline segmentation is post-hoc. Its out-of-sample confirmation is real but it
   did not carry a preregistered prediction.
4. Capability only. No throughput or latency claim.
5. The selector throughout is BM25 plus instruction protection — **not** the
   accumulated-attention mechanism, which loses to uncompressed at these budgets.
