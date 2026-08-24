# Open questions: results we cannot yet explain

Every anomaly in this programme that has an observation but not a cause. An
unexplained result is indistinguishable from a bug until it is explained, and this
programme has repeatedly found that it *was* a bug — `Policy.pool` defaulting to 1 so
every long-context result measured pooling disabled; `count_exponent` inert on the
self-compressing path; eviction running during prefill and reading as a 2.6× speedup;
`recent=512` consuming the whole budget so `heavy()==0` looked like a capability cliff.

Rule: prefer a **bug** explanation first, and concede a mechanism property only after
the bug is ruled out. Re-running an experiment to settle one of these is expected, not
a detour.

Status: **open** until the cause is named and demonstrated. Closing an entry requires a
measurement, not an argument.

---

## O1 — a 20-token span scores 0.528 where 5-token spans score 1.000

**Observed.** Stage 2, 128K, B=256, derived reach `grow=2.0`, 20-word payload:
0.83 / 0.42 / 0.33 across depths, mean **0.528**. The same policy scores 1.000 at
W=3-equivalent spans and 0.889 at d=7.

**Why it matters.** Natural-language evidence runs 20–40 tokens. If the mechanism
degrades at 20, its whole external-validity case is in question.

**Candidates.**
1. *Harness bug.* `max_new = max(12, 4W + 6)` = 86 decode tokens for W=20. If the model
   emits any preamble the answer could be cut off and scored wrong while being right.
2. *Scoring bug.* `hit = str(key) in got` is exact substring; a correct phrase with
   different spacing or a newline would score as a failure.
3. *Mechanism.* Longer spans genuinely fragment more — each additional token is another
   chance to lose one, and P2's failure signature was always losing exactly one.

**Blocked on.** The trial record truncated replies to 32 characters, so these failures
could not be classified at all. Fixed (`keep = max(32, len(key) + 24)`) but the cells
predate the fix.

**Decisive test.** Re-run W=20 with the record fix, then classify: a correct prefix that
stops early implicates fragmentation; a complete-but-differently-formatted phrase
implicates the hit test; an empty or truncated-at-86-tokens reply implicates `max_new`.

---

## O2 — the span law is marginal at d=7

**Observed.** `back ≥ d − 1` holds in 5 of 6 cells, but at the threshold itself the
margin narrows with span length: **0.972** (d=4, back=3), **1.000** (d=5, back=4),
**0.778** (d=7, back=6).

**Candidates.** The effective span may be longer than *d* — the leading space token
before the digits, or the `.` after — so the true requirement is `back ≥ d` or `d + 1`
and d=7/back=6 is simply below it. Or the law is approximate and something else
degrades with length.

**Decisive test.** d=7 at back=7 and back=8. If back=7 restores 1.000, the requirement
is `d`, not `d − 1`, and the whole law shifts by one — cheap and it would sharpen the
claim rather than weaken it.

---

## O3 — derived reach leaves 0.889 at d=7, not 1.000

**Observed.** `grow=2.0` beats the best tuned fixed reach at every span length
(1.000 / 1.000 / 0.889 against 0.972 / 1.000 / 0.778) but does not reach 1.000 at d=7.

**Candidates.** Growth stops early because a payload token dips below the median gate;
or `grow_max` truncates propagation; or the same unexplained long-span effect as O2.

**Decisive test.** Dump the grown score tensor for a d=7 failure and check whether every
digit position cleared the gate. This is a CPU-only inspection of saved tensors and
needs no re-run if the trace tool is extended to record them.

---

## O4 — the multiplier is inert at short spans and decisive at long ones

**Observed.** `grow=2.0` vs `grow=5.0`: identical at d=4 and d=5 (p = 1.0000), but
0.889 vs 0.611 at d=7 (exact McNemar **p = 0.0020**).

**Candidate, not yet verified.** A stricter gate excludes weak payload positions, so
growth cannot traverse a long span. Plausible and mechanical — but *asserted*, not
demonstrated.

**Decisive test.** Same tensor dump as O3, at both multipliers, counting how many
payload positions clear each gate.

---

## O5 — B=16 scores 36/36 with only 8 heavy slots

**Observed.** 128K, B=16 → 4 sinks + 4 recent + **8 heavy**. 36/36 at every depth,
7,864× compression.

**Why it matters.** This is the headline number. Eight positions out of 125,826 is
extraordinary, and extraordinary is exactly when to look for a free lunch.

**Candidates.**
1. The **final chunk may never be compressed**, so a late needle survives without the
   mechanism at all. Depth 0.9 being lossless everywhere is consistent with this.
2. The needle sits inside the recent window or the sinks for some depths.
3. The task is trivially separable — the filler contains **no digits at all**, so the
   payload is the only object of its kind and a regex scores 1.000 with no model.

**Decisive test.** Count, for chunk=4096 at 128K, how many compression rounds a needle
at each depth actually survives, and whether the last chunk is compressed before
decoding. If a depth-0.9 needle faces zero rounds, that cell measures nothing and must
be excluded from the headline. Currently under review.

---

## O6 — a third of stage-1 failures are neither truncation nor substitution

**Observed.** Classifying 21 stage-1 failures: 52% fragmented true span, 14% crowded out
by a decoy, **33% "other"** — degenerate output such as `75526` → `555222`.

**Candidates.** Severe cache damage producing repetition loops; or a classifier gap
where a real category is being missed.

**Decisive test.** Read the full replies now that records are no longer truncated, and
either name the category or show it is a decode pathology rather than a retrieval one.

---

## O7 — depth 0.9 is immune, and the explanation has never been measured

**Observed.** Depth 0.9 is lossless in nearly every cell, including where 0.1 and 0.5
collapse — stage 1 at k=16 framed: 0.42 / 0.42 / **1.00**.

**Stated explanation.** A late needle enters in a late chunk and survives few
compression rounds. This is repeated throughout the findings and has **never been
verified by counting rounds**. It is also the same mechanism that would make O5 an
artefact, so the two are linked.

**Decisive test.** Instrument the compression loop to report, per depth, how many rounds
the needle's positions actually pass through. Cheap, and it either grounds a claim the
programme leans on constantly or overturns it.

---

## Closing an entry

State the cause, cite the measurement that demonstrates it, and move the entry to a
**Resolved** section with the date. Do not delete it — a resolved anomaly is a record
of how the mechanism actually behaves, and the reasoning is worth more than the verdict.

---

# Resolved

## O5 — RESOLVED 2026-08-23: the exemption is real, the headline is not a free lunch

**Cause found, in code.** `scripts/kv/chunked_prefill.py:120`:

```python
if compress_fn is not None and ci < n_chunks - 1:
```

The **final chunk is never compressed**. That is a genuine exemption and it was worth
suspecting. But it does not reach the needles.

At 128K the prompt is 125,826 real tokens in chunk=4096 → **31 chunks, 30
compressions**, with the exempt chunk covering positions [122,880, 125,826). Needle
positions, taken from the committed span trace rather than estimated:

| depth | needle span | chunk | in the exempt chunk? | compressions faced |
|---:|---|---:|---|---:|
| 0.1 | [12,606, 12,632) | 3 | no | **27** |
| 0.5 | [62,910, 62,936) | 15 | no | **15** |
| 0.9 | [113,214, 113,240) | 27 | no | **3** |

No needle lands in the exempt chunk at any depth, so no cell survives without the
mechanism. The exempt region holds the question, which is where it belongs.

**And the headline holds on the hardest depth alone.** 128K/B=16 causal scores 1.00 at
depth 0.1 — 12/12 needles that each survived 27 compression rounds with 8 heavy slots.
The claim does not depend on averaging in the easy depth.

## O7 — RESOLVED 2026-08-23: the depth asymmetry, finally measured

The findings repeated throughout that "a late needle enters in a late chunk and
survives few compression rounds". It was never measured. It is now, in the table
above: **27 rounds at depth 0.1 against 3 at depth 0.9 — a 9× difference in exposure.**

**A methodological consequence that applies to every cell in this programme.** The
three depths are *not* equally difficult, so the pooled three-depth mean overweights
the easy one. A cell reported as 0.972 may be 12/12 at depth 0.9 and 11/12 at depth
0.1, and those are not the same result. Depth-wise numbers are reported throughout and
should be read in preference to the mean wherever the two disagree — and the depth-0.1
column is the one that carries the claim.
