# Cornerstone 2, phase 3 — what is established, and what to do next

Written 2026-08-27, after the phase-2 experimental plan completed.
State: **234/234 pins**, 0 integrity problems, 11 open questions closed, 5 open.

---

## Part 1 — what works, ranked by strength of evidence

### 1. The score itself is what works *(strongest, most replicated)*

Not the budget, not the window, not the schedule. Against controls holding the *same
number of positions*:

| setting | significance | random | streaming |
|---|---:|---:|---:|
| passkey 128K/B=16 | **1.000** | 0.000 | 0.000 |
| rag_dev keep 0.25 | **0.59488** | 0.34304 | 0.42740 |
| rag_dev keep 0.50 | **0.65338** | 0.44191 | — |
| **Baya 16K keep 0.25** | **0.66108** | 0.42022 | — |

Certified at **n = 100** on passkey: 300/300 needles, Clopper–Pearson lower bound
**0.9705**. On a model that can *use* long context the score is worth **+0.24086** at 16K
while the compression itself costs **exactly zero**.

**The caveat that governs everything below, now measured (O11, resolved 2026-08-27):** the
passkey score is **query-conditioned retrieval, not an importance measure**. Scored with an
unrelated question the same configuration retrieves **0.0000–0.0556** against **1.0000**
with the real one. The language results come from a *different* scoring path — no query
pass, attention accumulated during generation — so the two must be described separately.

### 2. Backward-only pooling — the cornerstone-2 discovery

Reach *backward* decides; forward reach is inert. On language the effect is large and its
sign flips with pressure:

| keep | centred (3,3) | causal (6,0) | difference |
|---:|---:|---:|---:|
| 0.25 | 0.26096 | **0.41286** | **+0.15190** (p = 5.7e-37) |
| 0.75 | **0.65999** | 0.65136 | −0.00863 (p = 3.0e-6) |

Under pressure a centred window is *worse than not pooling at all* (0.26096 against
0.33863). The causal window repairs damage rather than adding a gain.

### 3. Budget allocation dominates scoring choices *(largest single language effect)*

| recent rule | quality |
|---|---:|
| fixed 32 (inherited) | 0.41286 |
| **proportional, rf = 0.15** | **0.60834** |

**+0.19331**, p = 3.1e-60. The fixed window left the score **zero slots on 17,360
compressions**. The optimum is *interior* (~78% of budget under the score) and the curve
self-validates: at 43% score share it lands within 0.010 of the separately-run streaming
control.

### 4. Sinks are load-bearing

sinks = 0 scores **0.000** while holding *more* score-chosen slots. Two suffice.

### 5. Where the score earns its keep

Gains concentrate exactly where evidence competes — ambiguous evidence **+0.6461**,
planted distractors **+0.5651** — and go slightly *negative* on social chat (**−0.0529**),
where there is nothing to retrieve. Mean **+0.2653**, median **+0.0105**, 40% of cases
unchanged. The claim is *"catastrophic failures on the hard subset are prevented"*, not
*"quality improves by 0.27"*.

---

## Part 2 — what does NOT work, recorded as plainly

- **Derived reach (grow_spans)** loses on language at both keep fractions (−0.03513,
  −0.07466) and costs **1.7–2.5x** the runtime. Its case rests entirely on passkey, and
  even there it loses to a per-span-tuned back=8 at d=7.
- **B1 self-sizing floor ties the tuned ratio** — −0.00537, p = 0.92. A null against a bar
  set before the runs. Its value is *safety* (the degenerate state becomes unreachable),
  not quality.
- **Three-hop chains**: the advantage is undetectable (p = 0.125, random nominally
  higher). At two hops it is decisive (p = 0.019).
- **Long context breaks chains regardless of policy** — 2-hop 0.8333 to 0.4722 from length
  alone, with compression free at both ends.
- **The span axis is dead on this model** — an ~11-digit verbatim recall ceiling with a
  full cache.

---

## Part 3 — the next plan

### P1. O11 — RESOLVED 2026-08-27. It is similarity to the query.

| scoring signal | mean |
|---|---:|
| the real question | **1.0000** |
| wrong question (weather) | 0.0556 |
| wrong question (sky) | 0.0000 |
| no query pass | 0.0000 |

Replicated on two independent wrong queries. The mechanism is a **query-conditioned
KV-cache retrieval index**. Consequences — the name, multi-turn re-scoring, and the
separation of the passkey and language scoring paths — are in FINDINGS_P2 and paper §4.12.

**What this promotes to P1 in its place: B4 multi-turn, now with a prediction.** O11
predicts a cache scored for turn 1 is not scored for turn 2, so multi-turn should fail
without per-turn re-scoring. That is a falsifiable claim about the mechanism's main
application surface, and it is cheap to test.

<details><summary>the original P1 design, kept for the record</summary>

### O11 — is the score measuring IMPORTANCE, or SIMILARITY TO THE QUERY?

**Every compressed result in this campaign uses query-aware scoring.** All seven
query-unaware runs on file are uncompressed memory probes. The mechanism's identity is
therefore untested: we do not know whether the score finds *what matters* or *what
resembles the question*.

This is not a detail. If the score is largely query-similarity then "significance matrix"
is the wrong name, the mechanism is a retrieval index rather than an importance measure,
and the multi-turn case — where the query changes between turns — is far weaker than
implied.

**Design.** The headline cell, one variable: 128K/B=16, back=6, query-aware on and off,
12 needles at 3 depths. The same pair on rag_dev at keep 0.25.

**Pre-registered reading.**
- Query-unaware holds near 1.000 → the score measures **importance**; the name is right
  and multi-turn is plausible.
- Query-unaware collapses → the score is **query-conditioned retrieval**; rename it, and
  every multi-turn claim needs the query held constant.
- Partial → state the split rather than picking the flattering half.

*Cost: about 40 minutes of GPU. Actual: ran on CPU at 8K in ~40 minutes.*

</details>

### P2. A5 / O12 — the runtime cost, thermally controlled

Every number in this programme is a benefit; the only cost measured is derived reach at
1.7–2.5x, taken back-to-back at 85–94 C. The programme's own rule is that no sub-2x
throughput claim survives a 2.17x thermal band. Fixed clocks, interleaved arms, cooldown
between. *Cost: about 2 hours.*

### P3. The three-hop anomaly at larger n

Random was nominally better at three hops on **n = 12**, p = 0.125. Either the score
genuinely stops helping when several spans are required, or it is noise. n = 60 (five
seeds, as A3 did) settles it. If real it points at a concrete mechanism — the score
concentrates where a spread is needed. *Cost: about 2 hours.*

### P4. O6 — the unclassified stage-1 failures

A third of them are neither truncation nor substitution and nobody has looked at what they
are. *Cost: about 2 hours, CPU only.*

### P5. Close O1 and O3 by writing, not running

O1 is superseded by the O1/O8 withdrawal. O3 is answered — back=8 gives 1.000 at d=7, so
derived reach loses to per-span tuning. Both need a paragraph, not a GPU.

### P6. The paper

It is the deliverable and it is current, but it has never been read end to end for flow
rather than for correctness.

---

## The one idea worth adding to the mechanism

The three-hop result suggests a specific weakness: **the score is built to find one
region**. With several required spans and a tight budget it may concentrate where a spread
would serve better — which is exactly what the random arm reaching parity at three hops
would mean if it is real.

A **diversity term** — penalising keeps that cluster, or allocating the budget across
score *modes* rather than by a global top-k — is the first mechanism change this campaign
has produced evidence *for* rather than against. It should not be built until P3 says the
three-hop effect is real.
