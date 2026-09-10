# HLGA-1 P3: frozen local/global attention on real Rineya R2

Ten arms x 72 open-synthetic DEV cases, greedy, `max_new_tokens` 24, cuda/math,
block 128, allocator cap 6144 MiB, float32. Registered primary metric
`first_content_exact` (does the generation's first line equal the answer),
case set v2 `(512, 1024, 1280, 1536, 1792, 2048)`, readout v2.

Numbers below are from run `hlga-p3v2b-20260909-cuda-math`, gated by
`hlga-p2-gate`-lineage receipt on the same code identity. They are being
regenerated as `hlga-p3v2c-20260909-cuda-math` on the corrected metric-labelling
code; every arm completed so far reproduces the previous run **byte-identically**
(72/72 token_ids and 72/72 metric values per arm), confirming that fix was
label-only. `AGENT_RECEIPT_INDEX.md` records which receipts are live.

**Outcome: the registered "P2 pass + P3 degradation" case.** The retrofit is
mechanically faithful and capability-damaging when frozen. Within that, four
prefix sink tokens produce a large, one-sided, mechanistically legible repair.

---

## 1. Headline

| arm | | `first_content_exact` |
|---|---|---:|
| A0_global | all-global reference | **0.653** |
| A4_global_gate | identity-gated A0 | 0.653 |
| A7_sink1024 | W1024 + 4 sinks | 0.431 |
| **A3_sink512** | **primary treatment** | **0.250** |
| A6_sink512_gate | identity-gated A3 | 0.250 |
| A1_local1024 | W1024, no sinks | 0.181 |
| A8_all_local512 | no global layers | 0.069 |
| A2_local512 / A5 | W512, no sinks | 0.056 |
| A9_frontloaded512 | placement control | 0.042 |

Registered contrasts on the primary metric, with Holm-Bonferroni over all ten
at alpha 0.05:

| contrast | reference -> treatment | w/l | delta | raw p | Holm |
|---|---|---|---:|---:|---|
| pure_window_damage | A0 -> A2 | 0/43 | −0.597 | 2.3e-13 | survives |
| **primary_frozen_sink_retrofit** | **A0 -> A3** | **3/32** | **−0.403** | **4.2e-07** | **survives** |
| sink_at_1024 | A1 -> A7 | 18/0 | +0.250 | 7.6e-06 | survives |
| sink_at_equal_budget | A2 -> A3 | 14/0 | +0.194 | 1.2e-04 | survives |
| window_budget | A1 -> A2 | 0/9 | −0.125 | 3.9e-03 | survives |
| periodic_global_access | A8 -> A2 | 0/1 | −0.014 | 1.0 | zero power |
| placement_at_equal_work | A9 -> A2 | 1/0 | +0.014 | 1.0 | zero power |
| the three gate identities | — | 0/0 | +0.000 | 1.0 | null by design |

All five substantive contrasts survive multiplicity correction.

---

## 2. Damage separates into two mechanisms

Stratifying by the eviction map — computed from the window rule *before* the
run, for each case asking whether the evidence is still inside the local
window at generation time:

| arm | evidence evicted | evidence still visible |
|---|---:|---:|
| A2 (W512, no sinks) | −0.675 | **−0.500** |
| A1 (W1024, no sinks) | −0.650 | **−0.404** |
| A3 (W512 + sinks) | −0.650 | **−0.094** |
| A7 (W1024 + sinks) | −0.600 | **−0.077** |

Windowing inflicts two separable damages:

1. **Evidence loss.** Unavoidable once the answer leaves the window. Costs
   −0.60 to −0.68 and hits every windowed arm alike, sinks or not.
2. **Attention destabilisation.** Damage in cells where the context is fully
   visible: −0.40 to −0.50 without sinks, but only −0.08 to −0.09 with them.

Equivalently, as sink repair by stratum:

| | evidence still visible | evidence evicted |
|---|---:|---:|
| A2 -> A3, registered | **+0.406** (n=32) | +0.025 (n=40) |
| A2 -> A3, punctuation-tolerant | **+0.312** | +0.000 |
| A1 -> A7, registered | **+0.327** (n=52) | +0.050 (n=20) |
| A1 -> A7, punctuation-tolerant | **+0.269** | +0.050 |

**Four sink tokens repair almost exclusively where the evidence was never
lost, and essentially nothing where it was genuinely evicted.** The
dissociation holds at both window sizes and under both scoring rules. The
sinks cover absolute positions 0-3, which decode to `'Read the records.'`,
while the earliest evidence token in any of the 72 cases sits at position 21 —
so they cannot be retrieving evidence. A3 also attends a *strictly smaller*
recent window than A2 (508 slots against 512, because sinks consume the
budget) and still scores far higher.

That is consistent with attention-sink stabilisation. It remains an
inference from a behavioural dissociation, not a demonstration: no attention
map was inspected, and this experiment cannot exclude another mechanism that
would produce the same stratified pattern.

The same ordering appears independently in generation degeneracy, which is not
an accuracy metric at all: mean `max_token_repeat` A0 1.39, A2 5.07, A3 1.62;
`terminated_rate` 1.000, 0.722, 0.986.

---

## 3. Corrections to earlier reporting of this run

These were found by an adversarial audit of the interpretation and are
recorded because the superseded statements were circulated.

**RETRACTED — "N=512 is the cleanest proof of the sink mechanism."** It is not
evidence of the mechanism at all. At N=512, A3 is **bit-identical to A0 on
12/12 rows**: four sinks plus 508 recent slots covers the whole 512-token
prompt, so A3 is not effectively windowed there. The entire A2-vs-A3 gap at
that length is **three trailing full stops** — A2 wrote `'49770.'` and
`'UNKNOWN.'` against answers `'49770'` and `'UNKNOWN'`. That is exact-match
punctuation, not attention. Section 2 replaces it with a stratification that
survives the same scrutiny.

**QUALIFIED — the size of the sink effect.** 4 of 14 and 3 of 18 sink wins are
punctuation-only. Under punctuation-tolerant rescoring the effect is +0.139
(not +0.194) and +0.208 (not +0.250), still strictly one-sided at 10/0 and
15/0 genuine wins. **The primary contrast is unaffected: −0.403 under both
scorings.**

**CORRECTED — "three registered metrics" implies triangulation that does not
exist.** `strict_exact` is unreachable under this prompt protocol and is False
on **720/720** rows: 718/720 generations open with a BPE leading space because
the prompt ends `"Answer:"`, and every trimmed-correct generation additionally
ends in whitespace. All 30 of its contrast entries are 0/0/p=1.0 by
construction. `trimmed_exact` is **row-identical** to the primary on all 720
rows, because readout v2 made termination reachable so no generation has
content after a first newline. There is one informative metric. The other two
are retained for a pre-registered reason — side-by-side comparison with
superseded readout v1 — and `trimmed_exact` moving 0.000 -> 0.653 on A0 is
what evidences the readout fix.

**STRENGTHENED — the two nulls carry no information.** p = 1.0 was forced by
the marginals before any pairing; A2/A8/A9 score exactly 0.000 at every length
above 512, leaving an effective n of about 5. `periodic_global_access` and
`placement_at_equal_work` are **uninformative**, not evidence that global-layer
count or placement is irrelevant. The missing control is an all-local arm
*with* sinks, which the registered arm set does not contain.

**REFINED — abstention.** The collapse was quoted as 0.694 -> 0.028, which is
A0 against A2. For the primary treatment the figure is A3 at 0.083. Both
collapse; the treatment is the right number to quote.

---

## 4. What this establishes, and what it does not

**Establishes**, on this task and this checkpoint: the research runtime
reproduces native HF Qwen3 bit-exactly and the untrained gates are exactly
identity (72/72 token-for-token on real 512-2048 token generations, all three
pairs); frozen substitution of the registered periodic local/global topology
costs −0.403 on the primary endpoint; damage separates into evidence loss and
attention destabilisation; four prefix sinks remove most of the second and
none of the first, one-sidedly, surviving multiplicity correction and
punctuation-tolerant rescoring.

**Does not establish**: anything about trained performance, speed, memory, or
capability beyond this synthetic retrieval/abstention probe. Whether periodic
global layers help at all — that comparison had no power. Attention-sink
stabilisation as a *demonstrated* mechanism rather than an inference from a
behavioural dissociation. Generalisation beyond one seed and 72 correlated
cases with a reference arm itself at only 0.653.

P4 and P5 remain closed. No training was performed and none is authorised.

---

## 5. Highest-value next experiment

Add **A10 = all-local W512 + 4 sinks** (no global layers) and rerun the frozen
diagnostic. It is the one missing cell: with A3 (globals + sinks), A8 (globals
absent, sinks absent) and A2 (globals present, sinks absent) already measured,
A10 completes the 2x2 and is the only way to estimate the global-layer
contribution now that sinks are known to lift the arms off the floor where the
existing global-layer contrasts were trapped.

If A10 approaches A3, the periodic global layers are not carrying the repair
and the topology can be simplified before any training is registered. If A10
collapses, globals and sinks are complementary and the registered A4/A5/A6
gate-training design is the right next stage. Either outcome is decision-
relevant, and it costs one arm.
