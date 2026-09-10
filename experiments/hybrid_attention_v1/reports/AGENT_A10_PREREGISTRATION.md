# A10 preregistration — written BEFORE the arm was run

Registered 2026-09-09, before `A10_all_local_sink512` existed in code, so the
prediction below cannot be retrofitted to the outcome.

## The arm

`A10_all_local_sink512` — every layer local, window 512, **4 prefix sinks,
ZERO global layers**. Identical to `A8_all_local512` except for the sinks;
identical to `A3_sink512` except that A3 keeps the seven registered periodic
global layers (3, 7, 11, 15, 19, 23, 27).

## Why this arm

The two registered global-layer contrasts (`periodic_global_access` A8→A2 and
`placement_at_equal_work` A9→A2) both returned p = 1.0 with an effective n near
5, because every no-sink arm sits at exactly 0.000 above N=512. The global-layer
question was not answered; it was unmeasurable. Sinks are now known to lift arms
off that floor, so A10 completes a 2×2 and moves the question into the row where
it can actually be measured:

| | sinks absent | sinks present |
|---|---:|---:|
| **7 periodic globals** | A2 = 0.056 | A3 = 0.250 |
| **0 globals** | A8 = 0.069 | **A10 = ?** |

## Prior evidence that constrains the prediction

Overall / evidence-still-visible (n=32) / evidence-evicted (n=40), primary metric:

| arm | globals | sinks | overall | visible | evicted |
|---|---:|---:|---:|---:|---:|
| A0_global | 28 | 0 | 0.653 | 0.625 | 0.675 |
| A3_sink512 | 7 | 4 | 0.250 | 0.531 | 0.025 |
| A8_all_local512 | 0 | 0 | 0.069 | 0.156 | 0.000 |
| A2_local512 | 7 | 0 | 0.056 | 0.125 | 0.000 |
| A9_frontloaded512 | 7 | 0 | 0.042 | 0.094 | 0.000 |

Two facts do most of the work. First, in the sinks-absent row the seven global
layers buy **nothing** — A8 with zero globals (0.069) is if anything *above* A2
with seven (0.056) and A9 (0.042), and all three are exactly 0.000 on evicted
evidence. Second, A3's advantage over A2 lives almost entirely in the
**evidence-still-visible** stratum (0.531 vs 0.125), where by construction the
answer is inside the local window and a global layer is not needed to reach it.

## Registered predictions

**Primary prediction: H1 — A10 ≈ A3.** Overall **0.20–0.28**, visible stratum
**0.45–0.55**, evicted stratum **0.00–0.05**. The `global_layers_given_sinks`
contrast (A10 → A3) is predicted to be **null or small**, |delta| < 0.10, and
not significant after multiplicity correction.

Competing hypotheses and what would falsify H1:

| hypothesis | A10 overall | visible | evicted | falsifies H1 if observed |
|---|---|---|---|---|
| **H1** sinks do the work, globals are dead weight | 0.20–0.28 | 0.45–0.55 | 0.00–0.05 | — |
| H2 globals and sinks are complementary | 0.10–0.18 | 0.25–0.40 | 0.00 | A10 → A3 delta > +0.10, significant |
| H3 globals matter only for evicted evidence | ≈0.24 | ≈0.53 | 0.00 | A3 > A10 on the evicted stratum only |
| H4 sinks need a global layer to exploit | 0.05–0.09 | 0.12–0.18 | 0.00 | A10 ≈ A8, i.e. no sink repair at all |

I consider H1 most likely, and H3 hard to distinguish from H1 because A3 itself
only reaches 0.025 on the evicted stratum (1 case in 40) — that stratum has
almost no resolution left in either arm.

**What would be bad news for the architecture:** H4. If sinks only repair when a
global layer is present, then the repair is a property of the hybrid topology
rather than of sinks, and the cheap all-local variant is not available.

**What would be awkward rather than good news:** H1. If the seven global layers
contribute nothing, the registered periodic local/global topology is not what is
helping — the sinks are — and the honest architectural conclusion is that this
experiment has been measuring a *sink* effect wearing a hybrid-attention label.
That is a useful result but it is not a vindication of the registered design.

## What no outcome here can license

A10 is a **frozen** diagnostic on one synthetic retrieval/abstention probe, 72
correlated cases, one seed, with the all-global reference itself at only 0.653.

Every windowed arm measured so far, including the best one, is far **below** the
all-global reference: A3 is −0.403 and A7 is −0.222. A local/global retrofit that
is strictly worse when frozen is not a candidate for a published core on the
strength of that. It could only become one if either
(a) training recovers the gap — which is stage P4, closed, and untested; or
(b) it buys a measured efficiency advantage at matched quality — and no isolated
memory or throughput measurement has been taken, the protocol explicitly forbids
inferring one from FLOP or cache arithmetic, and `timing_causal` is registered
false.

So a "positive" A10 means *the cheaper topology is no worse than the more
expensive one, both being much worse than the reference*. That is a real and
useful architectural finding. It is not evidence of improved intelligence, and
it does not authorise touching a published Rineya core.

---

# AMENDMENT — A11 added, A10's role corrected (still before either arm was run)

An independent preregistration review returned **run-modified**, and its central
objection checks out against the completed run.

## The objection, verified

**A10 → A3 cannot measure the global-layer contribution.** Across all eight
windowed arms already run there are only **3 correct answers in 280 arm-case
opportunities** where the evidence lies outside that arm's own window — and
**0 of 140** on the unanswerable evicted cases, where being correct requires
only abstention and no retrieval at all. A0_global scores 27/40 on those same
cases. A3 itself is 17/32 reachable and **1/40 evicted**.

So the sinks-present row does not rescue the global-layer question, it
relocates the floor. A10 → A3 can only produce discordance in the stratum where
both arms already reach the answer locally and the global layers are causally
redundant. My original framing of that contrast was over-promising.

## What changed

1. **Contrast renamed** to `global_layers_given_visible_evidence_and_sinks`.
   It estimates the effect of restoring unbounded key access at 7 of 28 layers
   *conditional on the answer already lying inside the local window*. It is not
   "the global-layer contribution", and a null here is uninformative in either
   direction.
2. **`sink_without_global_layers` (A8 → A10) is pre-labelled a replication** of
   the established sink effect in a globals-free topology — not a new finding.
   It is the contrast most likely to look like a clean positive result, so it is
   labelled before the fact and must be reported eviction-stratified.
3. **New arm `A11_backloaded_sink512`** — A3's global *count* moved to the
   deepest layers (21-27), W512 + 4 sinks. Verified **exactly work-, cache- and
   FLOP-matched to A3** at every diagnostic length, so unlike A10 it carries no
   compute caveat. New contrast `global_placement_given_sinks` (A3 → A11), read
   on the **evicted stratum**, where the reference is a near-exact zero and the
   test is genuinely one-sided: with A3 at 1/40, k wins for A11 gives roughly
   p = 2/2^k, so about six wins would reach significance.
4. **The 2x2 interaction is dropped as an estimand.** The sinks factor is
   applied to 21 layers in the globals-present row and 28 in the globals-absent
   row; A2/A3's global layers already attend positions 0-3 as part of full
   causal attention, so the sinks-absent cells are not prefix-free; and A2 and
   A8 both score exactly 0 above N=512, so that row carries no variance. The
   square may be shown descriptively with the asymmetry stated alongside.

## Registered prediction for A11

**Null, like A10.** A3's placement already includes layers 19, 23 and 27, so
"deep globals" is not wholly untested, and I expect A11 ≈ A3 overall
(0.20-0.28) with **0-2 of 40 on the evicted stratum**.

Running it anyway is still worth 72 generations, because a null *at exactly
matched work* converts "we could not measure this" into a measured bound: 0/40
gives a one-sided 95% upper bound near 0.07 on evicted-evidence recovery,
against A0's 27/40 = 0.675 on the same cases. That makes "no 7-of-28 global
topology, at any of three placements, recovers evicted evidence" a measured
statement rather than an absence of evidence.

**What would overturn my prediction:** A11 winning six or more evicted cases
that A3 loses. That would mean placement, not count, governs whether global
layers can rescue evicted evidence — the first positive architectural lever
this study would have found, and it would immediately motivate a placement
sweep rather than a training stage.
