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
