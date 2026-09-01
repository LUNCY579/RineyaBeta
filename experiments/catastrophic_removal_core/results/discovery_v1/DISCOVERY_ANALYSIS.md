# CRCC v1 — what the discovery scan actually showed

**Scope: discovery only.** No recovery ran, no donor exists, no merge happened, no
sealed asset was opened. Every question below that needs a recovery delta is marked
**unanswerable in this run** rather than guessed at.

Source: `scan_summary.json` (stable summary `f5ea9f12…2884e3`), analysed by
`scripts/critical_core/analyze_discovery.py` into `discovery_analysis.json`.

---

## 1. Headline

The preregistered classifier elected **no layer**. G0 fails on depth (0 outside
1–14); the other six G0 checks pass. So H1 — "a nonempty subset of single-layer
deletions produces collapse jointly visible in reference NLL and generation
mechanics" — is **closed negative for this instrument revision on Qwen3-0.6B**.

That is a statement about single-layer deletion. It is not a statement about
whether recovery deltas make good donors, which remains untested.

## 2. Is catastrophic criticality the same signal as residual angular rewrite?

**Largely yes — they are correlated, not orthogonal.**

| Statistic | Value |
|---|---:|
| Pearson r (criticality vs angular distance) | **+0.5576** |
| Spearman ρ | **+0.6097** |
| Pearson r (log₂ NLL ratio vs angular distance) | **+0.6090** |
| Top-6 overlap | **5 of 6** — {0, 1, 2, 4, 27} |

This matters for the experiment's design. `control_angular_high` was specified as an
*orthogonal* matched control — "largest one-layer residual angular distance outside
CRCC". At ρ ≈ 0.61 with a 5/6 top-6 overlap, that control would have been drawn from
nearly the same population as the hypothesis arm. Had the core been nonempty, the
angular control would have been a much weaker discriminator than the plan assumed,
and the interpretation-table row "angular-high equals or beats CRCC" would have been
substantially more likely for reasons of construction rather than mechanism.

The two signals are not identical, though. Layer 3 ranks 4th by angular distance but
only 11th by criticality (ratio 1.21), and layer 12 ranks 5th by criticality but 23rd
by angular distance. Rewrite magnitude and removal damage disagree in the middle of
the stack.

## 3. Are catastrophic layers clustered by depth? Early, middle or final?

**Sharply clustered, and the answer is "the two ends, mostly the front".**

| Band | n | mean criticality | max |
|---|---:|---:|---:|
| early 0–2 | 3 | **0.5200** | 0.5519 |
| early-mid 3–9 | 7 | 0.0459 | 0.0728 |
| mid 10–19 | 10 | 0.0293 | 0.0804 |
| deep 20–26 | 7 | 0.0195 | 0.0615 |
| final 27 | 1 | 0.1730 | 0.1730 |

Only layers 0, 1 and 2 move reference NLL by more than 1.43× (5.47×, 4.42×, 4.15×).
Layer 27 is a distant fourth. Layers 3–26 are collectively near-free: mean
criticality 0.0313, and layer 14 is indistinguishable from intact (ratio 0.9994).

This reproduces, by a completely different method, the shape the roadmap's angular
profile already reported: layer 0 most precious, layer 27 next, and a broad
redundant interior. Two independent instruments agreeing is the most robust thing in
this run.

## 4. Why the classifier elected nothing

Two properties of the frozen instrument. Both are consequences of the preregistered
design; neither is a coding defect.

**(a) The composite path structurally requires generation mechanics.** With collapse
and intelligibility components at zero, the composite ceiling is
`0.45·1 + 0.10·1 = 0.55`, below the frozen `0.60`. NLL and EOS loss alone can never
elect a layer, no matter how extreme the NLL ratio. That is precisely what H1 asks
for — collapse must be *jointly* visible — so the threshold behaved as written.

**(b) Pooled-mean NLL dilutes the ratio.** The classifier divides pooled means, and
the two diagnostic sources have very different baselines (`rag_train` 0.3455,
`short_preservation_train` 4.4549). For layer 2 the per-source ratios are **25.96×**
and **2.46×**, but the ratio of pooled means is only **4.15×**.

The consequence is a genuine design inconsistency worth recording: the plan treats
two-source support as an *additional* restriction layered on top of the overall
classification, but the overall gate is systematically **less** sensitive than the
per-source gates it is combined with. Layer 2 is classified catastrophic in *both*
sources independently and still fails the overall test. Under a
"classify per source, then combine" rule, layer 2 would have been elected.

**Where the mechanical screen sits.** Bypassing layer 0 produces obvious word-salad
("1ải and1. generalsystem…"), yet that text is not empty, not repetitive, and not
control-character laden, so it scores collapse 0.100 and reads as "intelligible".
The screen detects *degeneracy*, not *semantic destruction* — §4.3 says so on
purpose, to avoid calling multilingual output corrupt. The cost of that choice is
that fluent-but-destroyed output is invisible to it, and fluent-but-destroyed is
exactly what removing layers 0–2 produces.

## 5. Instrument validity

The empty result is not a silently broken hook. Comparing bypassed against intact
logits on identical input:

| Layer | max abs Δlogit | argmax positions changed |
|---:|---:|---:|
| 0 | 27.97 | 100.0% |
| 14 | 5.50 | 10.5% |
| 25 | 6.09 | 15.8% |
| 27 | 12.09 | 21.1% |

Removing the hook restores bit-identical logits. Every bypass perturbs computation,
including layers that scored ≈0. Layer 14's 0.9994 ratio is real near-redundancy for
teacher-forced NLL, not a no-op.

## 6. Relationship to the historical "monster"

The historical 19-layer artefact deleted **nine layers simultaneously**:
`[6, 7, 8, 9, 10, 14, 18, 23, 24]`. Individually, every one of those is cheap in this
scan — 1.27×, 1.19×, 1.09×, 1.21×, 1.13×, 1.00×, 1.23×, 1.05×, 1.08×. None is
remotely catastrophic alone.

So the historical degeneracy was a **joint** effect of removing nine layers at once,
not the removal of any individually critical layer. This scan measured one layer at a
time and therefore could not have found it. That is a limitation of the instrument
relative to the motivating observation, not a contradiction of it — but it does mean
the phrase "catastrophic-removal layer" had no single-layer referent in this model.

## 7. Questions that remain unanswerable in this run

All of these require a nonempty core and a completed recovery, and none was run:

- Do CRCC recovery deltas have larger norms than controls?
- Does delta norm predict donor value?
- Are donor gains localized to the layers that caused catastrophic removal?
- Does random or low-criticality compression produce similar donor benefit?
- Does the high-angular control explain the same effect?
- Are Aegis and Monster deltas complementary, redundant, or conflicting?
- Does coefficient sensitivity indicate a broad basin or a lucky point?
- Does donor usefulness correlate poorly with standalone quality?

Reporting any of these would require inventing data.

## 8. What the layers are best described as

On this evidence, the layers whose single removal hurts most are **representation-
transition layers at the boundaries of the stack**, not a distributed "critical core":

- layer 0 performs the largest single residual rewrite in the network
  (angular distance 0.484) and its removal destroys generation outright;
- layers 1–2 continue that early transition;
- layer 27 is the output-side transition (angular 0.376);
- the interior 3–26 behaves like a deep, highly redundant refinement band where no
  single block is load-bearing.

Calling 0–2 and 27 a "transformation core" would be a stretch: they are the entry and
exit transitions, and the experiment's own classifier declined to elect them.
