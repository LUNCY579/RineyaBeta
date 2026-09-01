# Three-seed replication — findings

**Status: EXPLORATORY_DEV at the preregistered power level. The primary contrast is
complete across three seeds. This is NOT confirmatory: the preregistration in
`PREREGISTRATION.md` is still DRAFT and unsigned, so §8's gates have not formally
been run against a frozen protocol.**

**Branch:** `research/depth-aligned-semantic-transitions-v1` at `31cf2f2`
**Run dates:** 2026-08-31 → 2026-09-01
**Sealed evaluation:** CLOSED, never opened, never read.

Seeds **17, 29, 43**. 4,000 train / 800 DEV problems per run, 250 optimizer steps,
Qwen3-0.6B + LoRA r=16, `checkpoint_readout: final_norm` at layers 7/14/21/27,
`target_centering: within_problem`, greedy generation at `max_new_tokens: 384`,
scorer frozen at `f160f38`. Roughly 50 min per run.

Seed 17's three primary arms are the `g2_budget_*` runs; seeds 29 and 43 are
`g3_s*`. Analysis groups by `(arm, seed)` from each `run_manifest.json`, so the
directory prefix is irrelevant.

## 1. Results

| arm | s17 | s29 | s43 | mean | n |
|---|---:|---:|---:|---:|---:|
| `state_final_repeated` | 0.8850 | 0.8912 | 0.9100 | **0.8954** | 3 |
| **`cot_baseline`** | 0.8938 | 0.8950 | 0.8925 | **0.8937** | 3 |
| `state_reverse` | 0.8900 | 0.8912 | 0.8950 | 0.8921 | 3 |
| **`state_ordered`** | 0.8775 | 0.8862 | 0.8862 | **0.8833** | 3 |

All four arms now have all three seeds. Note `state_final_repeated`'s seed range of
**2.5 pp** (0.8850–0.9100) against `cot_baseline`'s 0.25 pp — it is by far the
noisiest arm, which is why its two-seed reading was misleading (§4a).

Mechanism metrics, seed-averaged:

| arm | order margin | checkpoint top-1 | monotonic | `L_state` | LM ppl |
|---|---:|---:|---:|---:|---:|
| `cot_baseline` | −0.0011 | 0.2503 | 0.5667 | — | 1.0069 |
| `state_final_repeated` | 0.0071 | 0.2500 | 1.0000 | 0.1433 | 1.0068 |
| **`state_ordered`** | **0.9350** | **0.8332** | **0.9846** | 0.2629 | 1.0076 |
| `state_reverse` | −0.1915 | 0.1033 | 0.0000 | **0.2534** | 1.0075 |

Paired by problem id, clustered bootstrap by `generator` (30 clusters), exact
McNemar on discordant pairs:

All pooled over three seeds, 2,400 paired problems:

| contrast | pooled diff | discordant | p |
|---|---:|---:|---:|
| **ordered − final_repeated** | **−0.0121** | 50:79 | **0.0134** |
| **ordered − baseline** | **−0.0104** | 54:79 | **0.0370** |
| ordered − reverse | −0.0088 | 68:89 | 0.1102 |
| reverse − final_repeated | −0.0033 | 65:73 | 0.5514 |
| reverse − baseline | −0.0017 | 68:72 | 0.8000 |
| final_repeated − baseline | +0.0017 | 62:58 | 0.7843 |

Per-seed detail for the primary contrast:

| ordered − baseline | diff | clustered 95% CI | discordant | p |
|---|---:|---|---:|---:|
| s17 | −0.0163 | [−0.0317, −0.0025] | 13:26 | 0.0533 |
| s29 | −0.0088 | [−0.0209, +0.0036] | 19:26 | 0.3713 |
| s43 | −0.0063 | [−0.0276, +0.0136] | 22:27 | 0.5682 |

## 2. The result

> **Depth-aligned semantic states are highly learnable, and aligning them to
> forward reasoning order is the only variant that costs reasoning accuracy.**

The trained projector reaches order margin **0.935**, checkpoint top-1 **0.833**
against a 0.25 chance level, and monotonic decoded progress **0.985**, with the
language model untouched (perplexity 1.0076 against the baseline's 1.0069). §2a
explains why those numbers do **not** demonstrate that training created the depth
trajectory.

What is unambiguous is that it does not help. `state_ordered` is **below** plain CoT
SFT in all three seeds, pooled **−1.04 pp** across 2,400 paired problems at
**p = 0.037**.

```text
state_final_repeated  0.8954  ─┐
cot_baseline          0.8937   ├─ mutually indistinguishable
state_reverse         0.8921  ─┘   (both vs baseline: p = 0.78, p = 0.80)
state_ordered         0.8833  ←    the only arm with a detectable deficit
```

**G3 passes.** `state_ordered − state_final_repeated = −1.21 pp, p = 0.0134` over
three seeds. Carrying a consistent auxiliary depth target is **free** — repeating
the final state at every depth lands on baseline (+0.17 pp, p = 0.78). Only the
canonical forward progression costs. That is the direct contrast `SSOT §8` G3 asks
for, and it is now the *strongest* contrast in the study, stronger than
ordered−baseline.

**Wording correction.** An earlier draft of this document said the cost "is specific
to the canonical forward assignment". That is stronger than the data supports, and
the error is a classic one:

> *significant* versus *not significant* is **not** the same as *significantly
> different*.

Ordered−baseline reaching p = 0.037 while reverse−baseline sits at p = 0.80 does not
establish ordered ≠ reverse. The direct test is **ordered − reverse = −0.88 pp,
p = 0.110** — consistently negative across all three seeds, but not crossing the
conventional threshold.

The defensible statement is therefore:

> **Forward DAST shows a small statistically detectable deficit relative to the CoT
> baseline. Reverse DAST does not. Forward is numerically worse than reverse in
> every seed, but their direct pooled comparison is not significant (p = 0.110).**

Two further reasons not to harden this: ~7 contrasts were computed without
multiplicity correction (Bonferroni at α = 0.05/7 = 0.007 would not admit p = 0.037),
and `SSOT §5` states that ordered-versus-baseline is *"informative but not
sufficient because active parameter use and losses differ"* — the contrast that
reached significance is the one the plan designates insufficient, while the
designated decisive contrasts are not significant.

## 2a. The mechanism metrics do NOT show that training built the trajectory

This is the most important correction in this document, and it makes the negative
result **sharper**, not weaker.

`PHASE0_READINESS.md` §3.6 already warned that after target centering the untrained
model reaches an order margin of 0.977–0.995, so a high margin becomes *necessary
but non-discriminative*. That warning was not applied when writing up these runs.
Applying it properly requires the frozen floor measured under **exactly** the final
metric and the final geometry, which is now done.

Two distinct reference points, both measured, both needed:

| reference | order margin | top-1 | `L_state` |
|---|---:|---:|---:|
| untrained MLP projector (`cot_baseline`, `state_weight: 0`, run's own metric) | **+0.0141 / −0.0243 / +0.0068** | 0.275 / 0.223 / 0.253 | — |
| **fitted linear probe on the FROZEN backbone**, same geometry, same metric, held out | **0.9947** | 0.8250 | **0.2229** |
| trained shared projector (`state_ordered`, 250 steps) | 0.9350 | 0.8332 | 0.2629 |

Read together:

- Against an **untrained projector**, training clearly does something: margin ~0.00
  to 0.935. That is the comparison an earlier draft implicitly made.
- Against a **fitted ridge probe on the untouched backbone**, training does not win.
  The frozen probe reaches margin **0.9947** versus the trained projector's 0.9350,
  and `L_state` **0.2229** versus 0.2629. Only top-1 is a near tie (0.825 vs 0.833).

So the depth-state trajectory was **already linearly decodable from the frozen
model**, at least as well as SGD training the projector achieves. The honest
statement:

> **The high order margin is not evidence that DAST training created a semantic
> depth trajectory. A least-squares readout on the untouched backbone matches or
> exceeds it. Training the projector learned to read something that was already
> present.**

Caveat on comparability: these are different estimators — a closed-form ridge
solution with a held-out split versus an SGD-trained MLP evaluated on DEV. They are
not identical procedures. But both compute diagonal-minus-off-diagonal cosine on
held-out data under the same readout and the same centred targets, so the comparison
is fair in kind, and the direction is not marginal.

## 2b. Does the untouched model prefer some orderings? All 24 permutations

`L_state` for reverse (0.2534) is consistently below ordered (0.2629) across seeds
(§3). One hypothesis was an intrinsic preference: perhaps depth natively hosts the
trajectory backwards. That is directly testable **without training anything** —
for K=4 there are 4! = 24 assignments of states to depths, and a shared linear
readout can be fitted for each on the frozen backbone.

Result over **20 random half/half splits** of 600 DEV problems:

| assignment | best rank | median rank | worst rank | times ranked #1 |
|---|---:|---:|---:|---:|
| identity (canonical forward) | 1 | **3** | 15 | **5 / 20** |
| reverse | 3 | **12** | 20 | **0 / 20** |

**This inverts the intrinsic-preference hypothesis.** On the frozen model the
canonical **forward** ordering fits *better* than reverse — median rank 3 against 12
— and reverse is never the best permutation in any split.

Two consequences:

1. The trained `L_state` asymmetry (reverse easier) **cannot** be attributed to the
   representation geometry preferring reverse, because the frozen geometry mildly
   prefers forward. It must arise from training dynamics — plausibly interference
   between forward depth supervision and the language-model objective, which is
   also the arm that loses accuracy. That is a sharper and more specific claim than
   "transformers reason backwards", which this experiment does **not** support.
2. The preference is weak in absolute terms. The `L_state` spread across all 24
   permutations averages **0.0096**, about **4.3%** of `L_state` itself, and identity
   tops the ranking in only 5 of 20 splits. The frozen model is close to
   *indifferent* about which state goes at which depth.

Point 2 is itself informative for the original thesis: if depth genuinely
implemented a chronological reasoning trajectory, the canonical ordering should be
markedly easier to decode than a shuffle of it. It is not.

## 3. No directional advantage — the cleanest single result

`SSOT §5` assigns `state_reverse` the job of testing "directional compatibility of
reasoning and depth". The answer is that there is none detectable, and it is the
result least dependent on the noisy generation endpoint:

| | s17 | s29 | s43 |
|---|---:|---:|---:|
| `L_state`, ordered | 0.2622 | 0.2644 | 0.2621 |
| `L_state`, reverse | **0.2520** | **0.2518** | **0.2563** |

`L_state` is each arm's fit to **its own** targets, so it compares arms on equal
footing. Under *training*, a fully reversed assignment is easier to fit than the
canonical forward one, in every seed, stable to three decimals. Ordered − reverse on
capability is −0.88 pp pooled (p = 0.110), negative in all three seeds but not
significant.

This must **not** be read as "transformers reason backwards through depth". §2b
tests that directly on the frozen model and finds the opposite: forward fits better
than reverse there (median rank 3 of 24 versus 12). The trained asymmetry therefore
reflects optimisation, not representational preference. Other explanations remain
open and untested — later semantic states may simply be easier targets, early and
late checkpoints may differ in capacity, cumulative-state construction is
asymmetric by nature, and forward supervision may conflict more with the LM
objective.

The order margin and monotonic columns must **not** be used for this comparison:
both are computed against the canonical order and therefore penalise reverse by
construction. That is why reverse shows margin −0.19 and monotonic 0.0000 while
fitting its own targets better than ordered fits theirs.

Similarly, `state_final_repeated`'s `L_state` of 0.1433 and monotonic score of
exactly 1.0000 are artifacts of its target set: four identical targets are much
easier to fit, and "progress" over four identical states is trivially
non-decreasing. Neither number is comparable across arms.

## 4. Four constraints on reading this

1. **−1.04 pp is BELOW the 2.0 pp practical margin** proposed in
   `PREREGISTRATION.md` §6 *before* these runs. Statistically detectable at this n;
   **not** practically significant by the bar set in advance. That bar is not being
   moved now that the effect landed under it.
2. **Not confirmatory.** The preregistration is unsigned, so this is
   EXPLORATORY_DEV. The §8 gates have not been formally executed against a frozen
   protocol.
3. **`state_shuffled` is absent from this matrix entirely.** `SSOT §5` requires the
   primary claim to rest on ordered versus **both** shuffled and reverse. Shuffled
   exists only in the G1 pilot at seed 17 and 200 DEV problems, never at 800 DEV
   across three seeds. So this document establishes the **primary directional
   result** and does **not** constitute closure of the originally specified G2 gate.
   (Its omission was deliberate and is defensible on its own terms — the permutation
   is drawn per problem from SHA-256 of seed and problem id, so no consistent
   function from (depth, prompt) to target exists and its failure to fit measures an
   impossibility rather than a fact about reasoning. But "defensible to omit from an
   exploratory step" is not "satisfies the gate".)
4. **G3 has one seed of three.** `state_final_repeated` completed only at seed 29;
   seeds 17 and 43 failed on an environmental accelerator constraint (§5). A
   single-seed null carries ~57% power, so "final-repeated is harmless" is
   suggestive, not established. It is the control that decides whether the ordered
   deficit is about reasoning order or about auxiliary depth losses in general.
5. **Narrow scope.** One 0.6B model, one procedurally-generated corpus, one LoRA
   configuration, one depth schedule. `SSOT §12` lists template overfitting as a
   live threat and no amount of seeds here addresses it.

## 4a. Three times I over-read an under-seeded contrast

Recorded because it is a methodological result about this endpoint, not merely an
apology, and because the same trap will recur in any follow-up.

| when | claim made | what the full data showed |
|---|---|---|
| 1 seed (`g2_budget`) | ordered−baseline "excludes zero", read as a finding | replicated to p = 0.037 pooled, but the s17 CI was the outlier of three |
| 1 seed | `final_repeated` "harmless" at −0.37 pp | at 3 seeds it is **+0.17 pp** |
| 2 seeds | `final_repeated` "intermediate cost", "a gradient not a binary" | seed 43 came in at **+1.75 pp** and the pooled figure is ~0 |

The arithmetic forbidding all three was computed and written down *before* each
claim: one seed at 800 DEV carries ~57% power at 6.0% discordance.

The sharpest illustration is that **single-seed contrasts in this setup reach
p < 0.05 in both directions from noise alone**: `state_ordered − cot_baseline` at
seed 17 gives p = 0.0533 negative, while `state_final_repeated − cot_baseline` at
seed 43 gives **p = 0.0288 positive**. Neither survives pooling. Any future
single-seed result here should be treated as uninformative regardless of its p-value.

## 4b. Multiplicity

Roughly 13 paired contrasts have now been computed across this study. Bonferroni at
α = 0.05/13 ≈ 0.0038 would admit **neither** ordered−final_repeated (0.0134) nor
ordered−baseline (0.0370).

That correction is conservative for a preregistered set — `PREREGISTRATION.md` §6
names a small decisive family, not thirteen — but the honest position is that both
headline contrasts are **nominally** significant and neither is robust to a strict
family-wise correction over everything that was looked at.

## 5. Why two runs are missing

The two `state_final_repeated` failures were accelerator out-of-memory faults on a
GPU shared with the desktop, not code faults; batch size, sequence length and every
other config value were left untouched so re-queued runs stay protocol-identical to
their seed-mates. An OOM therefore means *run incomplete*, never *scientific
observation*: no partial run contributes a number to any table here.

Full chronology, failure modes and the mitigations added
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `--min-free-mib` so a run
waits for headroom instead of dying into a squeeze) are in `PROGRESS.md` and commits
`9735aed`, `9bfaecf`, `31cf2f2`. It is engineering provenance, not evidence.

## 6. Verdict, graded by how well each claim is supported

**Strongly supported.** The original thesis — that aligning cumulative reasoning
states to increasing depth in *chronological* order improves or preserves reasoning
— is **not supported**. Forward DAST is below the CoT baseline in all three seeds
(0.8775 / 0.8862 / 0.8862 against 0.8938 / 0.8950 / 0.8925), while reverse sits at
baseline. The observed ordering is `baseline ≈ reverse > forward`.

**Moderately supported.** Forcing forward chronological states onto depth produces a
small interference effect: −1.04 pp pooled, p = 0.037. Two things keep this from
being firm — it is **below the 2.0 pp practical margin** fixed in advance, and it is
the contrast `SSOT §5` calls insufficient, while the designated decisive contrasts
are not significant.

**Refuted, and worth recording.** The idea that depth intrinsically prefers a
reversed trajectory. §2b tests all 24 permutations on the frozen model: forward
fits *better* than reverse (median rank 3 of 24 versus 12; reverse never ranks
first). The trained `L_state` asymmetry is an optimisation phenomenon, not a
representational one.

**Not established.** That forward ordering is *uniquely* responsible relative to
every auxiliary-state control. Ordered − reverse is p = 0.110, `state_final_repeated`
has one seed of three, and `state_shuffled` is absent from this matrix entirely.

**Now known to be non-evidence.** The mechanism metrics. A ridge probe on the
untouched backbone matches or beats the trained projector (§2a), so margin 0.935
cannot support any claim that training built the trajectory.

### What DAST appears to have actually found

The framing that survives all of the above is not "DAST failed" but:

> **Representations can accommodate depth-aligned semantic supervision, and doing so
> does not help reasoning — and forcing chronological progression onto depth mildly
> interferes with useful computation.**

That is a statement about architecture rather than about a training trick. A
Transformer has the whole prompt at every layer; its computation need not resemble
*read premise → read premise → derive → answer*. It may instead run something closer
to global hypothesis formation, then constraint integration, then refinement, then
output preparation. Mapping a human-authored chain-of-thought chronology onto that
axis may distort a perfectly good internal computation. **Depth is not reasoning
time.**

### The experiment this points to

Stop *assuming* the forward trajectory and start *measuring* the one the network
already has. §2b is a first pass — 24 permutations, frozen model, no training — and
its answer is that the network is nearly indifferent (spread ~4.3% of `L_state`).
Extending that is cheap and does not spend the `SSOT §13` one-mechanism revision
budget:

- more checkpoints, so the permutation space is informative rather than flat;
- targets that are not mean-pooled bags of embeddings, since `PHASE0_READINESS.md`
  §3.3 showed that geometry is dominated by problem identity;
- semantic **motion** rather than absolute state — between-problem same-stage delta
  cosine was 0.94–0.99 against within-problem cross-stage −0.32
  (`PHASE0_READINESS.md` §3.8), an exact reversal of the absolute-state pattern.

The motion variant *is* a new mechanism and does spend the revision budget, so it is
a recorded decision rather than an implementation detail.

## 7. Reproducing this table

```bash
python scripts/summarize_runs.py                              # arms x seeds + contrasts
python scripts/summarize_runs.py --baseline-arm state_reverse  # ordered vs reverse
```

Both read completed runs from `runs/`, group by manifest, and name any incomplete
run rather than omitting it silently.

## 8. Claim boundary

Three seeds. 800 DEV problems each. 250 optimizer steps. Unsigned preregistration.
TEST sealed and never read. Nothing here is a Rineya capability claim.

The single contrast reaching p < 0.05 sits below the practical margin fixed in
advance, is uncorrected for ~7 comparisons, and is the contrast `SSOT §5` designates
insufficient. `state_shuffled` is absent and `state_final_repeated` has one seed of
three, so this is the **primary directional result**, not closure of the G2/G3
gates. The mechanism metrics are explicitly not evidence of learning (§2a).
