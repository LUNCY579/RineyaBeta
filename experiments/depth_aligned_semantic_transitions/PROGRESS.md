# DAST progress

## 2026-09-01 — five corrections to the three-seed write-up, after external review

**Status unchanged: BUILT — UNRUN on a qualifying run. TEST CLOSED. EXPLORATORY_DEV.**

External review of `docs/FINDINGS_G3_SEEDS.md` raised five corrections. All five were
valid; all five are now applied. Two of them make the negative result **sharper**,
and one **refutes** a reading I had encouraged.

**1. "significant vs not significant" is not "significantly different".** The draft
said the cost "is specific to the canonical forward assignment". Ordered-baseline at
p = 0.037 alongside reverse-baseline at p = 0.80 does not establish ordered != reverse;
the direct test is ordered - reverse = -0.88 pp at **p = 0.110**. Also recorded: ~7
contrasts with no multiplicity correction (Bonferroni would need p < 0.007), and
`SSOT §5` designates ordered-vs-baseline *insufficient* while the designated decisive
contrasts are not significant.

**2. The mechanism metrics are NOT evidence of learning.** This is the most important
correction. `PHASE0_READINESS.md` §3.6 had already warned that after target centering
the untrained model reaches margin 0.977-0.995, so a high margin is necessary but
non-discriminative -- and I quoted margin 0.935 as "mechanism learned superbly"
anyway. Measured now under exactly the final metric and geometry:

| reference | margin | top-1 | `L_state` |
|---|---:|---:|---:|
| untrained MLP projector (`cot_baseline`) | ~0.00 | 0.25 | - |
| **fitted ridge probe, FROZEN backbone** | **0.9947** | 0.8250 | **0.2229** |
| trained projector (`state_ordered`) | 0.9350 | 0.8332 | 0.2629 |

A least-squares readout on the untouched backbone **matches or beats** the trained
projector. The trajectory was already linearly decodable; training learned to read
something already present.

**3. The reverse-preference reading is REFUTED.** Ran all 4! = 24 state-to-depth
permutations on the frozen model over 20 random splits
(`scripts/permutation_preference.py`):

| assignment | best | median | worst | #1 finishes |
|---|---:|---:|---:|---:|
| identity (forward) | 1 | **3** | 15 | 5/20 |
| reverse | 3 | **12** | 20 | **0/20** |

The frozen geometry mildly prefers **forward**, and reverse is never best. So the
trained `L_state` asymmetry is an optimisation effect, not a representational
preference -- "transformers reason backwards" is not supported. Also: the spread
across all 24 permutations is 0.0096, about **4.3%** of `L_state`, so the network is
close to indifferent about which state sits at which depth. That indifference is
itself evidence against the original thesis.

**4. `state_shuffled` is absent, so G2 is not closed.** `SSOT §5` requires ordered
versus shuffled AND reverse. Shuffled exists only in the G1 pilot at one seed and 200
DEV. This document establishes the primary directional result, not gate closure.

**5. The OOM chronology is engineering provenance, not evidence** — trimmed to two
paragraphs pointing at PROGRESS and the commits.

### Verdict, graded

Strongly supported: the forward-order thesis is not supported (`baseline ~ reverse >
forward`). Moderately supported: a small forward-ordering interference effect, below
the prefixed 2 pp margin. Refuted: intrinsic reverse preference. Not established:
that forward is uniquely responsible. Non-evidence: the mechanism metrics.

The framing that survives: representations can accommodate depth-aligned semantic
supervision, doing so does not help reasoning, and forcing chronological progression
onto depth mildly interferes. **Depth is not reasoning time.**

---

## 2026-09-01 — three-seed replication: mechanism learned, forward ordering COSTS accuracy

**Status: BUILT — UNRUN on a qualifying (signed-preregistration) run. TEST CLOSED.
EXPLORATORY_DEV at the preregistered power level. The primary contrast is complete
across three seeds; it is not confirmatory because the preregistration is unsigned.**

Full report: `docs/FINDINGS_G3_SEEDS.md`. Seeds 17/29/43, 4,000 train / 800 DEV,
250 optimizer steps, ~50 min per run.

| arm | s17 | s29 | s43 | mean | n |
|---|---:|---:|---:|---:|---:|
| **`cot_baseline`** | 0.8938 | 0.8950 | 0.8925 | **0.8937** | 3 |
| `state_reverse` | 0.8900 | 0.8912 | 0.8950 | 0.8921 | 3 |
| `state_ordered` | 0.8775 | 0.8862 | 0.8862 | **0.8833** | 3 |
| `state_final_repeated` | - | 0.8912 | - | 0.8912 | **1** |

| contrast | pooled diff | discordant | p |
|---|---:|---:|---:|
| **ordered - baseline** | **-0.0104** | 54:79 | **0.0370** |
| reverse - baseline | -0.0017 | 68:72 | 0.8000 |
| ordered - reverse | -0.0088 | 68:89 | 0.1102 |

**The result.** The shared projector learns the depth-to-state alignment extremely
well - order margin 0.935, checkpoint top-1 0.833 against 0.25 chance, monotonic
progress 0.985, LM perplexity untouched. It also does not help: `state_ordered` is
below plain CoT SFT in all three seeds, pooled -1.04 pp at p = 0.037 over 2,400
paired problems. And the cost is specific to the canonical FORWARD assignment -
reversing it costs -0.17 pp (p = 0.80, positive at seed 43) and flattening it to a
repeated final state costs -0.37 pp (p = 0.75, one seed).

**No directional advantage, and this is the cleanest result.** `L_state` scores each
arm on its OWN targets: reverse 0.2520/0.2518/0.2563 against ordered
0.2622/0.2644/0.2621. A fully reversed assignment is EASIER to fit, in every seed.
The margin and monotonic columns cannot be used for that comparison because they are
computed against canonical order and penalise reverse by construction.

**Four constraints.** -1.04 pp is BELOW the 2.0 pp practical margin fixed in
`PREREGISTRATION.md` before these runs, and that bar is not being moved now the
effect landed under it. The preregistration is unsigned, so nothing is confirmatory.
G3 has one seed of three. Scope is one 0.6B model on one procedurally-generated
corpus.

**Two runs missing, environmental.** Both `state_final_repeated` failures were
accelerator-memory faults, not code: with no training running, ~20 desktop processes
held 5,592 of 8,188 MiB, and desktop consumption drifted from ~2.5 GiB overnight to
5.6 GiB by 07:00. One CUDA OOM (partly self-inflicted - GPU-touching tests were run
during training, see 9735aed) and one CUBLAS_STATUS_INTERNAL_ERROR in a bf16 GEMM.
Mitigations added, both protocol-neutral: expandable_segments in the launcher, and
`--min-free-mib` in resume_runs.py so a run waits for headroom instead of wasting 50
minutes dying into a squeeze. Batch sizes and sequence length were deliberately NOT
reduced - that would have changed the config hash and made the re-queued runs a
different protocol from their own seed-mates. Re-queued after the reboot with 7.6 GiB
free.

### Next permitted action

Finish G3 at seeds 17 and 43, which decides the interpretation: whether the cost is
specific to progressive forward supervision, or comes from carrying any consistent
auxiliary depth target. Then the Stage-C freeze decision, which is the operator's.
The recorded revision candidate remains semantic MOTION rather than absolute state
(`PHASE0_READINESS.md` §3.8), and spending the SSOT §13 one-mechanism budget is a
decision, not an implementation detail.

---

## 2026-08-31 — budget escalation: mechanism learned, capability HARMED

**Status: BUILT — UNRUN on a qualifying frozen multi-seed arm run. TEST CLOSED.
EXPLORATORY_DEV, one seed. Not confirmatory, and the hypothesis is not closed.**

Full report: `docs/FINDINGS_G2_BUDGET.md`. Three arms, seed 17, 4,000 train /
800 DEV, 250 optimizer steps — 4x the G1 pilot on both axes.

| arm | accuracy | order margin | top-1 | monotonic | `L_state` |
|---|---:|---:|---:|---:|---:|
| **`cot_baseline`** | **0.8938** | +0.0141 | 0.2750 | 0.6125 | - |
| `state_ordered` | 0.8775 | **+0.9356** | **0.8338** | **0.9862** | 0.2622 |
| `state_reverse` | 0.8900 | -0.1911 | 0.1047 | 0.0000 | **0.2520** |

| contrast | diff | clustered 95% CI | discordant | McNemar p |
|---|---:|---|---:|---:|
| **ordered - baseline** | **-0.0163** | **[-0.0317, -0.0025]** | 13:26 | **0.0533** |
| reverse - baseline | -0.0037 | [-0.0252, +0.0157] | 20:23 | 0.7608 |
| ordered - reverse | -0.0125 | [-0.0326, +0.0100] | 19:29 | 0.1934 |

**The budget confound is resolved on the mechanism side.** Four times the training
moved `L_state` 0.5441 -> 0.2622 and the order margin 0.5751 -> 0.9356, nearly
reaching the 0.977 that a fitted linear probe gets on the untrained backbone. Depth
-aligned semantic states are clearly learnable.

**But capability moved the other way.** `cot_baseline` is the best arm. This is the
first contrast in the program whose interval excludes zero, and it says the
depth supervision COSTS about 1.6 points of accuracy versus not doing it at all.
`SSOT §13` names this exactly: "ordered improves mechanism metrics but not
capability" is a **revise** condition.

Four reasons not to over-read it: one seed (three are required; one gives 56.7%
power); the clustered interval and exact McNemar disagree at the boundary (excludes
zero vs p = 0.0533); **-1.63 pp is below the 2.0 pp practical margin** proposed for
the freeze; and `ordered - reverse` still spans zero, so there is still no
detectable forward-direction advantage.

**Verified before reporting.** Seven checks passed: reverse is genuinely reversed;
distinct config hashes and adapters; identical data hashes, checkpoints, readout,
centering and seed; same 800 problems in order with 88.9% identical raw responses;
extraction routes identical across arms; discordance 6.000%, matching the pilot's
6.0% on 4x the data. One genuine scorer gap surfaced — `answer_type: state` was
compared by exact string although its gold is an order-insensitive bag of key=value
bindings — worth 1 problem in 800, fixed in `f160f38`.

### Next permitted action

Not the 18-19 h matrix. Two targeted runs answer go/no-go for ~40% of that cost:
seed replication of these three arms at seeds 29 and 43 (~5 h, converts a borderline
single-seed result into an answer at 97.8% power), then `state_final_repeated` at
three seeds (~2.5 h, **G3** — decides whether the deficit comes from aligning the
loss to reasoning order or merely from carrying any auxiliary depth loss at all).

---

## 2026-08-31 — G1 pilot RE-RUN, four arms, valid scorer: EXPLORATORY_DEV

**Status: BUILT — UNRUN on a qualifying (frozen, multi-seed) arm run. TEST CLOSED.
First real DEV evidence exists. It neither supports nor closes the hypothesis.**

Full report: `docs/FINDINGS_G1_PILOT.md`. Four arms, seed 17, 1000 train / 200 DEV,
corrected geometry, fixed scorer, 63 optimizer steps each, ~12-14 min per arm.

| arm | accuracy | order margin | top-1 | monotonic | LM ppl | `L_state` |
|---|---:|---:|---:|---:|---:|---:|
| `cot_baseline` | **0.7800** | 0.0132 | 0.2700 | 0.6550 | 1.0547 | - |
| `state_ordered` | **0.7800** | **0.5751** | **0.7100** | **0.9750** | 1.0536 | **0.5441** |
| `state_shuffled` | 0.7600 | -0.0908 | 0.1863 | 0.4100 | 1.0551 | 0.9745 |
| `state_reverse` | 0.7650 | -0.1571 | 0.1450 | 0.0000 | 1.0557 | **0.5403** |

`ordered - baseline` is **+0.0000**, discordant **5:5**, McNemar **p = 1.0000**.

**Read the power table first.** Measured discordance is 6.0%, so n=200 gives 10-15
discordant pairs and a real 2:1 effect would have been missed with **~82%**
probability. No null above is evidence of absence.

**The one result robust to that:** `L_state` compares each arm on its OWN targets.
`state_reverse` = 0.5403 versus `state_ordered` = 0.5441, so a fully reversed
state-to-depth assignment fits marginally BETTER. `SSOT §5` assigns reverse the job
of testing directional compatibility of reasoning and depth; on this evidence there
is none. The margin and monotonic columns cannot be used for that comparison because
they are computed against the canonical order and so penalise reverse by
construction.

**The ordered-vs-shuffled gap is shuffled being harmed, not ordered being helped.**
The baseline anchors it: ordered lands exactly on baseline while shuffled (-0.020)
and reverse (-0.015) sit below. And shuffled's failure to fit is near-tautological -
the permutation is drawn per problem from SHA-256 of seed and problem id, so no
consistent function exists for the shared projector to learn. That gap must not be
reported as evidence about reasoning.

**Most plausible alternative explanation:** 63 optimizer steps is very little. The
trained projector's margin (0.575) is still below the 0.977 a fitted linear probe
reaches on the UNTRAINED backbone under the same geometry.

### Preregistration TBDs now resolvable from measurement

- practical minimum effect: **1.56 pp** at 3 seeds x 800 DEV (80% power, alpha 0.05,
  6.0% discordance). Proposed freeze: **2.0 pp**.
- DEV size: 800 per seed with **three** seeds; one seed at 800 gives only 56.7%
  power and cannot be confirmatory.

### Next permitted action

Escalate the training budget before the seed count, because it is the cheapest test
of the most plausible confound: 4000 train records (250 steps) for `cot_baseline`,
`state_ordered` and `state_reverse`, one seed, 800 DEV. ~3 h, about a fifth of the
full matrix. Drop `state_shuffled`, which measures an impossibility. Do **not** run
the 18-run matrix to confirm a null, and do **not** declare the hypothesis closed.

### Local qualification at `f5ae82f`

```text
compileall  PASS (0.2s)   pytest_device_visible  PASS (20.9s)   validate_fixture PASS (2.5s)
ruff        PASS (0.7s)   pytest_cpu_only        PASS ( 8.0s)   smoke            PASS (7.4s)
VERDICT: PASS
```

Artifact: `qualification/latest.json`.

---

## 2026-08-31 — G1 pilot: capability metric WITHDRAWN, instrument failure found and fixed

**Status: BUILT — UNRUN on a qualifying arm run. TEST CLOSED.
The G1 pilot's capability metric is WITHDRAWN. Its mechanism metrics stand as
EXPLORATORY_DEV. No arm comparison is claimed.**

A three-arm G1 pilot (1000 train / 200 DEV, one seed, generation on) was started.
`cot_baseline` and `state_ordered` completed; `state_shuffled` — the **decisive**
control — was killed partway by an external process stop and produced no summary.

### The pilot reported strict accuracy 0.0000 for both arms

That is an instrument failure, not a model result. The raw responses were visibly
correct:

```text
gold: 246 bolts
raw : Reasoning:
      1. Apply the restock: 246 + 85 = 331.
      ...
      5. Therefore, the answer is 246 bolts.
extracted: "5. Therefore, the answer is 246 bolts."   -> scored WRONG
```

Cause: even a short SFT taught the model the corpus's own `full_response` format,
which ends `"N. Therefore, the answer is X."`. The shipped extraction pattern
required a **colon or equals** after "answer", so **200/200** responses fell through
to the last-line fallback — and the list enumeration `5.` then became the
prediction's first number, losing every numeric comparison.

So **SFT made the model better at the task and drove the scorer to zero.** Every
arm of a Phase-1 matrix would have tied at 0.000, producing a completely spurious
`CLOSED — NEGATIVE`. This is the second time in this program that an endpoint has
been caught measuring itself rather than the model.

### Size of the failure, measured on the saved predictions

| n=200 | as scored by the run | rescored, fixed extractor | gold present in response |
|---|---:|---:|---:|
| `cot_baseline` | **0.0000** | **0.7800** | 0.7750 |
| `state_ordered` | **0.0000** | **0.7850** | 0.7850 |

The fixed extractor sits at the achievable ceiling, so the remaining ~22% are
genuine model errors (wrong arithmetic, incomplete sets) — the endpoint has real
discriminative range once it is reading the answer at all.

### Fixes

- `extract_answer` accepts `answer is X` alongside `answer:` and `answer =`.
- The last-line fallback strips a leading `N.` / `N)` enumeration — but only when
  followed by whitespace, and only on that route. Two bugs were found in the first
  attempt at this and are pinned by tests: without the trailing `\s+` the pattern
  ate the integer part of decimals (`14.71` → `71`, `94.` → empty), and applying it
  to a labelled capture destroyed legitimate values.
- Markdown emphasis (`**94**`) is stripped; the underscore is **not**, because it is
  legitimate answer content (`43UZOZ_end`).
- `tests/test_extraction_real_responses.py` pins one verbatim final line per
  answer_type taken from the real pilot output, so extraction cannot regress on
  output a fine-tuned Qwen3-0.6B actually produced. Suite is now **110 tests**.

**Discipline note.** Changing a scorer after seeing results is normally forbidden.
This change was driven by the **absolute** value being 0.0000 with visibly correct
responses — an unambiguous instrument fault — and not by any arm *difference*. It is
applied identically to every arm. The corrected scorer must be frozen before the
comparative matrix, and the pilot itself must be **re-run**, not rescored into a
result.

### What the mechanism metrics show, and what they do not

| | `cot_baseline` | `state_ordered` |
|---|---:|---:|
| order margin | 0.013 | **0.575** |
| checkpoint top-1 (chance 0.25) | 0.268 | **0.713** |
| decoded progress monotonic | 0.635 | **0.975** |
| teacher-forced LM perplexity | 1.0556 | 1.0545 |

The shared projector clearly learns the depth-to-state alignment, and LM quality is
preserved. But two cautions, both of which cut against reading this as support:

1. **The baseline is not a control for this.** `cot_baseline` runs with
   `state_weight: 0`, so its projector is untrained and its margin is simply chance.
   A trained auxiliary head beating an untrained one is not evidence about depth.
2. **The trained margin is BELOW the frozen floor.** A fitted shared *linear* probe
   on the **untrained** backbone reaches margin 0.977 under the same corrected
   geometry (§3.8 of the readiness report). The trained MLP projector reached 0.575
   after 63 optimizer steps. So there is as yet **no evidence of learned
   reorganisation** beyond what a probe already extracts from the frozen model.

Capability difference between the two arms after rescoring is **+0.005**, i.e. one
problem out of 200, on one seed, without the decisive control. That is **not a
finding** in either direction, and `SSOT §5` already states that ordered-vs-baseline
is insufficient because the arms differ in active losses.

### Measured cost, for the Stage-E projection

| | wall-clock |
|---|---|
| `cot_baseline`, 1000 train / 200 DEV + generation | 11 min 41 s |
| `state_ordered`, same | 12 min 13 s |

Scaling 4× on both train and DEV gives roughly 45–50 min per arm, so the 18-run
matrix is **~14–15 h GPU**, consistent with the earlier 15–25 h estimate. Peak VRAM
observed 4.1 GiB of 8.

### Next permitted action

Re-run the **full three-arm** pilot under the fixed scorer, including
`state_shuffled`. Only its ordered-vs-shuffled contrast can say anything, and only
then is the 15 h matrix worth committing.

---

## 2026-08-31 — remote CI blocker RESOLVED as an account billing issue

The 2026-08-30 entry classified the Actions failures as
`EXTERNAL BLOCKER — GITHUB-HOSTED RUNNER NOT ASSIGNED` and recorded that "the
specific platform-side cause cannot be established from the available run
metadata". It can now. The annotation on **every** run, including both attempts
from that session, is:

```text
The job was not started because recent account payments have failed or your
spending limit needs to be increased. Please check the 'Billing & plans'
section in your settings
```

| run | head | duration | steps | runner_id | conclusion |
|---|---|---:|---:|---:|---|
| 33356304292 (2 attempts) | `91f2e1c` | 5s | 0 | 0 | failure |
| 33356616512 | `a343e1f` | 4s | 0 | 0 | failure |
| 33431122752 | `96644f7` | 3s | 0 | 0 | failure |

So the earlier classification was correct in substance and is now specific:
**GitHub Actions billing / spending limit, at the account level.** No repository
change can affect it, and the ruff error fixed in `30119fe` was never what
these runs were failing on — they never reached the Lint step, or any step.

Commit status for `96644f7`: 0 status contexts; 1 check-run
`offline-qualification: completed/failure`, 0 steps executed.

**Action required outside this repository:** resolve Billing & plans, then re-run
run 33431122752. Until then remote CI cannot qualify anything.

### Local CI equivalent, used as the substitute qualification artifact

The four steps of `.github/workflows/dast-ci.yml` run locally under CPU-only
conditions matching `ubuntu-latest`, at `96644f7`:

```text
compileall   PASS
ruff         PASS
pytest       PASS   (94 tests)
smoke        PASS
OVERALL      PASS
```

This is engineering evidence only, and it is weaker than a hosted run in one
specific way: it does not prove the workflow file itself is correct, only that the
commands it invokes succeed. That remains unqualified until a runner is assigned.

---

## 2026-08-31 — post-review corrections to the V1.1 commit

An adversarial review of `30119fe` over four lenses (correctness,
scientific-validity, reproducibility, test-quality) raised 29 findings, each
verified or refuted by an independent agent. It caught **a real regression
introduced by that commit**, plus several repairs whose tests did not actually pin
them. Recorded here rather than amended into `30119fe`, because "a review caught
this" is part of the provenance.

### The regression

Adding `time` and `statistic` to the scorer's numeric-tolerance set routed them
through `_numeric_value`, which uses `re.search` and therefore compared only the
**first** number in each answer. Verified by execution:

```text
answers_equal("16:20", "16:55", "time")                            -> True   WRONG
answers_equal("mean 3.5, range 12", "mean 3.5, range 99", ...)     -> True   WRONG
```

All 160 DEV `time` answers are `HH:MM`, so the minute field was always discarded;
24 of 80 `statistic` answers are `mean X, range Y`, so the range was always
discarded. In the shipped 800-problem eval window that is ~61 problems (7.6%).
Both were **False** before `30119fe` under exact string equality, so the commit
made the primary endpoint worse for those types: it inflates absolute accuracy and
converts genuinely discordant pairs into ties, which is exactly the paired power
the endpoint exists to provide.

Fixed by comparing **all** numbers elementwise when the gold carries more than
one, while keeping first-number comparison for single-number golds so the intended
unit tolerance (`129 N` vs `129`, `$14.50` vs `14.50`) is preserved. The reviewer's
`rate` claim did **not** hold and was not acted on: all 80 DEV `rate` answers are
single-number-plus-unit.

### A no-op hiding inside the readout repair

Chasing "these tests are vacuous" produced a more important finding. The shared
projector begins with its own `LayerNorm`, which absorbs any **uniform** affine
transform exactly. So `checkpoint_readout: final_norm` does nothing at all when the
final norm's gain is uniform — and the toy backbone's `final_norm` was at default
init (weight 1, bias 0), on top of `ToyBlock` already ending in a `LayerNorm`.

Consequence: the offline smoke and every unit test would have passed with the
readout entirely broken. The toy now carries a deterministic **per-dimension**
gain, which is what a trained model has and the only reason the readout measurably
de-correlated the checkpoints on Qwen3-0.6B (7v21: 0.954 → 0.689).

### Other confirmed defects, now fixed

| Defect | Fix |
|---|---|
| The `final_norm` guard tested the configured **fraction**, but `int(round(L·f))` maps all of `[1 − 0.5/L, 1.0)` onto layer `L`. `0.99` on an 8-layer model, or the shipped `0.964` on any model with `L ≤ 13`, resolved to the top layer while passing validation | assert on the **resolved depth** in `DASTModel.__init__`, where `checkpoint_layers` and `num_hidden_layers` are both known |
| `_run_status` took its scope from `eval_split` alone, so a run that **trained** on the sealed split reported `..__DEV__..` | consider both splits |
| `_encode_targets` defaulted `reference_texts=None`, silently centering on the arm's own grid — the mode its own docstring forbids, which would unmatch the controls and collapse `state_final_repeated` to a zero target | refuse without an explicit canonical reference |
| The three new `Literal` config fields were unvalidated, so a typo survived config load and surfaced as a bare `KeyError` after the model and corpus were in memory | validated by value; `confirmatory_dev` additionally requires `preregistration_frozen` |
| No arm config set `evidence_status`, so all six DEV arm runs would have emitted the same manifest status as the 16-problem rehearsal | `base.yaml` sets `exploratory_dev`; the rehearsal overrides to `engineering` |
| The smoke config pinned the **pre-repair** geometry, so the only end-to-end offline test never exercised either protocol correction | smoke now runs `final_norm` + `within_problem` at fractions ending 0.9 |

### Repairs that had no test, and now do

Left-padded generation, seed-before-construction, the feedback hook readout, the
shipped-init intervention no-op, and the smoke device fix were all revertible with
the suite still green. `tests/test_v11_review_fixes.py` pins each. Two assertions
in the first pass were tautological (`allclose(last, last)`) and are replaced with
the three facts actually verified on Qwen3-0.6B: `lm_head(hidden_states[-1])`
reproduces `logits`, `lm_head(hidden_states[-2])` does not, and the last entry must
never be normed twice.

Suite is now **94 tests**, passing with a GPU visible, with `CUDA_VISIBLE_DEVICES=-1`,
and under CI-like invocation (`PYTHONPATH=src` only). Smoke PASS, and the real
seam rehearsal still exits 0 with `checkpoints [7, 14, 21, 27]`,
`readout final_norm`, `centering within_problem`.

### Not acted on

One reviewer claim is **declined**: that per-split corpus files make
`validate_dataset`'s cross-split duplicate detection unable to fire. True as
stated, but the corpus itself asserts `cross_split_problem_overlap: false` in
`dataset_stats.json` and ships `source_pair_sha256` per record. The right response
is a cross-file contamination check as a separate tool, not a change to this
commit. Recorded in `docs/PHASE0_DEFECT_AUDIT.md`.

---

## 2026-08-31 — V1.1 readiness correction

**Status: BUILT — UNRUN on a qualifying arm run. TEST CLOSED.
G0 now passes on the GPU host. The real Qwen/LoRA seam executes end to end.**

The scientific status is unchanged: no arm has been trained, no DEV comparison
exists, and nothing here supports a Rineya capability claim. What changed is that
the real-model path now runs at all, and the Phase-0 diagnostics the SSOT asks for
have been measured on the real checkpoint.

Full measurements: `docs/PHASE0_READINESS.md`.
Triaged defect audit (mostly UNVERIFIED, read its header): `docs/PHASE0_DEFECT_AUDIT.md`.

### Why the previous entry's qualification record was misleading

The 2026-08-30 entry recorded `pytest -q PASS (37 tests)` and `smoke PASS` without
naming a host. Measured on this GPU host, same commit, same interpreter, varying
only device visibility:

```text
                                  GPU visible        CUDA_VISIBLE_DEVICES=-1
python -m compileall -q src tests  PASS               PASS
python -m ruff check src tests     FAIL (1 error)     FAIL (1 error)
pytest -q                          36 passed 1 FAIL   37 passed
python -m dast.cli smoke           FAIL               PASS
```

So the recorded PASS was reproducible only on a CPU-only host and failed on the
host that has to run the science. Qualification records now name the host.

### Blockers repaired

| # | Defect | Repair |
|---|---|---|
| B1 | `apply_chat_template(tokenize=True)` returns a mapping on transformers 5.x, so `_as_int_list` raised `ValueError: invalid literal for int(): 'input_ids'` and **every** real-model path died at tokenization | unwrap `input_ids` when the return value is a mapping |
| B2 | `smoke.py` compared a CPU snapshot against CUDA parameters, so the offline qualification crashed on any CUDA host | compare on a common device |
| B3 | one ruff error gated the CI Lint step red, so compile/pytest/smoke never ran in CI at all | fixed |
| B4 | `data.path` pointed at a metadata-only manifest (no `problem`, `answer`, `full_response`, `reasoning_steps`, `compact_state_trace`); `validate-data` failed on all 48,000 records, and train+eval were derived from one path while the corpus ships one file per split | optional `data.train_path` / `data.eval_path`, pointed at `data/full_context/{train,dev}.jsonl` |
| B5 | generation right-padded and decoded from the batch-max width | left-pad, decode per row, and build the mask from known pad counts rather than by value comparison |
| B6 | the endpoint measured the harness: 51% of generations hit the 128-token cap without EOS, and `answers_equal` gave numeric tolerance to only 30.0% of DEV answer types | cap 384 (hit-cap 0.010), and answer-type-aware comparison for measurement/decimal/money/time/set/ordering/boolean |
| B7 | `command_inspect_alignment` read records with no `assert_split_allowed`, so `--split test` printed compact states — including `"Final answer: ..."` — with `allow_sealed_test: false` | guarded; verified to raise |
| B8 | `set_global_seed` ran inside `train()`, i.e. **after** the LoRA adapter and shared projector were constructed, so the same config and seed produced different initial weights on every run | seed first in `command_train` |

### Phase-0 protocol corrections, selected in the shipped configs

Both are switchable, both default to the previously committed behaviour, and both
are recorded in the run manifest so a run is not interpretable without them.

- `model.checkpoint_readout: final_norm` with `checkpoint_fractions` ending at
  **0.964 → layer 27, not 28**. `hidden_states[28]` is post-final-RMSNorm
  (`lm_head(h[28])` reproduces `logits` to 0.0 absolute error) while 7/14/21 are
  raw residual states (elementwise RMS 49/49/49.5 versus 3.06), so a schedule
  pinned to 1.0 fed two representation classes into the single shared projector.
  Measured pooled cosine 7v28: −0.018 raw, −0.111 after norming 7/14/21 only,
  **+0.409** for the 7/14/21/27 schedule — the only variant with no
  near-orthogonal outlier. Config validation now rejects `final_norm` combined
  with a final fraction of exactly 1.0.
- `semantic.target_centering: within_problem`. As shipped, targets carried 56.6%
  of their variance in the problem factor and 9.3% in the stage factor, while
  hidden states carried 90.8% stage and 6.0% problem — the two sides of the
  objective encoded orthogonal factors. Consequence: `state_ordered` sat at
  **0.860** cosine from `state_shuffled`, so the arms shared ~86% of their target
  signal. Centering each problem on its own canonical-state mean moves that to
  **−0.019** and lifts the frozen shared-linear-probe order margin from **0.129 to
  0.977** (0.995 with the readout fix). The mean is taken over canonical states,
  never the arm's own targets, so it is identical across arms and
  `state_final_repeated` still receives a constant non-zero target.

`training.batch_size` 1 → 2 with accumulation 16 → 8. The effective batch and the
optimizer-step budget are unchanged, so arms stay matched, but
`multi_positive_contrastive_loss` returns exactly 0.0 when `batch < 2`, so the
documented anti-collapse term had been contributing nothing. The rehearsal below
reports `loss/contrastive: 0.855`, non-zero for the first time.

### Honesty repairs, no behaviour change

- The smoke previously overwrote the feedback decoder (`normal_(std=0.02)`) and
  the gates (`fill_(2.0)`) and reported only the perturbed logit delta. Because
  `feedback_decoder.weight` is zero-initialised by contract, `delta = decoder(state)
  * gate` is identically zero, so **every intervention is a bitwise no-op until the
  decoder has trained**. The smoke now reports both numbers:
  `shipped_init_max_logit_delta: 0.0`, `shipped_init_is_no_op: true`, alongside the
  perturbed delta that qualifies the path. An untrained feedback run cannot produce
  a causal result and must not be reported as one.
- The code comment claiming `tanh(-3)` "starts near zero" was wrong: it is −0.995,
  near saturation, which also attenuates the gate's own gradient ~100×.
- The toy backbone appended only raw post-layer residuals, so `hidden_states[-1]`
  was pre-norm while Hugging Face's is post-norm. The offline suite and CI were
  therefore structurally blind to the exact asymmetry that only appears on a real
  checkpoint. The toy now matches HF semantics.
- The feedback hook pooled the pre-norm layer output while the auxiliary path read
  post-norm `hidden_states[...]`, so a projector trained under the auxiliary config
  was silently fed a different tensor during the causal probe. Both paths now apply
  the same readout.
- Manifest status was derived from `eval_split` alone, so a 16-problem seam
  rehearsal was recorded as `DEV_COMPLETE`, indistinguishable from a qualifying arm
  run. `experiment.evidence_status` now defaults to `engineering` and the status is
  composed as `TIER__SCOPE__PHASE`; the sealed-split test is case-insensitive.
- `run_manifest` schema v2 hashes **every** corpus file a run touched, instead of
  one path.

### Qualification, on this host

```text
host        RTX 4070 Laptop 8188 MiB, driver 610.62
python      3.11.15   torch 2.13.0+cu130   transformers 5.15.0   peft 0.20.0
base model  Qwen/Qwen3-0.6B, HF cache revision c1899de289a04d12100db370d81485cdf75e47ca

python -m compileall -q src tests                       PASS
python -m ruff check src tests                          PASS
pytest -q                                               PASS (74 tests, GPU visible AND CPU-only)
python -m dast.cli smoke                                PASS (GPU visible)
dast validate-data .../full_context/dev.jsonl           PASS (2400 records, 0 errors)
dast inspect-alignment --split test                     PermissionError, as required
dast inspect-alignment ordered/shuffled/reverse         manual RUNBOOK section 3 checks pass
dast train --config configs/seam_rehearsal.yaml         EXIT 0
```

The RUNBOOK §3 checks were confirmed on real records for the first time: the
initial `"No derived facts yet."` state is excluded, ordered targets progress
forward, shuffled is non-identity with the same multiset, reverse is exact, and
variable-length traces record repeated source indices (canonical `[0,1,3,4]` from
five targets).

### Seam rehearsal — ENGINEERING ONLY

`configs/seam_rehearsal.yaml`, 16 train and 8 eval problems, generation disabled.
This exists to prove the seam executes and to measure cost. It is far too small to
support any scientific claim and must never be aggregated with arm runs.

```text
status                    ENGINEERING__DEV__COMPLETE
resolved checkpoints      7, 14, 21, 27
readout / centering       final_norm / within_problem
precision                 bf16
LoRA trainable params     10,092,544 of 606,142,464 (1.69%)
optimizer steps           1        global steps 8
loss/contrastive          0.855    (was identically 0.0 at batch_size 1)
loss/state                0.963
state/order_margin        0.040    at random projector init, 1 optimizer step
```

The order margin here is **not** comparable to the 0.977 probe figure: that was a
least-squares probe *fitted* to held-out data, this is an untrained MLP projector
after one step. The comparison to make later is Δmargin over the measured frozen
floor, and ordered's Δ against shuffled's and reverse's — not any absolute margin.

### Measured cost, for budgeting Phase 1

| | s/problem | 800 DEV | × 18 runs |
|---|---:|---:|---:|
| batch 4, cap 128 | 1.41 | 19 min | 5.6 h |
| batch 4, cap 384 | 2.78 | 37 min | 11.1 h |

Inference is ~2.5 GiB resident, so 8 GiB is not the binding constraint; decode
throughput is.

### Next permitted action

**Not** the Phase-1 matrix. In order:

1. Re-measure the Phase-0 floors *under the corrected readout and centering*, at
   n ≥ 2,000. Every number in `docs/PHASE0_READINESS.md` was measured under the
   old geometry, and the corrected geometry pushes the untrained order margin near
   saturation — which makes mechanism metrics necessary but non-discriminative.
2. Restate G1 as a delta over that measured floor, and as
   Δmargin(ordered) > Δmargin(shuffled) and > Δmargin(reverse). "Margin is
   positive" is satisfied by the frozen pretrained model and proves nothing.
3. Verify the `docs/PHASE0_DEFECT_AUDIT.md` backlog. 55 of its adversarial
   verifiers died on a session limit, so apart from the rows marked CONFIRMED it
   is a triage queue, not evidence.
4. Only then resolve and freeze `docs/PREREGISTRATION.md`, and record the Stage-C
   design decision.

Operators and feedback remain phase-gated and must not begin because the software
runs.

### Not done, and why

- The Phase-1 matrix, the preregistration freeze, and the Stage-C decision are the
  operator's calls, not consequences of these repairs.
- A stage-contrastive term, a hidden-state semantic-motion head
  (`P_Δ(H_{k+1} − H_k) → O_k`), and any depth-schedule search are deliberately
  **not** implemented. Each is a new mechanism, and SSOT §13 allows exactly one
  bounded revision; spending it is a recorded decision, not an implementation
  detail. The motion direction is supported by measurement (between-problem
  same-stage delta cosine 0.94–0.99 versus within-problem cross-stage −0.32) but
  scored **worse** than the corrected state route on held-out data (margin 0.819
  versus 0.995), so it belongs in Phase 2, not V1.
- Restricting Phase 1 to traces with ≥K distinct states was measured and
  **rejected**: it moves the ordered/shuffled contrast only 0.860 → 0.849 while
  discarding 43.8% of the corpus.
- The repository-wide `docs/EXPERIMENT_PROGRAM_AUDIT.md` still does not mention
  DAST and still asserts the remote exposes one live branch. That is a separate
  documentation commit.
