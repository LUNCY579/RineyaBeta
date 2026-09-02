# RETIRED — DAST (Depth-Aligned Semantic Transitions), V1 and its diagnosis tree

**Retired 2026-09-02.** Status label only: the branch, its history and its artifacts
are untouched and fully recoverable. See the retirement tag for un-retire instructions.

**Terminal state: CLOSED — NEGATIVE (EXPLORATORY_DEV).**
**Sealed TEST: never opened, never read.**
**SSOT §13 one-mechanism revision budget: UNSPENT.**

---

## Why it is retired

The result is a **complete negative**, not a stopping point. Every successor route was
measured and excluded rather than left unfunded.

The thesis — that aligning cumulative chronological reasoning states to increasing
Transformer depth improves reasoning — is not supported. Forward-ordered supervision
*costs* about **1.00 pp** (p = 0.0449, consistently scored), which is below the 2.0 pp
practical margin fixed in advance, below its own three-seed minimum detectable effect of
1.39 pp, and found at ~54% post-hoc power.

The deeper reason it closes: **`L_state`, the objective the mechanism optimises, is
satisfiable from the layer-0 token-embedding centroid, before any decoder layer runs.**
Four bottom layers fit it *better* than V1's spread-out sites (0.131 vs 0.143), and a
closed-form ridge beats 250 steps of training the real mechanism (0.263) by ~1.8×. So no
rearrangement of *where* the objective is applied could rescue it. **This closes on the
target object, not the depth mapping** — which is why every successor built on the same
target inherits the defect.

## Where each hypothesis landed

| | outcome |
|---|---|
| **H1 / H4** depth-localized, target-specific | **unreachable.** One-seed MDE is 2.27 pp against a 1.00 pp total effect; a per-checkpoint quarter needs ~88 seeds. Stage E did not supply a substitute endpoint. |
| **H2** gradient conflict | **not supported, twice.** At convergence all arms are within 2σ and ~0.01 of orthogonal. As a trajectory (B2-T, 5 init seeds), AMBIGUOUS — and *dissociated*: the conflict is significant for ordered−reverse where accuracy is null, and null for ordered−final_repeated where accuracy is significant. |
| **H3** overconstraint | **supported** — a diagnosis, not a mechanism to build. Permutation spread is 1.89% of `L_state`. |
| **H5** example-dependent trajectories | **reliable but negligible.** Split-half r ≈ 0.37–0.39 against a label-shuffled null of 0.08–0.10, over a spread of ~0.003. Adaptive DAST forbidden. |
| **H6** motion / ΔH | **disqualified.** Retains 0.96× absolute states' problem information — so the pessimistic reading of PHASE0 §3.8 was wrong — but fails the layer-0 control identically (−0.005 vs absolute's −0.011). Revision budget unspent. |
| **H7** scale / Rineya | **not licensed.** Replicating a defective objective propagates the defect. |

## The one positive fact in the record

Forward and reverse depth assignment differ reliably in **early-training gradient
geometry** at matched target multiset: ordered − reverse = −0.00357, t = −4.57,
p = 0.0103, negative in 5 of 5 initialisations. The first replicated non-null arm
difference in the programme. It is optimisation geometry, not reasoning, and it is
dissociated from capability, so it licenses nothing on its own. It would become
interesting only if a future mechanism made a capability prediction that tracked it.

## Durable outputs — reusable, and worth promoting off this branch

1. **The layer-0 recoverability control.** One forward pass plus CPU ridge fits.
   Disqualifies any depth-supervision target that a probe can recover from the
   token-embedding centroid. Had it existed at the start it would have saved this
   programme roughly 10 GPU-hours. `scripts/depth_information_controls.py`,
   `scripts/motion_target_control.py`.
2. **The frozen-probe framework.** `src/dast/depth_cache.py`, `src/dast/depth_probe.py`
   — exhaustive permutation landscapes, layer×stage profiles, per-problem preference
   with split-half reliability against label-shuffled nulls. Factorised so exhaustive
   K! search is cheap; verified against a naive estimator.
3. **Endpoint power arithmetic.** `scripts/endpoint_power.py` — exact-McNemar MDE from a
   measured discordance rate. Run before, not after, committing to a sweep.
4. **The gradient-conflict probe.** `src/dast/grad_conflict.py` — read-only, pinned
   bit-identically inert under gradient checkpointing with dropout active.

## Known-broken / do-not-cite

- **`FINDINGS_G3_SEEDS.md` §2b's rank and spread figures are WITHDRAWN.**
  `scripts/permutation_preference.py` has no RNG and does one fixed split, so its
  "20 random half/half splits" numbers were never computable. Superseded by
  `artifacts/c1b_permutation_k4_dev.json`.
- **The scorer was not frozen across runs.** All three seed-17 arms predate `f160f38`.
  Cite the consistently re-scored values (`artifacts/rescore_consistency.json`), not the
  originally published ones.
- **Stage E is not a capability screen.** It reaches p ≈ 1e-48 but its magnitude is
  uncorrelated with accuracy (r = +0.188). Keep it as an auxiliary-interference monitor
  only.
- **42.5% of DEV problems receive a duplicated depth target at K=4**, so `state_ordered`
  is partly a `state_final_repeated` arm on those. Known since PHASE0; the distinctness
  filter that addresses it was a no-op until 2026-09-01.

## Unfinished, and not required

The cleanest form of B2-T's snapshot check — re-running the step-250 snapshot procedure
against the B2-T run's *own* saved adapter, so the weights are identical and any gap is
purely procedural — was never run; the accelerator was taken by an unrelated job. It is
**not** needed: the caveat it would address has already been lifted analytically
(`docs/FINDINGS_B2T_TRAJECTORY.md` §5, all three arms agree within 2.2 combined SE). It
remains the tidier test if anyone reopens this.

## Process lessons worth carrying forward

- **Freezing a protocol does not audit it.** An adversarial design review found three
  *invalidating* faults in a B2-T protocol already called frozen — two statistical, one
  of them a bug `cli.py:68` records having been fixed once before. Run the review before
  spending GPU hours, especially on any experiment whose *null* verdict would close a
  research direction.
- **Never let a programme-closing verdict be the default output of missing data.** The
  first B2-T analyser printed "H2-T REFUTED … the closure is complete" from three empty
  files.
- **A sign test is the wrong validity check** when the quantity's per-step noise is
  several times its own mean. That is how a perfectly good instrument came to look
  suspect.
- **Check endpoint power first.** Three separate over-readings of under-powered
  contrasts are recorded in `FINDINGS_G3_SEEDS.md` §4a; a fourth was avoided only by
  computing the MDE before launching.

## Entry points for a future reader

`docs/DAST_V1_CLOSURE.md` (claim boundary) → `docs/DAST_V2_HYPOTHESIS_TREE.md` (H1–H7
and where each landed) → `docs/FINDINGS_C1_FROZEN.md` (the layer-0 finding) →
`docs/FINDINGS_B2T_TRAJECTORY.md` (the last experiment) → `PROGRESS.md` (chronology).
