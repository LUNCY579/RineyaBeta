# RGE-PEFT Live Progress Dashboard

**Governing SSOT:** `docs/PLAN_RGE_PEFT.md`  
**Program version:** v1.1  
**Target branch:** `research/rge-peft-v1`  
**Starting parent:** `main @ 3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Initialized:** 2026-08-30, America/Chicago  
**Current milestone:** M5 — corrected Delta-CMA interaction control
**Current status:** M0 GATE_PASSED; M1 GATE_PASSED; M2 GATE_FAILED; M3 GATE_PASSED; M4 CLOSED; M5 IN_PROGRESS; M6 BLOCKED_MATCHING
**Implementation started:** yes
**Experimental outcomes observed:** yes; all claims remain bounded by the ledgers
**Sealed evaluation accessed:** no

This dashboard records operational progress. It may update dates, status, commits, run IDs, compute, blockers, and gate decisions. It may not redefine hypotheses, arm membership, capacity matching, thresholds, or sealed-evaluation policy from the SSOT.

## 1. Status vocabulary

```text
NOT_STARTED
IN_PROGRESS
BLOCKED
GATE_PASSED
GATE_FAILED
CLOSED
RETIRED
```

## 2. Milestone dashboard

| ID | Work | Planned window | Weight | Status | Evidence / next gate |
|---|---|---|---:|---|---|
| M0 | Approved SSOT and dashboard committed to the remote research branch | Aug 30 | 3% | GATE_PASSED | canonical commit `fe34db9`; exact blobs and cleanup verified |
| M1 | Phase 0–1: substrate freeze, AdamW parity, geometry instrumentation | Aug 31–Sep 6 | 15% | GATE_PASSED | all ten exit items pass in `evidence/m1_gate_audit.json` |
| M2 | Phase 2: Muon optimizer pilot | Sep 7–Sep 13 | 12% | GATE_FAILED | Muon rejected: lower training loss but frozen retention failure |
| M3 | Phase 3: DeltaBall factorial | Sep 14–Sep 27 | 18% | GATE_PASSED | active eps1x retained on small positive named-retention frontier |
| M4 | Phase 4: curriculum and late-tail isolation | Sep 28–Oct 4 | 14% | CLOSED | curriculum mixed/unsupported; tail neutral; six CMA inputs valid |
| M5 | Phase 5: Delta-CMA and rank-matched compression | Oct 5–Oct 11 | 14% | IN_PROGRESS | first uniform control invalidated at −224 tail tokens; exact-tail-plan correction frozen |
| M6 | Phase 6: matched confirmation | Oct 12–Oct 25 | 20% | NOT_STARTED | frozen control/candidate, seed evidence, retention/efficiency decision |
| M7 | One-time sealed promotion gate | Oct 26–Oct 28 | 1% | NOT_STARTED | exactly one authorized sealed result after candidate freeze |
| M8 | Phase 7: rank/family transfer | Oct 29–Nov 15 | 1% | NOT_STARTED | conditional transfer-scope decision |
| M9 | Findings, reproducibility audit, and v1 closeout | Nov 16–Nov 22 | 2% | NOT_STARTED | promote, revise, retire, or authorize v2 |

**Core target:** Phase-6 evidence frozen by October 25, 2026.  
**Positive-path target:** v1 closed by November 22, 2026.  
**Operational buffer:** December 6, 2026. Dates never override evidence gates.

## 3. Compute ledger

| Work package | Planning envelope | Actual GPU-hours | Status |
|---|---:|---:|---|
| M1 parity/smoke | 0–3 | 0.071 measured RGE training-step hours | GATE_PASSED |
| M2 optimizer pilot | 2–6 | 1.681 measured training/eval hours | GATE_FAILED |
| M3 DeltaBall | 4–12 | 1.483 measured training/eval hours | GATE_PASSED |
| M4 curriculum/tail | 4–12 | 1.186 measured training/eval hours | CLOSED |
| M5 CMA validation | 1–6 | 0.235 measured evaluation hours | GATE_PASSED |
| M6 confirmation | 12–36 | 0 | NOT_STARTED |
| **Core v1 target** | **23–69** | **0** | NOT_STARTED |
| Conditional M8 transfer | 12–30 | 0 | NOT_STARTED |

**Review ceiling:** crossing 75 active GPU-hours before M7 requires an approved SSOT amendment.

## 4. Milestone records

### M0 — SSOT preservation

- **Opened:** 2026-08-30
- **Status:** GATE_PASSED
- **Closed:** 2026-08-30 (America/Chicago)
- **Parent:** `main @ 3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`
- **Branch:** `research/rge-peft-v1`
- **Required files:**
  - `docs/PLAN_RGE_PEFT.md`
  - `experiments/rge_peft/PROGRESS.md`
- **Implementation boundary:** documentation only; no optimizer or experiment implementation.
- **Completion condition:** remote files are readable and the introducing commit SHA is recorded below.
- **Canonical commit:** `fe34db9d668e463781750a94b5b3e39d625cceb7`
- **Transfer commits:** `887d35fa309bede6df099ad4a21cd44cd7f84d39`, `145e8169804d66dbd555d9dba9498d198db2e1d5`
- **Remote verification:** GitHub Actions run `33344630813` succeeded; roadmap blob `3345c18e6bf57eac57b3b4fb45f4e130013feda1` and tracker blob `3935726b6c5cae70da271e1e766c350114d49f00` match the approved local Git object IDs; temporary import data and workflow are absent from the final tree.
- **Branch diff at close:** exactly two added files, 1,957 roadmap lines and 161 tracker lines; branch was three commits ahead and zero behind `main`.

### M1 — substrate/parity

- **Opened:** 2026-08-30 23:01:11 -05:00 (America/Chicago)
- **Closed:** 2026-08-30 23:37:53 -05:00 (America/Chicago)
- **Status:** GATE_PASSED
- **Planned window:** 2026-08-31 to 2026-09-06
- **Required evidence:** frozen environment/model/tokenizer/data/eval hashes; sealed-boundary check; AdamW parity; exact resume; geometry side-effect test; instrumentation overhead.
- **Branch/start:** `research/rge-peft-v1 @ e66bcddec8869b7f9fbbc87627bad848cad2f963`
- **Implementation:** RGE AdamW harness; explicit optimizer routing; low-rank effective-delta geometry; format-v2 RGE resume state; recursive training-source fingerprint; strict configs; manifests and JSONL schemas.
- **Tier-0 math:** `\.venv\Scripts\python.exe scripts\verify\verify_delta_geometry.py` — PASS, 68 checks; dense-oracle agreement, gauge invariance at three aspect ratios, no RNG/parameter/gradient mutation, nested addition/modification/removal detection.
- **CPU smoke parity:** `verify_rge_baseline_parity.py --config configs/rge_peft/smoke_adamw.yaml --steps 2` — PASS; legacy/RGE and geometry-off/on bit equality; 112 LoRA tensors, 112 optimizer entries, 56 geometry targets, 112 geometry records.
- **Principal GPU smoke parity:** `verify_rge_baseline_parity.py --config configs/rge_peft/m1_principal_smoke.yaml --steps 2` — PASS on RTX 4070 Laptop; rank 16/alpha 32, BF16 base/FP32 trainables, all q/k/v/o/gate/up/down targets; 392 LoRA tensors and optimizer entries bit-identical; 196 geometry targets; peak reserved 2,708 MiB under the 5,120 MiB cap.
- **Actual B0-plan parity:** `verify_rge_baseline_parity.py --config configs/rge_peft/pilot_adamw.yaml --steps 1 --cache-from artifacts/heal_runs/b0/cache` — PASS; B0 mix SHA `53f87c...c178b`, 128,365 rows, 1,003-step plan hash `ded0ba...65c3`; first bounded principal step and all 392 adapter/optimizer entries bit-identical; peak reserved 1,974 MiB under cap.
- **RGE exact resume:** `verify_rge_resume.py --config configs/rge_peft/smoke_adamw.yaml --steps 3 --break-at 1` — PASS; adapter, generic optimizer, RNG, position, plan, geometry checkpoint state, loss/LR sequence, and geometry artifact equal.
- **RGE GPU exact resume:** `verify_rge_resume.py --config configs/rge_peft/m1_principal_smoke.yaml --steps 2 --break-at 1` — PASS; 392 adapter and optimizer entries, RNG, position, generic optimizer tree, geometry snapshots, loss/LR sequence, and 392 geometry records equal.
- **Legacy regression:** `verify_determinism.py --steps 2`, `verify_resume.py --steps 3 --break-at 1`, and `verify_interrupt.py --steps 4 --signal-after 1` — PASS; 112 LoRA and 336 Adam tensors bit-identical in each applicable comparison.
- **Sealed boundary:** `verify_rge_sealed_boundary.py` — PASS; evaluator refused before record/model loading and created no result file.
- **Instrumentation overhead:** first five-arm run was `INCONCLUSIVE_THERMAL_DRIFT` (disabled references differed +13.7% while an external pytest job started). An identical repeat also showed cross-process drift (-37.9%), so cross-arm timing remains explicitly inconclusive. The paired cadence arm itself measured nine uninstrumented steps and one due step in the same process/trajectory: 0.8681 s nonmeasurement mean, 1.4578 s measurement step, **6.79% amortized cadence-10 overhead**, passing the preferred ≤10% target. Full every-step collection is screened out (>20% in both cross-arm attempts). Peak reserved remained 2,700 MiB in every mode. Selected M1 policy: full metrics every 10 steps.
- **Evidence:** `experiments/rge_peft/evidence/m1_parity_smoke_adamw.json`, `m1_parity_m1_principal_smoke.json`, `m1_parity_pilot_adamw.json`, `m1_resume_smoke_adamw.json`, `m1_resume_m1_principal_smoke.json`, `m1_sealed_boundary.json`, `m1_overhead.json`, `m1_overhead_repeat.json`.
- **Compute so far:** RGE training-step manifests/logs record approximately 0.0576 active GPU-hours; legacy reference active time was not separately instrumented. CPU RGE subprocess manifests recorded at least 75.88 seconds aggregate process wall time (legacy verifier wall time not aggregated here).
- **Sealed data accessed:** no.
- **Gate audit:** all ten required exit items PASS in `experiments/rge_peft/evidence/m1_gate_audit.json`. Final-format artifact-schema verification passes on both actual-B0 and GPU-resumed checkpoints.
- **Next gate:** M2 minimal Muon math/routing/state verification, then bounded effective-step LR calibration; no optimizer pilot until Tier-0 is green.
- **SSOT amendment required:** no.

### M2 — Muon pilot

- **Opened:** 2026-08-30 (America/Chicago)
- **Closed:** 2026-08-31 02:20:03 -05:00 (America/Chicago)
- **Status:** GATE_FAILED
- **Planned window:** 2026-09-07 to 2026-09-13
- **Required evidence:** Muon math/parameter routing; bounded LR calibration; O-A/O-M fixed-token runs; proceed/reject decision.
- **Implementation:** minimal Apache-2.0 FP32 matrix-only Muon; explicit no-fallback routing; matched sequential pair launcher; fixed-token/post-run comparability gates.
- **CPU math/routing:** `verify_rge_muon.py` — PASS, 40 checks including pinned dense reference, rank-1 and 16x1024/3072 aspects, finite refusal, FP32 state, frozen-base protection, bit-exact state continuation, deterministic no-RNG trajectory, and AdamW factory identity.
- **Pair protocol:** `verify_rge_optimizer_pair.py` — PASS, 20 checks including substrate/budget/sealed/claim negative controls, post-run matching, M1 authorization, and full-pair refusal while calibration is provisional.
- **Execution isolation:** fresh selected-arm directories are mandatory for `--resume none`; pair output cannot overlap either run directory; a single-arm diagnostic cannot discover or combine an older opposite-arm summary.
- **Isolation evidence:** `m2_pair_isolation_cpu.json` — PASS, 6 CPU controls; sealed data untouched.
- **Import/provenance:** owner payload archive and 19 file hashes verified locally after hosted Actions run `33358150973` was blocked by billing; see `evidence/m2_import_audit.json`. Pinned NVIDIA upstream and Apache-2.0 license independently verified.
- **Execution status:** effective-step calibration is frozen; the corrected 104,371-requested-token exploratory pair is authorized.
- **GPU pair smoke:** two sequential B0 steps per arm — PASS. Both arms consumed exactly 2,164 supervised tokens with identical source/model/data/plan/initial-state hashes. Muon remained finite; bounded diagnostics were loss delta +0.00704, time ratio 1.0902, throughput ratio 0.9173, optimizer-state byte ratio 0.49999, and peak-reserved delta -38 MiB. These are integration diagnostics only, not optimizer selection evidence.
- **Muon GPU resume:** `verify_rge_resume.py --config configs/rge_peft/m2_muon_smoke.yaml --steps 2 --break-at 1` — PASS; 392 adapter tensors, 392 optimizer states, RNG, geometry state, loss/LR sequence, and 392 geometry records bit-identical.
- **Compute so far:** approximately 0.019 measured M2 optimizer-step GPU-hours; model-load/probe engineering time excluded by the run-manifest definition.
- **Initial LR/effective-step bracket:** stable but `BRACKET_INSUFFICIENT`. Muon raw LR `5e-5`, `1e-4`, `2e-4` produced median paired `rho_delta` ratios `0.118`, `0.236`, `0.472` over 588 values each. Only the 0.5x regime passed factor-1.5 coverage; no LR was frozen and loss/evaluation did not participate.
- **Calibration compute:** 0.081 measured optimizer-step GPU-hours; cumulative M2 measured optimizer-step time is approximately 0.100 GPU-hours.
- **Bounded expansion:** preregistered before execution at `{2e-4, 4e-4, 8e-4}` with identical four-step B0 protocol and selection rule.
- **Expanded bracket:** PASS. Median paired ratios were `0.472`, `0.944`, `1.889` over 588 values each, identifying 0.5x/1x/2x regimes at `2e-4`/`4e-4`/`8e-4`. The preregistered rule froze Muon LR `4e-4`; loss/evaluation/sealed data were not used.
- **Calibration compute:** expanded bracket used 0.0819 measured optimizer-step GPU-hours; cumulative M2 measured optimizer-step time is approximately 0.182 GPU-hours.
- **Sealed data accessed:** no.
- **Pilot-budget correction:** the staged 1,000,000-token value was inconsistent with the SSOT's 10% rule. Full B0 is 1,043,713 supervised tokens, so amendment v1.2 freezes the M2 request at 104,371 before any pair outcome. Smoke/calibration loss was already visible, but no pair endpoint or retention result existed and the correction is formula-driven.
- **Registered pair training:** COMPLETE after exact resumes at AdamW step 60 and Muon step 30. Both arms ended at step 101 and exactly 104,571 effective supervised tokens. AdamW weighted/final loss: `1.95586` / `0.94659`; Muon: `1.80056` / `0.58329`; deltas `-0.15530` / `-0.36331`. Muon had lower loss on 75/101 paired points, tied 1, and higher on 25.
- **External contention:** user-owned long-context inference jobs repeatedly overlapped both arms. Training tensors/loss remain valid, but all time/throughput/GPU-hour efficiency comparisons from this cohort are `INVALID_EXTERNAL_GPU_CONTENTION`.
- **Resume-artifact negative:** the first pair audit failed because an uncheckpointed Muon step-31 log row survived a forced pause at checkpoint 30 and was logged again on replay. Checkpoint/model/optimizer tensors and final positions matched. The log was reconciled to the checkpoint (one row removed), active time recomputed, and comparability then passed. Future resume now performs this reconciliation automatically; injected-tail regression passes.
- **Memory/state:** Muon optimizer-state tensor bytes are 0.49999x AdamW; Muon peak reserved is +434 MiB in repaired summaries. Cumulative measured M2 optimizer-step time is 1.435 GPU-hours, but efficiency interpretation is prohibited by contention.
- **Scientific interpretation:** exploratory positive evidence for training-objective improvement per supervised token only. Retention/capability improvement is not established.
- **Frozen retention:** Muon `chat_dev` delta `-0.01524` with paired 95% CI `[-0.02857, -0.00195]`, failing the `-0.01` guardrail (74/312/124 win/tie/loss). `rag_dev` delta `-0.00542` with CI `[-0.01120, -0.00045]` passes the mean margin but adds four critical decoding-corruption cases, lowers intelligibility, and doubles contract violations.
- **M2 decision:** `GATE_FAILED`; Muon is `REJECTED` for the general Rineya v1 recipe. Do not tune Muon on retention. Lower training loss is retained only as optimizer-dynamics evidence.
- **M2 compute:** approximately 1.681 measured GPU-hours including 0.247 hours of frozen retention evaluation; model-load/probe time is not fully instrumented.
- **Next bounded action:** begin M3 DeltaBall under AdamW only, as H3 is logically separable from the rejected Muon transfer.

### M3 — DeltaBall factorial

- **Opened:** 2026-08-31 (America/Chicago)
- **Closed:** 2026-08-31 (America/Chicago)
- **Status:** GATE_PASSED
- **Planned window:** 2026-09-14 to 2026-09-27
- **Required evidence:** projection verification; bounded radius screen; G-A/G-M/G-AD/G-MD results; boundary-hit diagnostics; recipe freeze or retirement.
- **M2 dependency decision:** Muon is rejected. M3 proceeds only with G-A AdamW control and G-AD radius arms, as allowed by the SSOT's separability rule.
- **Radius derivation:** AdamW step-100 geometry across 196 modules gives `d_ref = p95(relative_delta) = 0.01435526856`. Frozen candidates are `0.5x/1x/2x`: `0.00717763428`, `0.01435526856`, `0.02871053712`, with 40-step linear warm-in.
- **Selection rule:** stable post-warm-in matrix-hit rate must be 1%–50%; among eligible arms choose lowest token-weighted loss, exact ties choose larger radius. Retention/evaluation does not select the screen radius.
- **Mechanism verification:** `verify_deltaball_projection.py` — PASS, 9 checks: inside no-op, exact boundary, symmetric `sqrt(c)`, dense product scale, frozen base, linear schedule, gauge equivalence, exact counter restore, no RNG.
- **GPU integration/resume:** PASS at rank 16/all targets. Straight versus resumed trajectories match across 392 adapter/optimizer entries, RNG, geometry state, projection counters, 588 geometry records, 588 projection records, and an injected stale train-log tail. Format-v2 artifact schema passes.
- **Radius screen:** complete. Control weighted/final loss `1.95586/0.94659`. Epsilon 0.5x: 37.14% post-warm-in matrix hits, 83.61% hit steps, `1.98109/1.04580` loss (overconstrained and worse). Epsilon 1x: 1.054% matrix hits, 34.43% hit steps, `1.95628/0.95234` (eligible, near-neutral/slightly worse). Epsilon 2x: zero hits through fully warmed checkpoint 40 and stopped early as inactive.
- **Selected radius:** epsilon `0.014355268562212586` (1x `d_ref`) by the preregistered best-eligible weighted-loss rule. Selection used no retention/evaluation result.
- **Control reuse proof:** the M3 G-A checkpoint is bit-identical across all 392 adapter tensors to the already evaluated M2 AdamW control, with identical position and plan. Its frozen retention results may be reused without regeneration.
- **M3 compute so far:** approximately 1.36 measured optimizer-step GPU-hours for the screen; timing is exploratory single-order only.
- **Sealed data accessed:** no.
- **Frozen retention:** selected eps1x `chat_dev` delta `+0.001493` (CI `[-0.004195, +0.007363]`, 19/476/15 W/T/L); `rag_dev` `+0.000806` (CI `[0, +0.002419]`, 1/751/0), with no critical regression and control contract violations removed.
- **M3 decision:** `GATE_PASSED` as exploratory mechanism/named-retention evidence. Freeze AdamW + DeltaBall epsilon `0.014355268562212586` for M4. No targeted capability, multi-seed, or speed claim is established.
- **M3 compute:** approximately 1.483 measured GPU-hours including selected endpoint evaluation.
- **Next bounded action:** build deterministic same-membership curriculum manifests and run M4 under the frozen AdamW + eps1x DeltaBall recipe.

### M4 — curriculum/tail

- **Opened:** 2026-08-31 (America/Chicago)
- **Closed:** 2026-08-31 (America/Chicago)
- **Status:** CLOSED
- **Planned window:** 2026-09-28 to 2026-10-04
- **Required evidence:** deterministic same-membership order hashes; C-U/C-C/C-CT/C-CTM results; ingredient decision.
- **Frozen recipe entering M4:** AdamW + DeltaBall epsilon `0.014355268562212586`; Muon excluded.
- **Metadata audit:** 120,000 rows carry explicit `basic/intermediate/compositional` difficulty and receive component-local ranking. 3,955 preservation rows lack difficulty and 4,410 RAG rows lack a comparable score; both receive deterministic fixed-random replay rather than inferred ranking.
- **Full order manifest:** 128,365 rows, 78 components, 10 aligned buckets, membership hash `688110...8820`, curriculum order `da146c...7607`.
- **Exact pilot subset:** 12,875 rows selected before outcomes by `sha256(seed|m4_subset|source_id) mod 10 == 0`; both arms consume the complete identical subset. Membership hash `8365fe...8e28`; uniform/curriculum order hashes `0d7151...9575` / `7e8814...350f`.
- **Schedule fairness:** amendment v1.3 freezes LR and DeltaBall warm-in by supervised tokens (41,383 warmup tokens), preventing order-dependent packing/step counts from changing schedule-versus-token semantics.
- **Verification:** `verify_rge_curriculum.py` PASS, 10 membership/hash/planner checks.
- **C-U/C-C training:** PASS comparability. Both consumed exactly 12,875 rows and 105,402 supervised tokens in 101 steps, with identical initial-state hash and verified token-indexed LR schedule. Uniform weighted/final loss `2.012459/1.224880`; curriculum `2.011093/1.370037`.
- **Order selection:** component-local selected by preregistered token-weighted loss delta `-0.001366`; its final-step loss is worse by `+0.145157`, so retention evaluation is required before keeping it.
- **Interruption:** disk exhaustion stopped C-C after uncheckpointed step 46. Regenerable calibration/smoke/intermediate checkpoints were pruned (25.98 GiB freed), checkpoint 40 resumed exactly, and six stale log rows were reconciled automatically. No model tensor was repaired.
- **M4 compute so far:** 0.798 measured optimizer-step GPU-hours for C-U/C-C training; timing is exploratory single-order only.
- **Sealed data accessed:** no.
- **Frozen retention:** curriculum versus uniform `chat_dev -0.003678` (CI `[-0.01375, +0.00624]`, 63/385/62 W/T/L); `rag_dev +0.000810` (CI `[-0.00242, +0.00484]`, 3/747/2), with no critical regression and slightly fewer contract violations. Both pass non-inferiority; neither CI establishes superiority.
- **Curriculum decision:** retain component-local order for the constant-tail experiment only, labeled weak/mixed exploratory evidence rather than a confirmed ordering gain.
- **M4 compute:** approximately 1.03 measured training/evaluation GPU-hours so far.
- **Constant-tail freeze:** 12 steps at LR `1e-5`; checkpoints `{2,4,6,8,10,12}`; retain adapter/AdamW/RNG/geometry state and selected curriculum membership/order; eps1x DeltaBall begins fully warmed; reset only tail plan position and projection counters. No averaging in M4.
- **Constant-tail run:** PASS. 12 steps / 12,713 supervised tokens at LR `1e-5`; six checkpoints saved exactly at `{2,4,6,8,10,12}`; tail weighted/final loss `1.32867/1.26110`; 0 DeltaBall hits; 0.0427 active GPU-hours; no averaging performed.
- **Tail retention:** versus pre-tail C-C, `chat_dev -0.000884` (CI crosses zero, 26/455/29 W/T/L); `rag_dev` exactly identical on 752/752 cases; no critical or contract regression.
- **M4 decision:** phase `CLOSED`. Component-local order has mixed/unsupported standalone benefit; constant tail is endpoint-neutral. The six tail checkpoints remain valid inputs for the distinct M5 CMA hypothesis.
- **M4 compute:** approximately 1.186 measured GPU-hours including evaluation.
- **Next bounded action:** implement and verify exact Delta-space averaging plus rank-16 QR/SVD recompression over tail steps `{2,4,6,8,10,12}`.

### M5 — Delta-CMA

- **Opened:** 2026-08-31 (America/Chicago)
- **Closed:** 2026-08-31 (America/Chicago)
- **Status:** GATE_PASSED
- **Planned window:** 2026-10-05 to 2026-10-11
- **Required evidence:** gauge-invariant averaging; QR/SVD recompression; retained energy; PEFT export/reload; endpoint decision.
- **Inputs:** frozen constant-tail checkpoints `{2,4,6,8,10,12}`, equal weights, source rank 16/scale 2.
- **Math verification:** `verify_cma_math.py` PASS, 7 checks covering exact dense average, gauge invariance, QR/core-SVD, rank/scale, energy/error, factor-averaging negative control, and rectangular aspects.
- **Real construction:** PASS from tail steps `{2,4,6,8,10,12}`. Diagnostic exact rank 96 and primary rank 16 adapters contain 196 target modules. Rank 16 retains `0.999999999678495` global delta energy; global relative reconstruction error `1.793e-5`; worst module `2.874e-5`.
- **Loader verification:** both adapters load through normal PEFT, produce finite logits, and merge/unload successfully on GPU. Exact config is r96/alpha96; matched config r16/alpha32.
- **Sealed data accessed:** no.
- **Endpoint results:** CMA-exact versus tail: chat `+0.001282`, RAG exactly tied. CMA-r16 versus tail: chat `+0.000873`, RAG exactly tied, no critical regression. CMA-r16 versus exact: chat `-0.000409`, RAG tied; compression effect is tiny/non-material.
- **M5 decision:** `GATE_PASSED`; freeze rank-16 CMA as M6 candidate. This is a tiny exploratory named-retention gain only—no targeted capability, speed, seed, or sealed claim.
- **Matching correction:** M6 audit found the uniform endpoint lacked the candidate's 12 tail steps/CMA. Amendment v1.5 reopens M5 for the optional C-UTM control before capability evaluation; prior CMA candidate selection remains frozen.
- **Capability bank freeze:** 4,080 balanced validation puzzles across 15 families; SHA `f717d9...3b95`; zero exact B0 matches, zero ≥0.8 approximate matches, maximum candidate bigram Jaccard `0.0667`. Test/challenge remain untouched.
- **M5 compute:** approximately 0.235 evaluation GPU-hours; construction/math primarily CPU.
- **First interaction-control audit:** INVALID. The uniform tail used its own shuffled tail plan and consumed 12,489 supervised tokens/1,535 rows, versus 12,713/1,536 for the candidate. The resulting CMA/retention comparison is superseded and scientifically ineligible despite passing pointwise retention gates.
- **Corrective preregistration:** initialize from the unchanged 105,402-token uniform main endpoint, then reuse the candidate's exact tail mix/plan hash, 12 batches, 12,713 tokens, checkpoint steps, LR, DeltaBall, and CMA weights. This restores the v1.5 contract so only the main-segment order differs.
- **Capability implementation:** deterministic exact-match runner and paired audit pass 14 CPU checks. A 16-row control-only diagnostic verified GPU integration; it is not eligible evidence, no candidate output was observed, and the full bank remains unopened.
- **Next bounded action:** construct the corrected control tail/CMA, prove post-run matching, repeat retention, and only then evaluate the full frozen capability bank.

### M6 — confirmation

- **Status:** NOT_STARTED
- **Planned window:** 2026-10-12 to 2026-10-25
- **Required evidence:** frozen control/candidate; matched full-budget run(s); seed plan; capability, retention, and GPU-hour comparison.

### M7 — sealed gate

- **Status:** NOT_STARTED
- **Planned window:** 2026-10-26 to 2026-10-28
- **Prerequisite:** M6 passed and candidate/config/checkpoint frozen before sealed access.

### M8 — transfer

- **Status:** NOT_STARTED
- **Planned window:** 2026-10-29 to 2026-11-15
- **Prerequisite:** positive or mechanistically informative core result.

### M9 — closeout

- **Status:** NOT_STARTED
- **Planned window:** 2026-11-16 to 2026-11-22
- **Required evidence:** findings, claims, negative results, reproduction commands, final program decision.

## 5. Decision log

| Date | Milestone | Decision | Evidence | Commit/run references |
|---|---|---|---|---|
| 2026-08-30 | M0 | Project owner approved the SSOT; v1.1 adds execution governance without changing the scientific design | approval recorded before implementation/results | canonical `fe34db9` |
| 2026-08-30 | M0 | Close M0 and advance the operational focus to M1 without starting implementation | exact Git blobs, successful hash gate, clean two-file diff | Actions `33344630813`; `fe34db9` |
| 2026-08-30 | M1 | Open implementation after re-verifying remote/current SHAs and frozen SSOT | Tier-0 math, CPU parity/resume, legacy regression, sealed refusal all pass; GPU gates still open | pending implementation commit; evidence under `experiments/rge_peft/evidence/` |
| 2026-08-30 | M1 | Close M1 as GATE_PASSED and authorize M2 engineering | ten-item gate audit PASS; final training source `840b04e`; no sealed access | gate record `a59a340`; `experiments/rge_peft/evidence/m1_gate_audit.json` |
| 2026-08-31 | M2 | Reject Muon for the general Rineya recipe and advance DeltaBall only under AdamW | training loss improved, but chat non-inferiority failed and RAG gained four critical defects | gate record `8f69c9d`; `experiments/rge_peft/evidence/m2_gate_audit.json` |
| 2026-08-31 | M3 | Retain AdamW + active eps1x DeltaBall for M4 | small positive/non-inferior chat and RAG retention with no critical regression; targeted capability untested | gate record `b689530`; `experiments/rge_peft/evidence/m3_gate_audit.json` |
| 2026-08-31 | M4 | Close curriculum/tail isolation without a standalone benefit; authorize CMA construction | curriculum mixed, tail neutral, exact six-checkpoint trajectory available | gate record `888d3e6`; `experiments/rge_peft/evidence/m4_gate_audit.json` |

## 6. Blocker log

| Opened | Closed | Milestone | Blocker | Effect | Resolution |
|---|---|---|---|---|---|
| 2026-08-30 | 2026-08-30 | M0 | First transfer run detected a CRC mismatch in one staging chunk | canonical files were not committed | split the affected chunk into four content-addressed pieces; rerun passed all hashes |

## 7. Weekly closeout template

Append one entry each Sunday in America/Chicago:

```markdown
### YYYY-MM-DD weekly closeout

- Current milestone/status:
- Commits added:
- Runs completed or interrupted:
- Supervised tokens this week / cumulative:
- Active GPU-hours this week / cumulative:
- Evidence obtained:
- Open blockers:
- Next bounded action or gate decision:
- Sealed data accessed: no/yes (authorization reference required)
- SSOT amendment required: yes/no
```

### 2026-08-30 weekly closeout

- Current milestone/status: M1 GATE_PASSED; M2 NOT_STARTED.
- Commits added: `13ee3cd`, `a860afc`, `0033d91`, `8db1e95`, `fe632a6`, `840b04e`, gate record `a59a340`, and merge `0bd2614` preserving concurrent M2 staging.
- Runs completed or interrupted: Tier-0 geometry suite; CPU parity/resume; legacy determinism/resume/interrupt; principal GPU parity/resume; actual B0 parity; two overhead attempts; substrate dry-run. No interrupted RGE run.
- Supervised tokens this week / cumulative: engineering smoke only; actual-B0 parity consumed one 1,121-supervised-token step per arm. No M2 pilot or confirmatory token budget consumed.
- Active GPU-hours this week / cumulative: approximately 0.071 measured RGE optimizer-step hours; model-load/probe engineering time and legacy reference active time are not separately instrumented.
- Evidence obtained: all ten M1 gate items pass; see `experiments/rge_peft/evidence/m1_gate_audit.json`.
- Open blockers: none for M1. M2 has not started.
- Next bounded action or gate decision: implement minimal attributed Muon and pass synthetic math/routing/state tests.
- Sealed data accessed: no.
- SSOT amendment required: no.
