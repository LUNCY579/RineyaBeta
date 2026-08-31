# RGE-PEFT Live Progress Dashboard

**Governing SSOT:** `docs/PLAN_RGE_PEFT.md`  
**Program version:** v1.1  
**Target branch:** `research/rge-peft-v1`  
**Starting parent:** `main @ 3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Initialized:** 2026-08-30, America/Chicago  
**Current milestone:** M2 — Muon transfer pilot
**Current status:** M0 GATE_PASSED; M1 GATE_PASSED; M2 IN_PROGRESS
**Implementation started:** yes
**Experimental outcomes observed:** none  
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
| M2 | Phase 2: Muon optimizer pilot | Sep 7–Sep 13 | 12% | IN_PROGRESS | 40 Muon and 20 pair-protocol CPU checks pass; GPU smoke/calibration pending |
| M3 | Phase 3: DeltaBall factorial | Sep 14–Sep 27 | 18% | NOT_STARTED | G-* evidence and frozen geometry recipe or retirement |
| M4 | Phase 4: curriculum and late-tail isolation | Sep 28–Oct 4 | 14% | NOT_STARTED | C-* evidence and ingredient decision |
| M5 | Phase 5: Delta-CMA and rank-matched compression | Oct 5–Oct 11 | 14% | NOT_STARTED | CMA math/export proof and endpoint decision |
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
| M2 optimizer pilot | 2–6 | 0.019 measured training-step hours | IN_PROGRESS |
| M3 DeltaBall | 4–12 | 0 | NOT_STARTED |
| M4 curriculum/tail | 4–12 | 0 | NOT_STARTED |
| M5 CMA validation | 1–6 | 0 | NOT_STARTED |
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
- **Status:** IN_PROGRESS
- **Planned window:** 2026-09-07 to 2026-09-13
- **Required evidence:** Muon math/parameter routing; bounded LR calibration; O-A/O-M fixed-token runs; proceed/reject decision.
- **Implementation:** minimal Apache-2.0 FP32 matrix-only Muon; explicit no-fallback routing; matched sequential pair launcher; fixed-token/post-run comparability gates.
- **CPU math/routing:** `verify_rge_muon.py` — PASS, 40 checks including pinned dense reference, rank-1 and 16x1024/3072 aspects, finite refusal, FP32 state, frozen-base protection, bit-exact state continuation, deterministic no-RNG trajectory, and AdamW factory identity.
- **Pair protocol:** `verify_rge_optimizer_pair.py` — PASS, 20 checks including substrate/budget/sealed/claim negative controls, post-run matching, M1 authorization, and full-pair refusal while calibration is provisional.
- **Execution isolation:** fresh selected-arm directories are mandatory for `--resume none`; pair output cannot overlap either run directory; a single-arm diagnostic cannot discover or combine an older opposite-arm summary.
- **Isolation evidence:** `m2_pair_isolation_cpu.json` — PASS, 6 CPU controls; sealed data untouched.
- **Import/provenance:** owner payload archive and 19 file hashes verified locally after hosted Actions run `33358150973` was blocked by billing; see `evidence/m2_import_audit.json`. Pinned NVIDIA upstream and Apache-2.0 license independently verified.
- **Execution status:** bounded GPU smoke authorized; full 1M-token pair blocked until effective-step LR bracket is frozen.
- **GPU pair smoke:** two sequential B0 steps per arm — PASS. Both arms consumed exactly 2,164 supervised tokens with identical source/model/data/plan/initial-state hashes. Muon remained finite; bounded diagnostics were loss delta +0.00704, time ratio 1.0902, throughput ratio 0.9173, optimizer-state byte ratio 0.49999, and peak-reserved delta -38 MiB. These are integration diagnostics only, not optimizer selection evidence.
- **Muon GPU resume:** `verify_rge_resume.py --config configs/rge_peft/m2_muon_smoke.yaml --steps 2 --break-at 1` — PASS; 392 adapter tensors, 392 optimizer states, RNG, geometry state, loss/LR sequence, and 392 geometry records bit-identical.
- **Compute so far:** approximately 0.019 measured M2 optimizer-step GPU-hours; model-load/probe engineering time excluded by the run-manifest definition.
- **Sealed data accessed:** no.
- **Next bounded action:** pre-register and execute the short AdamW reference plus log-spaced Muon effective-step LR bracket; keep the full 1M-token pair blocked.

### M3 — DeltaBall factorial

- **Status:** NOT_STARTED
- **Planned window:** 2026-09-14 to 2026-09-27
- **Required evidence:** projection verification; bounded radius screen; G-A/G-M/G-AD/G-MD results; boundary-hit diagnostics; recipe freeze or retirement.

### M4 — curriculum/tail

- **Status:** NOT_STARTED
- **Planned window:** 2026-09-28 to 2026-10-04
- **Required evidence:** deterministic same-membership order hashes; C-U/C-C/C-CT/C-CTM results; ingredient decision.

### M5 — Delta-CMA

- **Status:** NOT_STARTED
- **Planned window:** 2026-10-05 to 2026-10-11
- **Required evidence:** gauge-invariant averaging; QR/SVD recompression; retained energy; PEFT export/reload; endpoint decision.

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
