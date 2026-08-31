# RGE-PEFT Live Progress Dashboard

**Governing SSOT:** `docs/PLAN_RGE_PEFT.md`  
**Program version:** v1.1  
**Target branch:** `research/rge-peft-v1`  
**Starting parent:** `main @ 3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Initialized:** 2026-08-30, America/Chicago  
**Current milestone:** M0 — SSOT preservation  
**Current status:** IN_PROGRESS until the remote commit is verified  
**Implementation started:** no  
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
| M0 | Approved SSOT and dashboard committed to the remote research branch | Aug 30 | 3% | IN_PROGRESS | verify remote branch, files, and introducing commit |
| M1 | Phase 0–1: substrate freeze, AdamW parity, geometry instrumentation | Aug 31–Sep 6 | 15% | NOT_STARTED | `GATE_PASSED` only after parity/resume/overhead evidence |
| M2 | Phase 2: Muon optimizer pilot | Sep 7–Sep 13 | 12% | NOT_STARTED | O-A/O-M fixed-token evidence and optimizer decision |
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
| M1 parity/smoke | 0–3 | 0 | NOT_STARTED |
| M2 optimizer pilot | 2–6 | 0 | NOT_STARTED |
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
- **Status:** IN_PROGRESS
- **Parent:** `main @ 3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`
- **Branch:** `research/rge-peft-v1`
- **Required files:**
  - `docs/PLAN_RGE_PEFT.md`
  - `experiments/rge_peft/PROGRESS.md`
- **Implementation boundary:** documentation only; no optimizer or experiment implementation.
- **Completion condition:** remote files are readable and the introducing commit SHA is recorded below.
- **Introducing commit:** pending
- **Remote verification:** pending

### M1 — substrate/parity

- **Status:** NOT_STARTED
- **Planned window:** 2026-08-31 to 2026-09-06
- **Required evidence:** frozen environment/model/tokenizer/data/eval hashes; sealed-boundary check; AdamW parity; exact resume; geometry side-effect test; instrumentation overhead.

### M2 — Muon pilot

- **Status:** NOT_STARTED
- **Planned window:** 2026-09-07 to 2026-09-13
- **Required evidence:** Muon math/parameter routing; bounded LR calibration; O-A/O-M fixed-token runs; proceed/reject decision.

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
| 2026-08-30 | M0 | Project owner approved the SSOT; v1.1 adds execution governance without changing the scientific design | approval recorded before implementation/results | introducing commit pending |

## 6. Blocker log

| Opened | Closed | Milestone | Blocker | Effect | Resolution |
|---|---|---|---|---|---|
| — | — | — | none at initialization | — | — |

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
