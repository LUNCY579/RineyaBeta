# M2 Optimizer-Pilot Implementation Readiness

**Status:** IMPLEMENTATION_READY / REGISTERED EXPLORATORY PAIR AUTHORIZED
**Date:** 2026-08-30 (America/Chicago)
**Sealed data accessed:** no

## Delivered

- Self-contained FP32 Muon optimizer route for two-dimensional LoRA factors.
- Explicit optimizer contract and strict rejection of unsupported trainables.
- Byte-identical preservation check for the AdamW factory path.
- Registered, fixed-token O-A versus O-M configs and pair manifest.
- Sequential consumer-GPU pair launcher with strict preflight, disposable
  smoke run roots, and reversible arm order.
- Post-run scientific comparability gate using model/data/plan/initial-state
  hashes and exact effective supervised-token counts.
- Loss/token/time/memory/optimizer-state evidence aggregation, including
  environment provenance and steady-state timing distributions.
- Optional frozen endpoint evaluation on `chat_dev` and `rag_dev`.
- Fail-closed sealed-set and claim-boundary enforcement.
- Apache-2.0 third-party notice and license copy.

## CPU verification

| Evidence | Result |
|---|---|
| `m2_muon_unit_cpu.json` | PASS — 40 checks |
| `m2_pair_protocol_cpu.json` | PASS — 20 checks |
| `m2_pair_isolation_cpu.json` | PASS — 6 checks |
| pair validate-only preflight | PASS — zero protocol problems |
| Python compilation | PASS |

The proofs cover Newton–Schulz math properties, rank-1 and rank-16 rectangular
aspects, finite-value refusal, scale formulas, FP32 state, state-dict continuity,
deterministic/no-RNG behavior, frozen-base protection, parameter routing, AdamW
control identity, negative-control rejection, post-run hash matching, blocked
execution, and provisional-calibration fail-closed behavior.

## Not yet claimed

- Two-step B0 GPU integration and rank-16 Muon exact resume have passed; these
  are mechanism smoke evidence only.
- No O-A/O-M quality or speed result exists yet.
- No capability/OOD bank is frozen.
- No general-intelligence, broad-generalization, or confirmatory consumer-speedup
  claim is authorized.
- M1 is `GATE_PASSED`; bounded GPU integration smoke is authorized.
- Effective-step calibration is frozen at Muon LR `4e-4` (median AdamW-relative
  ratio `0.944`); the registered 104,371-requested-token exploratory pair is
  authorized under SSOT amendment v1.2.

## Execution-cohort hardening

The launcher summarizes only arms selected by the current invocation, rejects
non-empty selected-arm directories under `--resume none`, and refuses overlap
between pair-output and run directories. This prevents a one-arm diagnostic from
silently combining with an older opposite-arm summary. Six CPU negative controls
are committed in `evidence/m2_pair_isolation_cpu.json`. This changes no optimizer
math and authorizes no full M2 result.
