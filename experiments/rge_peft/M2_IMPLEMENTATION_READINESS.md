# M2 Optimizer-Pilot Implementation Readiness

**Status:** IMPLEMENTATION_READY / BOUNDED SMOKE AUTHORIZED / FULL PAIR BLOCKED BY LR CALIBRATION
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
- The full 1M-token pair remains blocked until effective-step LR calibration is
  frozen as required by the SSOT.
