# RGE-PEFT Claims Ledger

Allowed statuses are `PLANNED`, `IMPLEMENTED_UNVERIFIED`, `VERIFIED_MECHANISM`,
`EXPLORATORY_POSITIVE`, `EXPLORATORY_NEGATIVE`, `CONFIRMED_POSITIVE`,
`CONFIRMED_NEGATIVE`, `REJECTED`, and `SUPERSEDED`.

| Claim | Status | Required evidence | Supporting runs | Falsifiers |
|---|---|---|---|---|
| RGE AdamW reproduces the legacy trainer | VERIFIED_MECHANISM | bit-exact legacy/RGE parity on smoke and rank-16 principal substrate | CPU smoke, rank-16 GPU smoke, and actual B0-plan first-step parity evidence | any unexplained tensor, optimizer, plan, loss, or LR divergence |
| Effective-delta logging is observational | VERIFIED_MECHANISM | geometry-off/on bit equality plus RNG/parameter/gradient unit checks | 68-check Tier-0 suite; CPU and rank-16 GPU geometry-off/on parity; GPU exact resume | any trajectory or RNG divergence |
| The registered FP32 Muon route has deterministic, checkpointable matrix-update mechanics without altering the AdamW control path | VERIFIED_MECHANISM | independent math reference, routing rejection, state-dict continuation, no-RNG proof, and AdamW identity | `m2_muon_unit_cpu.json` | mismatch against pinned math, resume divergence, hidden fallback, or AdamW drift |
| The M2 launcher enforces a matched fixed-token O-A/O-M experiment and fails closed at the M1 boundary | VERIFIED_MECHANISM | negative controls for data/capacity/schedule/budget/sealed/claim drift plus post-run hash matcher | `m2_pair_protocol_cpu.json` | any mismatched pair launches, sealed asset is accepted, or blocked gate loads a model |
| Muon improves LoRA token efficiency | REJECTED | frozen matched O-A/O-M evidence after M1 | Training objective improved, but `m2_retention_comparison.json` shows chat −0.01524 and four new RAG critical defects; timing invalid | observed: frozen retention failure despite lower loss |
| DeltaBall improves retention | PLANNED | active-constraint G-* factorial evidence | — | inactive constraint or worse matched frontier |
| Curriculum ordering improves the endpoint | PLANNED | C-U/C-C with identical membership and token budget | — | no gain or regression |
| Rank-matched Delta-CMA improves the endpoint | PLANNED | C-CT/C-CTM plus compression diagnostics | — | gain only in higher-rank exact average |
