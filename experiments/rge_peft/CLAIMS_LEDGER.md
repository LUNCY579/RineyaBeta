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
| DeltaBall projection mechanics are correct | VERIFIED_MECHANISM | dense-product boundary, gauge, schedule, state, and RNG proofs | `m3_deltaball_unit_cpu.json` (9 checks) | product misses boundary, base mutation, or resume divergence |
| DeltaBall improves retention | EXPLORATORY_POSITIVE | active-constraint AdamW G-A/G-AD evidence after radius screen | eps1x: 1.054% matrix hits; chat +0.00149, rag +0.000806, no critical regression | later capability/seed failure or loss of gain under matched confirmation |
| Curriculum ordering improves the endpoint | EXPLORATORY_NEGATIVE | C-U/C-C with identical membership and exact tokens | weighted loss −0.00137, but final loss +0.145; chat −0.00368, RAG +0.00081; CIs cross zero | observed: no consistent endpoint gain |
| Constant tail improves the endpoint | EXPLORATORY_NEGATIVE | C-C versus C-CT frozen retention | chat −0.000884, RAG exactly tied on 752/752 cases | observed: no standalone endpoint gain; tail retained only for CMA |
| Rank-matched Delta-CMA improves the endpoint | EXPLORATORY_POSITIVE | C-CT/C-CTM plus compression diagnostics | rank16 vs tail: chat +0.000873, RAG tied, no critical regression; 0.99999999968 energy retained | multi-seed/capability failure or neutral confirmation |
| Delta-CMA construction is gauge-correct and rank-controlled | VERIFIED_MECHANISM | dense, gauge, QR/SVD, scaling, and raw-factor negative controls | `m5_cma_unit_cpu.json` (7 checks) | dense mismatch, loader failure, or rank/scale drift |
