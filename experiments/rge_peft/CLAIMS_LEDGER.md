# RGE-PEFT Claims Ledger

Allowed statuses are `PLANNED`, `IMPLEMENTED_UNVERIFIED`, `VERIFIED_MECHANISM`,
`EXPLORATORY_POSITIVE`, `EXPLORATORY_NEGATIVE`, `CONFIRMED_POSITIVE`,
`CONFIRMED_NEGATIVE`, `REJECTED`, and `SUPERSEDED`.

| Claim | Status | Required evidence | Supporting runs | Falsifiers |
|---|---|---|---|---|
| RGE AdamW reproduces the legacy trainer | VERIFIED_MECHANISM | bit-exact legacy/RGE parity on smoke and rank-16 principal substrate | CPU smoke, rank-16 GPU smoke, and actual B0-plan first-step parity evidence | any unexplained tensor, optimizer, plan, loss, or LR divergence |
| Effective-delta logging is observational | VERIFIED_MECHANISM | geometry-off/on bit equality plus RNG/parameter/gradient unit checks | 68-check Tier-0 suite; CPU and rank-16 GPU geometry-off/on parity; GPU exact resume | any trajectory or RNG divergence |
| Muon improves LoRA token efficiency | PLANNED | frozen matched O-A/O-M confirmation after M1 | — | equal or worse frozen capability |
| DeltaBall improves retention | PLANNED | active-constraint G-* factorial evidence | — | inactive constraint or worse matched frontier |
| Curriculum ordering improves the endpoint | PLANNED | C-U/C-C with identical membership and token budget | — | no gain or regression |
| Rank-matched Delta-CMA improves the endpoint | PLANNED | C-CT/C-CTM plus compression diagnostics | — | gain only in higher-rank exact average |
