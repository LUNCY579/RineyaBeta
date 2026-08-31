# RGE-PEFT v1 findings

## Outcome

RGE-PEFT v1 is **retired without promotion**. The final frozen curriculum-order
candidate passed the rank/capacity/token and point-retention controls, but it was
significantly worse than the fully matched uniform-order control on the primary
targeted-capability validation bank. Multi-seed escalation, sealed evaluation,
and transfer studies were therefore not authorized.

This is a clean negative result, not a failed implementation. The substrate,
resume path, DeltaBall projection, and Delta-CMA algebra all passed their proof
obligations.

## Final matched result

Both endpoints used rank 16 / alpha 32, the same 12,875-row membership, 118,115
supervised tokens, AdamW, DeltaBall radius, exact 12-batch constant tail, six CMA
checkpoints and weights, and rank-16 recompression. A 36-check audit found only
the intended main-segment ordering difference.

| Metric | Uniform control | Curriculum candidate | Candidate delta |
|---|---:|---:|---:|
| Puzzle capability exact accuracy (4,080) | 17.7941% (726) | 17.0343% (695) | **−0.7598 pp** |
| Capability paired 95% CI | — | — | **[−1.4216, −0.0980] pp** |
| Exact McNemar p | — | — | **0.0297** |
| `chat_dev` mean quality | 0.86093 | 0.85289 | −0.00804 |
| `rag_dev` mean quality | 0.82054 | 0.82216 | +0.00162 |

Capability win/tie/loss was 80/3,889/111. The largest family regressions were
`mislabeled_boxes` (−6.25 pp) and `object_tracking` (−5.15 pp). The candidate
therefore fails the committed clear-improvement rule and the SSOT definition of
success.

## Component conclusions

| Component | v1 conclusion |
|---|---|
| RGE AdamW harness | Verified parity, observational geometry, and exact resume |
| FP32 Muon | Rejected: better training loss did not survive retention |
| DeltaBall | Small exploratory retention positive; targeted capability unconfirmed in isolation |
| Component-local curriculum | Rejected by the fully matched capability comparison |
| Constant-LR tail | Endpoint-neutral exploratory negative |
| Delta-CMA rank 16 | Mechanically verified; tiny exploratory retention gain, targeted capability unconfirmed in isolation |

## Reproducibility

The compact evidence is committed; regenerable checkpoints/evaluations remain
under `artifacts/rge_peft/`. The capability asset must match SHA256
`f717d9f19a904a0278ba932afca76976a331dd4e184fd8a997af93843cd93b95`.

```powershell
& .\.venv\Scripts\python.exe scripts\verify\verify_rge_matched_cma_control.py
& .\.venv\Scripts\python.exe scripts\verify\verify_rge_capability.py
& .\.venv\Scripts\python.exe scripts\eval\run_rge_capability.py --adapter artifacts\adapters\rge_m5_uniform_matched_cma_rank16 --out artifacts\rge_peft\eval\m6_uniform_matched_cma_rank16_capability_repro --batch-size 16 --cap-mib 5120
& .\.venv\Scripts\python.exe scripts\eval\run_rge_capability.py --adapter artifacts\adapters\rge_m5_cma_rank16 --out artifacts\rge_peft\eval\m6_curriculum_cma_rank16_capability_repro --batch-size 16 --cap-mib 5120
& .\.venv\Scripts\python.exe scripts\verify\compare_rge_capability.py --control artifacts\rge_peft\eval\m6_uniform_matched_cma_rank16_capability_repro --candidate artifacts\rge_peft\eval\m6_curriculum_cma_rank16_capability_repro --evidence experiments\rge_peft\evidence\m6_capability_reproduction.json
```

No puzzle test/challenge split and no Rineya sealed RAG asset was accessed.
