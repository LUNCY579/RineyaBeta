# RGE-PEFT experiment artifacts

The governing design is [`docs/PLAN_RGE_PEFT.md`](../../docs/PLAN_RGE_PEFT.md).
This directory contains compact, reviewable evidence and ledgers. Regenerable run
payloads live under `artifacts/rge_peft/` and are intentionally ignored by Git.

## M1 commands

Use the project CUDA environment on this Windows host:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONHASHSEED='0'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& .venv\Scripts\python.exe scripts\train\rge_peft.py --config configs\rge_peft\smoke_adamw.yaml --max-steps 3 --resume none
```

Every run directory contains `resolved_config.yaml`, `run_manifest.json`,
`memory_profile.json`, `train_log.jsonl`, optional `geometry.jsonl`, and
format-version-2 checkpoints. Geometry artifacts are evidence logs, not resume
state; the last measured factor snapshot is separately stored in the checkpoint.

No RGE command in this directory authorizes sealed evaluation. The existing
evaluator must continue to refuse `rag_test_SEALED` until the M7 boundary.
