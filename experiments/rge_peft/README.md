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

## M2 optimizer-pair commands

The registered AdamW-LoRA versus Muon-LoRA protocol is documented in
[`docs/PROTOCOL_RGE_M2_OPTIMIZER_PILOT.md`](../../docs/PROTOCOL_RGE_M2_OPTIMIZER_PILOT.md).

Safe preflight:

```powershell
& .venv\Scripts\python.exe scripts\experiment\rge_optimizer_pair.py `
  --pair manifests\rge_peft\m2_optimizer_pair.yaml `
  --arm none
```

CPU proof obligations:

```powershell
& .venv\Scripts\python.exe scripts\verify\verify_rge_muon.py `
  --evidence experiments\rge_peft\evidence\m2_muon_unit_cpu.json
& .venv\Scripts\python.exe scripts\verify\verify_rge_optimizer_pair.py `
  --evidence experiments\rge_peft\evidence\m2_pair_protocol_cpu.json
```

M1 is now `GATE_PASSED`. M2 execution remains governed by the checked-in pair
manifest: arms run sequentially, scientific substrate and effective token counts
must match, and only frozen `chat_dev`/`rag_dev` retention evaluation is eligible.

No RGE command in this directory authorizes sealed evaluation. The existing
evaluator must continue to refuse `rag_test_SEALED` until the M7 boundary.
