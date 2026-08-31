# Catastrophic-Removal Critical Core experiment

This directory anchors the **CRCC / Monster** experiment. The full preregistration is
[`docs/PLAN_CATASTROPHIC_REMOVAL_CORE.md`](../../docs/PLAN_CATASTROPHIC_REMOVAL_CORE.md).

**Current state: BUILT — UNRUN.** The checked-in code and CPU-only verifier prove
structural contracts only; they contain no model result. The first hosted execution
attempt was blocked before runner allocation, so no CRCC membership is currently frozen.

## The three separable changes

1. **Discovery:** `scan_layer_ablation.py` measures intact Qwen and every one-layer
   bypass with reference NLL plus mechanical generation-collapse signals. It rejects
   sealed paths and resumes per layer.
2. **Matched recovery:** `design_arms.py`, `build_reduced_model.py`, and
   `make_recovery_configs.py` create source-mapped CRCC/random/low-criticality/
   angular-high arms and reuse the existing deterministic trainer without modifying
   `scripts/train/`.
3. **Delta reinsertion:** `apply_layer_deltas.py`, `plan_merge_grid.py`, and
   `run_dev_matrix.py` construct Qwen/Aegis/Halo/Supa tests as explicit mapped deltas,
   validate compatibility, and keep selection on unsealed development gates.

## Fast structural check

```bash
python -m compileall -q scripts/critical_core scripts/verify
python scripts/verify/verify_critical_core.py
python scripts/verify/verify_crcc_discovery_evidence.py
```

The verifiers construct temporary synthetic fixtures at runtime. They prove that arm
selection is deterministic, controls are depth matched, recovery configs differ only in
allowed fields, merge-plan hashes reproduce, sealed names are refused, the delta formula
is exact, incomplete discovery evidence is rejected, and scientific hashes are stable.
They must never be reported as model evidence.

## Canonical local/private discovery execution

After fetching this branch, Windows users can run the isolated reproducible path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/critical_core/bootstrap_and_run_crcc.ps1
```

That command creates `.venv-crcc-v1`, installs `requirements-crcc-v1.txt`, materializes
the exact pinned Qwen checkpoint revision when necessary, verifies the official
`model.safetensors` SHA-256, runs the runtime doctor and layer-bypass compatibility smoke,
validates any resumable baseline/layer files, executes the intact baseline plus all 28
one-layer bypass conditions, validates the completed raw evidence again, and invokes the
independent evidence assembler/G0 decision.

Useful diagnostic-only forms:

```powershell
# Validate environment/checkpoint/hook/resume state without running discovery inference.
powershell -ExecutionPolicy Bypass -File scripts/critical_core/bootstrap_and_run_crcc.ps1 -DoctorOnly

# Use an existing environment instead of creating the isolated venv.
powershell -ExecutionPolicy Bypass -File scripts/critical_core/run_discovery_local.ps1 -DoctorOnly

# Refuse any model download and require Qwen3-0.6B/ to exist locally.
powershell -ExecutionPolicy Bypass -File scripts/critical_core/bootstrap_and_run_crcc.ps1 -Offline
```

Machine-readable diagnostics are written under:

```text
runs/critical_core/crcc-v1/diagnostics/
  discovery_contract.json
  runtime_doctor.json
  execution_state.json
```

`execution_state.json` identifies the exact failed stage if execution stops. Existing
baseline/layer files are reused only after their case IDs, condition/layer metadata, and
stored summaries reproduce from the raw per-case records. The final assembler repeats
its own independent validation before membership can freeze.

The pinned discovery checkpoint is:

```text
Qwen/Qwen3-0.6B
revision: c916fa4defd319b7d4e4da17604ca7338f4d99f5
model.safetensors SHA-256:
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
```

## Execution order

```text
critical_core_scan_v1.yaml
  -> doctor_crcc_runtime.py
  -> verify_crcc_checkpoint.py
  -> resume_integrity.py
  -> scan_layer_ablation.py
  -> resume_integrity.py (post-scan)
  -> assemble_scan_evidence.py
  -> G0 PASS: freeze automatic CRCC membership
       OR G0 FAIL: stop for instrument review
  -> design_arms.py
  -> arms.json
  -> build_reduced_model.py (one checkpoint per arm)
  -> make_recovery_configs.py
  -> scripts/train/heal.py + export_adapter.py (commands recorded in matrix)
  -> plan_merge_grid.py
  -> apply_layer_deltas.py
  -> run_dev_matrix.py
  -> canonically registered historical 64-row targeted evaluator
  -> one final sealed evaluation
```

Generated artifacts belong under `runs/critical_core/` and remain untracked. Commit only
small audited manifests or summaries when a result changes the canonical program state.

## Important boundary

The current main branch contains unsealed `chat_dev` and `rag_dev` gates, but not a
canonical executable copy of the historical 64-row long-context family comparison.
Those development gates can reject regressions; they cannot establish a replacement for
Halo's reported `30/64`. The exact 64-row protocol and known anchors must be registered
and reproduced before the targeted donor claim is evaluated. The local discovery runner
never invokes DEV, the historical 64-row evaluator, recovery training, donor merging, or
sealed evaluation.