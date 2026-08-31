# Catastrophic-Removal Critical Core experiment

This directory anchors the **CRCC / Monster** experiment. The full preregistration is
[`docs/PLAN_CATASTROPHIC_REMOVAL_CORE.md`](../../docs/PLAN_CATASTROPHIC_REMOVAL_CORE.md).

**Current state: BUILT — UNRUN.** The checked-in code and CPU-only verifier prove
structural contracts only; they contain no model result.

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
```

The verifier constructs temporary synthetic fixtures at runtime. They prove that arm
selection is deterministic, controls are depth matched, recovery configs differ only in
allowed fields, merge-plan hashes reproduce, sealed names are refused, and the delta
formula is exact. They must never be reported as scientific evidence.

## Execution order

```text
critical_core_scan_v1.yaml
  -> scan_layer_ablation.py
  -> scan_summary.json
  -> design_arms.py
  -> arms.json
  -> build_reduced_model.py (one checkpoint per arm)
  -> make_recovery_configs.py
  -> scripts/train/heal.py + export_adapter.py (commands recorded in matrix)
  -> plan_merge_grid.py
  -> apply_layer_deltas.py
  -> run_dev_matrix.py
  -> frozen 64-row targeted evaluator
  -> one final sealed evaluation
```

Generated artifacts belong under `runs/critical_core/` and remain untracked. Commit only
small audited manifests or summaries when a result changes the canonical program state.

## Important boundary

The current main branch contains unsealed `chat_dev` and `rag_dev` gates, but not a
canonical executable copy of the historical 64-row long-context family comparison.
Those development gates can reject regressions; they cannot establish a replacement for
Halo's reported `30/64`. The exact 64-row protocol and known anchors must be registered
and reproduced before the targeted donor claim is evaluated.
