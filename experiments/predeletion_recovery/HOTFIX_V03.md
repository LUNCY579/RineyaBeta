# PDRR v0.3 Corrective Entry Points

A post-publication integration review found two issues in the initial real-model trainer entry point:

1. the intact `full` anchor followed the compact export path and could therefore retain a temporary gate wrapper or delete the nominated block;
2. the recipient anchor compared acquired recipient deltas with zero, which regularized all handoff activity rather than selectively protecting pre-existing function.

The original files remain unchanged for provenance. Use these corrected, fail-closed entry points:

```text
experiments/predeletion_recovery/scripts/train_hf_hotfix.py
experiments/predeletion_recovery/scripts/orchestrate_hotfix.py
```

The corrective trainer performs exact source transformations and refuses to run unless both expected legacy blocks are found exactly once. It changes the behavior as follows.

## Intact anchor export

For `--arm full`, it now:

- restores the target gate to exactly one;
- captures logits before removing the temporary wrapper;
- unwraps the gate without deleting the Transformer block;
- verifies maximum logit error, greedy-token identity, parameter-count identity, and full layer count;
- writes `export_full/` and marks the deployment as an intact export.

Every pruning arm still uses gate-zero versus physical-deletion verification and writes `export_compact/`.

## Recipient stability anchor

The corrected stability term decomposes each acquired recipient residual into a component aligned with the doomed block's missing residual and an orthogonal component. Aligned acquisition is permitted; orthogonal drift is penalized. The penalty is strongest at positions where source ablation has low causal significance, because changes there are least justified as responsibility transfer.

This is a functional trust region. It does not claim to prove preservation of every latent duty; that remains an empirical question for the behavioral and mechanistic evaluations.

## Commands

Single real-model arm:

```bash
python experiments/predeletion_recovery/scripts/train_hf_hotfix.py \
  --config experiments/predeletion_recovery/configs/rineya_pilot.yaml \
  --arm causal_state_position_recovery \
  --target-layer 14 \
  --seed 20260823 \
  --out artifacts/pdrr/hotfix_pilot \
  --max-steps 2
```

Corrected matrix orchestration:

```bash
python experiments/predeletion_recovery/scripts/orchestrate_hotfix.py \
  --config experiments/predeletion_recovery/configs/rineya_confirmatory.yaml
```

CPU contract suite:

```bash
export PYTHONPATH="$PWD/experiments/predeletion_recovery/src"
python -m pytest -q experiments/predeletion_recovery/tests
```

PowerShell uses the same commands with `$env:PYTHONPATH` and backticks for line continuation.
