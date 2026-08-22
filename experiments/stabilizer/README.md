# The stabilizer matrix

Pre-registration: [`docs/EXPERIMENT_DESIGN_STABILIZER.md`](../../docs/EXPERIMENT_DESIGN_STABILIZER.md).

Phase 0 is three kill-tests with thresholds declared before any measurement.
Nothing here is built until they pass.

```
src/
  concentration_mlp.py     P0.1  is there sparsity in a SwiGLU FFN to exploit?
  probe_sparse_matmul.py   P0.2  is sparse even faster on this GPU? (no model)
  predict_firing.py        P0.3  can one global matrix predict the firing set,
                                 and does the KV significance matrix help?
results/
  *.json                   one file per probe, read by the analysis in the docs
```

Run with `../../.venv/Scripts/python.exe`.
