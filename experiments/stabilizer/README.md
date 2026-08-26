# The stabilizer matrix

Pre-registration: [`docs/EXPERIMENT_DESIGN_STABILIZER.md`](../../docs/EXPERIMENT_DESIGN_STABILIZER.md).

Phase 0 is three kill-tests with thresholds declared before any measurement.
They did not pass. P0.1 passed (ρ(25%) = 0.9586), P0.2 failed the speed
criterion, and P0.3 was a qualified negative: a global selector reaches 0.70 recall but
the significance matrix adds +0.0002 over the hidden state alone. The stabilizer
hypothesis was abandoned and nothing was built from it.

**What this directory actually contains is cornerstone 2**, produced by following that
failure onto the KV axis: span-aware selection — an asymmetric max-pool over the
significance scores before top-k. It was never pre-registered. It makes 128K lossless
at B=16 (7,864x on real tokens, 1,501 MiB). See
[`results/FINDINGS_P2.md`](results/FINDINGS_P2.md); 15 of its figures are pinned by
`scripts/design/verify_claims.py`.

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

## Training-time follow-on: peer-conditioned sparse native updates

A separate follow-on was added on 2026-08-26 under
[`peer_optimizer/`](peer_optimizer/). It does **not** revive the failed
inference-time firing-selector hypothesis above. It asks whether compatible layers can
generate a useful support and step-scale prior for sparse native-weight fine-tuning,
while the task gradient retains control of update direction. Its implementation,
mathematical specification, tests, controlled feasibility results, and real-model
protocol are isolated in that subdirectory. No Rineya quality result is claimed yet.
