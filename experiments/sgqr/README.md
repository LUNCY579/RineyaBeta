# SGQR Fundamental Research Track

## Branch

`research/sgqr-core-quantized-remnant`

This experiment intentionally removes the confidence-gated portion of CSGQR.

SGQR means:

# Significance-Gated Quantized Remnant

The fixed-budget research question is:

> After retiring a full-precision MLP, can a frozen INT8/packed-INT4 remnant
> preserve useful donor computation, and can significance select a smaller
> subset of additive donor groups better than random or magnitude baselines?

There is no intermediate vocabulary projection, confidence calibration, entropy
threshold, probability margin, or confidence-controlled budget in this branch.

Confidence can be added later as an optional budget controller only after the
quantized-remnant and significance hypotheses are established.

See:

- `docs/SGQR_DESIGN.md`
- `configs/sgqr_reference.yaml`
- `scripts/verify/verify_sgqr.py`

---

## Implemented reference modes

| Mode | Description |
|---|---|
| `dense` | Original full-precision model |
| `fp_frozen` | Original MLP frozen, otherwise unchanged |
| `deleted` | Exact zero MLP residual branch |
| `int8_all` | Complete frozen INT8 remnant |
| `int4_all` | Complete frozen physically packed INT4 remnant |
| `random_topk` | Fixed random groups at a matched budget |
| `magnitude_topk` | Groups ranked by coupled weight magnitude |
| `static_topk` | Groups ranked by calibrated marginal recovery utility |
| `dynamic_topk` | Hidden-state probe predicts group utility at a fixed budget |

The reference backend dequantizes only groups that execute. It is designed to
establish correctness and the quality frontier. It does not claim fused-kernel
speedups.

---

## 1. Verify the implementation

From the repository root:

```powershell
python scripts/verify/verify_sgqr.py
```

The verifier checks:

- exact INT4 nibble packing;
- INT4 and INT8 storage accounting;
- zero-row behavior;
- additive quantized MLP groups;
- token-specific selected execution;
- exact deleted and all-on wrapper modes;
- static and dynamic significance routing;
- frozen donor buffers;
- state-dict reload;
- utility sign;
- exposure-normalized significance;
- Recovery Ratio.

Do not run long GPU experiments if verification fails.

---

## 2. Scan candidate layers

Use a small paired prompt set first:

```powershell
python scripts/sgqr/scan_layers.py `
  --model artifacts/rineya-core-en32k-28L `
  --dataset chat_dev `
  --layers all `
  --group-size 128 `
  --limit 24 `
  --cap-mib 5120 `
  --out artifacts/sgqr_runs/layer_scan.json
```

The scan reports, per MLP:

- deletion teacher KL;
- always-on INT8 teacher KL;
- always-on INT4 teacher KL;
- mean remnant utility;
- packed storage;
- approximate fraction of deletion KL recovered.

Choose a moderately sensitive layer for the first full experiment.

---

## 3. Calibrate significance

Example for layer 14:

```powershell
python scripts/sgqr/calibrate.py `
  --model artifacts/rineya-core-en32k-28L `
  --dataset chat_dev `
  --layer 14 `
  --bits 4 `
  --group-size 128 `
  --limit 64 `
  --exposure-exponent 1.0 `
  --cap-mib 5120 `
  --out artifacts/sgqr_runs/calibration_l14_g128
```

For every calibration prompt, the script runs paired paths:

1. dense teacher;
2. exact deletion;
3. full INT4 remnant;
4. each individual quantized group.

Group utility is:

\[
U_{t,g}
=
\operatorname{KL}(p_t^T\|p_t^0)
-
\operatorname{KL}(p_t^T\|p_t^{+g}).
\]

Outputs:

- `summary.json` — compact significance and storage summary;
- `utility_trace.pt` — hidden states and utility labels for router training;
- `trace.jsonl` — per-prompt audit rows.

These run artifacts are ignored by Git.

---

## 4. Train the optional dynamic significance router

```powershell
python scripts/sgqr/train_router.py `
  --trace artifacts/sgqr_runs/calibration_l14_g128/utility_trace.pt `
  --out artifacts/sgqr_runs/router_l14_g128.pt `
  --rank 32 `
  --epochs 200 `
  --batch-size 32 `
  --budget-groups 12 `
  --positive-weight 2.0
```

This probe predicts marginal group utility directly from the current MLP input.

It does **not** predict next-token confidence.

The training report compares utility captured by the learned route against a
random matched-budget route.

---

## 5. Run evaluation arms

Dense:

```powershell
python scripts/sgqr/evaluate.py `
  --dataset chat_dev `
  --mode dense `
  --layer 14 `
  --out artifacts/sgqr_runs/evaluation/chat_dense
```

Deletion:

```powershell
python scripts/sgqr/evaluate.py `
  --dataset chat_dev `
  --mode deleted `
  --layer 14 `
  --out artifacts/sgqr_runs/evaluation/chat_deleted
```

Always-on INT4:

```powershell
python scripts/sgqr/evaluate.py `
  --dataset chat_dev `
  --mode int4_all `
  --layer 14 `
  --group-size 128 `
  --out artifacts/sgqr_runs/evaluation/chat_int4_all
```

Random matched budget:

```powershell
python scripts/sgqr/evaluate.py `
  --dataset chat_dev `
  --mode random_topk `
  --layer 14 `
  --bits 4 `
  --group-size 128 `
  --budget-fraction 0.5 `
  --out artifacts/sgqr_runs/evaluation/chat_random_half
```

Static significance:

```powershell
python scripts/sgqr/evaluate.py `
  --dataset chat_dev `
  --mode static_topk `
  --layer 14 `
  --bits 4 `
  --group-size 128 `
  --budget-fraction 0.5 `
  --scores artifacts/sgqr_runs/calibration_l14_g128/summary.json `
  --out artifacts/sgqr_runs/evaluation/chat_static_half
```

Dynamic significance:

```powershell
python scripts/sgqr/evaluate.py `
  --dataset chat_dev `
  --mode dynamic_topk `
  --layer 14 `
  --bits 4 `
  --group-size 128 `
  --budget-fraction 0.5 `
  --router artifacts/sgqr_runs/router_l14_g128.pt `
  --out artifacts/sgqr_runs/evaluation/chat_dynamic_half
```

Repeat the frozen configuration on `rag_dev`.

Do not use `rag_test_SEALED` during development.

---

## 6. Sweep the active group fraction

At minimum test:

```text
0
0.0625
0.125
0.25
0.50
0.75
1.00
```

Compare at each matched budget:

- random;
- magnitude;
- static significance;
- dynamic significance.

The primary question is whether there is a useful interior operating point:

\[
0 < \rho < 1.
\]

If only \(\rho=1\) preserves quality, the result is an always-on quantized layer,
not successful significance gating.

---

## 7. Compile a comparison

```powershell
python scripts/sgqr/report.py `
  --dense artifacts/sgqr_runs/evaluation/chat_dense/summary.json `
  --deleted artifacts/sgqr_runs/evaluation/chat_deleted/summary.json `
  --run int4_all=artifacts/sgqr_runs/evaluation/chat_int4_all/summary.json `
  --run random_half=artifacts/sgqr_runs/evaluation/chat_random_half/summary.json `
  --run static_half=artifacts/sgqr_runs/evaluation/chat_static_half/summary.json `
  --run dynamic_half=artifacts/sgqr_runs/evaluation/chat_dynamic_half/summary.json `
  --out artifacts/sgqr_runs/comparison_chat.json
```

The report includes:

- absolute quality;
- Recovery Ratio;
- active group fraction;
- packed remnant bytes;
- peak VRAM;
- tokens per second;
- P90 case latency.

---

## Required interpretation

Report four quantities separately:

1. logical parameters;
2. trainable parameters;
3. packed storage bytes;
4. measured active execution.

Do not call INT4 a parameter-count reduction.

Do not claim compute savings from a post-hoc mask.

Do not infer latency from theoretical group count.

Do not attribute a slow gather/scatter reference backend automatically to an
algorithmic quality limitation.

---

## Advancement gates

Proceed only when:

### Gate A — Correctness

`verify_sgqr.py` passes and dense/all-off/all-on invariants hold on the real model.

### Gate B — Donor viability

Always-on INT8 or INT4 recovers meaningful deletion damage.

### Gate C — Significance value

Static or dynamic significance beats random at a matched group budget.

### Gate D — Interior frontier

A partial group budget preserves useful quality.

### Gate E — Systems evidence

Storage, VRAM, latency, and routing overhead are measured on the actual GPU.

Only after these gates are understood should confidence be reintroduced in the
separate CSGQR track.

---

## Research outputs to preserve

Create and commit compact summaries supporting every documented claim.

Keep negative results, including:

- INT4 donor failure;
- significance equal to random;
- magnitude equal to utility;
- dynamic routing slower than static;
- no useful interior point;
- reference backend latency regression;
- layer-specific failures;
- utility interactions that break independent top-k selection.

The objective is to establish the real quantized-remnant foundation, not to make
SGQR appear successful.
