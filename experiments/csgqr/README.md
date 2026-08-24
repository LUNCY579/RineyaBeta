# CSGQR — Confidence- and Significance-Gated Quantized Remnant

**Branch:** `research/csgqr-quantized-remnant`
**Checkpoint:** `artifacts/rineya-core-en32k-28L` (Qwen3, 28L, hidden 1024, d_ff 3072, |V| 32768, tied embeddings, bf16)

## What this is

A structurally retired transformer component is kept as a **frozen low-bit
remnant**. Before running it, the model estimates how mature its current
next-token decision already is: **confidence** decides *how much* recovery
compute to spend, **significance** decides *where* that compute goes, and
**quantization** decides what the retained donor costs to keep.

For the MLP case, with `h` the pre-remnant residual at layer `j`:

| state | update |
| --- | --- |
| deleted | `h' = h` |
| full remnant | `h' = h + F_j^Q(h)` |
| adaptive remnant | `h' = h + Σ_{g∈A(h)} α_g F_{j,g}^Q(h)` |

## Provenance note — read this first

This branch **contained no CSGQR code** when this campaign began. Its tip
(`1664f54`) was a significance-matrix paper commit, shared with
`research/pre-deletion-recovery-routing-v0.1` and `-v0.2`, eleven commits behind
`main`. There was no `docs/CSGQR_DESIGN.md`, no `scripts/csgqr/`, no
`scripts/verify/verify_csgqr.py`, and no CSGQR config — verified across every
ref, the working tree, the stash, the reflog and dangling objects.

So this is **not** an independent audit of someone else's prototype. The
implementation here was written as part of this campaign, which weakens audit
independence in a way no amount of testing repairs. The compensation is
structural: the adversarial controls (random routing, shuffled significance,
wrong-layer donor, always-on remnant, matched-compute baselines) were written
*before* any CSGQR result was looked at, and the analytic ceilings in
`results/budget.json` were computed before any GPU measurement. See
`METHOD_AUDIT.md`.

## The two structural facts that shape everything

Both are exact arithmetic on this checkpoint's dimensions, not measurements, and
both were fixed before the experiments ran (`src/budget.py`).

**1. Group decomposition is exactly additive.** Qwen3's MLP is
`F(x) = W_down (silu(W_gate x) ⊙ W_up x)` with `bias=False` on all three
projections. The elementwise product is per-intermediate-unit and `W_down` is
linear over that axis, so for any partition `{I_g}` of the intermediate
dimension, `F(x) = Σ_g F_g(x)` **exactly**. Executing a subset of groups is
mathematically identical to selecting those intermediate neurons; there is no
interaction term. Verified in float64 to `<1e-14` relative error, and the
constructor refuses a biased `down_proj` because a `down_proj` bias would be
counted once per active group instead of once overall.

**2. The exact-confidence monitor costs more than the MLP it gates.** The LM-head
projection is `2·|V|·d_model` MACs; the MLP it decides about is `2·3·d_model·d_ff`.
The ratio is

```
|V| / (3 · d_ff)  =  32768 / 9216  =  3.556×
```

so an exact-confidence gate that skipped the remnant on **every single token**
would still cost 3.56× more than it saved. This is a closed form, not a
measurement, and it generalises: an exact LM-head gate on a single MLP remnant is
only viable when `|V| < 3·d_ff`. For current models (`|V|` ~128–256k,
`d_ff` ~8–28k) that inequality essentially never holds. The hidden-state probe
is therefore not an optimisation of the method — it is the only viable
deployment form of it.

## Ceiling on the systems claim

One MLP layer is **1.99%** of this model's parameters (9,437,184 of 474,021,888).

| remnant format | packed | vs bf16 | saved | as share of model VRAM |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 18.000 MiB | 1.00× | — | — |
| int8-asym | 9.211 MiB | 1.95× | 8.79 MiB | 0.97% |
| int4-asym | 4.711 MiB | 3.82× | 13.29 MiB | **1.47%** |
| int2-asym | 2.461 MiB | 7.31× | 15.54 MiB | 1.72% |

Routing caps out just as low: even `ρ = 0` (never run the remnant) removes only
**1.99%** of model MACs per token; `ρ = 0.25` removes 1.49%.

**Consequence for the research question.** At single-layer scope the mechanism
cannot produce a systems win that exceeds allocator noise, whatever the routing
does. So this campaign treats single-layer CSGQR as a *mechanism* study — does
confidence predict remnant need, does significance beat random, does INT4 retain
donor function — and reports the systems ceiling analytically rather than
claiming a measured speedup that the arithmetic forbids. Gate E is assessed on
that basis.

## Numerical noise floor (important for reading every quality number)

The remnant reconstructs the donor's weight matrices **bit-identically** and
reproduces `nn.Linear` output bit-identically at most shapes — but not all. At
prompt length `T=22` the reconstructed path differs from `nn.Linear` by
`3.9e-3` max absolute, because cuBLAS selects a different bf16 GEMM tile for
that shape. The weights are identical; only the accumulation order differs.

This is the same confound `experiments/attention_shape` documented (switching
kernels moved 23% of greedy replies while changing mean quality by +0.00138).
It means:

- **B0 (dense)** is comparable to the stored gate baselines.
- **B2 (FP donor through the remnant path)** is the correct reference for every
  quantized arm, because it travels the identical code path.
- `|B0 − B2|` is the **noise floor**. Any quality difference smaller than that
  is not attributable to CSGQR.

## Layout

```
src/quant.py         blockwise INT2/3/4/8 with genuinely packed storage
src/remnant.py       group-addressable frozen remnant; alpha calibration
src/confidence.py    lens, features, temperature calibration, ECE
src/significance.py  marginal-utility accumulation and ranking controls
src/router.py        three-level policy, cost-aware top-k, routing trace
src/surgery.py       installs the remnant into a Qwen3 layer
src/budget.py        analytic storage / FLOP / monitor-cost accounting
src/envinfo.py       environment + model identity manifest
src/verify_csgqr.py  86-check verification suite (CPU, deterministic)
```

All three projections are stored `[d_ff, d_model]` — `down_proj` transposed — so
a remnant group is a **contiguous, byte-aligned row slice** of each. Inactive
groups are sliced away *before* dequantization and before any matmul; nothing is
computed and then masked.

## Reproducing

GPU work requires the repo venv — the ambient `python` on this machine carries a
CPU-only torch and would silently produce CPU numbers indistinguishable from GPU
ones in a summary file. `envinfo.require_cuda()` refuses rather than warns.

```bash
.venv/Scripts/python.exe experiments/csgqr/src/verify_csgqr.py --json experiments/csgqr/results/verify_csgqr.json
```

```bash
.venv/Scripts/python.exe experiments/csgqr/src/budget.py
```

## Evaluation protocol

Taken unchanged from `manifests/eval_assets.json` and `scripts/eval/run_gates.py`:
chat template with `add_generation_prompt=True`, `enable_thinking=False`;
`max_new_tokens=32`; greedy (`do_sample=False`); `response_control="eos_only"`;
scored by the vendored `quality_evaluation.py` v4.5 with the corpus profile built
from dataset records. Stored references: `chat_dev` 0.8335958 (510 cases),
`rag_dev` 0.6681789 (752 cases).

**Sealed sets are untouched.** `rag_test_SEALED`, `puzzle_test_SEALED` and
`puzzle_challenge_SEALED` are not read by any script here; development uses
`chat_dev` and `rag_dev`.
