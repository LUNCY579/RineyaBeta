# Peer-Conditioned Stabilizer Matrix (PCSM)

**Branch:** `exp/peer-stabilizer-matrix`  
**Base:** `exp/stabilizer-matrix`  
**Status:** implementation-complete research prototype; software invariants and a controlled synthetic feasibility study pass. No Rineya/LLM quality claim has been made.

## What this is

PCSM is a training-time method for **sparse native-weight fine-tuning**. Compatible transformer layers describe their weight geometry and calibration behavior to a target layer. Their messages generate:

1. a binary support matrix `M_t`, which says **where** the target may learn; and
2. a bounded positive scale matrix `A_t`, which says **how strongly** selected coordinates may learn.

The task loss retains authority over the update direction:

```math
W_t^{(n+1)}
=
W_t^{(n)}
-
\eta\,
M_t\odot A_t\odot
\nabla_{W_t}\mathcal L.
```

This is the mathematical version of the "coworker" idea: peer layers communicate structural working preferences, but they do not directly overwrite the target's knowledge.

## What this is not

This directory does **not** revive the original inference-time stabilizer selector. The original experiment tried to predict which SwiGLU neurons should fire and was closed after its pre-registered kill tests. PCSM asks a different question:

> Can other layers provide a useful prior over which native coordinates of a target matrix should remain plastic during fine-tuning?

The old negative result therefore remains intact and is not rewritten.

## Repository layout

```text
peer_optimizer/
├── README.md
├── THEORY.md
├── EXPERIMENT_PROTOCOL.md
├── run_checks.ps1
├── run_checks.sh
├── src/
│   ├── peer_features.py         compact descriptors and structural property maps
│   ├── sparse_selection.py      exact top-K masks, STE, and support audits
│   ├── peer_stabilizer.py       peer attention, support scores, and step scales
│   ├── sparse_optim.py          exact-support, scale-aware AdamW
│   ├── rineya_integration.py    Qwen/Rineya plan-build/save/load/apply workflow
│   ├── feasibility.py           controlled one-seed study
│   └── stress_feasibility.py    independent-seed repetition
├── tests/
│   ├── test_peer_stabilizer.py
│   ├── test_sparse_optim.py
│   └── test_rineya_integration.py
└── results/
    ├── RESULTS.md
    ├── feasibility.json
    └── feasibility_multiseed.json
```

## Core workflow

### 1. Collect calibration gradients

Run a representative calibration split through the unmodified model and call `loss.backward()`. No optimizer step is taken. Every candidate matrix must have a gradient from the **same objective and data split**.

### 2. Generate a fixed plan

Matrices are grouped by semantic role (`mlp.down_proj`, `self_attn.q_proj`, and so on). A target receives advice only from compatible peers by default.

```python
from peer_stabilizer import PeerStabilizerMatrix
from rineya_integration import build_sparse_plan, save_sparse_plan

selector = PeerStabilizerMatrix(hidden_dim=64).to(model.device)
plan = build_sparse_plan(
    model,
    selector,
    module_paths=["mlp.down_proj"],
    budget=0.001,  # exactly 0.1% of each matrix after rounding
)
save_sparse_plan(plan, "artifacts/peer_stabilizer/plan.pt")
```

The plan stores exact masks, step scales, peer-attention vectors, feature weights, and SHA-256 hashes of every support mask.

### 3. Restart from the exact base checkpoint

The calibration backward pass is a measurement, not training. Reload the byte-identical starting checkpoint before applying the plan. This prevents selection and optimization from being conflated.

### 4. Freeze the support and train

```python
from rineya_integration import load_sparse_plan, apply_sparse_plan

plan = load_sparse_plan("artifacts/peer_stabilizer/plan.pt")
optimizer = apply_sparse_plan(
    model,
    plan,
    lr=1e-4,
    weight_decay=0.01,
)

for batch in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

`SparseNativeAdamW` masks all five paths through which a supposedly frozen coordinate could otherwise move:

- task gradient;
- first-moment state;
- second-moment state;
- Adam update;
- decoupled weight decay.

The peer scale `A_t` multiplies the **realized Adam update and selected-coordinate weight decay**, not the raw gradient. Putting a positive scale only inside Adam's moments would be largely cancelled by second-moment normalization and would not reliably control the actual step. Optimizer states remain fp32 for fp16/bf16 parameters.

### 5. Audit the realized delta

```python
from peer_stabilizer import audit_sparse_delta

audit = audit_sparse_delta(base_weight, trained_weight, plan_entry.mask)
assert audit.changed_outside == 0
```

A run is invalid if even one undeclared coordinate changes beyond the declared audit tolerance.

## How peer advice is generated

For each layer `j`, PCSM builds a compact descriptor `z_j` from scale-normalized weight, activation, gradient, shape, and depth statistics. For target `t`, attention over eligible peers is

```math
\alpha_{jt}
=
\operatorname{softmax}_{j\in\mathcal P_t}
\left[
\frac{\cos(z_t,z_j)}{\tau}
+
\sigma(\gamma)
\frac{q(z_t)^\top k(z_j)}{\sqrt d}
-
\lambda_d |d_j-d_t|
\right].
```

Each peer contributes four non-negative structural property maps:

1. centered element magnitude;
2. row-energy profile;
3. column-energy profile;
4. joint row-column leverage proxy.

The attention-weighted property maps, target properties, and task-gradient magnitude form eight coordinate features. A small hypernetwork produces non-negative feature weights, yielding a score matrix `S_t`. Exact top-
`K` selection gives

```math
M_t=\operatorname{TopKMask}(S_t,K).
```

During meta-calibration, a custom straight-through operator returns an **exact binary mask in the forward pass** and a smooth surrogate gradient in the backward pass. This avoids floating-point leakage from the common `hard + soft - stopgrad(soft)` expression.

See [`THEORY.md`](THEORY.md) for the full derivation and invariants.

## Reproduce the checks

### Windows / PowerShell

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\experiments\stabilizer\peer_optimizer\run_checks.ps1
```

An explicit interpreter can be supplied:

```powershell
powershell -ExecutionPolicy Bypass -File .\experiments\stabilizer\peer_optimizer\run_checks.ps1 `
  -Python .\.venv\Scripts\python.exe
```

### Linux/macOS

```bash
bash experiments/stabilizer/peer_optimizer/run_checks.sh
```

The workflow compiles all modules, runs the unit/integration suite, reproduces the full one-seed study, and repeats the reduced stress configuration across five independent seeds.

## Current evidence

The software suite contains **21 tests**. It checks exact top-`K`, self-exclusion, semantic peer filtering, differentiable meta-calibration, gradient-direction preservation, plan persistence, bf16/fp32-state behavior, and bitwise frozen-coordinate preservation under momentum plus weight decay.

The committed five-seed controlled study reports:

| arm | true loss, mean across seeds | support recall |
|---|---:|---:|
| peer-conditioned | **0.022917** | **0.932552** |
| shuffled peer coordinates | 0.035245 | 0.102344 |
| random support | 0.039252 | 0.091146 |
| gradient top-`K` | 0.069804 | 0.142969 |
| dense noisy update | 0.127059 | 1.000000 |
| no update | 0.030312 | 0.000000 |
| oracle | 0.000000 | 1.000000 |

All declared controlled checks passed on 5/5 seeds. The synthetic optimum was intentionally constructed from peer structural properties; this validates that the implementation can learn and exploit the proposed signal. It does **not** show that real transformer layers contain the same useful signal.

See [`results/RESULTS.md`](results/RESULTS.md) and the raw JSON files.

## Design safeguards

- **No self-advice:** a target's attention to itself is exactly zero.
- **Same-role peers by default:** q-projections do not casually advise down-projections.
- **Fail closed:** no compatible peer is an error unless fallback is explicitly enabled.
- **No direct peer overwrite:** peer matrices cannot choose update sign independently of the task gradient.
- **Fixed support for SFT:** changing `M_t` every step is excluded from the first experiment because the union of touched weights could become dense.
- **Structural, not identity, regularization:** the optional regularizer matches row/column energy profiles rather than forcing raw matrices to become equal.
- **Persisted audit trail:** masks, scales, attention, feature weights, and hashes are saved before training.

## Known limitations

1. The controlled task is favorable by construction; it is a mechanism test, not external validation.
2. Cross-layer neuron coordinates may not be aligned. Property maps and semantic grouping reduce this risk but do not eliminate it.
3. Sparse updates do not automatically make dense PyTorch backward passes faster or smaller. The current implementation primarily controls optimizer state, stored deltas, and exact plasticity—not dense matmul cost.
4. A full score matrix is materialized transiently. For large matrices, a later implementation should use blockwise top-`K` or streaming selection.
5. The generator requires calibration or meta-training. An uncalibrated bootstrap is included for diagnostics, not presumed optimal.
6. Tied/shared parameters require special accounting before a real model run.

The next valid claim must come from the real-data controls in [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md), especially **real peers versus coordinate-shuffled peers** at exactly matched native-weight budgets.
