# RGE-PEFT M2 Optimizer Pilot Protocol

**Status:** IMPLEMENTED / BOUNDED SMOKE AUTHORIZED / FULL PAIR BLOCKED PENDING LR CALIBRATION
**Program:** Rineya Geometry-Efficient PEFT v1.1
**Branch:** `research/rge-peft-v1`
**Experiment ID:** `rge_v1__m2__optimizer_pair__r16__t1m__seed20260819`
**Governing SSOT:** `docs/PLAN_RGE_PEFT.md`
**Pair manifest:** `manifests/rge_peft/m2_optimizer_pair.yaml`

## 1. Purpose

This protocol implements the first strict comparison between traditional
AdamW-LoRA and the Puro-inspired optimizer component of RGE-PEFT. It asks a
narrow question:

> At identical model, LoRA capacity, data order, supervised-token budget,
> precision, schedule, batching, and consumer-GPU memory policy, does an
> FP32 Muon update route improve capability acquisition per token or per unit
> training time relative to AdamW-LoRA without damaging the named frozen
> retention sets?

This is the **O-A versus O-M optimizer isolation arm** from the SSOT. It is not a
test of DeltaBall, curriculum ordering, a constant-learning-rate tail,
Delta-CMA, DoRA, or the complete future RGE recipe.

## 2. Claim boundary

The current learning mix is the accepted deterministic B0 healing/capability
mix. The repository's broader v3 corpus remains train-only and pending human
review, and no dedicated RGE capability/OOD bank has yet been frozen.
Therefore M2 may establish only:

- optimizer mechanics and stability;
- convergence behavior on the registered learning mix;
- supervised-token and wall-clock efficiency on the measured host;
- memory and optimizer-state cost; and
- preservation or regression on `chat_dev` and `rag_dev`.

M2 **must not** be described as proving general intelligence improvement,
broad OOD generalization, universal consumer-hardware speedup, or quality
preservation outside the named retention sets. The sealed RAG test remains
forbidden until M7.

## 3. Registered arms

| Property | O-A control | O-M candidate |
|---|---|---|
| Base | `artifacts/rineya-core-en32k-28L` | identical |
| LoRA | rank 16, alpha 32, dropout 0 | identical |
| Targets | q/k/v/o + gate/up/down | identical |
| Data | B0 mix | identical bytes/order/plan |
| Precision | BF16 base, FP32 adapters | identical |
| Schedule | cosine, 40-step warmup | identical |
| Budget | 1,000,000 supervised tokens | identical effective token count |
| Optimizer | `torch.optim.AdamW` | `rge.optim.Muon` |
| Nominal LR | `1e-4` | `1e-4` |
| Momentum | AdamW betas 0.9/0.999 | EMA 0.95 + Nesterov |
| Update geometry | elementwise Adam normalization | 5-step quintic Newton–Schulz |
| Muon scaling | n/a | spectral × 0.2 |
| Weight decay | 0 | 0 |
| Delta projection | none | none |
| Curriculum/CMA | reference / off | identical |

The same nominal LR plus Muon's `0.2` extra multiplier is a **provisional
screening calibration**, chosen to approximately align update RMS with AdamW.
It is not a confirmatory optimal-LR claim. Any bounded LR calibration performed
before Phase 6 must be recorded as exploratory and must not use the sealed set.

## 4. Muon implementation contract

The repository carries a small self-contained Apache-2.0 Muon subset in
`scripts/train/optim/muon.py`, adapted from the pinned NVIDIA NeMo Emerging
Optimizers implementation at commit
`02b6ec843b392475a9dd29fc68dbdb27c48bea4c`.

M2 freezes the following behavior:

1. Only ordinary two-dimensional FP32 LoRA A/B tensors are eligible.
2. Non-matrix trainables are rejected; there is no implicit AdamW fallback.
3. Momentum is an EMA of gradients with optional Nesterov interpolation.
4. Orthogonalization uses five optimized quintic Newton–Schulz steps.
5. Scaling is `sqrt(max(size_out, size_in)) * 0.2`.
6. Weight decay, when nonzero, is decoupled.
7. Optimizer math is FP32, with no Triton, tensor parallelism, Hyperball,
   reduced-precision optimizer state, or hidden partitioning.
8. Optimizer state is checkpointable through the existing format-v2 RGE
   checkpoint and exact-resume path.

The optimizer contract, coefficient schedule, parameter names, shapes, dtypes,
and fallback policy are written into every run manifest and compared at resume.

## 5. Pair preflight and fail-closed rules

`scripts/experiment/rge_optimizer_pair.py` is the canonical launcher and audit
surface. Before any model is loaded it verifies:

- exactly one AdamW control and one Muon candidate;
- byte-addressable config files;
- identical configuration outside `optimizer` and `run`;
- identical positive supervised-token budgets;
- identical learning-data path;
- distinct run IDs and directories;
- no sealed asset in training or active evaluation;
- sealed access policy `forbidden_until_m7`; and
- a `mechanism_and_retention_only` claim while the capability bank is unfrozen.

M1 is `GATE_PASSED`, so bounded integration smoke is authorized. The full
1M-token pair remains fail-closed until the SSOT's effective-step LR bracket is
completed and `optimizer_calibration.status` is changed from
`provisional_screening_configuration` to `frozen_exploratory`. This operational
gate may not silently change arm contents, data, capacity, or claims.

## 6. Run protocol

### 6.1 Validate only

```bash
python scripts/experiment/rge_optimizer_pair.py \
  --pair manifests/rge_peft/m2_optimizer_pair.yaml \
  --arm none
```

Expected output: `pair_preflight.json` with `status: PASS`, followed by no
training. This command is safe while the execution gate is blocked.

### 6.2 Bounded integration smoke after M1 authorization

```bash
python scripts/experiment/rge_optimizer_pair.py \
  --pair manifests/rge_peft/m2_optimizer_pair.yaml \
  --arm both \
  --resume none \
  --max-steps 2 \
  --run-root artifacts/rge_peft/smokes/m2_pair_2step \
  --output artifacts/rge_peft/smokes/m2_pair_2step/pair
```

This proves GPU construction, stepping, checkpointing, manifest generation,
and post-run comparability. The disposable run root prevents a smoke run from
contaminating the canonical 1M-token directories. It is marked
`bounded_smoke: true` and is never confirmatory evidence.

### 6.3 Effective-step calibration gate

Before the registered pair, run the AdamW reference and a small log-spaced Muon
bracket, then identify approximately 0.5x, 1x, and 2x robust
`base_relative_effective_step` regimes. Freeze exactly one stable exploratory
candidate and record the selection without sealed evaluation. A provisional
nominal-LR match cannot authorize the full pair.

### 6.4 Registered exploratory pair

```bash
python scripts/experiment/rge_optimizer_pair.py \
  --pair manifests/rge_peft/m2_optimizer_pair.yaml \
  --arm both \
  --resume none \
  --evaluate-final
```

The arms run sequentially so one consumer GPU is never oversubscribed. On
interruption, restart with `--resume auto`. Exact resume remains subject to the
existing environment, source, plan, data, and optimizer-contract guards. A
reverse-order timing replicate can be launched with `--order candidate-control`
and a distinct `--run-root`/`--output`; one execution order alone cannot support
a confirmatory hardware-speed claim on a thermally variable laptop.

## 7. Post-run comparability gate

The launcher refuses interpretation unless both completed arms match on:

- model and tokenizer hashes;
- data mix hash;
- seed;
- precision;
- complete LoRA configuration;
- schedule, curriculum, and CMA state;
- requested and effective supervised-token count;
- plan hash;
- first-batches hash; and
- initial trainable-state hash.

A mismatch writes `FAIL_COMPARABILITY` and aborts. Optimizer identity is
expected to differ and is reported rather than included in this equality gate.

## 8. Recorded evidence

Every arm already writes the standard RGE artifacts, including:

- `run_manifest.json`;
- `resolved_config.yaml`;
- `memory_profile.json`;
- `train_log.jsonl` with step time and supervised tokens;
- `geometry.jsonl` at cadence 10;
- `checkpoint_ledger.jsonl`;
- format-v2 checkpoints with optimizer and geometry state; and
- `summary.json` with wall time, active GPU-hours, peak memory, final position,
  and `sealed_data_accessed: false`.

The pair layer adds:

- `pair_preflight.json`;
- `pair_summary.json`;
- full loss-versus-supervised-token curves;
- token-weighted mean loss;
- final and minimum loss;
- supervised tokens per second and full/steady-state step-time distributions;
- candidate/control training-time, throughput, and steady-median ratios;
- optimizer-state tensor bytes and ratio; and
- optional full frozen `chat_dev` and `rag_dev` summaries for both endpoints.

## 9. Interpretation

The M2 result is encouraging only if the candidate is finite and stable, the
comparability gate passes, and at least one efficiency axis improves without a
material named-retention regression. Raw tokens/second may be lower while total
compute-to-quality is better; these are reported separately.

A lower training loss alone is not enough to claim a better Rineya model. A
candidate with faster convergence but degraded frozen retention is a tradeoff,
not a free improvement. A candidate that merely increases per-step cost without
improving token efficiency is rejected for the consumer-hardware objective.

The full RGE claim remains reserved for later DeltaBall, curriculum/tail,
Delta-CMA, matched confirmation, and one-time sealed phases.

## 10. Verification commands

```bash
python scripts/verify/verify_rge_muon.py \
  --evidence experiments/rge_peft/evidence/m2_muon_unit_cpu.json

python scripts/verify/verify_rge_optimizer_pair.py \
  --evidence experiments/rge_peft/evidence/m2_pair_protocol_cpu.json

python -m py_compile \
  scripts/train/optim/muon.py \
  scripts/train/optim/factory.py \
  scripts/experiment/rge_optimizer_pair.py \
  scripts/verify/verify_rge_muon.py \
  scripts/verify/verify_rge_optimizer_pair.py
```

CPU proofs are implementation evidence only. The GPU integration smoke and
registered O-A/O-M runs remain future bounded actions after M1 authorization.
