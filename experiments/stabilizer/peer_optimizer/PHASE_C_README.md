# PCSM Phase C — real-gradient causal screen

This directory-level experiment answers one question before full SFT:

> Does intact coordinate structure from real `mlp.down_proj` peers improve a
> one-step held-out update relative to the same peers with coordinate identity destroyed?

The existing synthetic study proves mechanism feasibility only. Phase C is the
first test on a real Rineya causal-language-model gradient.

## Pre-registered design

Primary module family: `mlp.down_proj` across all retained transformer layers.

Primary budget: `0.1%` of every candidate matrix.

Partitions are grouped by `metadata.template_family`:

- `META_TRAIN`: generator calibration only.
- `MASK_CAL`: real task gradient and fixed-plan construction only.
- `ONE_STEP_DEV`: held-out pre/post one-step loss only.
- `SEALED`: never tokenized or used for model computation in Phase C.

Exact normalized-prompt duplicates crossing partitions invalidate preparation.

Required arms:

1. seeded random support;
2. target gradient top-K;
3. target-only structural diagnostic;
4. calibrated PCSM with independently coordinate-shuffled peers;
5. real-peer bootstrap PCSM;
6. calibrated real-peer PCSM;
7. calibrated PCSM with intact peers assigned to incorrect layer identities.

The calibrated real, coordinate-shuffled, and layer-shuffled arms share the
same generator checkpoint. The control differs only in the peer transformation.

## Important implementation detail

The measured update uses the exact *empty-state first step* of
`SparseNativeAdamW` without allocating dense Adam state. For selected coordinate
`i`, zero weight decay gives

```math
\Delta w_i
=
-\eta A_i\frac{g_i}{|g_i|+\epsilon}.
```

This is exactly the first AdamW step after bias correction. It also preserves
the PCSM correction that `A_i` scales the realized adaptive update rather than
being normalized away inside Adam moments.

Only selected coordinates are written, so the evidence runner does not need to
materialize persistent dense masks, scales, or optimizer states for all 28
matrices. Full score/property maps remain transient during plan generation.

## Coordinate-shuffle control

Each non-target peer receives deterministic independent row and column
permutations. The same permutations are applied to its weight and optional
gradient.

This preserves:

- the exact peer value multiset;
- the peer's weight/gradient coordinate pairing;
- matrix shape;
- row-energy and column-energy distributions;
- global descriptor statistics based on those distributions.

It destroys correspondence between peer coordinates and the target coordinate
system. The target itself is never shuffled.

## Layer-identity control

All peer tensors remain intact, but peer contents are deranged among the
non-target layer identities. The destination depth metadata stays fixed.

This asks whether *which coworker* supplied the information matters, separately
from whether coordinate structure matters.

## Generator calibration

`META_TRAIN` performs a small first-order meta-calibration. For each meta step:

1. choose one target `down_proj`;
2. compute a real inner task gradient;
3. let PCSM generate an exact-forward STE support and step scale;
4. form the differentiable first AdamW update for that one target matrix;
5. evaluate an independent `META_TRAIN` outer example through
   `torch.func.functional_call`;
6. backpropagate only into the PCSM generator.

No `MASK_CAL`, `ONE_STEP_DEV`, or `SEALED` example participates in generator
training.

## Running

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\experiments\stabilizer\peer_optimizer\run_phase_c.ps1
```

or:

```bash
bash experiments/stabilizer/peer_optimizer/run_phase_c.sh
```

The wrappers execute:

```text
prepare -> run --resume -> verify
```

The audited v3 source JSONL is intentionally ignored by Git because it is
deterministically regenerable. Rebuild the corpus if that local file is absent.
Both source SHA-256 values are pinned in the config and a mismatch fails closed.

The local checkpoint must contain `.safetensors` weight shards under
`artifacts/rineya-core-en32k-28L`; the runner will not silently substitute a
remote model.

## Decision gate

For each seed define

```math
d_s=L^{shuffle}_{post,s}-L^{real}_{post,s}.
```

Positive is beneficial to real peer coordinates.

Advance only if, across all five pre-registered seeds:

- real calibrated peers beat coordinate-shuffled peers in at least 4/5 seeds;
- the 95% paired t interval for `d_s` lies strictly above zero;
- real calibrated peers also beat the layer-identity control in at least 4/5;
- that paired 95% interval also lies strictly above zero;
- every required arm is valid and has the exact same selected-count signature;
- every arm begins from the byte-identical candidate checkpoint snapshot.

A pass means **only** `ADVANCE_TO_MATCHED_SFT`. It is not yet a Rineya quality
claim.

## Smallest decision-ready result package

`runs/pcsm_phase_c/down_proj_0p1_v1/decision_bundle/` contains:

- `DECISION.md`
- `SUMMARY.json`
- `TRIALS.jsonl`
- `RUN_MANIFEST.json`
- `SPLIT_AUDIT.json`
- `PLAN_MANIFESTS.jsonl`
- `GENERATOR_MANIFESTS.jsonl`
- `ARTIFACT_HASHES.json`

Raw prompts and `SEALED` examples are not copied into the decision bundle.

Per-seed compact support indices/scales, generator checkpoints, calibration
gradient hashes, and support-overlap diagnostics remain in the parent run
directory.

## Mechanistic diagnostics

The runner records Jaccard overlap for:

- real PCSM vs gradient top-K;
- real PCSM vs coordinate-shuffled PCSM;
- real PCSM vs target-only support.

A real-peer win with low overlap against gradient top-K is stronger evidence
that PCSM found a genuinely different plasticity subset rather than simply
reimplementing large-gradient selection.

## What to do after the result

- `ADVANCE_TO_MATCHED_SFT`: run a short matched SFT pilot before a full run.
- `DO_NOT_ADVANCE`: preserve the negative result; inspect whether bootstrap,
  target-only, or gradient top-K won before changing the hypothesis.
- `INVALID`: repair only the execution/provenance defect and rerun the same
  preregistered design.
- If one-step benefit later disappears during SFT, the next PCSM variant to test
  is bounded periodic support refresh, with the union of touched coordinates
  explicitly controlled.
