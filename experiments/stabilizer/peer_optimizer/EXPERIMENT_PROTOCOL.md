# PCSM real-model experiment protocol

**Status:** proposed protocol. Software qualification and controlled synthetic feasibility are complete. No Rineya fine-tuning result is claimed.

## 1. Hypothesis and decisive comparison

At an exactly matched native-coordinate budget, peer-conditioned support may improve adaptation or capability preservation beyond random support, target-gradient top-`K`, target-only features, and matched LoRA.

The coworker-specific causal comparison is:

```math
\text{real peer coordinates}\quad\text{versus}\quad\text{coordinate-shuffled peers}.
```

Both arms must use the same generator architecture, task gradients, data splits, support count, optimizer, step-scale bounds, and tuning budget. If they tie, there is no evidence that peer coordinate structure matters, even if both beat gradient top-`K`.

## 2. Qualification gates already implemented

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\experiments\stabilizer\peer_optimizer\run_checks.ps1
```

The suite must establish all of the following before a model run:

- exact binary top-`K` masks;
- zero self-advice and zero attention to incompatible module roles;
- differentiable meta-calibration through an exact-forward straight-through operator;
- task-gradient ownership of update direction;
- fp32 Adam states for bf16/fp16 native weights;
- no movement outside declared support under momentum or decoupled weight decay;
- peer scales control the realized AdamW step, rather than being normalized away inside Adam moments;
- plan build, save, load, hash, apply, and post-training support audit;
- failure when calibration gradients or compatible peers are missing.

The committed controlled study includes no update, random, target-only, gradient top-`K`, coordinate-shuffled peers, real peers, dense noisy update, and oracle controls. It verifies mechanism feasibility only because its hidden optimum is deliberately constructed from peer properties.

## 3. Phase C: real-gradient predictive test without SFT

This is the cheapest honest test of whether real Rineya layers contain useful coworker information.

### Leakage-resistant splits

Create grouped splits with no source-family overlap:

- `META_TRAIN`: fit generator parameters;
- `MASK_CAL`: compute calibration gradients and fixed masks;
- `ONE_STEP_DEV`: measure post-update held-out loss;
- `SEALED`: untouched until every choice is frozen.

No `ONE_STEP_DEV` or `SEALED` prompt may contribute activations, gradients, descriptors, masks, scales, or hyperparameter selection.

### First module family

Begin with one semantically aligned family:

```text
mlp.down_proj across every retained transformer layer
```

Only after a stable result expand independently to `mlp.up_proj`, `mlp.gate_proj`, and the attention projection families. Do not pool unlike module roles in the first experiment.

### Budgets

Use exact fractions such as:

```text
0.01%, 0.05%, 0.10%, 0.50%, 1.00%
```

Resolve each fraction once to an integer `K`, then record both `K` per matrix and the global selected count. Every sparse-native arm must have identical support counts.

### Required arms

| arm | support source | step scale |
|---|---|---|
| random | fixed seeded random | common scalar |
| gradient top-`K` | `E[|g|]` from `MASK_CAL` | common scalar |
| target only | target weight/gradient features | bounded head |
| shuffled peers | PCSM with peer coordinates independently permuted | bounded head |
| real peers, bootstrap | PCSM initial policy | bounded head |
| real peers, calibrated | generator trained only on `META_TRAIN` | bounded head |

Restart from the same checkpoint before every one-step arm. Apply an actual functional or copied-model update and score held-out loss; score correlation alone is insufficient.

### Advancement gate

Advance to full SFT only when, over at least five seeds:

1. real peers beat shuffled peers on paired one-step held-out loss in at least four seeds;
2. the aggregate paired confidence interval excludes zero in the beneficial direction;
3. real peers receive no larger support, wider learning-rate search, or extra data;
4. the result survives layer-identity and coordinate-shuffle controls;
5. the entire mask-generation path is isolated from evaluation leakage.

## 4. Phase D: matched SFT

For every seed and arm:

1. load and hash the exact base checkpoint;
2. run the frozen `MASK_CAL` backward measurement without an optimizer step;
3. save the exact plan, support hashes, attention, feature weights, and scales;
4. reload the byte-identical base checkpoint;
5. apply the saved plan and freeze every unplanned parameter;
6. train with identical data order, steps, effective batch size, schedule, precision, clipping, and objective;
7. audit every final native delta against the declared mask.

Required comparison arms are frozen base, random sparse native, gradient top-`K` sparse native, target-only sparse native, shuffled-peer PCSM, real-peer PCSM bootstrap, real-peer PCSM calibrated, exactly parameter-matched LoRA, and an appropriately capacity-matched dense native control. A full dense upper bound may be reported only as explicitly unmatched.

LoRA contributes

```math
r(d_{in}+d_{out})
```

trainable parameters per matrix, while sparse native tuning contributes `K`. Match global trainable counts and report optimizer-state bytes, not merely nominal percentages.

## 5. Causal ablations

The minimum ablation set is:

1. coordinate shuffle: preserve values, destroy coordinate structure;
2. layer shuffle: assign intact peers to incorrect depths;
3. module-role shuffle: deliberately mismatch matrix roles;
4. uniform peer attention;
5. nearest-layer-only communication;
6. target descriptor only;
7. task-gradient feature only;
8. no step-scale head;
9. random generator weights;
10. optional structural regularizer off versus on.

These distinguish useful peer content from generic hypernetwork capacity, local depth bias, support quality, and optimizer conditioning.

## 6. Evaluation and invalidation

Preserve task score, ordinary-chat preservation, long-context score where applicable, abstention behavior, train/dev loss, selected count and union, delta norm inside/outside support, peak allocated/reserved VRAM, wall-clock, tokens/second, and all checkpoint/plan hashes.

A run is invalid when any undeclared coordinate moves, support counts differ across matched arms, calibration begins from a different checkpoint, evaluation data enters mask generation, or controls receive asymmetric tuning.

Do not claim speed merely from sparse parameter count. Dense PyTorch backward and matrix kernels may dominate runtime. The current claim concerns exact plasticity and optimizer-state control, not sparse-kernel acceleration.

## 7. Failure interpretations

| result | interpretation |
|---|---|
| real peers = shuffled peers | no evidence for coordinate-level coworker information |
| real peers > shuffled but = target only | peer pathway adds capacity, not useful peer content |
| one-step gain disappears in SFT | local predictive signal does not survive optimization dynamics |
| adaptation rises while preservation falls | selected coordinates are plastic but destructive |
| only step scale helps | benefit is optimizer conditioning, not support generation |
| only neighboring layers matter | simplify to a local peer graph |
| no wall-clock gain | expected with dense kernels; retain only quality/plasticity claims |

A clean negative result is valid and should be preserved.

## 8. Evidence ledger

For every run save the base commit/branch, checkpoint hashes, source and split manifests, generator checkpoint/configuration, calibration seed and command, exact masks and SHA-256 hashes, scales, attention, feature weights, resolved `K`, training command/environment, optimizer state precision, per-step metrics, final delta audit, evaluation outputs, and GPU/driver/PyTorch/Transformers versions.

The plan produced by `rineya_integration.py` is the authoritative fixed-support artifact; its `manifest()` output belongs in the experiment ledger.
