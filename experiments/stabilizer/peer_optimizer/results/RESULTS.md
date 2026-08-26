# PCSM implementation and feasibility results

**Date:** 2026-08-26  
**Scope:** software validation plus controlled synthetic feasibility only  
**Real Rineya training performed:** no

## 1. Development environment

The committed results were generated in the available CPU development container:

```text
Python 3.13.5
PyTorch 2.10.0+cpu
```

The Qwen/Rineya integration path is implemented and unit-tested with a structurally equivalent fake `model.model.layers[*].mlp.down_proj` hierarchy. It has not yet loaded the full private checkpoint in this environment.

## 2. Software qualification

Command:

```bash
python -m compileall src tests
python -m unittest discover -s tests -p 'test_*.py' -v
```

Result:

```text
21 tests passed
```

Covered invariants:

- descriptor dimensionality and finite values;
- property-map shape projection;
- exact integer and fractional top-`K` budgets;
- exact binary forward mask with differentiable surrogate backward;
- zero self-advice;
- zero attention to incompatible module roles;
- task gradient owns update direction;
- peer-coordinate changes alter the generated score matrix;
- meta-gradients reach generator parameters;
- optional structural regularizer is finite;
- support audit catches undeclared changes;
- missing compatible peers fail closed;
- plan build, save, load, hash, apply, and train workflow;
- missing calibration gradients fail closed;
- bf16 weights use fp32 Adam moments;
- momentum and decoupled weight decay cannot move frozen coordinates;
- malformed masks and shapes are rejected;
- package-style and direct-script imports both work;
- peer step scales proportionally control the realized Adam update;
- negative or non-finite step scales are rejected.

### Defects found by the tests and mathematical audit

The first implementation used the usual algebraic straight-through expression

```math
M_{hard}+M_{soft}-\operatorname{stopgrad}(M_{soft}).
```

A strict bitwise test showed that floating-point cancellation could make the returned tensor differ minutely from the exact binary mask. The implementation now uses a custom autograd function: forward returns a cloned hard mask exactly; backward routes the derivative through the soft surrogate.

The first AdamW implementation also multiplied `A_t` into the raw gradient before first- and second-moment accumulation. Because Adam approximately normalizes a positive constant out through `m / sqrt(v)`, that did not reliably implement "how strongly this coordinate should learn." The corrected optimizer accumulates moments from `M_t * G_t` and multiplies both the realized adaptive update and selected-coordinate weight decay by `A_t`. A regression test verifies a 0.25 versus 1.0 scale produces the corresponding first-step ratio.

## 3. Full one-seed controlled run

Command:

```bash
python src/feasibility.py --out results/feasibility.json
```

Configuration:

```text
seed: 20260826
layers: 6
matrix shape: 16 x 16
support budget: 24 / 256 = 9.375%
gradient noise: 1.35
peer noise: 0.20
meta-training: 140 steps x 6 tasks
held-out evaluation: 96 tasks
```

The task constructs a sparse optimum from peer structural maps and corrupts the target-only gradient. This makes the hidden support known and lets the test distinguish implementation success from vague benchmark movement.

| arm | mean true loss | support recall | mean improvement over no update |
|---|---:|---:|---:|
| no update | 0.030812 | 0.0000 | +0.000000 |
| random | 0.039606 | 0.0946 | -0.008794 |
| target only | 0.057652 | 0.1220 | -0.026840 |
| gradient top-`K` | 0.071099 | 0.1380 | -0.040287 |
| coordinate-shuffled peers | 0.034193 | 0.0985 | -0.003381 |
| **peer-conditioned** | **0.022548** | **0.9405** | **+0.008264** |
| dense noisy update | 0.127679 | 1.0000 | -0.096867 |
| oracle | 0.000000 | 1.0000 | +0.030812 |

All declared checks passed.

## 4. Five-seed stress repetition

Command:

```bash
python src/stress_feasibility.py \
  --train-steps 40 \
  --eval-tasks 32 \
  --out results/feasibility_multiseed.json
```

Seeds:

```text
20260822, 20260823, 20260824, 20260825, 20260826
```

| arm | loss mean | loss SD | recall mean | recall SD |
|---|---:|---:|---:|---:|
| no update | 0.030312 | 0.000344 | 0.000000 | 0.000000 |
| random | 0.039252 | 0.000392 | 0.091146 | 0.006484 |
| target only | 0.056830 | 0.000836 | 0.120052 | 0.011802 |
| gradient top-`K` | 0.069804 | 0.000757 | 0.142969 | 0.008534 |
| coordinate-shuffled peers | 0.035245 | 0.001838 | 0.102344 | 0.009903 |
| **peer-conditioned** | **0.022917** | **0.000488** | **0.932552** | **0.018714** |
| dense noisy update | 0.127059 | 0.001077 | 1.000000 | 0.000000 |
| oracle | 0.000000 | 0.000000 | 1.000000 | 0.000000 |

Check pass counts:

```text
peer beats random loss:          5 / 5
peer beats gradient top-K loss:  5 / 5
peer beats shuffled-peer loss:   5 / 5
peer recall beats random:        5 / 5
meta loss decreases:             5 / 5
```

## 5. Interpretation

The result supports exactly one claim:

> When a sparse optimum genuinely depends on peer matrix properties, the implemented PCSM pathway can learn that relationship, generate exact support, survive peer-specific ablations, and apply a lower-loss native update without changing undeclared coordinates.

It does not support any of the following claims:

- real Rineya layers necessarily contain this signal;
- PCSM improves language-model quality;
- PCSM is faster than LoRA or dense fine-tuning;
- the learned attention graph is a causal explanation;
- raw cross-layer coordinates are aligned;
- the method is novel relative to all prior learned-optimizer/hypernetwork work.

The next decisive result is Phase C in `EXPERIMENT_PROTOCOL.md`: real peer coordinates versus coordinate-shuffled peers on held-out one-step loss, before spending compute on full SFT.
