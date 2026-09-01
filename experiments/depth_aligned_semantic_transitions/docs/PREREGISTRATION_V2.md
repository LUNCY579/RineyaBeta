# DAST V2 preregistration — DRAFT, PARKED. TEST CLOSED.

**PARKED.** The nine-run `v2_s{17,29,43}_*` training matrix is **not authorized**.
H2 (optimization conflict) remains a hypothesis in
[`DAST_V2_HYPOTHESIS_TREE.md`](DAST_V2_HYPOTHESIS_TREE.md). Do not sign this file
or launch those configs unless Stage B/C diagnosis selects H2 as the successor.
This file is not frozen. Its hash cannot open TEST.

**Status:** `DRAFT`. This file is **not frozen**. Its hash cannot open the sealed
TEST split, and **no V2 training run may start**, until Stage B/C selects H2
*and* the freeze block in §10 is completed and approved.

V2 is a **new** protocol. It does not inherit confirmatory status from V1, and it
does not reopen V1.

## 0. What this experiment is not

V1 is **CLOSED — NEGATIVE (EXPLORATORY DEV)**
(`FINDINGS_G3_SEEDS.md`, `PROGRESS.md`). V2 **does not** retest whether forward
depth alignment improves held-out reasoning. That hypothesis is closed for the
tested formulation.

Generated-answer accuracy is recorded only as a **consistency check**. A V2
accuracy result, in either direction, cannot reopen G2/G3, cannot license
operators or causal feedback, and cannot open TEST.

## 1. Primary question

Does **forward** state supervision produce greater **optimization conflict** with
the language-model objective than matched **reverse** or **flat-target**
supervision?

Conflict is measured from training and teacher-forced DEV **losses**, not from
capability.

This question is licensed by V1: trained `L_state` was higher for ordered than
for reverse in every seed (0.2629 vs 0.2534 seed-mean), while a frozen-backbone
probe mildly prefers forward order. That pattern is an optimization claim, not a
capability claim, and V1 did not preregister it.

## 2. Base checkpoint and protocol constants

Inherited from V1 without change:

| | |
|---|---|
| model | `Qwen/Qwen3-0.6B` |
| architecture | 28 decoder layers; resolved checkpoints **7 / 14 / 21 / 27** with `checkpoint_readout: final_norm` |
| LoRA | r=16, alpha=32, dropout 0.05, on q/k/v/o/gate/up/down |
| batch | 2, gradient accumulation 8, **250** optimizer steps |
| learning rate | 2e-4, auxiliary 2e-4, warmup ratio 0.03, cosine |
| target geometry | `target_centering: within_problem` |
| state / contrastive weight | 1.0 / 0.10 |
| data | same train/DEV files and hashes as `PREREGISTRATION.md` §3; 4,000 train / 800 DEV |
| seeds | **17, 29, 43** |
| scorer | frozen at `f160f38` (consistency-check only; not a V2 gate) |

A run that changes any of the above is not a V2 run.

## 3. Arms

Three matched arms. Auxiliary target **count**, projector architecture, LoRA
capacity, prompt/response tokens, source problems, and optimizer schedule are
identical.

| Arm | Config | Role |
|---|---|---|
| `state_ordered` | `configs/v2_s{seed}_state_ordered.yaml` | treatment: canonical forward assignment |
| `state_reverse` | `configs/v2_s{seed}_state_reverse.yaml` | matched reverse assignment |
| `state_final_repeated` | `configs/v2_s{seed}_state_final_repeated.yaml` | flat auxiliary target (final state at every depth) |

`cot_baseline` is **optional** and not a decisive contrast. It may be reused from
V1 artifacts for LM-only reference. It must not be retrained to chase a V2 gate.

`state_shuffled` is out of scope. V2 is not a G2 redo.

## 4. Primary and secondary outcomes

### Decisive family (pre-enumerated; Bonferroni α = 0.05/3 ≈ 0.0167 if p-values are quoted)

| ID | Contrast | Metric | Predicted direction |
|---|---|---|---|
| C1 | ordered − reverse | Final teacher-forced DEV `loss/state` (`L_state`) | ordered **higher** (harder fit) |
| C2 | ordered − final_repeated | Final teacher-forced DEV `loss/state` | ordered **higher** |
| C3 | ordered − reverse | Training `loss/lm` trapezoidal AUC over `optimizer_step`, plus the final logged `loss/lm` | ordered **higher**, or diverges earlier |

C1 and C2 together are **V2-G1**. C3 is **V2-G2**. No other contrast is a gate.

### Predeclared margins (from V1, not tuned after V2 starts)

V1 seed-mean `L_state`: ordered 0.2629, reverse 0.2534, final-repeated 0.1433.

| Contrast | Margin |
|---|---|
| C1 ordered − reverse `L_state` | **≥ 0.005** in the three-seed mean, **and** the same sign in all three seeds |
| C2 ordered − final_repeated `L_state` | **≥ 0.05** in the three-seed mean, **and** the same sign in all three seeds |
| C3 ordered − reverse `loss/lm` AUC | three-seed mean AUC(ordered) > AUC(reverse), **and** the same sign in all three seeds. No numeric pp-style margin: V1 final perplexities were 1.0076 vs 1.0075, so the signal, if any, is in the **path**. |

`L_state` is each arm's fit to **its own** targets. That is the correct comparison
for an optimization-conflict claim. Order margin and monotonicity must **not**
be used: they are computed against canonical order and penalise reverse by
construction.

### Consistency check (not a gate)

Strict greedy generated-answer accuracy, paired by `problem_id`. Expectation from
V1: ordered ≤ reverse and ordered ≤ final-repeated. A reversal does not pass V2;
a replication of the V1 deficit does not reopen V1.

## 5. Analysis

Rebuild from `runs/` with:

```bash
python scripts/summarize_training_dynamics.py --prefix v2_
python scripts/summarize_runs.py --prefix v2_   # accuracy consistency check only
```

- Group by `(arm, seed)` from `run_manifest.json`, never from the directory name.
- Training curves come from `metrics.jsonl` (`loss/lm`, `loss/state`, keyed by
  `optimizer_step`).
- Final `L_state` comes from `dev_summary.json`.
- Incomplete runs are named, not silently dropped.
- Quote seed-level values and the three-seed mean. With n = 3 seeds, a paired
  t-test is underpowered and is **not** a gate.

## 6. Gates

### V2-G1 — state-fit conflict (required)

C1 **and** C2 both pass their predeclared margins and sign pattern. If either
fails, close the optimization-conflict hypothesis as stated.

Final-repeated is structurally easier (four identical targets). C2 is therefore
a sanity check that ordered is not somehow *easier* than a flat target. The
scientifically load-bearing contrast is C1.

### V2-G2 — LM-objective conflict (secondary)

C3 passes. Failure of V2-G2 with a pass on V2-G1 means: forward state targets
are harder to fit than reverse, but that difficulty does not show up in the LM
loss path at this budget. Record that distinction; do not promote it to a
capability claim.

V2-G2 cannot rescue a failed V2-G1.

### Not gates

- Generated-answer accuracy.
- Order margin, checkpoint top-1, monotonicity.
- Any contrast against `cot_baseline`.
- `state_shuffled`.
- Operators (V1 G5) and causal feedback (V1 G6).

## 7. Exclusion and failure handling

Same as V1 `PREREGISTRATION.md` §9. A seed may be rerun only for a documented
infrastructure failure (OOM, host crash), never because `L_state` or accuracy is
unfavourable. Failed seeds are reported. Leakage or budget mismatch withdraws
the comparison.

## 8. Stop rules

- If V2-G1 fails, close the optimization-conflict hypothesis. Do not add seeds,
  schedules, or arms to recover it.
- **Do not open TEST** to resolve an ambiguous DEV result.
- **Do not run additional V1 seeds.**
- Do not start operators or causal feedback from a V2 result, including a V2
  pass. Those phases still require a **positive capability** preregistration,
  which this file is not.

## 9. Claim boundary

A V2 pass supports only:

> Under this protocol, forward state supervision is harder to optimize, and/or
> interferes more with the LM loss path, than matched reverse or flat-target
> supervision.

It does **not** support: DAST improves reasoning; transformers reason backwards;
depth is a chronological trace; V1 should be reopened; operators should start.

## 10. Freeze block — REQUIRES APPROVAL

```text
status: DRAFT
commit: PENDING
sha256_of_this_file: (compute after approval, before any V2 config references it)
frozen_at_utc: PENDING
approved_by: PENDING
```

Until this block is filled:

1. `experiment.preregistration_frozen` remains `false` and
   `preregistration_sha256` remains empty on every `configs/v2_*.yaml`.
2. No V2 training run is a qualifying run.
3. TEST stays sealed.
