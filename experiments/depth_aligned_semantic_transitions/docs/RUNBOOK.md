# DAST runbook

## 0. Claim boundary

The branch is currently **BUILT — UNRUN** on a real checkpoint. Do not edit that
status because tests pass. A real run must leave raw artifacts and a run manifest.

## 1. Environment

Recommended first real path: the same RTX 4070 Laptop / 8 GiB environment used by
other bounded Rineya experiments.

```bash
cd experiments/depth_aligned_semantic_transitions
python -m venv .venv
# activate the environment
pip install -e ".[all]"
export PYTHONHASHSEED=0
```

Windows PowerShell:

```powershell
$env:PYTHONHASHSEED = "0"
pip install -e ".[all]"
```

The default config is local-only. Confirm the model is already cached or change
`model.name_or_path` to an audited local path. Do not set `trust_remote_code: true`
merely to bypass a loading error.

## 2. Offline qualification

The authoritative gate is **local**, not GitHub Actions:

```bash
python scripts/qualify.py          # full gate
python scripts/qualify.py --fast   # skip smoke and the CPU-only pass
```

It writes a machine-readable record to `qualification/latest.json` carrying the
commit, whether the tree was dirty, host and package versions, and a per-step
result. Commit that artifact as the qualification evidence.

**Why local is the gate, and why it is stronger here.** Every Actions run on this
branch failed before a runner was assigned — 0 steps executed, `runner_id: 0` —
with the annotation "recent account payments have failed or your spending limit
needs to be increased". A red result that never ran a step carries no information
about the code, and there is no budget to resolve it. Beyond that, a hosted runner
is CPU-only, so it structurally **cannot** catch a defect that only appears when a
GPU is present — which is exactly what made the 2026-08-30 "pytest PASS (37 tests)"
record reproducible only on a machine without one. `qualify.py` therefore runs the
suite in **both** device configurations. The workflow is preserved at
`.github/workflows/dast-ci.yml` as `workflow_dispatch` only.

The one thing local qualification does **not** establish: that the workflow file
itself is correct. That stays unqualified.

The narrower smoke-only path remains available:

```bash
scripts/run_smoke.sh
```

Expected:

- data status PASS;
- all tests pass;
- finite joint loss;
- shared projector parameters change;
- depth checkpoints are 2/4/6/8 for the toy 8-layer model;
- feedback intervention changes logits;
- no real-model claim is emitted.

## 3. Validate the full dataset and assignment

```bash
dast validate-data \
  ../../Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1/data/manifests/problem_manifest.jsonl

dast inspect-alignment --config configs/state_ordered.yaml --limit 10
dast inspect-alignment --config configs/state_shuffled.yaml --limit 10
dast inspect-alignment --config configs/state_reverse.yaml --limit 10
```

Manually confirm:

- initial unsolved states are excluded;
- ordered targets progress forward;
- shuffled targets are non-identity but retain the same multiset;
- reverse targets are exactly reversed;
- variable-length trajectories record repeated source indices;
- the resolved real-model checkpoints are 7/14/21/28 for a 28-layer model.

## 4. Freeze the DEV protocol

Before any comparative real run, complete `PREREGISTRATION.md`:

- base checkpoint and revision;
- seeds (default proposal: 17, 29, 43);
- training-record count and budget matching;
- primary metric and practical margin;
- OOD/deeper-trace strata;
- phase gates;
- hardware and package versions.

Commit the frozen file and put its SHA-256 in the eventual TEST config. DEV configs
must keep `allow_sealed_test: false`.

## 5. Resource rehearsal

Start with one seed and a bounded training slice. The shipped configs use 4,000
train and 800 DEV problems.

```bash
python -m dast.cli train --config configs/cot_baseline.yaml
python -m dast.cli train --config configs/state_ordered.yaml
```

Check after both runs:

- no out-of-memory or NaN events;
- actual trainable parameter counts are identical in the backbone;
- ordered adds only the documented shared auxiliary head;
- run manifests have identical data hashes and resolved layer checkpoints;
- prompt truncation is rare and balanced;
- no TEST record appears in a manifest;
- metrics contain every batch and failed response remains in denominators.

Do not tune only the ordered arm. Any learning-rate or loss-weight adjustment must
be applied symmetrically to matched controls and recorded as exploratory.

## 6. Phase 1 DEV matrix

Run the same immutable matrix for every frozen seed:

```bash
scripts/run_dev_matrix.sh
```

The shipped launcher covers:

1. CoT baseline;
2. ordered semantic states;
3. shuffled semantic states;
4. reversed semantic states;
5. final-state repeated deep supervision;
6. literal reasoning steps.

Copy configs per seed rather than editing outputs in place. Never overwrite a run.

## 7. Phase 1 decision

Proceed only when the frozen aggregate supports all required conditions:

- ordered beats shuffled and reverse on the primary generated-answer metric;
- ordered beats final-state repeated on at least one preregistered mechanism or OOD metric;
- the effect is not driven by a single seed/category;
- standard LM quality does not cross the preservation-loss boundary;
- canonical order margin is positive and stronger than controls;
- no leakage or budget mismatch is found.

If not, record **CLOSED — NEGATIVE** for the tested V1 formulation. Do not advance to
operators because an attractive visualization is not a gate pass.

## 8. Phase 2 operator matrix

Only after Phase 1:

```bash
python -m dast.cli train --config configs/operator_ordered.yaml
python -m dast.cli train --config configs/operator_shuffled.yaml
```

Add held-out operator/content combinations before claiming transfer. Inspect:

- transition cosine;
- direct-state consistency;
- router entropy/load;
- operator diversity;
- same-operation/different-content transfer;
- novel composition performance.

Router specialization alone is not sufficient.

## 9. Phase 3 causal feedback

The config `feedback_causal_probe.yaml` disables gradient checkpointing by contract.
First run teacher-forced state interventions only. Required controls:

- correct counterfactual state swap;
- wrong-stage swap;
- same-norm random vector;
- zero state;
- unmodified forward pass.

A qualified effect is directionally appropriate answer change, not simply a
non-zero logit delta. Cached generation is deliberately unsupported in V1.

## 10. Checkpoints and recovery

Each checkpoint contains:

- DAST auxiliary state;
- PEFT adapter when supported by the base model;
- optimizer/scheduler state;
- Python/NumPy/Torch RNG state;
- file checksums;
- config hash.

The current trainer writes checkpoints at configured intervals, but automatic
resume is not promoted as evidence until a real interrupted-run equivalence check
has compared resumed and uninterrupted outputs. Treat resume as engineering beta.

## 11. TEST opening

TEST remains closed until:

1. DEV arms, seeds, weights, budgets, and metric definitions are frozen;
2. the preregistration file is committed;
3. its SHA-256 is copied into a dedicated TEST config;
4. `preregistration_frozen: true` and `allow_sealed_test: true` are explicitly set;
5. approval to open TEST is recorded.

The code rejects TEST without all three machine-readable conditions. Once opened,
do not tune any model or threshold from TEST outcomes.

## 12. Reporting

Update `PROGRESS.md` with:

- branch/commit;
- exact status vocabulary;
- commands run;
- environment;
- artifact paths and hashes;
- pass/fail gate table;
- deviations;
- next permitted phase.

Then update the repository-wide experiment audit in the same merge commit. Preserve
negative and withdrawn evidence.

Attach the `qualification/latest.json` produced on the committed tree. An artifact
whose `environment.git_dirty` is true describes no committed state and is not
evidence.
