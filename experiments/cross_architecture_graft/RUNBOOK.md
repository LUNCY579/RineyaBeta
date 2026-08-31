# Cross-Architecture Graft v1 — Runbook

This runbook operationalizes `EXPERIMENT_PROTOCOL.md`. It intentionally stops
before sealed or showcase evaluation.

## 1. Checkout and environment

```bash
git fetch origin
git checkout research/cross-architecture-graft-v1
git status --short
```

Run commands from the repository root. The experiment adds no dependency beyond
the repository's existing Python/PyTorch/Transformers environment.

For strict provenance, the tree should be clean when real evidence is produced.
The manifest records commit, branch, dirty state, Python, PyTorch, CUDA, model
hashes, data hashes, and the complete canonical config snapshot.

## 2. Verify the implementation first

Linux/macOS/Git Bash:

```bash
bash experiments/cross_architecture_graft/run_checks.sh
```

PowerShell:

```powershell
./experiments/cross_architecture_graft/run_checks.ps1
```

Expected state:

- all unit/integration tests pass;
- all three real-arm plans pass strict config validation;
- zero gate is exact;
- primary/control parameter budgets are within 1%;
- synthetic smoke returns `INFRASTRUCTURE_PASS_ONLY`;
- evidence verifier returns `PASS`;
- no result claims Rineya quality.

Stop on any failure.

## 3. Select an explicitly unsealed corpus

Use a training-derived corpus or a separately designated open development
source. Do not use the historical long-context demo, a showcase pack, a hidden
holdout, or any sealed path.

Inspect field coverage:

```bash
python -m experiments.cross_architecture_graft.src.inspect_data \
  --input path/to/open_cag_mix.jsonl \
  --out artifacts/cross_architecture_graft/data/inspection.json
```

Proceed only when the status is `READY_FOR_GROUPED_SPLIT` and at least one
provenance field is present on every row with enough distinct groups.

A blocked inspection is useful evidence. Repair the data-generation metadata;
do not manufacture a row-level group ID or fall back to a random split.

## 4. Freeze the grouped split

Choose the provenance field precedence from the inspection result. Example:

```bash
python -m experiments.cross_architecture_graft.src.prepare_split \
  --input path/to/open_cag_mix.jsonl \
  --train-out artifacts/cross_architecture_graft/data/unsealed_train.jsonl \
  --dev-out artifacts/cross_architecture_graft/data/unsealed_dev.jsonl \
  --manifest-out artifacts/cross_architecture_graft/data/unsealed_split_manifest.json \
  --group-fields template_id,generator_id,group_id,cag_group_id,source_family \
  --dev-fraction 0.10 \
  --seed 20260831
```

Review the manifest before running models:

- `group_overlap` must equal `0`;
- both arms must have multiple groups;
- target and preservation categories must be represented;
- hashes and row counts must match the intended source;
- the exact split seed and field precedence must be acceptable.

Do not reroll the split seed because the category balance looks inconvenient
after observing model outcomes. Any pre-run balance correction must be documented
and the protocol/config version updated before training.

## 5. Materialize all matched configs together

```bash
python -m experiments.cross_architecture_graft.src.materialize_campaign \
  --model-path artifacts/rineya-core-en32k-28L \
  --train-mix artifacts/cross_architecture_graft/data/unsealed_train.jsonl \
  --dev-mix artifacts/cross_architecture_graft/data/unsealed_dev.jsonl \
  --split-manifest artifacts/cross_architecture_graft/data/unsealed_split_manifest.json \
  --output-dir artifacts/cross_architecture_graft/campaign_v1
```

This writes 15 configs:

```text
5 seeds × {latent_kv, nope_gqa, swiglu}
```

and a `campaign_manifest.json` whose contract hash excludes only the intended
arm-specific graft definition, output path, notes, and experiment ID.

Do not hand-edit individual arm configs. Regenerate the whole campaign if a
shared field must change.

For a tiny operational rehearsal, add:

```bash
--seeds 20260831 --max-train-rows 32 --max-dev-rows 32
```

That rehearsal is not one of the five scientific runs unless it uses the frozen
full contract.

## 6. Run the arms

Run one config at a time to keep failures attributable and peak-memory records
clean:

```bash
python -m experiments.cross_architecture_graft.src.runner \
  --config artifacts/cross_architecture_graft/campaign_v1/configs/20260831/latent_kv.json \
  --command real
```

Repeat for `nope_gqa` and `swiglu`, then for all five seeds.

Each run performs, in order:

1. strict config validation and closed-evaluation guard;
2. split-manifest and content-hash verification;
3. local-only tokenizer/model load;
4. baseline unsealed DEV NLL;
5. deterministic multi-row host logit probe;
6. host freeze and graft attachment;
7. bit-exact zero-gate logit comparison;
8. trainability-boundary audit;
9. graft-only training;
10. frozen-parameter mutation audit;
11. post-training unsealed DEV NLL by category;
12. peak-memory and duration capture;
13. model/data/config/runtime evidence manifest;
14. graft-only export pack.

Expected output per run:

```text
runs/<seed>/<arm>/
├── evidence_manifest.json
└── graft_pack/
    ├── graft_manifest.json
    └── graft_state.pt
```

No host weights are duplicated in the graft pack.

## 7. Verify each run immediately

```bash
python -m experiments.cross_architecture_graft.src.verify_evidence \
  artifacts/cross_architecture_graft/campaign_v1/runs/20260831/latent_kv/evidence_manifest.json
```

Reject a run if any of these are false:

- `zero_gate_bit_exact`
- `only_graft_parameters_trainable`
- `frozen_parameters_unchanged`
- `sealed_evaluation_opened == false`

Also reject a run if its source tree is dirty without a documented reason.

## 8. Compare only matched same-seed runs

```bash
python -m experiments.cross_architecture_graft.src.compare_manifests \
  artifacts/cross_architecture_graft/campaign_v1/runs/20260831/latent_kv/evidence_manifest.json \
  artifacts/cross_architecture_graft/campaign_v1/runs/20260831/nope_gqa/evidence_manifest.json \
  artifacts/cross_architecture_graft/campaign_v1/runs/20260831/swiglu/evidence_manifest.json \
  --reference-id cag-v1-latent_kv-seed-20260831 \
  --out artifacts/cross_architecture_graft/campaign_v1/runs/20260831/comparison.json
```

The comparator refuses:

- different seeds;
- different host hashes;
- different train/dev/split hashes;
- different training, host, evaluation, or memory contracts;
- invalid evidence payload hashes;
- failed invariants;
- differing host baseline NLL.

`COMPARABILITY_PASS` means only that the NLL screening records can be placed next
to each other. It is not a scientific win.

## 9. Run the registered open semantic DEV gates

This branch intentionally does not guess or duplicate the repository's semantic
evaluator. Before P4, register the exact open-DEV evaluator with:

- case file hashes;
- scorer implementation hash;
- prompt/chat template;
- decoding parameters;
- per-case output schema;
- target versus preservation membership;
- bootstrap procedure.

Run A0–A3 under identical evaluator settings. Retain per-case outputs. Apply the
criteria in `EXPERIMENT_PROTOCOL.md`; do not select only favorable categories.

The historical 64-row long-context evaluator is outside this step and stays
closed until canonical registration.

## 10. Failure matrix

| Failure | Action |
|---|---|
| Config rejects a field | Fix the shared plan or document a protocol revision; never bypass validation. |
| `BLOCKED_NO_COMPLETE_GROUP_KEY` | Restore generator/template provenance. Do not row-split. |
| Split hash mismatch | Stop; identify data drift and rematerialize all configs. |
| Zero-gate logit mismatch | Stop; no optimizer step is valid. |
| Non-graft parameter trainable | Stop; correct attachment/trainability logic and rerun all arms. |
| Frozen parameter version changed | Mark run invalid and investigate in-place mutation. |
| OOM | Change the same operational contract for every arm and restart the entire campaign. |
| Non-finite loss | Mark the arm failed; do not skip the batch or silently lower its loss. |
| One arm interrupted | Restart that arm from step zero. V1 does not claim exact resume. |
| Source code changes mid-campaign | Restart all arms unless the change is evidence-only and source provenance remains explicit. |
| NLL improves but semantic DEV does not | NO-GO; NLL was only a health screen. |
| Primary wins one control only | NO-GO under the two-control rule. |

## 11. What to commit after a real campaign

Commit small, reviewable evidence rather than large model artifacts:

- campaign manifest;
- each evidence manifest;
- same-seed comparison summaries;
- open semantic DEV per-case results and aggregate report;
- a result interpretation document with explicit claim boundary.

Do not commit:

- local base model weights;
- tokenizer caches;
- optimizer state;
- full graft binaries unless the repository's artifact policy explicitly calls
  for them;
- sealed/showcase data or outputs.

A final result commit should state one of:

```text
SUPPORTED under preregistered v1 gates
NO-GO under preregistered v1 gates
INCONCLUSIVE because <specific invalidating condition>
```

It must not use a broader conclusion than the evidence supports.
