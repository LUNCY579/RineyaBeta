# DAST: Depth-Aligned Semantic Transitions

> **Status: CLOSED — NEGATIVE (EXPLORATORY DEV).** DAST V1 does not support the
> hypothesis that aligning cumulative semantic reasoning states to increasing
> Transformer depth improves held-out reasoning. The sealed TEST split remains
> closed. Operator and causal-feedback phases remain gated. See
> `docs/FINDINGS_G3_SEEDS.md` and `PROGRESS.md`.

DAST tests whether Transformer depth can be given an explicit semantic coordinate
system for reasoning. A normal causal-LM objective remains intact, while one
**shared** projector reads prompt-only hidden states at ordered depth checkpoints:

\[
h^{(d_1)},h^{(d_2)},\ldots,h^{(d_K)}
\xrightarrow{\;P_\phi\;}
z_1,z_2,\ldots,z_K.
\]

For Qwen3-0.6B's 28 layers and the default quarter-depth schedule, the checkpoints
are layers **7, 14, 21, and 28**. The main arm aligns them to cumulative compact
reasoning states. Matched controls assign the same states in shuffled, reversed,
or final-state-repeated order. A literal-step arm tests the original
"sentence/reasoning-step per depth region" proposal directly.

The operator phase additionally learns a reusable low-rank transformation bank:

\[
T_\alpha(z)=\operatorname{norm}\left(z+\sum_j\alpha_j(zA_j)B_j\right).
\]

This package is additive. It does not replace or modify the existing
`Rineya_state_experiment_bundle`. That earlier lab asks whether **explicit textual
compact states** are sufficient between separate generated transitions. DAST asks
whether **hidden states inside one forward pass** can be aligned to semantic
reasoning progress.

## What is implemented

- strict YAML configs with inheritance and unknown-key rejection;
- deterministic mapping of variable-length reasoning traces to arbitrary model depth;
- full-CoT LM targets held constant across all primary arms;
- prompt-only hidden-state pooling, excluding assistant and padding tokens;
- one shared semantic projector across every depth checkpoint;
- frozen text-target geometry derived from the base input embedding plus a fixed projection;
- ordered, shuffled, reversed, final-repeated, and literal-step controls;
- low-rank operator dictionary, router, transition, consistency, diversity, and load diagnostics;
- optional causal semantic feedback at the last prompt token, with state interventions;
- LoRA/QLoRA-capable Hugging Face path with local-only loading by default;
- strict TEST sealing tied to a frozen preregistration SHA-256;
- deterministic manifests, dataset hashes, metrics, checkpoints, and paired comparison tools;
- offline toy model, fixture, unit/integration tests, Linux/Windows launchers, and CI.

## Installation

```bash
cd experiments/depth_aligned_semantic_transitions
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
pip install -e ".[all]"
```

The default real-model configs set `local_files_only: true` and
`trust_remote_code: false`. Put `Qwen/Qwen3-0.6B` in the Hugging Face cache or
change `model.name_or_path` to an audited local Rineya/R2 checkpoint.

## Qualification

The authoritative gate is **local**. GitHub Actions is retired to
`workflow_dispatch` only, because every run on this branch failed before a runner
was assigned — 0 steps executed, `runner_id: 0`, a billing annotation — and there is
no budget to resolve it.

```bash
python scripts/qualify.py            # writes qualification/latest.json
python scripts/qualify.py --fast     # compileall + ruff + pytest only
```

The gate runs the suite with the accelerator **both** visible and hidden. That is
not redundancy: a CPU-only hosted runner structurally could not catch the CPU/CUDA
defect that made an earlier "pytest PASS" record reproducible only on a machine
without a GPU. The one thing local qualification does not establish is that the
workflow file itself is correct.

## Sixty-second engineering check

```bash
# Linux/macOS
scripts/run_smoke.sh

# Windows PowerShell
scripts\run_smoke.ps1
```

The smoke path validates the fixture, performs joint LM/state/operator training on
an offline toy causal model, verifies the shared projector updates, and confirms
that a semantic feedback intervention changes downstream logits.

## Inspect the exact depth assignment

```bash
dast inspect-alignment --config configs/state_ordered.yaml --limit 3
dast inspect-alignment --config configs/state_shuffled.yaml --limit 3
```

No target assignment is implicit. The command prints canonical targets, training
targets, source indices, and the deterministic permutation for every example.

## First real DEV ladder

V1 on this ladder is **closed negative**. Do not re-run these arms to hunt for
significance. A V2 training matrix is allowed only after
`docs/PREREGISTRATION_V2.md` is signed. Read `docs/PREREGISTRATION.md` and
`docs/RUNBOOK.md` before any remaining engineering work.

```bash
python -m dast.cli train --config configs/cot_baseline.yaml
python -m dast.cli train --config configs/state_ordered.yaml
python -m dast.cli train --config configs/state_shuffled.yaml
python -m dast.cli train --config configs/state_reverse.yaml
python -m dast.cli train --config configs/state_final_repeated.yaml
python -m dast.cli train --config configs/literal_ordered.yaml
```

Run the same frozen matrix for seeds 17, 29, and 43 by copying configs and changing
only `experiment.seed`, `experiment.name`, and `experiment.output_dir`. Those
commands are the historical V1 ladder, retained for provenance. Operator arms
remain gated and must not begin merely because the software runs.

## Primary comparisons

| Comparison | What it isolates |
|---|---|
| ordered vs shuffled | reasoning order aligned to depth, holding target multiset and loss count fixed |
| ordered vs reversed | whether forward reasoning order is structurally compatible with increasing depth |
| ordered vs final-repeated | progressive semantic milestones vs generic deep supervision |
| ordered semantic vs literal ordered | semantic state transitions vs literal textual steps |
| ordered vs CoT baseline | contextual benefit/cost, not a parameter-matched causal comparison |
| operator ordered vs operator shuffled | transferable transitions under matched operator capacity |

## Important safety and validity boundaries

1. Assistant targets are teacher-forced, but semantic pooling uses only prompt
   positions. A test changes every assistant hidden state by a huge amount and
   verifies that the pooled state is unchanged.
2. The shared target encoder is frozen. DAST refuses a trainable input embedding by
   default because moving targets would make the objective ill-defined.
3. TEST is inaccessible unless `allow_sealed_test`, `preregistration_frozen`, and a
   64-character preregistration hash are all present.
4. Feedback mode is incompatible with gradient checkpointing and cached generation
   until those paths receive separate qualification.
5. A decodable state is not automatically causal. Feedback interventions are a
   later gate, not evidence retroactively inferred from probe accuracy.
6. A toy smoke, green CI, or low training loss is not evidence that Rineya reasons
   better.

## Layout

```text
configs/       frozen arm definitions and local model defaults
docs/          algorithm, evidence standard, V1/V2 preregistration, findings, runbook
fixtures/      tiny offline engineering dataset
scripts/       Linux/macOS and Windows launchers; run and training-dynamics summarizers
src/dast/      data, alignment, targets, model, operators, losses, training, evaluation
tests/         unit and end-to-end offline qualification
PROGRESS.md    branch-local status and claim boundary
```

The program-level single source of truth is
`../../docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md`.
