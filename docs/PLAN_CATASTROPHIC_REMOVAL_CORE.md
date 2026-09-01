# Catastrophic-Removal Critical Core (CRCC) v1

**Status:** **MEASURED — DISCOVERY ONLY. G0 FAILED (empty automatic core); H1 CLOSED — NEGATIVE in this instrument revision.**  
**Working nickname:** *Monster*  
**Scientific name:** *Catastrophic-Removal Critical Core*  
**Branch:** `research/catastrophic-removal-core-v1`  
**Source coordinate system:** official full-depth Qwen3-0.6B, 28 decoder layers

> **Execution record (2026-08-31).** The discovery scan in §9 step 2 ran to
> completion on the pinned checkpoint: intact baseline plus all 28 one-layer
> bypass conditions, CUDA/bfloat16, 32 deterministic diagnostics.
> **No layer satisfied the frozen classifier, so the automatic CRCC set is empty
> and G0 fails on the depth check.** Per §4.4 the scientific progression stops
> here: no matched arms were designed, no reduced checkpoint was built, no
> recovery ran, no donor was merged, and no sealed asset was opened. H2, H3 and
> H4 are therefore **untested**, not refuted. Evidence:
> `experiments/catastrophic_removal_core/results/discovery_v1/`.

This document is the preregistration and execution contract for one question:

> Do decoder layers whose individual removal catastrophically destroys coherent
> generation form a structurally critical transformation core, and does recovery
> training of that core create unusually useful layer deltas when they are restored
> to a full Rineya scaffold?

The existence of this implementation is not evidence that the hypothesis is true. No CRCC
scan, recovery run, donor merge, long-context score, or sealed evaluation is claimed
by this branch.

---

## 1. Why this is a new experiment rather than a revival of the old 19-layer soup

The historical 19-layer precursor removed layers
`[6, 7, 8, 9, 10, 14, 18, 23, 24]`, retained the complementary 19 layers, and
produced degenerate direct behavior before later soup work. The current roadmap
correctly retires **unrelated LoRA averaging as structural repair**.

CRCC changes the scientific contract in four ways:

1. layer membership is measured by one-layer causal bypass against an intact baseline;
2. subjective recollections of “alien” output are retained only as a separate manual
   hypothesis field, never silently promoted to automatic labels;
3. the recovered reduced model is treated as a **delta donor**, not as proof of
   standalone capability;
4. donor value is compared against equal-depth random, low-criticality, and
   high-residual-rewrite controls trained with the same data, optimizer, seed, and
   token budget.

The old result motivates the question. It does not answer it.

---

## 2. Concepts that must remain separate

A layer can be:

- **necessary:** deleting it breaks the intact network;
- **sufficient:** a reduced network containing it can solve the task;
- **a useful donor:** recovery changes its weights in a direction that improves a
  compatible full scaffold;
- **high value per parameter:** retaining it gives a favorable quality/memory point.

None implies the others. In particular,

\[
Q_{\setminus i}\;\text{collapses}
\]

shows that layer \(i\) is necessary in the original composition. It does not show that
layer \(i\) is independently capable. Likewise, a recovered CRCC model may score poorly
standalone yet carry a valuable update:

\[
\Delta_{M,j}=M^{\text{recovered}}_j-M^{\text{initial}}_j.
\]

The primary donor construction is

\[
T_i' = T_i + \beta\Delta_{M,j},
\qquad j\leftrightarrow i
\]

for mapped source layer \(i\), while embeddings, final norm, LM head, and unmapped
layers remain untouched.

---

## 3. Preregistered hypotheses

### H1 — structural criticality

A nonempty subset of single-layer deletions produces baseline-relative collapse that
is jointly visible in reference NLL and generated-output mechanics. These layers form
the automatic CRCC set.

### H2 — donor specificity

At equal reduced depth and a fixed primary coefficient \(\beta=0.50\), the recovered
CRCC donor improves the targeted capability more than all three matched controls:
random layers, lowest-criticality layers, and highest angular-rewrite non-CRCC layers.

### H3 — complementary recovery

If H2 survives, adding the CRCC delta to an Aegis-derived reconstruction or to Halo
provides capability not already contained in the Aegis/Halo trajectory.

### H4 — family transfer

If H2 survives and the target is coordinate-compatible, the same CRCC delta can
improve a baked family checkpoint such as Supa A1.1 without violating preservation
and coherence gates.

### Null explanations

The pipeline is explicitly capable of supporting any of these negatives:

- catastrophic deletion merely identifies dependency bridges, not a useful core;
- every severely compressed model produces similarly useful recovery deltas;
- residual angular rewrite, not catastrophic behavior, explains donor value;
- the CRCC update duplicates what Aegis/Halo already contains;
- a gain exists only at a post-hoc coefficient or only in one recovery seed;
- the merge raises the 64-row score while damaging ordinary chat/RAG behavior.

---

## 4. Change 1 — automatic critical-core discovery

### 4.1 Diagnostic assets

`configs/critical_core_scan_v1.yaml` selects 32 records by deterministic hash:

- 16 from the committed Aegis RAG training corpus;
- 16 from the committed short-response preservation training slice.

These are discovery diagnostics, not final evaluation cases. Any path containing
`SEALED` is rejected before it is opened. The selection records source hashes, selected
case IDs, prompt/target token lengths, model inventory, code hashes, and a stable run
identity.

### 4.2 Intervention

For each of the 28 decoder layers, the scan installs a temporary forward hook that
replaces exactly that block's visible hidden-state output with its input. The hook
preserves tensor-versus-tuple return shape across Transformers versions, keeps the model
depth, embeddings, positional system, final norm, and output head fixed, and is removed
in a `finally` block. `use_cache=False` prevents stateful cache effects while the layer
is bypassed.

The intact model is evaluated once on the same cases. Each layer condition is then
measured independently against that baseline.

### 4.3 Signals

The classifier combines:

- mean teacher-forced reference NLL relative to intact Qwen;
- generated-output collapse rate;
- intelligibility loss under a mechanical screen;
- EOS-rate loss;
- non-finite NLL as a hard failure.

The mechanical screen records, rather than hides:

- empty output;
- invalid/control Unicode rate;
- dominant-token and repeated-ngram concentration;
- longest token/character runs;
- max-length termination without EOS.

It deliberately does **not** classify all non-English Unicode as corruption. A model
can produce valid multilingual text without being mechanically broken.

The composite score is

\[
C_i=0.45N_i+0.25R_i+0.20I_i+0.10E_i,
\]

where the components are clipped baseline-relative NLL, collapse-rate increase,
intelligibility loss, and EOS loss. Thresholds are frozen in the scan config. Every
classification stores its components and reasons. The automatic *structural* CRCC set
also requires catastrophic support in both configured diagnostic sources; a deletion
that collapses only one domain remains visible as domain-specific evidence but is not
automatically promoted to the core.

### 4.4 Discovery gate G0

Advance only if:

1. all 28 single-layer conditions complete against the identical case set;
2. the intact baseline has finite reference NLL;
3. the automatic CRCC depth is within the preregistered range 1–14;
4. no layer is included solely because it appeared in `manual_hypothesis_layers`;
5. the scan manifest and summary hashes reproduce.

If the automatic set is empty or larger than 14, stop and revise the instrument before
training. Do not choose a convenient set by eye.

---

## 5. Change 2 — matched reduced models and recovery

`design_arms.py` creates the following source-ordered reduced networks, all with CRCC
depth \(k\):

| Arm | Selection rule | Role |
|---|---|---|
| `crcc_monster` | automatic catastrophic-removal layers | hypothesis |
| `control_random` | deterministic sample from non-CRCC layers | matched control |
| `control_low_criticality` | lowest CRCC classifier scores | matched control |
| `control_angular_high` | largest one-layer residual angular distance outside CRCC | orthogonal matched control |

The existing 19-layer mapping is also recorded as `aegis_19_reference`, but it is not
called a matched control because its depth differs from \(k\).

### 5.1 Reduced checkpoint construction

`build_reduced_model.py`:

- copies only selected blocks from the same full Qwen checkpoint;
- preserves original order;
- renumbers runtime attention `layer_idx` fields for cache correctness;
- writes original source indices into `config.json` and `layer_map.json`;
- records exact parameter counts and checkpoint inventories;
- never vendors model weights into Git because the output lives under `runs/`.

A donor may not be merged later without this mapping.

### 5.2 Matched recovery contract

`make_recovery_configs.py` reuses `scripts/train/heal.py`; it does not fork or edit the
fingerprinted trainer. Starting from one template, it changes only:

- reduced model path;
- run directory;
- replication seed;
- explicit CRCC provenance metadata.

Within a seed, every other training field must hash identically. The generator forces
`resume.allow_code_drift: false`, removing the temporary B0 calibration waiver. The
training mix is rejected if its path is marked sealed.

The committed v1 config preregisters seeds `20260831`, `20260901`, and `20260902`.
Seed `20260831` is the screening run; the other two are frozen confirmatory
replications and need be executed only if the screening gate advances. Their identities
must not be changed after inspecting the screening merge outcomes.

### 5.3 Recovery gate G1

For all four matched arms:

- build manifests must identify the same full source checkpoint;
- reduced depth must equal \(k\);
- the same template/data hash and training-contract hash must be recorded;
- each run must complete the same planned token budget;
- exported adapters must reference the correct reduced base;
- standalone evaluation is descriptive only and cannot substitute for donor tests.

If any matched run fails or receives a different budget, the comparison is incomplete.

---

## 6. Change 3 — layer-delta reinsertion and merge matrix

### 6.1 Merge operator

For a donor arm with reduced index \(j\) mapped to source index \(i\):

\[
\Delta_{d,j}=W^{\text{recovered}}_{d,j}-W^{\text{initial}}_{d,j}
\]

\[
W'_{T,i}=W_{T,i}+\alpha_d\Delta_{d,j}.
\]

This is not literal averaging between differently structured checkpoints. It is an
explicit source-mapped delta operation. `apply_layer_deltas.py` refuses:

- missing or non-monotonic layer maps;
- state-key or tensor-shape differences;
- architecture differences in hidden/FFN/head dimensions;
- non-finite deltas or merged tensors;
- coefficients above the declared safety bound;
- ambiguous donor input (both or neither of adapter/merged checkpoint).

It records per-layer L2 norms, maximum absolute deltas, changed element counts, donor
order, target identity, and output identity.

### 6.2 Preregistered candidates

`configs/critical_core_merge_v1.yaml` expands these families:

| Family | Construction | Purpose |
|---|---|---|
| Qwen + CRCC | \(Q+\beta\Delta_M\) | primary donor test |
| Qwen + each control | \(Q+\beta\Delta_C\) | matched nulls |
| Qwen + Aegis | \(Q+\alpha\Delta_A\) | reference reconstruction |
| Qwen + Aegis + CRCC | \(Q+\alpha\Delta_A+\beta\Delta_M\) | “Aegis + Monster” on a full scaffold |
| Halo + CRCC | \(H+\beta\Delta_M\) | complementarity with current soup trajectory |
| Supa A1.1 + CRCC | \(S+\beta\Delta_M\) | family-transfer test |

The primary matched comparison is fixed at \(\beta=0.50\). The broader sweep
`[0.10, 0.25, 0.50, 0.75, 1.00, 1.25]` is exploratory. A coefficient selected from the
sweep must be frozen before confirmatory or sealed evaluation.

“Aegis + Monster” is represented primarily as both deltas restored to full Qwen. A
literal merge into a 19-layer Aegis target would omit CRCC layers not present in Aegis
and would answer a different question.

### 6.3 Development preservation gates

`run_dev_matrix.py` can build chosen candidates and invoke the current unsealed
`chat_dev` and `rag_dev` gates. It defaults to the four primary matched candidates and
requires `--execute` to spend compute. It never uses a shell and writes an atomic
execution ledger after each step.

A candidate may advance only if it does not introduce:

- coherence collapse;
- a material critical-defect increase;
- a preservation regression beyond the tolerance frozen with the exact target
  evaluator;
- mapping or merge-manifest failure.

### 6.4 Missing canonical 64-row evaluator

The repository's current `main` does not contain a canonical executable registry entry
for the historical 64-row long-context showcase used for the `14/64`, `4/64`, and
`30/64` family comparisons. Therefore this branch **cannot claim that it reproduces or
beats 30/64** merely by running `chat_dev`/`rag_dev`.

Before the targeted donor test:

1. import or identify the exact 64 cases without opening any sealed successor set;
2. pin dataset SHA, prompt template, tokenizer, context construction, generation
   settings, scorer, and model-loading behavior;
3. reproduce the known Qwen and Halo anchors under that same executable path;
4. freeze the candidate-selection rule;
5. keep any final sealed bank closed until one recipe is chosen.

This is a blocking evidence requirement, not optional documentation.

---

## 7. Statistical and advancement contract

### Primary donor comparison

At \(\beta=0.50\), compare Qwen+CRCC against Qwen+random, Qwen+low-criticality, and
Qwen+angular-high on identical cases. Use paired case outcomes. Report:

- exact correct count out of 64;
- all pairwise discordant counts;
- exact McNemar tests;
- Holm correction across the three primary control comparisons;
- bootstrap interval for the paired score difference;
- preservation-gate results.

A screening result advances only if CRCC beats **all three controls**, has no preservation
failure, and the direction is not carried by a single diagnostic category. It becomes a
confirmatory result only if the frozen coefficient shows the same direction in at least
two of three independently recovered seeds.

### Family transfer

Halo/Supa tests are conditional on donor qualification. Running them early is allowed as
engineering smoke work but cannot be used to rescue a failed primary matched comparison.

### Sealed evaluation

Open a sealed target set at most once, after:

- one target scaffold;
- one donor set;
- one coefficient tuple;
- one recovery-seed aggregation rule;
- all preservation thresholds

are frozen in a signed manifest. Failure on sealed is final for v1; do not tune against
it.

---

## 8. Interpretation table

| Outcome | Valid interpretation |
|---|---|
| CRCC standalone strong, donor strong | critical layers are both comparatively sufficient and useful donors in this setting |
| CRCC standalone weak, donor strong | standalone capability and merge contribution are distinct; recovery created useful compensatory deltas |
| CRCC donor equals random/low controls | severe compression pressure, not catastrophic identity, is the likely source of donor value |
| Angular-high equals or beats CRCC | residual rewrite magnitude may explain the effect better than generation collapse |
| CRCC improves Qwen but not Halo | CRCC and Aegis/Halo updates are likely redundant or conflict on the same coordinates |
| Halo/Supa improve but primary matched test fails | exploratory family result only; not evidence for the CRCC hypothesis |
| One coefficient/seed works | unstable exploratory observation; do not promote |
| No automatic CRCC set | the historical “monster” behavior was not reproduced by this instrument/model revision |

---

## 9. Reproduction commands

All commands run from the repository root with the project environment.

```bash
# 1. Record the scan identity and verify paths/model load without evaluating layers.
python scripts/critical_core/scan_layer_ablation.py \
  --config configs/critical_core_scan_v1.yaml --dry-run --hash-weights

# 2. Run intact + 28 one-layer bypass conditions. Results resume per layer.
python scripts/critical_core/scan_layer_ablation.py \
  --config configs/critical_core_scan_v1.yaml --hash-weights

# 3. Freeze CRCC and matched controls.
python scripts/critical_core/design_arms.py \
  --config configs/critical_core_recovery_v1.yaml

# 4. Materialize each reduced checkpoint. Repeat for every arm in arms.json.
python scripts/critical_core/build_reduced_model.py \
  --arm-manifest runs/critical_core/crcc-v1/design/arms.json \
  --arm crcc_monster --model Qwen3-0.6B \
  --out runs/critical_core/crcc-v1/models/crcc_monster --smoke

# 5. Generate matched trainer configs, then execute/export commands from the matrix.
python scripts/critical_core/make_recovery_configs.py \
  --config configs/critical_core_recovery_v1.yaml

# 6. After adapters exist, expand all target/donor/coefficient plans.
python scripts/critical_core/plan_merge_grid.py \
  --config configs/critical_core_merge_v1.yaml

# 7. Build and run only the four fixed-beta primary candidates by default.
python scripts/critical_core/run_dev_matrix.py \
  --matrix runs/critical_core/crcc-v1/merge/merge_matrix.json \
  --build --evaluate --execute

# 8. CPU structural verification; no checkpoint required.
python scripts/verify/verify_critical_core.py
```

PowerShell uses the same Python entry points; replace line continuations with backticks
or place each command on one line.

---

## 10. Artifact contract

All generated weights and bulky per-case outputs live below `runs/`, which is already
ignored. Evidence intended for review should be copied as small manifests/summaries only
after a run is complete and audited.

Required artifacts are:

- scan `run_manifest.json`, `baseline.json`, per-layer JSON, `scan_summary.json`;
- deterministic `arms.json` with source mappings and overlap table;
- one `build_manifest.json` and `layer_map.json` per reduced arm;
- `recovery_matrix.json`, completed run checkpoints, exported adapter manifests;
- immutable merge plans, merge matrix, per-model `crcc_merge_manifest.json`;
- development execution ledger and evaluator summaries;
- exact 64-row evaluator registry/anchor reproduction before a targeted claim;
- final frozen-selection manifest before sealed evaluation.

Do not commit model weights, adapters, optimizer checkpoints, or imported sealed rows.

---

## 11. Merge-readiness checklist

This branch is code-complete for the built/unrun state when:

- `python -m compileall -q scripts/critical_core scripts/verify` passes;
- `python scripts/verify/verify_critical_core.py` reports all groups passing;
- no file under `scripts/train/` has changed, preserving existing resume fingerprints;
- all default discovery/recovery datasets are unsealed training assets;
- configs label Halo/Supa paths as local prerequisites rather than pretending the
  checkpoints are committed;
- the canonical experiment audit is updated to `BUILT — UNRUN` in the eventual merge
  commit, then updated again only when qualifying evidence changes the status.

The correct current conclusion is:

> CRCC is an executable, controlled hypothesis about structurally critical layer
> transformations and donor utility. It has not yet established that “monster” layers
> exist as an automatic set or that any resulting merge improves Rineya.

---

## 12. Measured discovery result — 2026-08-31

This section is appended, not a rewrite. Sections 1–11 remain the preregistration
as it stood before any model was loaded.

### 12.1 What ran

Intact Qwen3-0.6B baseline plus all 28 one-layer bypass conditions on the same 32
deterministic diagnostics, resolved to `cuda:0` / `bfloat16` under the pinned
runtime (torch 2.8.0+cu128, transformers 4.57.1). The amendment's CPU/FP32 hosted
plan was never executed; this is a CUDA/BF16 run and is labelled as such in the
evidence manifest.

- Run identity `1b91c6d2d8cf68aed50f28e5f1aff838561538eacdcfd3871b3d77f9c7b69896`
- Stable summary `f5ea9f1247eef5ebb4f6d710b0cdddfddcee16f73e4603b811b076f7042884e3`
- Membership `002d55d1d3e825e40246945546376ffb0c857f022847bc499c989dcadfb4b7a1`

### 12.2 G0 outcome

Six of seven G0 checks pass. The single failure is
`automatic_core_depth_in_preregistered_range_1_to_14`, because the automatic set
is empty (depth 0). Coverage, baseline finiteness, manual-promotion policy,
cross-source support, case count and weight hashing all pass, so this is a
measured negative rather than an aborted run.

### 12.3 Intact baseline

Mean reference NLL 2.400199; collapse-case rate 0.0000; intelligibility 1.0000;
EOS rate 1.0000; mean generated tokens 14.19. Per source, mean NLL is 0.3455
(`rag_train`) and 4.4549 (`short_preservation_train`).

### 12.4 The measured depth profile

Criticality is sharply concentrated rather than distributed:

| Band | n | mean criticality | max |
|---|---:|---:|---:|
| early 0–2 | 3 | 0.5200 | 0.5519 |
| early-mid 3–9 | 7 | 0.0459 | 0.0728 |
| mid 10–19 | 10 | 0.0293 | 0.0804 |
| deep 20–26 | 7 | 0.0195 | 0.0615 |
| final 27 | 1 | 0.1730 | 0.1730 |

Only layers 0, 1 and 2 raise reference NLL by more than 1.43x (5.47x, 4.42x,
4.15x). Layer 14 is indistinguishable from the intact model (ratio 0.9994).
Layers 6, 7 and 8 — three of the nine the historical Aegis surgery deleted — are
individually near-free at 1.27x, 1.19x and 1.09x.

### 12.5 Why no layer qualified

Two properties of the frozen instrument, both consequences of the preregistered
design rather than defects:

1. **The composite path requires generation mechanics.** With collapse and
   intelligibility components at zero the composite ceiling is
   `0.45*1 + 0.10*1 = 0.55`, below the frozen `0.60` threshold. NLL and EOS loss
   alone can never elect a layer. This is exactly what H1 demands — collapse must
   be *jointly* visible in reference NLL and generated-output mechanics — so the
   threshold is doing what it was written to do.
2. **Pooled-mean NLL dilutes the ratio.** The classifier divides pooled means, and
   the two sources have very different baselines (0.3455 vs 4.4549). Layer 2's
   per-source ratios are 25.96x and 2.46x, but the ratio of pooled means is only
   4.15x. The overall gate is therefore systematically less sensitive than the
   per-source gates it is combined with: layer 2 is catastrophic in *both* sources
   individually yet fails the overall test.

The mechanical screen behaved as specified. Bypassing layer 0 produces obvious
word-salad, but that output is neither empty, repetitive, nor control-character
laden, so it scores collapse 0.100 and reads as "intelligible". The screen detects
degeneracy, not semantic destruction, and §4.3 says so deliberately.

### 12.6 Instrument validity

The negative is not a no-op bug. Directly comparing bypassed against intact logits
on the same input: layer 0 changes the argmax at 100% of positions
(max |Δlogit| 27.97), layer 14 at 10.5% (5.50), layer 25 at 15.8% (6.09) and
layer 27 at 21.1% (12.09); removing the hook restores bit-identical logits. Every
bypass genuinely perturbs computation, including the layers that scored near zero.

### 12.7 What this does and does not say

- **H1 is closed negative for this instrument revision and this model.** No
  single-layer deletion in Qwen3-0.6B produces collapse that is jointly visible in
  reference NLL and generation mechanics under the frozen thresholds.
- **H2, H3 and H4 are untested.** They are conditional on a nonempty core. Nothing
  here says a recovery delta would or would not be a useful donor.
- The historical "monster" behaviour was a **nine-layer simultaneous** deletion.
  This scan measures one layer at a time and found that removal damage is
  concentrated in layers 0–2 and 27, none of which the historical surgery touched.
  Those are consistent facts, not a contradiction.

### 12.8 Revising the instrument is a new preregistration

§4.4 permits revising the instrument after a stop; it forbids choosing a
convenient set by eye. Any successor must freeze its changes *before* re-running.
Candidates worth considering, in order of how much they change the question:

1. score per source and combine, instead of classifying a ratio of pooled means;
2. add a semantic-destruction signal so "fluent but wrong" is separable from
   "degenerate", which is the gap that made layers 0–2 unelectable;
3. measure multi-layer joint removal, which is what the historical result
   actually did.
