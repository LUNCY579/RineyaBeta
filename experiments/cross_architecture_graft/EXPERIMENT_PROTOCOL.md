# Cross-Architecture Graft v1 — Preregistered Experiment Protocol

## 0. Status and authority

This document is the experiment-level source of truth for
`research/cross-architecture-graft-v1`.

- Branch base: `main@3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`
- Protocol state: **FROZEN BEFORE ANY REAL RINEYA RUN**
- Implementation state: **BUILT**
- Synthetic end-to-end state: **PASS, INFRASTRUCTURE ONLY**
- Real Rineya state: **UNRUN**
- Open semantic DEV state: **UNRUN**
- Sealed/showcase state: **CLOSED**

If implementation behavior and this protocol conflict, the run is invalid until
the conflict is resolved and a new protocol version is explicitly recorded.
Infrastructure commits may not silently redefine the hypothesis or promotion
gates.

## 1. Research question

Can a compact Qwen-derived Rineya gain useful computation from a small,
architecture-distinct residual sidecar, while retaining the original host
function at initialization and outperforming controls with approximately the
same trainable parameter budget?

The v1 intervention at layer \(l\) is:

\[
\tilde h_{l+1}
  = F_{\text{host},l}(h_l)
  + g_l(h_l)P_{\mathrm{out}}
      F_{\mathrm{foreign}}(P_{\mathrm{in}}h_l).
\]

The gate is bounded and initialized so that

\[
g_l(h_l)=0
\]

for every input at step zero. Therefore

\[
\tilde h_{l+1}=F_{\text{host},l}(h_l)
\]

exactly on the evaluation bypass path.

## 2. Scope: mechanism transfer, not donor-knowledge transfer

The primary arm implements a newly initialized latent-KV attention sidecar. It
borrows the architectural hypothesis that K/V content can pass through a shared
low-dimensional latent. It does **not** load DeepSeek weights and is not claimed
to be an exact MLA reproduction.

The first causal claim under test is therefore:

> Under a frozen host and matched training contract, latent-KV architecture is a
> more useful allocation of additional trainable capacity than a full NoPE/GQA
> attention sidecar or a conventional SwiGLU residual MLP.

A positive v1 result would support only this bounded mechanism claim. It would
not establish cross-family knowledge transfer.

A later functional-transfer phase may wrap an actual frozen donor block with
`ForeignModuleAdapter`. That later phase requires its own preregistration and at
least these controls:

1. trained frozen donor block;
2. same architecture with reinitialized weights;
3. coordinate-permuted or subspace-shuffled donor control;
4. same-family Qwen/Rineya donor block;
5. bridge-only capacity control;
6. matched trainable-parameter and training-compute accounting.

## 3. Primary hypothesis and null

### H1 — architecture-specific benefit

Across the five frozen seeds, the latent-KV graft improves the preregistered
open target-capability endpoint relative to:

- the untouched host;
- the parameter-matched full NoPE/GQA attention sidecar; and
- the parameter-matched SwiGLU sidecar;

while satisfying all preservation, invariance, comparability, and stability
criteria.

### H0 — no architecture-specific benefit

After parameter, data, optimization, placement, and seed controls, the latent-KV
arm does not reliably exceed both matched controls, or it causes unacceptable
preservation loss.

A failure to reject H0 retires this exact v1 configuration. It does not by itself
rule out every cross-architecture graft, every layer placement, or functional
transfer from trained donor weights.

## 4. Frozen primary design

### 4.1 Host

- One content-addressed local Rineya/Qwen causal-LM checkpoint.
- Expected architecture for the current plan: 28 decoder layers, hidden size
  1,024, layer container `model.layers`.
- Host weights are frozen before graft attachment and excluded from the
  optimizer.
- Real mode is local-only; implicit model downloads are forbidden.
- All three arms in a seed bundle must have the same aggregate checkpoint hash.

The exact host hash is recorded by the first real campaign manifest. A different
host checkpoint constitutes a different experiment.

### 4.2 Placement

- One graft only.
- Zero-indexed layer: `14`.
- The placement is fixed before the first real run.
- No layer sweep may be used to reinterpret v1 after seeing primary results.

Layer 14 is the single middle-region probe for a 28-layer scaffold. A later
placement study is allowed only after v1 and must include a multiplicity-aware,
predeclared search protocol.

### 4.3 Arms and exact planned budgets

| ID | Intervention | Planned trainable graft parameters | Relative gap vs A1 |
|---|---|---:|---:|
| A0 | Frozen host, no active graft | 0 | — |
| A1 | Latent-KV attention, foreign width 512, bridge rank 64 | 721,409 | 0.00% |
| A2 | Full causal NoPE/GQA attention, foreign width 424, bridge rank 64 | 725,097 | +0.51% |
| A3 | Conventional SwiGLU residual branch, width 235 | 722,945 | +0.21% |

A2 isolates latent compression from the generic benefit of adding another
attention path. A3 isolates architecture-specific effects from ordinary residual
capacity.

A parameter gap above 1% invalidates the matched-control claim unless a revised
protocol is committed before any result is inspected.

### 4.4 Fixed seeds

The primary campaign seeds are:

```text
20260831, 20260832, 20260833, 20260834, 20260835
```

The seed controls donor initialization, bridge initialization, data order, and
optimizer trajectory. Arms are compared only within the same seed first, then
aggregated across the five paired seed bundles.

### 4.5 Training contract

The frozen plan uses:

- steps: 400
- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `0.0`
- gradient clip: `1.0`
- microbatch rows: 1
- gradient accumulation: 8
- maximum sequence length: 512
- dropout in the graft mechanisms: 0.0
- host: frozen
- graft gate: exact-zero scalar gate, bound 1.0
- cache during training/evaluation: disabled
- deterministic PyTorch algorithms: enabled
- TF32: disabled

Any learning-rate, step-count, sequence-length, accumulation, data-order,
checkpoint, layer-placement, or evaluator change creates a different campaign.

This short v1 run is a mechanism-screening study, not a claim that 400 steps are
an optimal training budget.

## 5. Data contract

### 5.1 Allowed data

Only explicitly unsealed training/development data may be used. A useful first
curriculum should contain both:

- target context-integration cases, such as multi-fact support, supported cases
  with distractors, procedure-with-distractor, conflicts, partial support,
  ambiguity, and prompt-injection resistance; and
- preservation cases spanning concise conversation, basic reasoning, grounding,
  abstention, and instruction following.

The exact category mixture and row counts must be frozen in the split/curriculum
manifest before the first arm runs. The protocol does not authorize changing the
mixture after inspecting arm outcomes.

### 5.2 Forbidden data

CAG v1 structurally rejects paths containing markers associated with:

- sealed or hidden evaluation;
- final/private holdouts;
- the historical long-context demo pack;
- showcase assets.

The literal term `unsealed` is explicitly recognized as open data and is not
mistaken for `sealed`.

No training or tuning may use:

- any sealed evaluation row;
- the historical 64-row long-context showcase;
- test answers or scorer-derived labels;
- examples selected because a particular arm failed them;
- duplicated or paraphrased evaluation items.

### 5.3 Grouped split requirement

A row-random split is forbidden. The corpus must contain an explicit provenance
key that identifies a generator/template/source family. Preferred fields are:

```text
template_id, generator_id, group_id, cag_group_id, source_family
```

The chosen field must be present on every included row and provide enough
independent groups. All rows from one group go to exactly one side. The split
builder writes a manifest containing:

- input, train, and dev SHA-256 hashes;
- row, group, and category counts;
- the exact seed and requested DEV fraction;
- field precedence;
- `group_overlap: 0`.

The real runner rehashes both data files and refuses to run if they differ from
the split manifest.

### 5.4 Data inspection stop condition

If no explicit provenance field covers every row, the campaign is blocked. The
operator must repair provenance from the generator records. It is not acceptable
to use prompt text, answer text, row index, or a random hash of each row as a
surrogate group identifier.

## 6. Evidence phases

### P0 — static and synthetic proof

Required before real weights are touched:

- config schema and forbidden-path tests;
- causal attention tests;
- zero-gate bit-exact unit tests;
- host/donor freeze tests;
- layer-attachment and graft-pack round trip;
- deterministic grouped split tests;
- evidence payload integrity tests;
- synthetic end-to-end train/export/manifest smoke;
- preflight parameter counts for A1–A3.

Passing P0 means **BUILT**, not scientifically validated.

### P1 — data and campaign freeze

Required artifacts:

- data inspection report;
- grouped split manifest;
- category/curriculum manifest;
- content-addressed host checkpoint fingerprint;
- materialized A1–A3 configs for all five seeds;
- campaign manifest proving the non-graft run contract is identical.

P1 ends with `CONFIGS_MATERIALIZED_UNRUN`.

### P2 — zero-gate real-model proof

Before the first optimizer step in each arm:

- compare logits over at least eight deterministic probe rows;
- require `torch.equal` between host and zero-gated model;
- require maximum absolute logit error exactly `0.0`;
- record the layer path and model signature;
- prove only `.graft_branch.` parameters are trainable.

Any failure invalidates the run; approximate equivalence is insufficient.

### P3 — unsealed train/DEV health screen

Each real run records:

- loss at first and last step;
- unsealed DEV NLL before and after training;
- per-category DEV NLL;
- train duration;
- peak allocated/reserved CUDA memory;
- exact model/data/config hashes;
- frozen-parameter version checks;
- graft-only export pack.

This phase detects broken optimization, data drift, memory failure, and gross
regression. NLL cannot promote an architecture.

### P4 — matched open semantic DEV evaluation

The same frozen, non-sealed semantic evaluator must score A0–A3. It must produce
per-case outputs so paired uncertainty can be computed. The evaluator command,
scorer version, decoding settings, prompt template, and case hashes must be
registered before first use.

Required endpoints:

1. **Primary target endpoint:** mean score over the frozen target-capability DEV
   cases.
2. **Preservation endpoint:** mean score over the frozen general-capability DEV
   cases.
3. **Category endpoints:** target and preservation score by preregistered category.
4. **Operational endpoints:** trainable parameters, run time, and peak memory.

### P5 — decision

A1 advances only if all conditions hold:

1. **Two-control rule:** A1 exceeds both A2 and A3 on the primary target endpoint.
2. **Paired uncertainty:** for each comparison, the 95% paired bootstrap lower
   bound of A1 minus control is greater than zero on pooled seed/case results.
3. **Minimum practical effect:** paired mean improvement is at least 0.01 on a
   0–1 semantic score against each control.
4. **Seed stability:** A1 beats each control on at least four of five seeds.
5. **Host improvement:** A1 exceeds A0 on the target endpoint.
6. **Preservation non-inferiority:** aggregate preservation loss versus A0 is no
   worse than 0.005, and no required category loses more than 0.02.
7. **Invariant pass:** every run retains exact zero-gate proof, frozen-host proof,
   content hashes, and closed-evaluation status.
8. **Matched budget:** parameter gap stays below 1%; training steps/data are
   identical; meaningful compute imbalance is reported and cannot be hidden.

If one criterion fails, the primary v1 result is `NO-GO`. Exploratory positives
may be documented, but they cannot override the preregistered decision.

## 7. Interpretation matrix

| Outcome | Permitted interpretation |
|---|---|
| A1 beats A3 but not A2 | Extra attention helped; latent-KV-specific benefit not established. |
| A1 beats A2 but not A3 | Latent-KV may be competitive with attention, but ordinary capacity explains the gain. |
| A1 beats both but loses preservation | Target specialization occurred; architecture is not acceptable for the compact family contract. |
| A1 wins only one or two seeds | Unstable exploratory signal; v1 NO-GO. |
| A1 passes all criteria | Narrow evidence that latent-KV is a better graft allocation under this host/layer/data/budget. |
| All arms fail to improve A0 | Training signal/budget or graft premise failed; do not search sealed evaluation. |

A positive outcome does not show that the same mechanism works at another layer,
scale, context length, checkpoint, or task distribution.

## 8. Efficiency and deployment boundary

The reference latent-KV implementation reconstructs K/V tensors and uses a
manual causal attention path with `use_cache=False`. It tests representational
and optimization behavior, not production KV-cache savings.

No speed, memory-saving, or deployment claim is permitted from v1. A later
efficiency phase must implement cache-aware decoding and compare:

- identical prompt/output tokens;
- TTFT and decode tokens/s;
- allocated and resident memory;
- KV-cache bytes by context length;
- numerical/behavioral equivalence;
- matched runtime kernels.

## 9. Historical long-context boundary

The historical 64-row evaluator must be canonically registered before any result
is described as comparable to Halo's historical `30/64` figure. This branch does
not register, inspect, tune on, or run that evaluator. Even a positive open-DEV
result may not be translated into a Halo comparison without that separate
registration boundary.

## 10. Failure handling

- OOM: reduce only a preregistered operational parameter that affects every arm
  identically, then restart the entire campaign under a new contract hash.
- NaN/Inf: mark the arm failed; do not silently clip model outputs or skip rows.
  The zero-gate path sanitizes malformed donor deltas only to prevent an inactive
  branch from contaminating the host.
- Data hash drift: stop and regenerate all affected arm configs/runs.
- Interrupted run: v1 has no checkpoint-resume claim. Restart from step zero to
  preserve exact comparability.
- Code change after any arm runs: rerun every arm unless the change is proven
  evidence-only and leaves the source commit attached to the run unchanged.
- Evaluator defect: freeze all decisions, repair/version the evaluator, and rerun
  every arm. Do not selectively rescore the favored candidate.

## 11. Promotion and merge policy

This branch may be proposed for merge as reusable research infrastructure after
P0 evidence is committed. No architecture should enter a production Rineya model
until P5 passes and a separate deployment/performance review succeeds.

The merge message must distinguish:

- `BUILT` — implementation and synthetic proof;
- `UNSEALED-RUN` — real train/DEV evidence exists;
- `SUPPORTED` — preregistered semantic gates pass;
- `PRODUCTION-READY` — separate runtime, safety, and packaging evidence passes.

At the current branch head, only the first label is authorized.
