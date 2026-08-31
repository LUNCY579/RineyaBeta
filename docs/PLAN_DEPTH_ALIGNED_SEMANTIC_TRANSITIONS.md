# Depth-Aligned Semantic Transitions (DAST)

**Repository:** `LUNCY579/RineyaBeta`  
**Working branch:** `research/depth-aligned-semantic-transitions-v1`  
**Created:** 2026-08-30  
**Program status:** **BUILT — UNRUN on a real checkpoint**  
**Sealed evaluation:** **CLOSED**

This document is the branch-level single source of truth for investigating whether
Transformer depth can be trained as a semantic reasoning trajectory rather than
left as an unconstrained sequence of hidden transformations.

---

## 1. Research thesis

Conventional supervised fine-tuning presents the complete problem and rationale to
a model and allows the final loss to update every trainable layer. It does not
specify what computational progress should exist at intermediate depth.

The original intuition motivating this program was more radical: divide a
reasoning trace among depth regions and train each region on a different sentence
or step. The literal formulation is valuable as a control but has a serious
interface problem. A sentence is a surface-language object, while a Transformer
layer consumes and produces a distributed residual representation. Independently
trained blocks can invent incompatible latent coordinate systems.

DAST tests a more defensible formulation:

> Assign **cumulative semantic reasoning states** to increasing Transformer depth,
> read every checkpoint through one shared semantic interface, and later test
> whether reusable low-rank state transformations and local training can be made
> causal and transferable.

For a problem with latent milestones \(S_1,\ldots,S_K\) and a Transformer with
checkpoints \(d_1<\cdots<d_K\):

\[
H^{(d_1)}\rightarrow S_1,\quad
H^{(d_2)}\rightarrow S_2,\quad\ldots,\quad
H^{(d_K)}\rightarrow S_K.
\]

The long-term hypothesis is that depth may support an internal reasoning trace:

\[
S_0\xrightarrow{O_1}S_1\xrightarrow{O_2}\cdots\xrightarrow{O_K}S_K,
\]

where \(S_k\) is what is currently represented and \(O_k\) describes how that
representation changes.

---

## 2. Relationship to adjacent Rineya work

### 2.1 State-sufficiency lab

The existing `Rineya_state_experiment_bundle` asks whether a compact textual state
is sufficient for predicting the next explicit reasoning transition. Its SPLIT
condition receives `(problem, current compact state)` and generates the next step
and updated state, potentially in globally shuffled order.

DAST is not a replacement or rerun of that hypothesis:

| State-sufficiency lab | DAST |
|---|---|
| separate generated transition calls | one causal-LM forward pass |
| compact state supplied as text | semantic state read from hidden depth |
| tests Markov sufficiency of explicit state | tests alignment of computation progress to depth |
| sequence order manipulated across records | state-to-depth assignment manipulated within a record |
| no claim about internal layer roles | explicit shared depth interface |

The existing dataset is reused because it already contains canonical full traces
and compact post-step states under controlled splits. The existing lab remains
**BUILT — UNRUN** and is not treated as prior positive evidence.

### 2.2 DiffusionBlocks inspiration

DiffusionBlocks makes blocks independently trainable by defining valid
intermediate distributions along a denoising trajectory. DAST borrows the
scientific pattern, not the result:

- DiffusionBlocks: known noise state \(x_{\sigma_i}\rightarrow x_{\sigma_{i+1}}\);
- DAST: hypothesized semantic reasoning state \(S_i\rightarrow S_{i+1}\).

DAST V1 does **not** claim independent blocks or DiffusionBlocks-like memory
reduction. It first asks whether ordered semantic depth supervision works at all
under ordinary end-to-end backpropagation. Local gradients are a later gate.

### 2.3 Latent/recurrent reasoning

Latent reasoning approaches move computation from generated token time into hidden
state or recurrence. DAST's distinctive question is whether fixed network depth
can receive explicit progress supervision and a shared semantic interface, then be
validated with shuffled/reversed controls and interventions.

---

## 3. Core representation

A reasoning checkpoint is modeled as:

\[
\mathcal R_k=(S_k,O_k),
\]

where:

- \(S_k\): a compact semantic state describing current entities, relations,
  constraints, intermediate values, uncertainty, and resolved facts;
- \(O_k\): a transformation such as bind, compare, eliminate, compose, substitute,
  count, retrieve, resolve, or verify.

No manual claim is made that individual dimensions or slots correspond to these
English labels. The labels motivate transfer tests; the learned representation is
continuous.

### 3.1 Shared semantic ABI

A single projector \(P_\phi\) maps pooled hidden states at all depths into the same
unit sphere:

\[
z_k=P_\phi(\operatorname{pool}_{prompt}(H^{(d_k)})).
\]

The shared projector is essential. Separate probes would permit each checkpoint to
encode the target in an unrelated basis and would not create a transferable
boundary.

### 3.2 Prompt-only pooling

Full rationale targets are teacher-forced for the LM loss. Hidden states at
assistant positions therefore contain target information. DAST never pools them.
The state mask includes only system, user, and generation-prompt tokens. Padding
and assistant tokens are excluded. This is a hard validity contract with unit
tests.

### 3.3 V1 target geometry

V1 uses the frozen model input embedding to encode compact-state text, followed by
a fixed deterministic projection. This choice is intentionally bounded:

- no external teacher is required;
- target coordinates cannot drift;
- all arms receive identical target construction;
- the method is runnable offline with a local model.

It is not a proof of a language-independent semantic manifold. A later multi-view
phase may align paraphrases, structured states, and execution states only if V1
passes.

---

## 4. Model and depth schedule

The initial target is Qwen3-0.6B or the audited compatible Rineya/R2 core. Runtime
architecture inspection determines layer count. The default four checkpoint
fractions are:

\[
(0.25,0.50,0.75,1.00).
\]

For 28 layers:

| Stage | Fraction | Post-layer checkpoint |
|---:|---:|---:|
| 1 | 0.25 | 7 |
| 2 | 0.50 | 14 |
| 3 | 0.75 | 21 |
| 4 | 1.00 | 28 |

The code rejects schedules that collapse to duplicate layers. Variable-length
reasoning traces are mapped by normalized progress with both endpoints fixed.
Repeated states are retained and recorded when the trace has fewer steps than
checkpoints.

The 7/14/21/28 partition is a first controlled schedule, not an assumption that
natural reasoning roles are exactly equal in depth. Phase 0 diagnostics may later
motivate a frozen alternative schedule, but schedule search cannot use sealed TEST.

---

## 5. Experimental arms

Every Phase-1 arm uses the same source problems, full-CoT assistant targets,
tokenizer, maximum length, LoRA capacity, optimizer schedule, and checkpoint count.

| Arm | Depth target | Purpose |
|---|---|---|
| `cot_baseline` | none | contextual full-CoT SFT baseline |
| `state_ordered` | cumulative states in canonical order | main hypothesis |
| `state_shuffled` | same state multiset, deterministic non-identity order | controls auxiliary supervision and target quantity |
| `state_reverse` | states in reverse order | tests directional compatibility of reasoning and depth |
| `state_final_repeated` | final state at every checkpoint | controls generic deep supervision |
| `literal_ordered` | literal reasoning-step text in order | tests the original literal proposal |
| `operator_ordered` | ordered states + low-rank operators | Phase 2 treatment |
| `operator_shuffled` | shuffled states + identical operator capacity | Phase 2 control |

The primary causal evidence is ordered versus **both** shuffled and reverse.
Ordered versus baseline is informative but not sufficient because active parameter
use and losses differ.

---

## 6. Objectives

### 6.1 Language-model preservation

All arms retain assistant-only causal language modeling:

\[
\mathcal L_{LM}=-\sum_{t\in assistant}\log p(y_t\mid x,y_{<t}).
\]

The prompt is masked from token loss. The full rationale/answer target is identical
across Phase-1 arms.

### 6.2 State alignment

\[
\mathcal L_{state}=\frac1K\sum_k(1-z_k^\top s_k^*).
\]

### 6.3 Contrastive anti-collapse

A symmetric stage-wise multi-positive InfoNCE objective distinguishes examples.
Identical target strings are treated as multiple positives.

### 6.4 Operator transition

For operator atoms \(O_j=A_jB_j\) and router weights \(\alpha_k\):

\[
\hat z_{k+1}=\operatorname{norm}\left(z_k+
\sum_j\alpha_{k,j}(z_kA_j)B_j\right).
\]

The operator predicts the next target state and agrees with the next direct depth
state under stop-gradient. Diversity and bounded load regularizers discourage
identical atoms and router collapse.

### 6.5 Total objective

\[
\mathcal L=\mathcal L_{LM}+\lambda_s\mathcal L_{state}
+\lambda_c\mathcal L_{contrastive}
+\lambda_t\mathcal L_{transition}
+\lambda_{cons}\mathcal L_{consistency}
+\lambda_d\mathcal L_{diversity}
+\lambda_b\mathcal L_{balance}.
\]

Weights are frozen symmetrically before confirmatory comparison. Loss improvement
alone is not a capability result.

---

## 7. End-to-end pipeline

### Phase 0 — Audit and observation

1. validate the existing state dataset and checksums;
2. inspect trace lengths, categories, generators, answer types, and split isolation;
3. resolve real model depth and tokenizer prefix behavior;
4. run untrained/frozen hidden-state diagnostics across layers;
5. freeze the initial schedule and compute budget.

**Output:** readiness report and frozen preregistration.  
**No claim:** hidden-state decodability alone is descriptive.

### Phase 1 — Ordered semantic depth supervision

1. run the six-arm matrix on frozen seeds;
2. preserve full-CoT LM target and matched exposure;
3. record generated answers, LM loss, state matrices, VRAM, timing, and failures;
4. compare ordered against shuffled, reverse, and final repeated;
5. stratify by trace depth, generator, domain, and category.

**Gate:** ordered must pass the preregistered primary metric against both decisive
controls and preserve baseline quality.

### Phase 2 — Transferable operators

1. enable the operator bank under ordered and shuffled matched arms;
2. define held-out operator/content combinations;
3. test same-operation/different-content similarity;
4. test transition prediction and novel operator composition;
5. audit router entropy and atom collapse.

**Gate:** operator ordered must beat operator shuffled on held-out transfer, not
merely reconstruct training states.

### Phase 3 — Causal state use

1. enable bounded semantic feedback with gradient checkpointing disabled;
2. establish unmodified preservation;
3. swap correct counterfactual states between matched problems;
4. compare wrong-stage, random norm-matched, zero, and no-op controls;
5. require directionally correct answer changes.

**Gate:** semantic interventions must causally alter downstream reasoning in the
predicted direction.

### Phase 4 — Progressive gradient isolation

Only after Phases 1–3:

1. stop gradient at one semantic boundary;
2. compare against fully end-to-end training;
3. add boundary reconciliation or synthetic downstream signal if necessary;
4. progressively isolate more boundaries;
5. measure memory, compute, and capability separately.

**Gate:** local training must preserve the ordered-state effect within a frozen
non-inferiority margin.

### Phase 5 — Independently sampled semantic boundaries

The true DiffusionBlocks analogue would generate a valid hidden boundary state
from the problem and semantic milestone without executing preceding blocks:

\[
G_k(x,S_k)\rightarrow H^{(d_k)}.
\]

This is unresolved. No efficiency claim is allowed until sampled boundary states
match real activation distributions and independently trained blocks compose under
end-to-end evaluation.

---

## 8. Data strategy

The initial dataset has 48,000 controlled problems and 189,922 transitions. It is
valuable because complete trajectories and compact states are paired from the same
canonical construction.

Required analysis strata:

- constraint logic and candidate elimination;
- arithmetic and variable update;
- set membership;
- relational ordering;
- graph reachability;
- Boolean state;
- symbolic/program execution where available;
- trace length and repeated checkpoint mapping;
- seen versus held-out generator/template;
- paraphrase and entity substitution.

Future multi-view targets should supply semantically equivalent natural-language
paraphrases and structured forms. They must be generated or curated without
cross-split leakage and frozen before confirmatory evaluation.

---

## 9. Evaluation

### 9.1 Capability

- strict generated final-answer accuracy;
- category and generator stratification;
- deeper-than-training reasoning traces;
- held-out composition and paraphrase;
- refusal, formatting failure, and nontermination rates.

### 9.2 Mechanism

For cross-stage cosine matrix \(C_{ij}=z_i^\top s_j^*\):

- diagonal canonical cosine;
- final-state cosine;
- diagonal-minus-off-diagonal order margin;
- checkpoint top-1 state identity;
- monotonic decoded progress;
- transition reconstruction;
- operator routing entropy/diversity;
- intervention directionality.

### 9.3 Preservation and efficiency

- teacher-forced LM loss/perplexity;
- existing Rineya preservation gates where applicable;
- trainable/total parameter counts;
- target tokens, optimizer updates, and source-problem exposure;
- peak allocated/reserved VRAM;
- wall time and tokens/second;
- checkpoint/restart fidelity.

Mechanism metrics cannot replace generated capability. Efficiency metrics cannot
replace preservation. A single aggregate cannot hide category failures.

---

## 10. Statistical design

- pair results by original problem ID;
- use at least three frozen seeds for confirmatory DEV;
- report paired accuracy differences and problem-cluster bootstrap intervals;
- report exact McNemar discordance for binary correctness;
- freeze a practical margin before comparative results;
- correct exploratory subgroup multiplicity or label it descriptive;
- retain every seed and failed run;
- do not open TEST to resolve ambiguous DEV evidence.

The shipped comparison utility enforces identical problem sets. More complete
statistics can reuse the mature state-sufficiency evaluator after an explicit
adapter bridge; its results must retain arm provenance.

---

## 11. Compute strategy for 8 GiB VRAM

Initial real runs use LoRA rank 16, assistant-only loss, gradient checkpointing,
auto BF16/FP16 selection, batch size 1, and gradient accumulation. QLoRA is
available if the unquantized rehearsal does not fit.

Resource gates:

1. toy offline smoke;
2. real tokenizer/model seam test;
3. 64–128 problem overfit/gradient rehearsal;
4. 4,000-problem one-seed DEV pilot;
5. frozen full Phase-1 matrix;
6. operators only after the state-order gate;
7. feedback with checkpointing disabled only after preservation is established.

Do not describe LoRA memory savings as DiffusionBlocks-style independent training.
DAST V1 still backpropagates through the full model graph.

---

## 12. Threats and mitigations

| Threat | Mitigation |
|---|---|
| answer-token leakage | prompt-only mask; perturbation unit test |
| target drift | frozen input embedding and fixed projection; fail closed |
| incompatible depth bases | one shared projector |
| generic auxiliary-loss gain | shuffled, reverse, and final-repeated controls |
| literal wording memorization | compact-state target; literal arm kept as control |
| variable trace confound | normalized endpoint-pinned mapping, repeated-state record |
| parameter/update mismatch | matched LoRA and budget manifests |
| probe-only epiphenomenon | phase-gated semantic interventions |
| operator collapse | diversity, load diagnostics, held-out transfer |
| TEST leakage | machine-enforced preregistration hash gate |
| selective seed reporting | immutable run directories and all-seed aggregation |
| false efficiency claim | separate capability, VRAM, FLOP, and local-gradient gates |
| overfitting synthetic templates | held-out generators, paraphrases, deeper traces |

---

## 13. Success, revision, and retirement criteria

### Support V1 only when

- ordered beats shuffled and reverse under frozen paired criteria;
- final-repeated does not explain the effect;
- gains generalize beyond training templates and trace depths;
- preservation remains within the frozen boundary;
- audit finds no leakage or budget mismatch.

### Revise when

- semantic states are learnable but checkpoint schedule is poorly aligned;
- ordered improves mechanism metrics but not capability;
- target geometry appears too tied to surface text;
- only particular domains/operators benefit.

A revision must specify one bounded new mechanism and retain the negative V1
result. It cannot simply search many schedules until one passes.

### Retire the tested formulation when

- ordered does not beat both decisive controls across frozen seeds;
- final-repeated matches ordered;
- causal interventions are non-directional;
- preservation costs erase capability gains;
- effects vanish under held-out generators/deeper traces.

Retirement applies to the tested formulation, not automatically to every possible
semantic-depth method.

---

## 14. Deliverables

The V1 branch must contain:

- this SSOT;
- algorithm and evidence documents;
- strict configs and split guard;
- reusable data/alignment/tokenization modules;
- shared semantic and operator modules;
- real Hugging Face/LoRA path;
- offline toy qualification;
- unit/integration tests and CI;
- branch progress record;
- no fabricated real-model results.

A later findings document is created only after real artifacts exist.

---

## 15. Current decision

The idea is classified as **scientifically plausible and under-explored**, while
the literal sentence-exclusive/freezing formulation is considered high-risk. The
implemented V1 therefore begins with end-to-end differentiable, cumulative
semantic-state supervision and decisive ordering controls. Independent block
training remains a gated research endpoint rather than an assumption.
