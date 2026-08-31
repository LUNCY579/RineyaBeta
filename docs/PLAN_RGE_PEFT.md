# Rineya Geometry-Efficient PEFT (RGE-PEFT)
## Single Source of Truth: Research & Development Roadmap

**Status:** APPROVED SINGLE SOURCE OF TRUTH / DOCUMENTATION ONLY — implementation has not started and no experimental result is claimed by this document  
**Program version:** SSOT v1.1  
**Date frozen:** 2026-08-30  
**Date approved for repository governance:** 2026-08-30  
**Target repository:** `LUNCY579/RineyaBeta`  
**Canonical repository path:** `docs/PLAN_RGE_PEFT.md`  
**Current remote snapshot observed while drafting:** `main @ 3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Working project name:** **Rineya Geometry-Efficient PEFT (RGE-PEFT)**  
**Inspiration:** Puro-2B / MuonH / effective-LR control / curriculum model averaging (CMA)  
**Scope:** parameter-efficient post-training of an already capable Rineya checkpoint, not from-scratch pretraining

---

# 0. Authority, governance, and how this document must be used

This document is the **single source of truth (SSOT)** for the RGE-PEFT research program until it is explicitly superseded by a versioned amendment. Future coding agents, experiment agents, and reviewers should read this document before changing the implementation, launching runs, interpreting results, or writing claims.

The following rules are binding:

1. **The design is pre-registered.** Results must not be back-edited into the original hypotheses, thresholds, or arm definitions to make an outcome appear stronger.
2. **Amendments are append-only.** If implementation reveals that a design assumption is impossible or materially wrong, add an amendment with date, commit SHA, rationale, and whether the change occurred before or after observing outcome data.
3. **Code does not silently redefine the experiment.** Any implementation behavior that differs from this SSOT must be documented before the affected confirmatory run.
4. **Negative results are first-class results.** A failed Muon transfer, inactive DeltaBall constraint, harmful curriculum, or failed CMA compression must be recorded rather than removed from the program history.
5. **Exploratory and confirmatory evidence must remain separate.** Hyperparameter discovery, debugging, and mechanism probes do not count as confirmatory evidence.
6. **The sealed evaluation boundary remains sealed.** RineyaBeta's sealed test assets must not be used for optimizer selection, curriculum selection, checkpoint selection, rank selection, or learning-rate tuning.
7. **Matched comparisons are mandatory.** Token budget, trainable capacity, data membership, data order where appropriate, target modules, decoding protocol, evaluation cases, and hardware conditions must be controlled according to the arm being tested.
8. **No "Puro fine-tuning" overclaim.** RGE-PEFT is a Puro-inspired transfer experiment. Puro established results in pretraining and some post-training settings; it did **not** establish that MuonH/CMA will provide the same efficiency improvements for LoRA on Rineya.
9. **No "intelligence amplification" claim without operational definition.** The program measures capability on frozen tasks and quality gates, retention, token efficiency, and GPU-hour efficiency. Any broader language must be downstream of evidence.
10. **Repository artifacts govern over chat summaries.** Once this file is committed into RineyaBeta, the committed SSOT plus linked manifests/results supersede conversational descriptions.

### Planned evidence files

When implementation begins, the program should eventually maintain:

```text
docs/PLAN_RGE_PEFT.md                 # this SSOT; pre-registration + amendments
docs/FINDINGS_RGE_PEFT.md             # evidence-backed findings only
docs/NEGATIVE_RESULTS.md               # existing shared negative-results ledger; append program results
experiments/rge_peft/README.md         # run map and artifact conventions
experiments/rge_peft/PROGRESS.md       # living milestone/status dashboard; no scientific claims
experiments/rge_peft/RESULTS_LEDGER.jsonl
experiments/rge_peft/CLAIMS_LEDGER.md
```

The SSOT should explain **what was planned**. Findings should explain **what happened**. They should not be collapsed into one hindsight-edited document.

---

# 1. Executive decision

The program will **not** attempt to reproduce Puro-2B's trillion-token pretraining pipeline inside RineyaBeta. Instead, it will transfer the subset of Puro ideas that plausibly improve **optimization efficiency per trainable parameter** into RineyaBeta's existing LoRA-scale training system.

The canonical v1 hypothesis is:

> An already capable Rineya checkpoint may gain more capability per supervised token and per GPU-hour if its low-rank update is trained with better matrix geometry, explicit base-relative update diagnostics/constraints, curriculum-aware scheduling, and mathematically correct late-checkpoint averaging.

The first implementation should therefore extend the existing deterministic RineyaBeta PEFT infrastructure rather than importing Megatron-LM as a new training stack.

### v1 components

The program will test, in order:

1. **Matched AdamW-LoRA control** — preserve the current reference behavior.
2. **Muon-LoRA** — apply Muon-style optimization to eligible two-dimensional adapter matrices while retaining AdamW for any non-matrix trainables.
3. **Effective-update instrumentation** — measure the induced low-rank update relative to each frozen base matrix.
4. **DeltaBall** — a PEFT-specific trust region on the induced update, inspired by Hyperball but mathematically distinct from Puro's fixed-radius full-weight sphere.
5. **Component-local curriculum** — reorder the same examples while preserving membership and mixture counts.
6. **Late constant-LR continuation** — create a short, controlled averaging trajectory rather than averaging arbitrary checkpoints.
7. **Delta-space CMA** — average actual adapter-induced weight deltas, not raw LoRA factors, then recompress to the original rank for a matched-capacity endpoint.

### deliberately deferred components

These are **not** part of the initial proof:

- FP8 training. Puro's FP8 result is hardware/system specific and would confound the optimizer-transfer question. RGE-PEFT v1 should remain BF16-base / FP32-trainable as the existing Rineya trainer does unless the host forces a different precision.
- Full Puro-Megatron integration.
- From-scratch pretraining.
- Dynamic online curriculum scoring during the production run.
- Rank increases as a substitute for algorithmic improvement.
- DoRA spherical projection.
- Native manifold/projected low-rank optimization.
- Simultaneous combination with unrelated Rineya research mechanisms.

Those become later phases only after the low-cost transfer hypothesis survives matched controls.

---

# 2. Why this is a sensible RineyaBeta experiment

Puro-2B reports a from-scratch 2B model trained on approximately 1.4T tokens, with a canonical run consuming 22,514 active GPU-hours. Its efficiency stack combines consumer hardware, blockwise FP8, MuonH/Hyperball optimization, curriculum model averaging, and data design. Puro reports a matched scaling-ladder estimate of roughly a **1.19x quality-equivalent compute multiplier** for the complete MuonH recipe versus its tuned Muon baseline, and its curriculum/CMA system changes the Phase-2 cost-efficiency frontier. Those are pretraining results and must not be assumed to transfer automatically.

What makes Rineya different is that we do **not** begin at random initialization. Rineya already contains pretrained linguistic and reasoning representations. The expensive problem solved by trillion-token pretraining is therefore not our problem. RGE-PEFT asks a narrower question:

> Can a small trainable subspace reorganize and specialize an existing model more efficiently if the optimizer controls the geometry of that subspace more deliberately?

This is precisely the kind of question that can be tested at consumer-GPU scale.

RineyaBeta already has unusually useful infrastructure for such a test:

- deterministic LoRA training;
- exact-resume state handling;
- memory preflight and a hard allocator cap;
- adapter checkpoints;
- fixed data plans and plan hashes;
- frozen evaluation assets;
- `chat_dev` and `rag_dev` gates;
- a sealed RAG test boundary;
- a quality scorer with versioned configuration;
- explicit negative-result and experiment-design conventions.

The existing healing configuration already trains an all-linear rank-16 LoRA over `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` on the 474,021,888-parameter R2 core. That means RGE-PEFT should be introduced as a **new optimizer/geometry layer around a proven PEFT substrate**, not as a new training framework.

---

# 3. Program objective and non-goals

## 3.1 Primary objective

Maximize:

\[
\text{capability improvement per supervised token}
\]

and secondarily:

\[
\text{capability improvement per active GPU-hour}
\]

while preserving general Rineya behavior and keeping trainable parameter count matched to the AdamW-LoRA control.

The program is successful only if the new recipe yields a better **capability / retention / compute** tradeoff than the matched control.

## 3.2 What "capability" means in this program

Capability is not a single informal impression. It is a frozen vector including:

- targeted reasoning/correctness performance;
- RAG/grounding quality;
- general short-response quality;
- contract fulfillment;
- factual/logic consistency;
- critical defect rates;
- retention relative to the starting checkpoint;
- selected domain-specific exact or deterministic metrics where available.

The program may compute a pre-registered composite, but component results must always remain visible so that one gain cannot hide a serious regression.

## 3.3 Explicit non-goals

RGE-PEFT does **not** claim to:

- replace 1.4T-token pretraining with a few hours of fine-tuning;
- manufacture knowledge absent from both the base model and training signal;
- turn a sub-billion-parameter model into a universally equivalent much larger model;
- prove that all Puro efficiency factors transfer to PEFT;
- reproduce Puro's FP8/hardware results;
- use Puro's mixed-license corpus as Rineya training data;
- merge unrelated Rineya mechanisms before the optimizer effect is isolated;
- select a winner from the sealed test set.

---

# 4. Scientific questions and pre-registered hypotheses

## H0 — infrastructure parity

**Question:** Can the new experiment harness reproduce the existing AdamW-LoRA behavior without changing the model, data plan, or training semantics?

**Pass condition:** The reference AdamW arm must reproduce the legacy path under the same seed/config at the level expected by RineyaBeta's deterministic contract. If exact parameter equality is feasible, require exact adapter equality; otherwise the discrepancy must be explained and a stricter common baseline frozen before any optimizer comparison.

**If H0 fails:** Stop. No optimizer experiment is interpretable until parity is restored.

---

## H1 — Muon transfer to LoRA factors

**Question:** Does Muon-style optimization of eligible LoRA matrices improve learning efficiency over AdamW at fixed data, rank, target modules, and supervised-token budget?

**Null:** Muon-LoRA is no better than matched AdamW-LoRA after fair LR calibration.

**Evidence needed:** Better frozen-dev capability at equal token budget, or equal capability in materially fewer tokens/GPU-hours, without unacceptable retention loss.

**Important limitation:** LoRA factorization is non-identifiable. A representation `BA` can be transformed by `B Q` and `Q^{-1} A` without changing the induced delta. Therefore a factor-space optimizer may be sensitive to representation geometry even when the effective model is unchanged. This is a reason to measure delta-space behavior, not a reason to assume Muon will work.

---

## H2 — effective-update geometry is a useful descriptor

**Question:** Does the base-relative effective step size explain learning dynamics better than raw scalar LR alone?

For target matrix \(l\), define:

\[
W_l^{\mathrm{eff}}(t)=W_{0,l}+\Delta W_l(t)
\]

and the **base-relative effective step**:

\[
\rho_l^{\Delta}(t)=
\frac{\left\|\Delta W_l(t+1)-\Delta W_l(t)\right\|_F}
{\left\|W_{0,l}\right\|_F}.
\]

The program will log this quantity alongside raw LR, adapter gradient norm, loss, and accumulated delta magnitude.

**Supportive evidence:** Similar `rho_delta` trajectories produce more similar loss/capability trajectories than raw-LR matching, or `rho_delta` reveals systematic layer outliers correlated with instability/regression.

**If unsupported:** Keep it as a diagnostic only; do not force an ELR-control mechanism into the production recipe.

---

## H3 — DeltaBall improves the PEFT tradeoff

**Question:** Does constraining the adapter-induced delta relative to the frozen base prevent destructive updates while retaining useful learning?

Define the relative delta size:

\[
d_l(t)=\frac{\|\Delta W_l(t)\|_F}{\|W_{0,l}\|_F}.
\]

DeltaBall enforces:

\[
d_l(t)\leq \epsilon_l(t).
\]

This is a **trust-region ball around the base model**, not Puro's fixed-radius Hyperball sphere.

**Supportive evidence:** At matched token budget, DeltaBall improves the capability-retention frontier or prevents unstable/high-regression runs that occur without it.

**If the constraint almost never activates:** It has not been tested meaningfully; report it as inactive rather than crediting it for an improvement.

---

## H4 — curriculum ordering transfers to PEFT

**Question:** If the same training examples and component counts are consumed in a different deterministic order, does a component-local curriculum improve the final capability vector?

**Critical control:** Membership and total component token counts remain fixed. The first curriculum experiment changes **order**, not data quantity.

**Supportive evidence:** Curriculum ordering improves the frozen target metric while preserving general retention under the same optimizer, schedule, examples, and token budget.

---

## H5 — constant-tail Delta-CMA improves the endpoint

**Question:** Does a short constant-LR continuation followed by delta-space averaging provide a better endpoint than the final individual checkpoint?

**Critical controls:**

- compare against the final checkpoint from the same trajectory;
- compare against the best predeclared single checkpoint if checkpoint selection is allowed by the dev protocol;
- retain original rank for the primary CMA endpoint;
- report exact high-rank average as a diagnostic only.

**Supportive evidence:** Rank-matched Delta-CMA improves the frozen dev endpoint without adding inference capacity.

---

## H6 — interaction hypothesis

Muon, DeltaBall, curriculum, constant-tail continuation, and CMA may interact. The program must not infer that every ingredient is independently beneficial from one combined winner.

The staged arm structure is designed to identify which ingredients actually survive.

---

# 5. Mathematical contract

## 5.1 Baseline LoRA

For frozen base matrix \(W_0\):

\[
W_{\mathrm{eff}} = W_0 + \Delta W,
\qquad
\Delta W = s B A,
\qquad
s=\frac{\alpha}{r}.
\]

The current Rineya healing path uses rank \(r=16\), alpha 32, zero LoRA dropout, and all major attention/MLP projection modules. RGE-PEFT v1 keeps those values fixed for the principal comparison.

### Capacity rule

No optimizer arm may change rank, target modules, alpha, trainable biases, or other trainable parameter groups unless the arm is explicitly a capacity experiment. Capacity changes belong to a later phase.

---

## 5.2 Why raw A/B norms are insufficient

The factorization has gauge freedom:

\[
BA=(BQ)(Q^{-1}A)
\]

for invertible \(Q\). Therefore:

- `||A||` and `||B||` are not intrinsic properties of the effective model;
- independently projecting A and B to arbitrary spheres is not a well-defined delta-space constraint;
- independently averaging A and B across checkpoints is not mathematically equivalent to averaging the induced weights.

The program must therefore treat **\(\Delta W\)** as the canonical object for geometry diagnostics and checkpoint averaging.

---

## 5.3 Puro Hyperball versus RGE-PEFT DeltaBall

Puro MuonH fixes a selected full matrix to its initial Frobenius radius \(R=\|W_0\|_F\), normalizes the Muon update, takes a prescribed relative step, and projects the resulting full matrix back to that fixed-radius sphere.

A literal transfer to LoRA delta space is ill-defined because the conventional LoRA initialization has:

\[
\Delta W_0=0,
\]

which would give a fixed sphere of radius zero.

Therefore v1 defines **DeltaBall**:

\[
\mathcal{B}_l(t)=
\left\{\Delta W:\|\Delta W\|_F\leq
\epsilon_l(t)\|W_{0,l}\|_F\right\}.
\]

After the optimizer proposes \(A',B'\), compute:

\[
\Delta W'=sB'A'.
\]

If:

\[
\|\Delta W'\|_F > \epsilon_l(t)\|W_{0,l}\|_F,
\]

let:

\[
c=
\frac{\epsilon_l(t)\|W_{0,l}\|_F}
{\|\Delta W'\|_F}.
\]

A symmetric factor projection is:

\[
A'\leftarrow \sqrt{c}A',
\qquad
B'\leftarrow \sqrt{c}B',
\]

which gives:

\[
sB'A' \leftarrow c\Delta W'.
\]

This projection depends on the induced delta norm rather than arbitrary individual factor norms.

### Optimizer-state rule

v1 treats DeltaBall as a projected optimizer: parameters are projected, optimizer moments are not automatically rescaled. Boundary-hit frequency and repeated radial pressure must be logged. If moment mismatch causes pathological repeated boundary hits, an optimizer-state-aware projection may be investigated only as a separately declared variant.

---

## 5.4 DeltaBall schedule

A constant strict radius from step zero may be unnecessarily harsh. The first controlled version uses a deterministic radius schedule:

\[
\epsilon_l(t)=\epsilon_{l,\max}\,g(t)
\]

where `g(t)` is a predeclared monotonic warm-in function, initially linear unless a pilot demonstrates a numerical reason to change it.

The final `epsilon_max` candidates will be derived **before confirmatory runs** from the matched AdamW control's observed delta-scale distribution on pilot/training data. This avoids importing Puro's full-weight effective LR values into an unrelated PEFT scale.

The derivation procedure, not an ad hoc chosen number, is pre-registered:

1. run the matched control;
2. compute the end-of-pilot distribution of \(d_l\) across target matrices;
3. define a robust reference \(d_{ref}\), initially the 95th percentile across matrices;
4. screen `epsilon_max` multipliers `{0.5, 1.0, 2.0} × d_ref` on the exploratory pilot;
5. freeze the selected value before confirmation.

If the projection never activates at any screened radius, DeltaBall is considered unsupported in that regime.

---

## 5.5 Geometry diagnostics

For each targeted base matrix, log or periodically sample:

### Delta magnitude

\[
d_l(t)=\frac{\|\Delta W_l(t)\|_F}{\|W_{0,l}\|_F}.
\]

### Base-relative effective step

\[
\rho_l^{\Delta}(t)=
\frac{\|\Delta W_l(t+1)-\Delta W_l(t)\|_F}
{\|W_{0,l}\|_F}.
\]

### Effective-weight radius drift

\[
r_l(t)=
\frac{\|W_{0,l}+\Delta W_l(t)\|_F}{\|W_{0,l}\|_F}.
\]

### Angular displacement

\[
\theta_l(t)=
\arccos
\frac{\langle W_{0,l},W_{0,l}+\Delta W_l(t)\rangle_F}
{\|W_{0,l}\|_F\,\|W_{0,l}+\Delta W_l(t)\|_F}.
\]

### Projection activity

Log:

- whether the matrix hit the DeltaBall boundary;
- pre-projection delta ratio;
- post-projection delta ratio;
- projection scale `c`;
- cumulative boundary-hit rate.

### Runtime policy

Exact dense `BA` materialization for every module on every step may add unnecessary overhead. Phase 1 must benchmark three diagnostic modes:

1. every step;
2. every `N` steps;
3. low-rank algebra where possible.

The confirmatory configuration should use the cheapest mode that preserves the required mechanism semantics and evidence.

---

## 5.6 Muon parameter grouping

The initial Muon arm follows the conceptual parameter split used in Puro:

- eligible **two-dimensional LoRA matrices** → Muon-style optimizer;
- non-matrix trainables, if introduced later (e.g. scalar gains / DoRA magnitude) → AdamW;
- frozen base model remains untouched.

RGE-PEFT must not apply Muon to frozen base matrices merely to mimic Puro. The experiment is specifically about optimization inside the trainable PEFT subspace.

### Numerical policy

The existing trainer keeps trainable LoRA parameters in FP32 while the base model may use BF16. Preserve that policy initially. Orthogonalization/Newton-Schulz operations should be validated for the highly rectangular small-rank LoRA matrices and should fail loudly on NaN/Inf.

---

# 6. Delta-space Curriculum Model Averaging

## 6.1 Why factor averaging is prohibited as the primary method

Given checkpoint `i`:

\[
\Delta W_i=s_iB_iA_i.
\]

The correct equal-weight average of effective models sharing the same frozen base is:

\[
\overline{W}
=\frac{1}{K}\sum_i(W_0+\Delta W_i)
=W_0+\frac{1}{K}\sum_i\Delta W_i.
\]

Therefore:

\[
\overline{\Delta W}
=\sum_i w_i\,s_iB_iA_i,
\qquad \sum_i w_i=1.
\]

Averaging `A_i` and `B_i` independently is generally wrong because of factor gauge freedom and cross terms.

---

## 6.2 Exact average versus rank-matched average

The exact average of `K` rank-`r` deltas may have rank up to `Kr`.

Two outputs must be distinguished:

### CMA-exact

An exact rank-≤`Kr` representation used **only as a diagnostic/oracle**. It tells us whether averaging itself helps before rank compression.

### CMA-r

The primary matched-capacity output. Compress \(\overline{\Delta W}\) back to original rank `r` using truncated SVD (or an algebraically equivalent low-rank method).

Only **CMA-r** can be compared as an equal-rank deployment candidate to the single rank-r checkpoint.

---

## 6.3 Efficient low-rank averaging without materializing dense deltas

For equal or known weights, construct factor concatenations such that:

\[
\overline{\Delta W}=B_cA_c.
\]

One convenient construction is:

\[
B_c=[\sqrt{w_1s_1}B_1,\ldots,\sqrt{w_Ks_K}B_K]
\]

and

\[
A_c=
\begin{bmatrix}
\sqrt{w_1s_1}A_1\\
\vdots\\
\sqrt{w_Ks_K}A_K
\end{bmatrix}.
\]

Then perform thin QR decompositions:

\[
B_c=Q_BR_B,
\qquad
A_c^T=Q_AR_A.
\]

The small core is:

\[
M=R_BR_A^T.
\]

Compute:

\[
M=U\Sigma V^T.
\]

The rank-r approximation is:

\[
\overline{\Delta W}_r=
(Q_BU_r)\Sigma_r(V_r^TQ_A^T).
\]

Re-factor it into a PEFT-compatible `B_new A_new`, accounting explicitly for the desired LoRA scaling `s_new`.

This avoids constructing every full dense target matrix solely to perform SVD.

---

## 6.4 CMA diagnostics

For every averaged target matrix report:

- exact combined rank upper bound;
- retained rank `r`;
- retained Frobenius energy:

\[
E_r=\frac{\|\overline{\Delta W}_r\|_F^2}
{\|\overline{\Delta W}\|_F^2};
\]

- reconstruction error;
- singular-value spectrum summary;
- adapter file size;
- exact-vs-compressed evaluation delta.

If `CMA-exact` improves but `CMA-r` loses the gain and retained energy is poor, the finding is **rank bottleneck**, not "CMA failed."

---

# 7. Repository integration architecture

## 7.1 Preserve the current trainer as the reference

The current `scripts/train/heal.py` already provides:

- deterministic initialization;
- plan hashing;
- exact resume state;
- data/mix hashing;
- memory probing;
- checkpointing;
- signal-safe exits;
- LoRA model construction;
- AdamW baseline behavior.

The first RGE-PEFT implementation should **not perform a risky large refactor of `heal.py` before H0 is proven**.

Preferred architecture:

```text
scripts/train/heal.py                    # legacy/reference path; preserve
scripts/train/rge_peft.py                # new experimental harness
scripts/train/optim/
    __init__.py
    factory.py                           # AdamW / Muon / hybrid construction
    muon.py                              # minimal attributed implementation
    delta_geometry.py                    # norms, rho_delta, angles, DeltaBall projection
    schedules.py                         # LR + optional radius schedules
scripts/train/curriculum.py              # deterministic materialization/order logic
scripts/train/cma.py                     # delta-space checkpoint averaging + compression
scripts/measure/peft_geometry.py         # offline geometry analysis
scripts/verify/
    verify_rge_baseline_parity.py
    verify_muon_math.py
    verify_delta_geometry.py
    verify_deltaball_projection.py
    verify_cma_math.py
    verify_rge_resume.py
```

If code reuse requires extracting shared utilities from `heal.py`, that extraction must first pass baseline parity before optimizer experiments proceed.

---

## 7.2 Configuration layout

Use a dedicated namespace rather than mixing experimental knobs into historical healing configs:

```text
configs/rge_peft/
    smoke_adamw.yaml
    smoke_muon.yaml
    pilot_adamw.yaml
    pilot_muon.yaml
    pilot_adamw_deltaball.yaml
    pilot_muon_deltaball.yaml
    confirm_control.yaml
    confirm_candidate.yaml
    curriculum_uniform.yaml
    curriculum_ordered.yaml
    cma_continuation.yaml
```

Every config must specify or resolve:

```yaml
program_version:
seed:
model:
data:
lora:
optimizer:
geometry:
schedule:
curriculum:
cma:
train:
memory:
determinism:
run:
```

No hidden defaults may affect a confirmatory comparison. Resolved configuration must be copied into each run manifest.

---

## 7.3 Checkpoint schema additions

RGE-PEFT checkpoints must preserve all state required for exact resume:

- adapter state;
- optimizer state;
- scheduler state if stateful;
- RNG state;
- plan position;
- plan hash;
- data/mix hash;
- code/environment fingerprint;
- optimizer implementation/version;
- Muon momentum/state;
- DeltaBall schedule parameters;
- cached base-matrix norms or their deterministic derivation metadata;
- geometry sampling phase if sampling cadence affects logging only;
- curriculum manifest hash;
- current curriculum bucket/progress if not derivable from global step;
- CMA-eligibility metadata for saved checkpoints.

A resumed candidate that silently changes optimizer geometry is invalid.

---

# 8. Data and curriculum contract

## 8.1 First principle: hold membership constant

The first curriculum experiment must consume the **same examples** as its uniform control.

The experimental variable is order.

This prevents a common error:

> "curriculum improved the model" when the curriculum arm actually contained better or more specialized data.

---

## 8.2 Component-local ordering

Puro orders samples within each source component rather than comparing source-specific quality scores globally. RGE-PEFT should preserve that principle.

For every component/capability group:

1. identify a valid within-component ordering signal;
2. sort only within that component;
3. convert order to normalized cumulative supervised-token progress;
4. construct aligned buckets across components;
5. preserve total component membership and token mass as closely as the existing deterministic planner allows.

Possible ordering signals, in priority order:

1. existing human-reviewed quality/difficulty metadata;
2. explicit dataset-provided difficulty/quality label with documented meaning;
3. deterministic programmatic difficulty feature that was frozen before outcome evaluation;
4. fixed random order if no defensible score exists.

A source-local score must not be treated as globally comparable unless there is evidence that its scale is calibrated across components.

---

## 8.3 Quality and difficulty are different hypotheses

Do not silently conflate:

- low-quality → high-quality curriculum;
- easy → hard curriculum;
- broad → specialized curriculum;
- short → long sequence curriculum.

The v1 curriculum must choose one declared interpretation. Additional orderings are exploratory arms.

The recommended first Rineya curriculum is **broad/easier stabilization → harder capability examples while preserving replay**, but the exact score must be frozen after inspecting available metadata and before outcome runs.

---

## 8.4 Static materialization

The production curriculum order must be materialized before training into a manifest/shard/plan with:

- source/component identifiers;
- row IDs;
- score provenance;
- normalized rank;
- bucket assignment;
- token counts;
- seed;
- SHA256.

No online model-dependent re-ranking should occur during the confirmatory run.

---

# 9. Learning-rate and continuation contract

Puro's absolute Hyperball effective LRs must **not** be copied directly into PEFT. Full-weight pretraining and rank-16 SFT operate at different scales.

## 9.1 AdamW anchor

The current Rineya healing configuration uses an AdamW LR of `1e-4` with warmup and cosine decay. This remains the principal baseline anchor unless the implementation-day audit demonstrates that another existing Rineya control has already superseded it.

## 9.2 Fair Muon calibration

Rather than declaring a raw Muon LR by analogy to Puro, calibrate it against the induced update scale.

Exploratory calibration protocol:

1. run the AdamW reference for a short fixed token budget;
2. measure the distribution of \(\rho_l^{\Delta}\) after warmup;
3. run a small log-spaced Muon LR bracket;
4. identify candidates whose median/robust base-relative step magnitude is approximately `0.5×`, `1×`, and `2×` the AdamW reference;
5. select the stable pilot candidate according to the predeclared pilot metric;
6. freeze that LR before confirmatory comparisons.

This prevents an unfair comparison between a tuned baseline and an arbitrarily scaled experimental optimizer.

---

## 9.3 Short-run Puro-inspired schedule

The initial schedule experiment should not imitate a trillion-token curve literally. It should preserve the relevant structure:

1. **warmup**;
2. **main learning segment** with a controlled decay;
3. **late constant-LR continuation** long enough to produce several meaningfully separated checkpoints;
4. **checkpoint averaging** only across that continuation.

The constant tail is a separate causal ingredient and must not be silently bundled into every Muon run.

### Six-checkpoint default

Puro's canonical CMA averages six late checkpoints. RGE-PEFT will use six as the initial default because it is a concrete, testable transfer choice, not because six is assumed optimal.

Spacing should be defined in **supervised tokens or optimizer steps under a fixed plan**, recorded in the config, and identical across matched CMA comparisons.

---

# 10. Experimental roadmap

## Phase 0 — implementation-day preflight and freeze

### Purpose

Ensure the branch starts from a known current repository state and that the experiment is not built on stale assumptions.

### Required actions

- fetch current `main` and record the exact base SHA;
- create a dedicated branch, recommended name:

```text
research/rge-peft-v1
```

- record whether any new training/evaluation work landed after this SSOT snapshot;
- freeze model path, tokenizer, LoRA target list, rank, alpha, data mix, eval hashes, PyTorch/CUDA versions, GPU, and memory cap;
- verify Puro/Muon code licensing/attribution for any code ported into RineyaBeta;
- copy this SSOT into `docs/PLAN_RGE_PEFT.md` without rewriting hypotheses;
- create the experiment artifact skeleton;
- verify the sealed test set remains inaccessible to ordinary gating.

### Gate P0

Do not write optimizer code until the baseline configuration and evaluation assets are pinned.

---

## Phase 1 — substrate, baseline parity, and geometry instrumentation

### Purpose

Prove that we can observe LoRA delta geometry without changing baseline training.

### Work

1. Add RGE-PEFT experimental harness.
2. Reuse current dataset planner, memory probe, checkpointing, deterministic environment, and model-loading behavior.
3. Add base-matrix norm capture for every LoRA target.
4. Add geometry measurement functions.
5. Add run manifests and geometry JSONL.
6. Keep optimizer = AdamW.
7. Run smoke parity against legacy `heal.py`.

### Required verification

- identical trainable parameter names/count;
- identical LoRA initialization;
- identical plan hash;
- identical first batches;
- identical LR sequence;
- identical loss sequence under reference configuration if possible;
- identical final adapter for a short deterministic run if the code paths are mathematically identical;
- resume-vs-uninterrupted equivalence;
- geometry logging has no gradient side effect;
- no unbounded GPU-memory growth.

### Instrumentation-overhead rule

Benchmark training with geometry logging disabled versus candidate logging cadence.

Preferred confirmatory overhead: **≤10% wall-time** from measurement-only instrumentation.  
Hard screening threshold: if exact per-step geometry costs **>20%**, reduce measurement cadence or use low-rank formulas. Do not make the entire program slower solely to collect redundant diagnostics.

### Kill condition

Any unexplained baseline divergence stops the program before Phase 2.

---

## Phase 2 — optimizer pilot: AdamW versus Muon

### Purpose

Test the simplest transfer: optimizer geometry only.

### Arms

| Arm | Optimizer | DeltaBall | Curriculum | CMA | Role |
|---|---|---:|---:|---:|---|
| O-A | AdamW | no | uniform/reference | no | canonical control |
| O-M | Muon on LoRA matrices | no | same | no | optimizer transfer |

If non-matrix trainables exist, O-M becomes Muon + AdamW hybrid with the split recorded exactly.

### Pilot budget

Pilot budget is defined in **supervised tokens**, not merely epochs. On implementation day set:

```text
T_pilot = min(10% of the planned confirmatory supervised-token budget,
              a predeclared cap that keeps one pilot arm inexpensive)
```

The exact token count is frozen in the experiment manifest before seeing pilot evaluation results.

### LR calibration

Use the procedure in §9.2. LR discovery is exploratory. Only the frozen selected LR enters confirmation.

### Measurements

- train loss versus supervised tokens;
- dev loss if available without leaking sealed assets;
- `rho_delta` traces;
- delta magnitude and angle;
- throughput;
- peak VRAM;
- optimizer-state size;
- NaN/Inf incidents;
- gradient clipping incidence;
- frozen small development capability evaluation.

### Proceed rule

Muon proceeds if it is stable and shows either:

- a capability signal at equal tokens; or
- a favorable learning-curve/efficiency signal strong enough to justify a full matched run.

If Muon is neutral, DeltaBall may still be tested on AdamW because H3 is logically separable from H1.

---

## Phase 3 — DeltaBall factorial test

### Purpose

Separate optimizer effects from trust-region effects.

### Arms

| Arm | Optimizer | DeltaBall | Purpose |
|---|---|---:|---|
| G-A | AdamW | no | matched control |
| G-M | Muon | no | optimizer effect |
| G-AD | AdamW | yes | constraint effect under baseline optimizer |
| G-MD | Muon | yes | interaction |

All four must use the same:

- base checkpoint;
- rank/alpha/targets;
- training examples;
- example order;
- supervised-token budget;
- gradient accumulation plan;
- precision;
- evaluation cases;
- seed set for confirmation.

### Exploratory DeltaBall calibration

Screen the predeclared `{0.5, 1.0, 2.0} × d_ref` radius choices. Freeze one before the confirmatory factorial.

### Mechanism diagnostics

A DeltaBall arm must report:

- fraction of optimizer steps with ≥1 projected matrix;
- per-layer boundary-hit rate;
- median projection magnitude;
- whether the same modules repeatedly dominate projection;
- delta magnitude distribution;
- capability/retention impact.

### Interpretation rules

- **No boundary hits + better score:** do not attribute the gain to DeltaBall.
- **Frequent projection + worse score:** constraint may be too restrictive; only the predeclared exploratory sweep may be used to choose the confirmatory radius.
- **AdamW+DeltaBall wins but Muon+DeltaBall does not:** keep the constraint, reject the optimizer interaction.
- **Muon wins but DeltaBall adds nothing:** retain Muon only.

---

## Phase 4 — curriculum and schedule isolation

Only the best surviving geometry recipe from Phase 3 enters this phase.

### Purpose

Test Puro's second transferable idea: coordinated data ordering and late optimization.

### Minimum arms

| Arm | Data order | Main schedule | Constant tail | CMA |
|---|---|---|---:|---:|
| C-U | uniform/reference | frozen main schedule | no | no |
| C-C | component-local curriculum | same | no | no |
| C-CT | same curriculum | same main segment | yes | no |
| C-CTM | same curriculum | same main segment | yes | yes |

### Optional interaction control

If C-CTM wins and compute permits, add:

| Arm | Data order | Constant tail | CMA |
|---|---|---:|---:|
| C-UTM | uniform | yes | yes |

This determines whether CMA/constant-tail benefits require the curriculum interaction.

### Curriculum contract

The uniform and curriculum arms use identical example membership and total supervised-token counts. Materialized order hashes are recorded.

### Decision rule

Do not automatically keep CMA because Puro used it. Puro itself reports that checkpoint averaging was not independently beneficial in every ordering condition. RGE-PEFT must earn each ingredient.

---

## Phase 5 — Delta-CMA implementation and compression study

### Required tests before model evaluation

Synthetic matrices must demonstrate:

1. delta-space averaging equals explicit dense effective-weight averaging;
2. result is invariant to equivalent LoRA gauge transformations;
3. raw factor averaging is not silently used;
4. low-rank QR/SVD implementation matches dense truncated SVD within numerical tolerance;
5. requested final LoRA scaling reproduces the compressed delta;
6. saved adapter loads through the normal PEFT/evaluation path;
7. original rank and trainable/inference parameter count are preserved for CMA-r.

### Endpoint comparison

Evaluate:

- final individual checkpoint;
- predeclared best eligible individual checkpoint if protocol allows;
- CMA-exact diagnostic;
- CMA-r primary output.

### Compression gate

Always report retained energy. If CMA-r loses a large fraction of delta energy, do not describe it as equivalent to the exact average.

---

## Phase 6 — confirmatory full-budget comparison

### Purpose

Determine whether the complete selected recipe really improves Rineya at matched capacity.

### Required endpoints

At minimum:

```text
CONTROL = best fair AdamW-LoRA reference recipe
CANDIDATE = frozen RGE-PEFT recipe selected before confirmatory evaluation
```

### Seeds

Use multiple predeclared seeds for confirmation; default target is **three seeds** if compute allows. If the hardware budget cannot support three full seeds, the result must be labeled limited-seed evidence rather than silently treated as replicated.

### Evaluation order

1. training-time diagnostics;
2. frozen development capability suite;
3. `chat_dev`;
4. `rag_dev`;
5. freeze candidate and all analysis;
6. only then, if promotion criteria pass, open the sealed final evaluation once according to the repository's existing governance.

### No sealed checkpoint shopping

If the sealed result is disappointing, do not return to the sealed set with a new LR/rank/checkpoint. That would destroy its status.

---

## Phase 7 — rank and family transfer

Only after algorithmic improvement is established at rank 16.

### Rank scaling

Test ranks such as 32/64 only as a **separate capacity study**.

Questions:

- Does the recipe still outperform AdamW at the same higher rank?
- Does CMA compression require more rank?
- Does increased rank simply dominate the optimizer effect?
- What is the marginal capability gain per additional trainable parameter?

### Model-family transfer

Do not assume one optimizer/radius schedule transfers unchanged to every Rineya checkpoint. For each compatible family member:

- pin checkpoint and tokenizer hashes;
- recompute base matrix norms;
- rerun a cheap LR/effective-step calibration;
- preserve the algorithm, but allow predeclared scale calibration;
- evaluate whether the direction of the gain replicates.

Cross-repository model families must not be silently conflated with RineyaBeta artifacts.

---

## Phase 8 — advanced geometry research

These are follow-on projects, not v1 requirements.

### A. DoRA + spherical effective-weight control

DoRA introduces explicit magnitude/direction structure and may make a closer analogue of Puro's spherical full-weight projection possible. This should be tested only after the simpler DeltaBall study.

### B. Base-relative effective-LR-controlled PEFT

Instead of merely logging \(\rho_l^{\Delta}\), develop an optimizer that explicitly targets a predeclared base-relative step schedule.

This is scientifically closer to Puro's finding that effective LR can explain much of the MuonH advantage, but it requires careful low-rank step construction.

### C. Native projected low-rank delta optimizer

Conceptual target:

\[
\widetilde{\Delta W}_{t+1}
=\Delta W_t-\eta_t\,\mathcal{O}(G_t)
\]

followed by:

\[
\Delta W_{t+1}
=\Pi_{\operatorname{rank}\le r}
\left[
\Pi_{\mathrm{DeltaBall}}
(\widetilde{\Delta W}_{t+1})
\right].
\]

This would optimize the effective low-rank delta more directly instead of relying on arbitrary A/B factor geometry. It is a distinct research contribution and should not be mixed into the first Muon-LoRA experiment.

---

# 11. Evaluation and promotion protocol

## 11.1 Existing Rineya gates

The current evaluation stack includes:

- `chat_dev` — 510 cases;
- `rag_dev` — 752 cases;
- `rag_test_SEALED` — 738 sealed cases;
- quality scorer v4.5;
- fixed greedy generation with short-response controls.

These gates are useful for behavioral retention and grounding, but RGE-PEFT also needs a **targeted capability suite** because the program's goal is more than chat style.

---

## 11.2 Target capability suite

Before Phase 2 confirmatory runs, freeze a capability development bank covering at least:

- mathematical reasoning;
- general logical reasoning;
- verification/correctness judgments;
- factual/general knowledge where the answer is objectively scorable;
- selected long-context/retrieval cases if the training intervention targets them;
- instruction-following/contract tasks.

Prefer deterministic or externally checkable scoring where possible.

A separate sealed capability set should be created or identified before candidate selection if the existing repository does not already provide one.

### Contamination rule

Training rows must be checked against evaluation prompts/answers using exact and approximate deduplication appropriate to the data source. Any benchmark-derived training data must be documented.

---

## 11.3 Primary comparison unit

Use **paired per-case comparisons** whenever the same evaluation cases are run under both models.

Report:

- mean difference;
- median difference if meaningful;
- paired bootstrap confidence interval;
- win/tie/loss counts where a case-level comparison is defined;
- task-specific exact accuracy for deterministic benchmarks;
- category deltas;
- regression tail, not only average.

---

## 11.4 Retention gates

A candidate must not buy a targeted benchmark gain by damaging core behavior.

Initial promotion requirements:

1. targeted capability composite improves over control with a consistent direction across confirmatory seeds;
2. `chat_dev` and `rag_dev` remain non-inferior under the frozen scorer;
3. no material increase in critical defects or contract violations;
4. no unexplained severe category collapse;
5. trainable/inference capacity remains matched for the principal claim;
6. compute overhead is recorded and included in the efficiency conclusion.

### Initial numeric non-inferiority guardrail

Until a repository-native promotion engine is explicitly adopted for this program, use a conservative provisional rule:

- mean quality score should not fall by more than **0.01 absolute** on either `chat_dev` or `rag_dev` versus the matched control;
- critical-defect and contract-violation changes must be individually inspected even if the mean score passes.

If existing Rineya promotion rules are stricter, the stricter repository rule wins.

---

## 11.5 Compute-efficiency reporting

Every run reports:

- active GPU-hours;
- wall-clock duration;
- supervised tokens;
- padded tokens/positions;
- optimizer steps;
- tokens/sec or supervised tokens/sec;
- peak allocated/reserved VRAM;
- adapter parameter count;
- optimizer-state bytes where practical.

Primary efficiency comparisons:

\[
\frac{\Delta \text{target capability}}{10^6\text{ supervised tokens}}
\]

and

\[
\frac{\Delta \text{target capability}}{\text{active GPU-hour}}.
\]

Do not call an optimizer "more efficient" solely because it reaches lower training loss faster if the frozen capability metric does not improve.

---

# 12. Compute-budget ladder

The entire point of this program is to avoid thousand-GPU-hour experimentation. Compute is therefore governed by a **progressive evidence ladder**.

## Tier 0 — no/near-no training

- math tests;
- optimizer unit tests;
- synthetic projection tests;
- CMA algebra tests;
- parameter/memory accounting;
- dry-run planner/memory probes.

No expensive run is permitted until Tier 0 is green.

## Tier 1 — smoke

Tiny deterministic runs sufficient to test:

- forward/backward;
- optimizer state;
- resume;
- logging;
- DeltaBall activation;
- adapter export/load.

## Tier 2 — pilot

Short fixed-token arms used for LR/radius screening and mechanism evidence.

**Program rule:** no exploratory arm should consume a long full-training budget merely to answer a question that a pilot can kill.

## Tier 3 — matched full run

Only mechanisms that pass pilots receive a full matched control/candidate comparison.

## Tier 4 — replicated confirmation

Only a frozen candidate that beats the matched control receives multiple full seeds and sealed evaluation.

### Budget philosophy

The initial proof should target **single-GPU, same-session experiments** rather than cluster-scale training. Exact wall time depends on the host and dataset; this document does not promise a fixed number of hours. The governing rule is that every escalation must be justified by evidence from the cheaper tier.

A recommended program-level guard is to require an explicit SSOT amendment before the exploratory program exceeds **tens of active GPU-hours**. The central research claim is low-cost post-training; a design that requires hundreds or thousands of GPU-hours has missed its objective.

---

# 13. Verification suite

## 13.1 Optimizer math

`verify_muon_math.py` should cover:

- zero gradient;
- rank-1 and highly rectangular matrices;
- transpose branch if used by the Muon implementation;
- finite normalized update;
- deterministic update under fixed inputs;
- FP32 state policy;
- no mutation of frozen base weights;
- expected parameter-group routing.

If a Newton-Schulz orthogonalization is used, test convergence/finite behavior for the actual LoRA matrix aspect ratios in Rineya.

---

## 13.2 Delta geometry

`verify_delta_geometry.py` should prove:

- `BA` scaling includes `alpha/r` correctly;
- base norm caching matches direct calculation;
- `rho_delta`, radius drift, and angle match dense reference calculations;
- metrics are gauge-invariant when `BA` is unchanged;
- logging does not modify gradients or model parameters.

---

## 13.3 DeltaBall

`verify_deltaball_projection.py` should prove:

- deltas inside the ball are unchanged;
- deltas outside are projected to the boundary within tolerance;
- symmetric `sqrt(c)` factor scaling produces exact product scale `c`;
- projection result is product/gauge based rather than factor-norm based;
- projection never touches frozen base weights;
- resume reproduces the same projected trajectory.

---

## 13.4 CMA

`verify_cma_math.py` should prove:

- exact low-rank concatenation equals explicit dense weighted average;
- equivalent gauge-transform checkpoints average to the same delta;
- truncated low-rank reconstruction matches dense SVD;
- final adapter scaling reproduces target delta;
- rank-matched export has exactly the intended rank;
- evaluation loader merges the produced adapter correctly.

---

## 13.5 Determinism and interruption

RGE-PEFT must inherit the spirit of existing `verify_determinism.py`, `verify_resume.py`, and `verify_interrupt.py`.

Add checks for:

- Muon state restoration;
- DeltaBall schedule restoration;
- curriculum position restoration;
- geometry logging cadence restoration if it can affect RNG (prefer designs where it cannot);
- constant-tail continuation resume;
- exact checkpoint averaging input IDs/hashes.

---

# 14. Run artifact contract

Every run directory should contain or reference:

```text
run_manifest.json
resolved_config.yaml
train_log.jsonl
geometry.jsonl
memory_profile.json
checkpoints/
checkpoint_ledger.jsonl
eval/
    capability_dev/
    chat_dev/
    rag_dev/
summary.json
```

### Required manifest fields

- run ID;
- program version;
- exploratory/confirmatory status;
- Git branch;
- Git commit SHA;
- dirty-tree status;
- model path/ID and content hash;
- tokenizer hash;
- data manifest and SHA256;
- evaluation asset hashes;
- seed;
- device name;
- CUDA/driver/PyTorch/Transformers/PEFT versions;
- precision;
- trainable parameter count;
- LoRA rank/alpha/targets;
- optimizer and implementation version;
- optimizer hyperparameters;
- DeltaBall configuration;
- schedule;
- curriculum order manifest/hash;
- supervised-token budget;
- plan hash;
- checkpoint interval;
- CMA eligibility;
- wall time;
- active GPU-hours;
- peak VRAM;
- exit reason.

### Run ID convention

Recommended:

```text
rge_v1__<phase>__<arm>__r16__seed<seed>__<short-config-hash>
```

No two materially different runs should share a run ID.

---

# 15. Evidence and claims ledger

`CLAIMS_LEDGER.md` should maintain entries such as:

| Claim | Status | Required evidence | Supporting runs | Falsifiers |
|---|---|---|---|---|
| Muon improves LoRA token efficiency | planned | matched O-A/O-M confirmatory runs | — | equal/worse frozen capability |
| DeltaBall improves retention | planned | factorial G-* arms with active projection | — | inactive constraint or worse frontier |
| Curriculum ordering improves endpoint | planned | C-U vs C-C same membership | — | no gain / regression |
| Delta-CMA improves rank-matched endpoint | planned | C-CT vs C-CTM + compression diagnostics | — | gain only in rank-Kr exact average |

Statuses should be limited to a controlled vocabulary such as:

```text
PLANNED
IMPLEMENTED_UNVERIFIED
VERIFIED_MECHANISM
EXPLORATORY_POSITIVE
EXPLORATORY_NEGATIVE
CONFIRMED_POSITIVE
CONFIRMED_NEGATIVE
REJECTED
SUPERSEDED
```

No `CONFIRMED_POSITIVE` without the predeclared controls.

---

# 16. Failure modes and how to interpret them

## 16.1 Muon loses to AdamW

Interpretation: Puro's optimizer benefit may not transfer to low-rank factor-space SFT, or our factor representation may be the wrong substrate.

Next action: test DeltaBall on AdamW if H3 remains motivated; do **not** endlessly tune Muon until it wins.

---

## 16.2 Muon has lower loss but no capability gain

Interpretation: optimization efficiency on the training objective did not transfer to downstream capability.

Result: no capability claim. Retain as optimizer-dynamics evidence only.

---

## 16.3 DeltaBall never activates

Interpretation: the chosen trust region did not constrain this training trajectory.

Result: no causal DeltaBall credit. Use only the predeclared exploratory radius screen; otherwise close H3 for v1.

---

## 16.4 DeltaBall activates constantly and training stalls

Interpretation: radius too small, optimizer moments push radially, or low-rank geometry is incompatible with the constraint.

Only predeclared pilot calibration may adjust the radius. A post-confirmation radius rescue is a new experiment.

---

## 16.5 Curriculum wins but constant-tail/CMA does not

Keep curriculum ordering and drop averaging. Puro itself shows that averaging is not independently helpful in every condition.

---

## 16.6 CMA-exact wins but CMA-r loses

Interpretation: averaging creates useful directions outside the original rank-r subspace.

This is scientifically interesting. It suggests a **rank bottleneck**. It does not justify claiming a rank-matched CMA win.

Follow-up: separate capacity study at rank `2r` or structured compression method.

---

## 16.7 Dev gains disappear on sealed evaluation

Reject promotion. Record the failure. Do not tune on the sealed set.

---

## 16.8 Candidate improves reasoning but damages chat/RAG behavior

The candidate may be useful as a specialist adapter but is not a general Rineya-family upgrade under the original objective. Record the capability tradeoff explicitly.

---

## 16.9 Everything is neutral

A clean negative result is valid:

> Puro-inspired optimizer/curriculum geometry did not materially improve rank-16 Rineya PEFT under the tested compute and data regime.

This prevents future repeated effort and may motivate the more principled native-delta optimizer instead of further factor-space tuning.

---

# 17. Security, licensing, and provenance

## 17.1 Training code

Any Muon/MuonH code ported from external repositories must retain required copyright/license notices. Prefer a minimal, auditable implementation over importing the entire Puro-Megatron stack.

## 17.2 Puro dataset

Do not make the Puro mixed-source training dataset a dependency of RGE-PEFT v1. Puro's paper/release notes distinguish project-authored artifacts from upstream component data whose terms may vary.

## 17.3 Rineya data provenance

Every training component must retain its source/license/provenance record. Curriculum materialization changes order, not ownership or license.

## 17.4 Model ownership/release

The optimization recipe should remain separable from the model release policy. The Rineya output license should be determined by Rineya's base model/data obligations and project policy, not assumed from Puro's paper license.

---

# 18. Recommended first implementation sequence

When remote implementation begins, the agent should execute in this exact order:

1. **Read this SSOT and current repository audit.**
2. **Pin current `main` SHA.**
3. **Create `research/rge-peft-v1`.**
4. **Commit the SSOT alone or with only experiment skeleton files.**
5. **Freeze baseline hashes/config/eval assets.**
6. **Implement RGE harness with AdamW only.**
7. **Prove parity and exact resume.**
8. **Implement geometry diagnostics.**
9. **Benchmark geometry overhead.**
10. **Implement/test Muon parameter group.**
11. **Run Tier-1 smoke.**
12. **Run exploratory LR/effective-step calibration.**
13. **Run Phase-2 optimizer pilot.**
14. **Implement/test DeltaBall.**
15. **Run Phase-3 factorial pilot/confirmation.**
16. **Freeze the winning geometry recipe.**
17. **Build deterministic curriculum manifests.**
18. **Run curriculum/schedule arms.**
19. **Implement Delta-CMA and synthetic verification.**
20. **Run CMA endpoint study.**
21. **Freeze final candidate.**
22. **Run multi-seed confirmation.**
23. **Only after gates pass, open sealed evaluation according to repository policy.**
24. **Write findings/negative-results ledgers.**
25. **Decide whether to promote, revise, or retire RGE-PEFT v1.**

Do not skip directly to a combined Muon + DeltaBall + curriculum + CMA run. A combined run can tell us that *something* worked; it cannot tell us what worked.

---

# 19. Definition of success

RGE-PEFT v1 is considered a **program success** if a rank-matched candidate demonstrates all of the following under the frozen protocol:

1. **Target capability:** clear improvement over matched AdamW-LoRA at equal supervised-token budget.
2. **Retention:** chat/RAG/general behavior remains within the predeclared non-inferiority gates.
3. **Robustness:** direction of gain is consistent across confirmatory seeds or otherwise statistically/empirically compelling under the declared seed limitation.
4. **Capacity:** no hidden parameter-count advantage.
5. **Compute:** active GPU-hour cost is small enough that capability-per-GPU-hour improves, not merely raw score.
6. **Mechanism:** at least one component has isolated evidence, not only a combined endpoint.
7. **Reproducibility:** run can be reconstructed from committed config/manifests/checkpoints and exact source revision.
8. **Sealed evidence:** final candidate does not collapse when evaluated outside the selection loop.

A **strong success** is more specific:

> The candidate beats matched AdamW-LoRA on targeted capability, remains non-inferior on general quality, and achieves the improvement with the same rank and a training budget that remains in the consumer-GPU regime.

That would justify deeper study, family transfer, higher-rank scaling, and a possible paper-oriented evaluation.

---

# 20. Definition of retirement

RGE-PEFT v1 should be retired rather than endlessly tuned if:

- baseline parity cannot be established;
- Muon and DeltaBall both fail clean matched pilots;
- all observed gains disappear under proper capacity/token controls;
- curriculum effects are explained entirely by changed data membership;
- CMA gains exist only by increasing effective rank and vanish under rank-matched compression;
- repeated confirmatory runs show no stable capability advantage;
- general regressions outweigh targeted gains;
- compute overhead defeats the low-cost objective.

Retirement does not invalidate the broader idea. It localizes the failure and determines whether a more principled native-delta optimizer deserves a separate v2 program.

---

# 21. Candidate v2 research direction if v1 partially works

If v1 finds that **delta geometry matters** but factor-space Muon is unstable or gauge-dependent, the strongest follow-up is not more LoRA hyperparameter tuning. It is a direct low-rank delta optimizer.

A v2 program could optimize a low-rank manifold representation with explicit control over:

- tangent update norm;
- base-relative effective LR;
- rank projection;
- trust-region radius;
- singular-value allocation;
- layer-specific sensitivity budgets.

That would turn the central intuition into a more intrinsic algorithm:

> optimize the actual low-dimensional change to the model, not the arbitrary coordinates used to factor that change.

This is potentially the most scientifically novel extension, but v1 should first establish whether the simpler transfer signal exists.

---

# 22. References and provenance anchors

## Puro-2B

- Kairong Luo et al., **Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090**, arXiv:2608.27370v1, 2026.  
  https://arxiv.org/abs/2608.27370

Key facts motivating this SSOT:

- up to ~1.4T scheduled pretraining tokens;
- canonical run: 22,514 active-training GPU-hours;
- efficiency stack includes blockwise FP8, MuonH/Hyperball optimization, curriculum model averaging, hardware selection, and data design;
- MuonH applies normalized Muon updates to selected approximately scale-invariant matrices and projects full matrices to a fixed Frobenius radius;
- paper emphasizes effective LR rather than raw scalar LR as an important descriptor;
- curriculum is constructed component-locally while preserving mixture proportions;
- canonical CMA uses a late constant-LR continuation and an equal average of six checkpoints;
- paper reports that averaging alone is not beneficial in every ordering condition, motivating explicit ablations here.

## Puro code/release

- https://github.com/thu-pacman/Puro-Megatron
- https://huggingface.co/collections/thu-pacman/puro-2b

## RineyaBeta baseline anchors observed during SSOT drafting

Repository: `LUNCY579/RineyaBeta`

Relevant current files:

```text
scripts/train/heal.py
scripts/train/checkpoint.py
scripts/train/dataset.py
scripts/train/determinism.py
scripts/train/memory.py
scripts/train/plan.py
scripts/train/export_adapter.py
scripts/verify/verify_determinism.py
scripts/verify/verify_interrupt.py
scripts/verify/verify_resume.py
scripts/eval/run_gates.py
configs/heal_b0.yaml
configs/quality_evaluation.json
manifests/eval_assets.json
docs/EXPERIMENT_DESIGN_STABILIZER.md
docs/NEGATIVE_RESULTS.md
```

Current observed `heal_b0.yaml` anchor:

- R2 core: 474,021,888 parameters;
- LoRA rank 16;
- alpha 32;
- target modules: q/k/v/o + gate/up/down projections;
- BF16 base on CUDA, trainable LoRA parameters promoted to FP32 by trainer;
- AdamW LR `1e-4`;
- hard memory cap `5120 MiB` in that historical config;
- strict deterministic/resume design.

These are **snapshot anchors**, not eternal assumptions. Implementation Phase 0 must re-pin the then-current repository state.

---

# 23. Amendment log

## v1.5 — 2026-08-31

M5/M6 matched-control correction, recorded before any targeted capability-bank
evaluation or M6 confirmation. The selected curriculum candidate includes a
12-step tail and six-checkpoint CMA, while the M4 uniform endpoint does not.
Comparing them in M6 would confound order with extra tokens/tail/averaging.

Resolution: run the preregistered optional C-UTM interaction control—uniform
order with the identical 12-step constant tail and rank-16 Delta-CMA—and use it
as M6 CONTROL. Use curriculum CMA-r16 as CANDIDATE. Membership, total supervised
tokens, optimizer, DeltaBall, tail LR/steps, checkpoint weights, final rank, and
evaluation protocol must match; only main-segment order differs.

The capability validation bank was hash-frozen but had not been run on either
endpoint. No test/challenge/sealed asset was accessed. Commit reference: `857c85d`.

## v1.4 — 2026-08-31

M4 constant-tail transition details, frozen before tail training. Continue the
selected component-local endpoint for 12 optimizer steps at constant LR `1e-5`,
with checkpoints at steps `{2,4,6,8,10,12}`. Retain adapter tensors, AdamW state,
RNG, geometry reference factors, data membership/order, and the eps1x DeltaBall
radius. Reset only the new tail plan position and projection counters; the radius
begins fully warmed. No averaging is performed in M4.

This specifies a previously planned ingredient rather than changing a hypothesis
or evaluation gate. No tail/CMA outcome had been observed. Commit reference:
`57f4d26`.

## v1.3 — 2026-08-31

M4 curriculum fairness amendment, recorded before any M4 training outcome.

Exact issue: reference-shuffled and component-local row orders can pack into
different optimizer-step counts and per-step supervised-token masses. Reusing a
step-indexed cosine schedule or stopping both at the first step beyond a token
request would therefore change the LR-versus-token trajectory and actual token
count alongside order.

Resolution: freeze a deterministic 10% component-stratified subset (12,875 rows)
with identical IDs/content in both arms and consume the complete subset epoch.
Parameterize LR warmup/decay and DeltaBall warm-in by supervised tokens, using
41,383 warmup tokens (the frozen M3 control's step-40 cumulative position). This
keeps total membership, exact supervised tokens, and schedule-versus-token
semantics matched even when packing produces different step counts.

No M4 loss, retention, or capability outcome had been observed. The amendment
changes no curriculum ranking, optimizer, LoRA capacity, DeltaBall radius,
evaluation policy, promotion threshold, or sealed boundary. Commit reference:
`38d0c7a`.

## v1.2 — 2026-08-31

M2 pilot-budget correction, recorded before any O-A/O-M pilot endpoint or
retention result and before launching the registered pair. Training loss and
effective-step geometry from bounded integration/calibration runs had been
observed; the correction is formula-driven and does not use those outcomes to
select the budget.

Exact issue: the staged M2 pair requested 1,000,000 supervised tokens. The
current B0 plan contains 1,043,713 supervised tokens, so that request is about
96% of a full epoch and violates Phase 2's preregistered
`T_pilot = min(10% of confirmatory budget, bounded inexpensive cap)` rule. At
measured step times it would also exceed the default M2 compute envelope.

Resolution: freeze `T_pilot = 104,371` requested supervised tokens, exactly 10%
of the current B0 confirmatory epoch rounded to the nearest token. Both arms stop
at the first identical deterministic plan boundary at or beyond the request, so
effective token counts must still match exactly. Commit reference: `f99fdd3`.

This amendment changes no hypothesis, optimizer setting, LoRA capacity, data
membership/order, evaluation policy, promotion threshold, or sealed boundary.

## v1.1 — 2026-08-30

Project-owner approval and execution-calendar amendment. This amendment was made before implementation and before observing any RGE-PEFT experimental result. It changes no scientific hypothesis, arm, threshold, sealed-evaluation rule, or evidence standard.

Additions:

- records the roadmap as the approved governing SSOT;
- adds the expected calendar and compute envelopes in §25;
- establishes `experiments/rge_peft/PROGRESS.md` as the live milestone/status dashboard;
- defines unambiguous milestone states, weekly closeouts, schedule-slippage rules, and early-stop behavior;
- preserves progression as evidence-gated rather than date-gated.

## v1.0 — 2026-08-30

Initial pre-registration.

Major design decisions frozen:

- extend RineyaBeta's deterministic PEFT trainer rather than adopt Megatron as the runtime;
- preserve rank-16 all-linear LoRA as the principal matched-capacity substrate;
- separate Muon, DeltaBall, curriculum, constant-tail, and CMA effects;
- define DeltaBall as a base-relative trust region rather than falsely calling it an exact Hyperball transfer;
- make induced `Delta W` the canonical geometry object;
- require delta-space rather than factor-space checkpoint averaging;
- recompress CMA to original rank for the principal comparison;
- retain exact higher-rank average only as a diagnostic;
- preserve sealed evaluation and negative-result governance;
- defer FP8, DoRA, native projected low-rank optimization, and unrelated mechanisms until v1 earns them.

---

# 24. One-page operational summary for future agents

**Goal:** determine whether Puro-inspired optimization ideas can improve Rineya's capability per token/GPU-hour using rank-matched PEFT rather than full pretraining.

**Do first:** pin `main`, create or resume `research/rge-peft-v1`, read this SSOT, freeze model/data/eval hashes, and update `experiments/rge_peft/PROGRESS.md`.

**Never skip:** AdamW parity.

**Core sequence:**

```text
AdamW parity
  -> delta geometry logging
  -> AdamW vs Muon
  -> AdamW/Muon x DeltaBall factorial
  -> freeze winner
  -> uniform vs component-local curriculum
  -> late constant-LR continuation
  -> six-checkpoint Delta-CMA
  -> rank-r recompression
  -> matched multi-seed confirmation
  -> sealed evaluation only after freeze
```

**Canonical mathematical object:**

```text
DeltaW = (alpha/r) * B @ A
```

**Do not:** independently average A and B.

**Do not:** call DeltaBall exact Puro Hyperball.

**Do not:** increase rank while claiming an optimizer gain.

**Do not:** tune on sealed evaluation.

**Do not:** bundle all mechanisms into the first run.

**Success:** better targeted capability than matched AdamW-LoRA, no meaningful general-quality regression, same adapter capacity, favorable GPU-hour efficiency, reproducible evidence.

**Failure is acceptable:** record it, localize it, and stop rather than tune indefinitely.

---

# 25. Approved execution timeline and progress-tracking contract

## 25.1 Calendar interpretation

The dates below are **target windows, not permission to bypass a gate**. A phase begins only when the prior phase's evidence and exit criteria are satisfied. If a kill condition is reached, the program records the result and closes or revises the affected hypothesis instead of consuming the remaining calendar merely because time was reserved.

The initial schedule assumes:

- one primary consumer GPU, historically an RTX 4070 Laptop-class environment;
- rank-16 all-linear LoRA on the R2-sized Rineya core as the matched substrate;
- same-session screening runs rather than multi-day uninterrupted jobs;
- deterministic training and strict resume checks retained even when they reduce throughput;
- implementation work beginning from the SSOT branch created on **Sunday, August 30, 2026**.

After the first parity benchmark, measured step time and supervised-token throughput replace rough planning estimates. A schedule may move without changing the science, but any material expansion of experimental scope or compute ceiling requires an append-only amendment before the extra work is launched.

## 25.2 Milestone calendar

| Milestone | Target window | Governing phase(s) | Expected deliverable | Exit decision |
|---|---|---|---|---|
| **M0 — SSOT preservation** | **Aug 30, 2026** | governance | `research/rge-peft-v1`; approved SSOT; initialized progress dashboard; pinned parent SHA | repository-backed plan exists |
| **M1 — frozen substrate and AdamW parity** | **Aug 31–Sep 6** | Phase 0 + Phase 1 | environment/model/data/eval manifest; geometry logging; deterministic short-run parity; resume and overhead verification | `GATE_PASSED` only with explained or zero baseline divergence |
| **M2 — Muon transfer pilot** | **Sep 7–Sep 13** | Phase 2 | optimizer split; Muon math tests; LR/effective-step calibration; matched O-A/O-M pilot report | retain Muon, reject Muon, or keep only as exploratory |
| **M3 — DeltaBall factorial** | **Sep 14–Sep 27** | Phase 3 | projection tests; bounded radius screen; G-A/G-M/G-AD/G-MD comparison; boundary-hit diagnostics | freeze surviving geometry recipe or retire components separately |
| **M4 — curriculum and late-tail isolation** | **Sep 28–Oct 4** | Phase 4 | byte-identical membership/order manifests; C-U/C-C/C-CT/C-CTM comparison | freeze or remove ordering/tail ingredients |
| **M5 — Delta-CMA validation** | **Oct 5–Oct 11** | Phase 5 | gauge-invariant averaging tests; CMA-exact and CMA-r endpoints; retained-energy report; normal PEFT export/load | accept CMA-r, retain final checkpoint, or record a CMA/compression negative |
| **M6 — confirmatory matched comparison** | **Oct 12–Oct 25** | Phase 6 | frozen CONTROL/CANDIDATE; preferably three seeds; full dev gates; retention and efficiency report | promote, reject, or label limited-seed evidence |
| **M7 — one-time sealed promotion gate** | **Oct 26–Oct 28** | Phase 6 promotion boundary | exactly one evaluation of the already frozen candidate, only if all prerequisite gates pass | final v1 external-evidence status; no checkpoint shopping |
| **M8 — rank/family transfer** | **Oct 29–Nov 15** | Phase 7, conditional | rank-32/64 matched controls and/or transfer to one additional compatible Rineya checkpoint | establish transfer scope or localize non-transfer |
| **M9 — program closeout** | **Nov 16–Nov 22** | synthesis | findings ledger; negative results; claims audit; reproducibility commands; promotion/retirement decision | close v1 and authorize or reject a v2 program |
| **Phase 8 earliest start** | **Nov 23 onward** | advanced geometry, conditional | DoRA/spherical control, ELR-controlled PEFT, or native projected low-rank optimizer | separate amendment and authorization required |

M7 is prohibited unless M6 passes and the candidate/config/checkpoint are already frozen. M8 is conditional on a positive or mechanistically informative core result. A failed v1 is not extended into rank/family transfer merely to search for a favorable outcome.

### Delivery targets

- **Core proof target:** freeze Phase-6 evidence by **October 25, 2026**.
- **Positive-path target:** finish the sealed gate, transfer study, and v1 synthesis by **November 22, 2026**.
- **Operational buffer:** dependency, hardware, or reproducibility problems may extend closure to **December 6, 2026** without changing the scientific contract.
- **Early-retirement path:** a clean negative can close at M2 or M3. Stopping early is an efficiency success when the evidence does not justify further GPU spend.

## 25.3 Compute envelope

These are governance envelopes, not quotas that must be spent. Each run manifest records actual GPU model, active-training seconds, supervised tokens, throughput, peak VRAM, interruptions, and exit reason.

| Work package | Default planning envelope | Review ceiling / escalation rule |
|---|---:|---|
| M1 parity, unit, and smoke work | 0–3 GPU-hours total | no long run until parity passes |
| M2 optimizer screening | 2–6 GPU-hours total | longer work requires a written unresolved learning-curve question |
| M3 DeltaBall screening/factorial | 4–12 GPU-hours total | weak radii terminate early using only predeclared diagnostics; no exhaustive tuning |
| M4 curriculum/tail isolation | 4–12 GPU-hours total | identical membership/token budget remains mandatory |
| M5 CMA construction and validation | primarily CPU plus 1–6 GPU-hours | exact average is diagnostic; primary endpoint remains rank matched |
| M6 matched confirmation | 12–36 GPU-hours total | control/candidate and seed plan frozen before launch |
| **Core v1 through M6** | **23–69 GPU-hours target** | crossing **75 active GPU-hours** requires an SSOT amendment and explicit rationale |
| Conditional M8 transfer | 12–30 GPU-hours total | authorized only after a positive core result |

An individual screening run should remain a few hours. The aggregate is larger because fair controls, ablations, and replication are required to know whether the method works. Even the upper planning envelope remains orders of magnitude below from-scratch Puro-scale pretraining.

## 25.4 Status vocabulary

The live dashboard uses only:

```text
NOT_STARTED   approved but no work product exists
IN_PROGRESS   active work is linked to a concrete commit, command, or run
BLOCKED       cannot proceed; blocker and next action are written
GATE_PASSED   required evidence satisfied the predeclared phase gate
GATE_FAILED   predeclared gate failed; interpretation is recorded
CLOSED        phase/program is finished and no further work is authorized under this version
RETIRED       hypothesis/program was deliberately ended under the retirement rules
```

Do not use ambiguous labels such as “mostly done,” “looks good,” or “promising” as milestone status.

## 25.5 Progress dashboard requirements

`experiments/rge_peft/PROGRESS.md` is the living execution dashboard. It does not replace this SSOT and cannot redefine hypotheses. Every status update must contain:

- calendar date and America/Chicago timestamp;
- phase/milestone and status;
- branch and exact commit SHA;
- work completed;
- commands or run IDs;
- linked evidence paths;
- actual GPU-hours and supervised-token counts where applicable;
- open blocker or next gate;
- explicit statement of whether any sealed data was accessed;
- amendment reference if implementation departed from this document.

Update the dashboard:

1. after every implementation commit that changes experiment behavior;
2. immediately after a phase gate passes or fails;
3. after every completed or interrupted GPU run;
4. after selecting any LR, DeltaBall radius, checkpoint set, or curriculum recipe;
5. at least once per calendar week while the program is active, even if the update is only a blocker.

A weekly roll-up is entered each **Sunday** in America/Chicago with planned-versus-actual milestone status and cumulative active GPU-hours.

## 25.6 Milestone completion weights

Overall percentage is descriptive only and never substitutes for gate evidence. Fixed weights prevent progress inflation through many small implementation tasks.

| Milestone | Program weight |
|---|---:|
| M0 SSOT preservation | 3% |
| M1 substrate/parity | 15% |
| M2 optimizer pilot | 12% |
| M3 DeltaBall factorial | 18% |
| M4 curriculum/tail | 14% |
| M5 Delta-CMA | 14% |
| M6 confirmation | 20% |
| M7 sealed gate | 1% |
| M8 transfer | 1% |
| M9 closeout | 2% |

A milestone contributes its full weight only after `GATE_PASSED` or `CLOSED`. `IN_PROGRESS` may be visualized as half its weight operationally, but scientific claims remain zero until the evidence gate closes.

## 25.7 Schedule-slippage and early-stop rules

- A slip of **≤7 calendar days** with unchanged scope is recorded in the dashboard, not treated as an amendment.
- A slip of **>7 days** caused by a scientific or architectural discovery requires an SSOT amendment explaining whether the design changed.
- Hardware downtime may shift dates without changing hypotheses, but compute and interruption evidence must still be logged.
- No missed target date authorizes weaker controls, fewer required artifacts, opening the sealed set early, or merging an unvalidated implementation.
- If M1 parity is not closed by **September 13, 2026**, all optimizer experiments pause for a scope review before more GPU time is spent.
- If Muon is negative at M2, DeltaBall may still proceed under AdamW because H3 is logically separable.
- If Muon and DeltaBall both fail clean matched tests at M3, the program should retire the geometry v1 path rather than continue automatically into curriculum/CMA.
- If no component improves the matched frontier by M5, M6 runs only when a bounded confirmation of the strongest neutral/negative result is scientifically justified; otherwise v1 closes without a multi-seed campaign.

## 25.8 Current status at repository approval

```text
Program: RGE-PEFT v1.1
Current milestone: M0 — SSOT preservation
Current status: IN_PROGRESS until the remote commit is verified
Approved date: 2026-08-30
Implementation started: no
Experimental outcomes observed: none
Sealed evaluation accessed: no
Next gate: commit the approved SSOT and initialized dashboard to research/rge-peft-v1
```
