# RineyaBeta experiment program audit

> **Canonical history/status record.** This file is the single source of truth for
> the **existence, chronology, lineage, and current status** of experiments in this
> repository. History checkers should start here rather than inferring status from
> branch names, result-directory presence, old pre-registrations, or stale planning
> prose.
>
> **Snapshot:** `main@647c60ea9bf716ed05877ae2ccae8935aed71d9f`, 2026-08-30, extended
> with `research/depth-aligned-semantic-transitions-v1@1dce3cd`, 2026-08-31. Every
> "at this snapshot" C1/C2/PCSM statement below remains anchored to the `main` tree
> on which it was verified.
> The GitHub remote was created on 2026-08-20, but the imported Git history begins
> on 2026-08-19 (`5af0be4`). The remote now exposes **five live branches**, verified
> by `git ls-remote --heads origin` on 2026-08-31:
>
> ```text
> main                                          3532c7bc
> research/catastrophic-removal-core-v1         193f93d8
> research/cross-architecture-graft-v1          79cb0897
> research/depth-aligned-semantic-transitions-v1 1dce3cd
> research/rge-peft-v1                          5005561
> ```
>
> Only the DAST branch is audited below. The other four are recorded here so that
> the earlier "one live branch" claim is not carried forward, not because their
> contents have been reviewed.

This audit records what was actually built, run, killed, withdrawn, superseded, or
left unrun. It deliberately preserves negative results and invalidated interpretations.
It does **not** turn a mechanism test into a model-quality claim, and it does not
promote a historical branch name into a live ref when that ref no longer exists.

---

## 0. Authority and update contract

This document is authoritative for **program history and experiment status**. The
specialized records remain authoritative for exact measurements:

1. committed result JSON/manifests and their generating code;
2. `scripts/design/verify_claims.py` for pinned C1/C2 headline values and relational
   checks;
3. the experiment's final findings/result document;
4. this audit for chronology, lineage, and status;
5. pre-registrations and plans for what was intended **before** a run.

If an exact number in this audit ever conflicts with a committed result or a
`verify_claims.py`-checked source, the committed measurement wins and this file must be
corrected in the same change. A pre-registration is never evidence that the experiment
ran. A result directory is never evidence that a run is valid. A deleted branch name is
never evidence that the work disappeared.

### Status vocabulary

| Status | Meaning |
|---|---|
| **ESTABLISHED** | The claimed, bounded result has committed evidence and survived the relevant controls used by the program. |
| **POSITIVE — BOUNDED** | Positive evidence exists, but the scope is narrower than a general model-quality claim. |
| **CLOSED — NEGATIVE** | A declared or later-clean control killed the hypothesis in its tested form. |
| **WITHDRAWN** | The measurement/interpretation was confounded or invalid and must not be used as evidence. |
| **SUPERSEDED** | The old measurement may be valid for its configuration, but a later mechanism/configuration changed the live conclusion. |
| **BUILT — UNRUN** | Implementation/pipeline exists, but no qualifying real experiment result exists. |
| **SYNTHETIC ONLY** | Mechanism/software feasibility is established only on a constructed task. |
| **INFRASTRUCTURE** | Reproducibility, evaluation, or execution work; not itself a scientific quality result. |
| **ARCHIVAL / NOT AUDITABLE** | Mentioned historically, but current canonical refs do not contain enough evidence to state a result safely. |

### Required maintenance rule

Whenever an experiment changes state — for example `BUILT — UNRUN` to
`ESTABLISHED`, or a result is withdrawn — update this file in the same commit that
makes the new state canonical. Never delete a failed row to make the program look
cleaner; change its status and point to the evidence that closed it.

---

## 1. Executive summary

The repository's strongest results fall into three different layers of the system:

1. **Model surgery/recovery — R2.** Vocabulary trimming produced a 474M-parameter
   28-layer core from Qwen3-0.6B. Chat quality was indistinguishable from stock on the
   paired gate, while a small RAG degradation localized primarily to lost `conflict`
   abstentions. The preregistered B0 healing run restored the abstention gate and is a
   positive recovery result. Naive dense low-rank factorization, aggressive persistent
   FFN compression, and unhealed KV-head merging did not survive measurement.

2. **Inference-time cache selection — Cornerstones 1 and 2.** Sliding-window
   attention lost to the KV-head/cache-selection direction on this stack. C1 established
   accumulated-attention scoring, count normalization, chunked prefill, and query-aware
   scoring. C2 then showed that many remaining failures were **span fragmentation**, not
   wholesale evidence eviction, and that backward-reaching/causal pooling can preserve
   coherent evidence much more effectively. The mature passkey score is best described
   as a **query-conditioned KV retrieval index**, not intrinsic text importance.

3. **Training-time sparse plasticity — PCSM.** Peer-conditioned sparse native updates
   pass software qualification and a deliberately favorable synthetic causal study.
   That establishes mechanism feasibility only. The real-gradient Phase C harness is now
   on `main`, but **no Rineya real-model PCSM quality result exists yet**. Its decisive
   next comparison is intact peer coordinates versus coordinate-shuffled peers before
   any full SFT claim is allowed.

The repository also contains a mature **state-sufficiency experiment pipeline**. It is
validated infrastructure, not a completed training result: its own bundle states that no
adapter has been trained on the real checkpoint for that experiment.

---

## 2. Program arc

| Date | Track / event | Status at this snapshot | What changed |
|---|---|---|---|
| **2026-08-19** | Repository history begins; state-sufficiency bundle imported | **BUILT — UNRUN** | Controlled FULL vs compact-state transition pipeline enters the repo; tests/smokes are infrastructure, not model evidence. |
| **2026-08-19** | Vocabulary utilization / tokenizer sweep / R2 | **ESTABLISHED** | `rank` 32,768 adopted: 596,049,920 -> 474,021,888 resident parameters; logit-equivalent on the retained mapping before healing. |
| **2026-08-19** | R2 paired chat/RAG gates | **ESTABLISHED — BOUNDED COST** | Chat delta +0.00019 with CI spanning zero; RAG delta -0.01273, dominated by lost `conflict` abstentions. |
| **2026-08-19** | Healing corpus audit + deterministic VRAM-capped harness | **INFRASTRUCTURE** | Duplicate-heavy source mixing and long-form token domination are corrected before training; resume/VRAM contracts are made reproducible. |
| **2026-08-20** | Attention shape: SWA vs KV-head count | **CLOSED — SWA NEGATIVE** | SWA raises peak on this implementation, fails to extend reachable context, and cliffs when the window excludes sinks; GQA is the structurally better cache axis but needs healing. |
| **2026-08-20 -> 08-21** | Cornerstone 1 significance KV | **ESTABLISHED — BOUNDED** | Attention-derived eviction, `S/C^e`, chunked prefill and query-aware scoring become the working C1 stack. |
| **2026-08-20** | Geometry/NMS eviction | **CLOSED — NEGATIVE** | Diversity suppression loses to top-k and matched-random controls; similar attention profiles behave as corroboration rather than redundant evidence. |
| **2026-08-22 -> 08-23** | Stabilizer Strand A (FFN selector/compression) | **CLOSED — NEGATIVE** | SwiGLU concentration and prediction are real, but sparse gather is slower, firing sets turn over, and persistent low-rank/static compression fails. |
| **2026-08-22 -> 08-29** | Cornerstone 2 span-aware / backward-reaching retention | **ESTABLISHED — BOUNDED** | Position-identity traces expose fragmentation; backward reach, sink protection, controls, language transfer, runtime accounting and paper audit mature into the C2 submission draft. |
| **2026-08-23** | B0 healing result | **ESTABLISHED** | Preregistered abstention gate passes: R2 179/202 correct abstentions -> B0 202/202 while chat contract fulfillment remains 1.0000. |
| **2026-08-23** | Native weight-spectrum factorization | **CLOSED — NEGATIVE** | 99% spectral-energy factorization would add rather than remove parameters; no `surgery.factorize` implementation is justified. |
| **2026-08-23 -> 08-27** | Passkey-to-reasoning / distractor / multi-hop program | **PARTIAL / BOUNDED** | Competition is measured; word-phrase span experiment is withdrawn; two-hop score advantage is detected; three-hop advantage is not; context length/model capability becomes a binding limit. |
| **2026-08-26** | PCSM peer-conditioned sparse native optimizer | **SYNTHETIC ONLY** | Exact-support optimizer, integration, tests and five-seed constructed feasibility study pass. |
| **2026-08-27** | O11 mechanism-identity control | **ESTABLISHED** | Correct scoring query succeeds; unrelated scoring queries collapse retrieval. Passkey scoring is query-conditioned retrieval, not intrinsic importance. |
| **2026-08-28 -> 08-29** | Runtime/open-question closure | **ESTABLISHED — BOUNDED** | Directional pooling is essentially free relative to centered pooling; query-pass cost is measured; 13/13 distinct C1/C2 open questions are closed by 08-29. |
| **2026-08-29** | C1 manuscript + PCSM merged to `main` | **MERGED** | Previously separate research lines are preserved on the canonical branch. |
| **2026-08-30** | PCSM Phase C real-gradient causal screen | **BUILT — UNRUN** | Leakage-resistant one-step real-gradient experiment becomes executable on `main`; commit explicitly makes no Rineya quality claim. |
| **2026-08-30** | Depth-aligned semantic transitions (DAST) enters the repo | **BUILT — UNRUN** | Depth-checkpoint semantic-state supervision, its six-arm matched control set and a sealed evaluation become executable on `research/depth-aligned-semantic-transitions-v1`. No arm has run in a qualifying frozen multi-seed configuration, and TEST has never been opened. |
| **2026-08-31** | DAST instrument repair before any comparative run | **INFRASTRUCTURE** | The real-model path could not execute at all (transformers 5.x returns a mapping from `apply_chat_template`). Two further instrument failures were found and fixed, each of which would have produced a spurious result: the semantic targets encoded problem identity rather than stage, leaving `state_ordered` at 0.860 cosine from `state_shuffled`; and the answer scorer read **0.0000** on visibly correct post-SFT responses because short SFT taught the model the corpus's own `"N. Therefore, the answer is X."` format. |
| **2026-08-31** | DAST hosted-runner gate retired; local qualification authoritative | **INFRASTRUCTURE** | Every GitHub Actions run failed before a runner was assigned — 0 steps executed, `runner_id` 0 — with a billing annotation. The workflow becomes `workflow_dispatch` only, and `scripts/qualify.py` becomes the gate: run locally, with the accelerator both visible and hidden, emitting `qualification/latest.json`. A CPU-only hosted runner structurally could not catch the GPU-only defect that made an earlier "pytest PASS" record reproducible on CPU only. |
| **2026-08-31** | DAST G1 learnability pilot, four arms, one seed | **BUILT — UNRUN** | **EXPLORATORY_DEV only**, 1000 train / 200 DEV. `state_ordered` accuracy 0.7800 equals `cot_baseline` 0.7800 exactly (discordant 5:5, exact McNemar p = 1.0000), and `state_reverse` fits its own reversed targets marginally better than ordered fits the canonical ones (`L_state` 0.5403 vs 0.5441). Measured discordance is 6.0%, so n=200 carried only ~18% power at a true 2:1 split. Nothing is supported and nothing is closed. |
| **2026-09-01** | DAST V1 three-seed DEV matrix closed | **CLOSED — NEGATIVE (EXPLORATORY DEV)** | Four arms × seeds 17/29/43 on 4,000 train / 800 DEV. `state_ordered` mean strict accuracy 0.8833 vs `cot_baseline` 0.8937 (−1.04 pp, p = 0.0370), vs `state_reverse` 0.8921 (−0.88 pp, p = 0.1102), vs `state_final_repeated` 0.8954 (−1.21 pp, p = 0.0134). G3 fails as a directional efficacy gate (ordered did not beat final-repeated). G2 is not closed (`state_shuffled` absent at this scale). Bonferroni over ~13 contrasts admits neither headline p-value. A frozen-backbone linear probe matches or exceeds the trained projector, so mechanism metrics are non-evidence of a trained trajectory. TEST remains sealed. No further V1 seeds. |

---

## 3. Track registry — current canonical status

| Track | Status | Core question | Canonical reading | Primary sources |
|---|---|---|---|---|
| **State sufficiency** | **BUILT — UNRUN** | Can full reasoning supervision be decomposed into compact independent state transitions without losing capability? | Pipeline/dataset are mature; no real-checkpoint adapter has been trained for the experiment. | `Rineya_state_experiment_bundle/README_BUNDLE.md` |
| **19L soup precursor** | **RETIRED PRECURSOR** | Can heavy depth removal be repaired by averaging unrelated LoRA adapters? | Degenerate direct behavior; retained as a post-mortem, not a base. The experiment predates the clean R2 program and is imported as provenance. | `docs/MINI_MODEL_ROADMAP.md` §2 |
| **Vocabulary trim / R2** | **ESTABLISHED** | Can unused vocabulary capacity be removed without broad quality loss? | Yes at 32,768 vocab with a localized grounding/calibration defect that is later healed. | `docs/MINI_MODEL_ROADMAP.md`; `docs/RESEARCH_PROGRAM.md` |
| **Tokenizer selection rule** | **ESTABLISHED / NEGATIVE CONTROL** | Merge-rank selection or token-length selection? | Merge-rank 32,768 wins the measured compute/size trade; `len <= 4` inflates prose past break-even and is rejected. | `docs/MINI_MODEL_ROADMAP.md` §3 |
| **B0 healing** | **ESTABLISHED** | Can the localized post-trim abstention damage be recovered under a frozen gate? | Yes. The preregistered abstention criterion passes. Aggregate improvements are not evidence that trimming itself improved the model because stock did not receive the same adapter. | `docs/MINI_MODEL_ROADMAP.md` §18; commit `38bdc998` |
| **Depth axis** | **MEASURED / NOT CANONICALLY CUT** | Which layers are structurally least consequential? | Layer-profile evidence exists; the current canonical R2 remains 28 layers. Do not report a depth-pruned frontier checkpoint from this program. | `docs/MINI_MODEL_ROADMAP.md` §2; `docs/RESEARCH_PROGRAM.md` C1 |
| **Static FFN width compression** | **NEGATIVE AT 25% KEEP** | Can a persistent small FFN subset approximate the full block? | Static 25% keep is far too damaging; ordinary width pruning remains a separate higher-keep/healing question, not an established checkpoint. | `experiments/stabilizer/results/FINDINGS_P0_4.md` |
| **Native low-rank factorization** | **CLOSED — NEGATIVE** | Do R2 weight matrices have sufficient low rank to save parameters? | No at the measured retention levels; 99% energy is parameter-negative. | `docs/MINI_MODEL_ROADMAP.md` §18.3; commit `38bdc998` |
| **Attention shape: SWA** | **CLOSED — NEGATIVE AS MEMORY STRATEGY** | Can sliding windows extend context more effectively than reducing KV heads? | Not on this stack: peak/reach/quality all defeat the intended memory claim. It may still be a steady-state decode optimization with a native window kernel. | `experiments/attention_shape/README.md`; `docs/NEGATIVE_RESULTS.md` |
| **Attention shape: GQA 8->4/2/1** | **STRUCTURAL OPPORTUNITY; UNHEALED QUALITY NEGATIVE** | Can fewer KV heads reduce cache and parameters? | Surgery works structurally; unhealed 4-KV quality collapses. This proves uptraining/healing is essential, not that GQA is impossible. | `experiments/attention_shape/README.md`; commits `38bdc998`, `c842faa` |
| **C1 significance KV** | **ESTABLISHED — BOUNDED** | Does attention-derived retention beat position-only eviction? | Yes in the useful operating range; at very thin short-prompt budgets the crossover can favor recency/position. | `docs/FINDINGS_KV_PROGRAM.md`; `docs/PAPER_DRAFT_SIGNIFICANCE.md` |
| **Count normalization** | **ESTABLISHED** | Does persistent score accumulation create a survival ratchet? | Yes; ranking on `S/C^e` removes the major late-arrival bias and remains useful after span pooling. | `docs/FINDINGS_KV_PROGRAM.md`; `docs/NEGATIVE_RESULTS.md` |
| **Chunked prefill** | **ESTABLISHED** | Can peak cache cease scaling with total prompt length under bounded retention? | Yes for the measured mechanism; correctness is gated before compression. | `scripts/kv/chunked_prefill.py`; `docs/FINDINGS_KV_PROGRAM.md` |
| **Query-blind chunk scoring** | **CLOSED — NEGATIVE** | Can each chunk be scored only by its own trailing queries? | No when the real question arrives at the end; old evidence is evicted before it can earn score. | `docs/NEGATIVE_RESULTS.md`; `docs/FINDINGS_KV_PROGRAM.md` |
| **Query-aware scoring** | **ESTABLISHED — RETRIEVAL** | Does scoring the current cache against the real question fix the blind-eviction failure? | Yes on passkey; O11 later narrows the interpretation to query-conditioned retrieval. | `docs/FINDINGS_KV_PROGRAM.md`; `docs/OPEN_QUESTIONS.md` O11 |
| **Per-layer KV water-filling** | **POSITIVE PROXY / NOT END-TO-END AUTHORITY** | Can budget be allocated across layers from measured concentration curves? | Concavity held in the measured curves and the proxy objective improves vs uniform allocation; do not promote the proxy alone to a task-quality claim. | commit `c54b7c6`; C1 records |
| **Geometry/NMS eviction** | **CLOSED — NEGATIVE / HISTORICAL BRANCH** | Does diversity-aware suppression beat top-k? | No. The result is preserved in canonical negative documentation even though the historical `exp/geometric-significance` ref is no longer live. | `docs/NEGATIVE_RESULTS.md` §2; `docs/README.md` |
| **KV int4 / naive grouping / CLA** | **CLOSED — NEGATIVE** | Can retained cache be compressed further by uniform low precision or cross-layer prediction? | Pure int4, naive tokenwise K quantization and dense CLA are rejected in their tested forms. | `docs/NEGATIVE_RESULTS.md` §5; `experiments/kv_significance/src/store_probe.py` |
| **KV int8 + significance-gated mixed precision** | **POSITIVE — UNDEREXPLOITED** | Can score-conditioned precision improve the cache-store Pareto point? | Yes in the committed store probe; mixed precision beats pure int8 error at substantially more compression. No broad end-to-end deployment claim is implied. | `docs/NEGATIVE_RESULTS.md` §5a; `experiments/kv_significance/src/mixed_precision.py` |
| **Stabilizer Strand A P0.1-P0.3** | **CLOSED — NEGATIVE AS SYSTEM** | Can one global projector cheaply predict which SwiGLU neurons should fire? | Concentration and prediction are real; sparse gather is slower and the set turns over too quickly for the intended memory/speed story. | `experiments/stabilizer/results/FINDINGS.md` |
| **Stabilizer P0.4 persistent compression** | **CLOSED — NEGATIVE** | Can discarded FFN contribution be represented by a cheap fixed low-rank correction? | No. Required oracle rank rises with sample size; the apparent depth structure also weakens with more data. | `experiments/stabilizer/results/FINDINGS_P0_4.md` |
| **Predict future attention from content/key** | **CLOSED — NEGATIVE** | Can cheap content features replace seeing the future query? | No in the measured probe; key-only prediction is weak and past+predicted is no better than past alone. | `experiments/stabilizer/results/FINDINGS_P1.md` |
| **C2 span / backward-reaching pooling** | **ESTABLISHED — BOUNDED** | Is per-position retention the wrong unit for contiguous evidence? | Yes on the audited retrieval program: failures are often fragmentation; backward reach is the useful direction; sinks are load-bearing. | `experiments/stabilizer/results/FINDINGS_P2.md`; `docs/PAPER_DRAFT_CORNERSTONE2.md` |
| **Derived/geodesic reach** | **MIXED** | Can reach be inferred rather than fixed? | It beats a single fixed reach held constant across span sizes on passkey, but loses to per-span oracle tuning at d=7 and loses on natural-language RAG. Do not state that it beats the best tuned reach. | `docs/CORNERSTONE_LEDGER.md`; C2 draft |
| **C2 natural-language RAG** | **POSITIVE — BOUNDED** | Does C2 transfer beyond passkey? | Yes at short natural-language RAG under compression, but the spectacular passkey compression ratio does not transfer as a language headline. | `docs/PAPER_DRAFT_CORNERSTONE2.md` §4.10 |
| **Passkey distractor competition** | **ESTABLISHED — BOUNDED** | What happens when many query-shaped candidate spans compete? | Thin-budget performance collapses; increasing budget restores it. Competition is a capacity cost and dramatically reduces the headline compression ratio. | `docs/PLAN_LONG_REASONING.md`; C2 draft §5 |
| **Word-phrase Stage 2** | **WITHDRAWN** | Can word phrases extend the span-length axis? | No valid conclusion: word payload and word filler were confounded; filler recitation also fails in the uncompressed control. | `docs/NEGATIVE_RESULTS.md`; `docs/OPEN_QUESTIONS.md` O1/O8 |
| **Long digit-span generation** | **MODEL CEILING / NOT A CACHE RESULT** | Can the generation endpoint validate 12-30-token digit spans? | No on this model: full-cache generation itself fails around the ~11-digit boundary. Do not call compressed failures at that length cache failures. | C2 findings/open-question record |
| **2-hop / 3-hop program** | **PARTIAL / BOUNDED** | Does the cache score preserve several simultaneously required spans? | Two-hop score advantage is detectable in the valid matched analysis; at three hops it is not detectable. Longer context degrades chains largely independent of compression policy. | `docs/PLAN_LONG_REASONING.md`; commit `dd3a5d40` |
| **O11 wrong-query control** | **ESTABLISHED** | Is the passkey score intrinsic importance or query-conditioned retrieval? | Query-conditioned retrieval. Correct scoring query succeeds; unrelated scoring queries collapse while the answer question remains unchanged. | `docs/OPEN_QUESTIONS.md` O11; commit `a949a5fa` |
| **C2 runtime / O12** | **ESTABLISHED — BOUNDED** | What does the mechanism cost? | Directional pooling is essentially parity with centered pooling; decode scoring has a real overhead; query-aware prefill pass is costlier than naive token-ratio reasoning, while the bounded-cache complete prefill can still be faster than no compression. | C2 draft §4.13; `docs/OPEN_QUESTIONS.md` O12 |
| **PCSM synthetic feasibility** | **SYNTHETIC ONLY** | Can compatible peer layers generate useful exact sparse native supports/scales when the hidden optimum really depends on peers? | Yes in the constructed study and software suite. This is not evidence that real transformer layers contain the same causal signal. | `experiments/stabilizer/peer_optimizer/README.md`; `results/RESULTS.md` |
| **PCSM Phase C real-gradient screen** | **BUILT — UNRUN** | Do intact real peer coordinates improve a held-out one-step update over coordinate-shuffled peers? | Harness is on `main`; decisive five-seed causal gate is preregistered. No Rineya quality claim until it runs and passes. | `experiments/stabilizer/peer_optimizer/PHASE_C_README.md`; `EXPERIMENT_PROTOCOL.md`; commit `647c60ea` |
| **DAST V1 (depth-aligned semantic transitions)** | **CLOSED — NEGATIVE (EXPLORATORY DEV)** | Does aligning cumulative compact reasoning states to increasing Transformer depth improve held-out reasoning versus matched reverse/shuffled/flat-target controls? | No on Qwen3-0.6B + LoRA, three seeds, 800 DEV. Ordered is below baseline and below final-repeated; G3 fails directionally; G2 is not closed. Unsigned preregistration, so exploratory rather than confirmatory. TEST sealed. 9-run V2 H2 matrix is parked pending B1/B2/C1 diagnosis. | `experiments/depth_aligned_semantic_transitions/docs/DAST_V1_CLOSURE.md`; `FINDINGS_G3_SEEDS.md`; `DAST_V2_HYPOTHESIS_TREE.md` |

---

## 4. R2 model-surgery and healing line

### 4.1 Inherited 19-layer soup: precursor, not a frontier result

The repository's clean program does not begin from the older 19-layer soup. The
roadmap records that nine layers were removed and unrelated LoRA adapters were averaged,
a design that changed several variables at once and produced degenerate direct behavior.
The post-mortem is useful because it motivated measured layer profiling and the rule
that structural deletion requires actual recovery training. It is **retired as a
repair technique**, not silently reused as R2's base.

### 4.2 Vocabulary trimming: the clean reduction

The vocabulary census showed that only a small fraction of the stock vocabulary was
addressed by the in-repository English/RAG corpora, while byte-level BPE preserves a
fallback representation if the byte alphabet and special tokens are retained. The
measured sweep compared merge-rank sizes 24,576 / 32,768 / 40,960 / 49,152 and a
short-token (`len <= 4`) rule.

The adopted point is **merge rank 32,768**:

- stock resident parameters: 596,049,920;
- R2 resident parameters: 474,021,888;
- reduction: 20.47%;
- the roadmap reports a 14.72% per-text compute reduction after accounting for token
  inflation.

The length-only tokenizer is a real negative: it makes prose sufficiently longer that
the compute saving disappears. The result is therefore not simply "smaller vocab is
better"; the **selection rule and sequence inflation** are part of the experiment.

### 4.3 Real gate cost and B0 recovery

On the paired R2-vs-stock gates, chat is statistically indistinguishable at the reported
precision, while RAG shows a small one-directional cost dominated by `conflict`.
The key behavioral failure is nine lost abstentions: correct abstentions move from
188/202 on stock to 179/202 on R2.

B0's criterion was fixed before training: restore abstention to at least stock's 93.1%
while chat `contract_fulfillment` stays 1.0000. B0 reaches **202/202** correct
abstentions and keeps that chat contract at 1.0000. That is a valid recovery result.

Do **not** rewrite it as "vocabulary trimming made the model better." R2+B0 and stock
are not adapter-matched. The valid statement is that R2 is smaller, its localized defect
was measured, and the preregistered healing run recovered that defect.

### 4.4 What did not survive the model-surgery screen

- **Native low-rank factorization:** effective ranks are above the parameter
  break-even rank at 95/99% energy; 99% retention is parameter-negative.
- **Static 25% FFN keep:** the stabilizer follow-up measures roughly 53% relative
  output error, making that aggressive width point unsuitable.
- **Unhealed GQA head pooling:** structural cache reduction works, but the 4-KV model
  collapses semantically before healing. The experiment says **healing is part of the
  method**, not that GQA can never work.

---

## 5. Attention shape and Cornerstone 1

### 5.1 Sliding-window attention loses as the memory axis on this stack

The attention-shape experiment asks a controlled deployment question: at reduced cache,
should R2 spend the budget on a positional window or fewer KV heads?

SWA loses for three separate reasons in the measured Transformers/SDPA path:

1. its logical cache reduction does not reduce the resident prefill storage at the
   moment peak memory is set;
2. the dense sliding-window mask raises peak and blocks the fast path;
3. R2 is strongly sink-heavy, so once the window excludes the opening sink positions,
   quality falls off a cliff rather than degrading smoothly.

The result is stack-specific enough to preserve one caveat: a native window kernel may
change the efficiency story. The current result kills **SWA as this repository's
long-context memory strategy**, not every possible native-window implementation.

### 5.2 Significance-based eviction

C1 replaces position with observed use. Each cached position keeps accumulated received
attention at KV-head granularity and the policy protects sinks, recent positions, and a
score-selected heavy set.

The result has an operating range rather than a universal win: on short RAG prompts,
significance beats the positional/StreamingLLM control at moderate keep fractions but
loses at an aggressively thin 25% keep setting. That crossover is part of the result.

### 5.3 Count normalization, chunking and the query

Three later experiments are essential to the mature C1 mechanism:

- **count normalization `S/C^e`:** persistent survivors have more opportunities to
  accumulate raw score; dividing by observation count prevents the retained set from
  ratcheting shut against late arrivals;
- **chunked prefill:** caps the cache seen by later chunks so peak is governed by the
  retained budget plus chunk rather than full prompt length;
- **query-aware scoring:** old chunks cannot know what matters when the question appears
  only at the end. Running the actual question as a scoring query between compression
  rounds corrects that blind-eviction failure.

O11 later narrows the conceptual claim: on this passkey path, the system is best
understood as a **query-conditioned retrieval index**.

---

## 6. Alternative cache-selection and cache-storage experiments

### 6.1 Geometry/NMS: closed negative

The geometry branch asked whether top-k wastes slots on similar keys or similar
attention profiles. It did not. Key-space apparent redundancy is dominated by global
anisotropy, while attention-profile similarity behaves like functional
co-participation: suppressing similar profiles damages the attention output and loses to
a matched random reshuffle. This is why the result belongs in the negative registry
rather than an "open alternatives" list.

The historical branch name `exp/geometric-significance` is retained in docs for
provenance, but **no such live branch exists at this snapshot**.

### 6.2 Store probes

The store probe distinguishes eviction from representation. Important tested outcomes:

- uniform int4 is too damaging in the measured attention-error calibration;
- tokenwise K quantization is substantially worse than per-channel K at the same bit
  width;
- dense adjacent-layer CLA prediction has high held-out error and unfavorable parameter
  arithmetic;
- per-prompt low-rank SVD shows that cache information can be low-rank, but that is an
  oracle prompt-specific basis, not a measured deployable fixed MLA projection;
- K-per-channel int8 + V-per-token and **significance-gated mixed precision** survive as
  useful store-level results.

The mixed-precision result is underexploited, but it is still a **store-probe result**.
Do not silently promote it to an end-to-end Rineya deployment score until that stack is
run.

---

## 7. Stabilizer Strand A: useful mechanism, failed system claim

The original stabilizer was an inference-time FFN selector. Its preregistration is in
`docs/EXPERIMENT_DESIGN_STABILIZER.md`; the final Phase-0 record is
`experiments/stabilizer/results/FINDINGS.md`.

### P0.1 — concentration exists

Keeping the largest 25% SwiGLU contributions preserves about 0.9586 of the measured
layer-output norm on the probe. This refutes the initial expectation that smooth SwiGLU
activation would lack exploitable concentration.

### P0.2 — speed claim fails

Dynamic gather at 25% density measures about 0.84x of dense speed — a slowdown — while
a pregathered upper bound is faster. The gather is therefore a real cost, and the dense
FFN is too small a fraction of the measured decode floor for the idealized saving to
matter much end to end on this host.

### P0.3 — global predictor works; significance coupling does not

A global matrix reaches about 0.70 firing-set recall and preserves about 0.90 of the
output, comfortably above static/random controls. Adding KV-significance features moves
the result by only a few ten-thousandths. The firing set also changes strongly from token
to token, so "the other weights can stay unloaded" does not follow.

The correct conclusion is therefore:

> **SwiGLU concentration and global predictability are real, but the proposed sparse
> execution/memory system is not useful on this host.**

### P0.4 — persistent compression also fails

The attempted rescue permanently prunes a static subset and asks a low-rank correction
to reconstruct the discarded FFN contribution. Even the per-state oracle SVD needs
rank >=149 at the 90%-discarded-energy point as sampling grows, beyond the declared 128
threshold. The early apparent depth pattern also largely disappears with more prompts.
This closes the persistent-compression interpretation rather than inviting a fifth
post-hoc variant.

### P1 — future attention cannot be cheaply predicted from the key

A later reversal asks whether the global/content features can help the KV score predict
future attention. They do not: key-only prediction is weak and combining it with past
attention does not beat past attention. That negative is one reason the program moves
toward using the **actual question** rather than predicting future relevance.

---

## 8. Cornerstone 2: evidence lives in spans

C2 is not a new score. It is the discovery that the **unit being retained** matters.
The C2 submission record is `docs/PAPER_DRAFT_CORNERSTONE2.md`; the detailed mechanism
record is `experiments/stabilizer/results/FINDINGS_P2.md`.

### 8.1 Fragmentation, not wholesale eviction

Position-identity tracing shows that a passkey needle can remain present in most
layer/KV-head streams while generation fails. Per-token tracing then shows the missing
property: only part of the contiguous key survives. A five-token key with one position
missing is useless even though most of the evidence is still resident.

This is the decisive conceptual shift from C1's independent positions to C2's evidence
coherence.

### 8.2 Query-shaped labels are overprotected relative to payloads

The query-aware score preferentially reaches text resembling the query. In the traced
passkey format, framing such as `The pass key is` resembles the question and survives
more often than the digits it introduces. Centered max-pooling propagates that high score
to nearby payload positions, but not uniformly through the whole payload.

That diagnosis predicts an asymmetric repair: payload tokens should inherit score from
what **precedes** them.

### 8.3 Backward reach is the active direction

Backward-reaching/causal pooling produces the major C2 gain. Matched controls later show
that forward reach is largely inert and raw window width is not the explanatory variable:
a narrower backward-reaching window can beat a wider centered one.

The exact closed-form span law proposed during development is **retracted**. The valid
qualitative finding is that required backward reach **tracks evidence span size** and is
not a single universal constant.

### 8.4 Headline passkey result and its conditions

At the thin 128K passkey operating point, the corrected policy is not "eight magic
positions." The load-bearing structure is:

- protected attention sinks;
- a small recent set;
- eight score-chosen heavy positions at B=16.

The sink ablation is decisive: removing sinks collapses the cell even though it gives
more slots to the score. Random and StreamingLLM matched-budget controls also fail where
the significance-based mechanism succeeds.

The extreme compression ratio must always be quoted with its task condition. Passkey
filler contains no digits, so the target is uniquely typed and there are no plausible
competitors. Once query-shaped distractor spans are added, the required budget rises
sharply and the available compression ratio falls correspondingly. The headline is a
**single-distinctive-span retrieval result**, not a universal context-compression ratio.

### 8.5 Natural-language transfer

C2 does transfer to natural-language RAG under compression, but the scope is much less
spectacular than passkey:

- prompts are hundreds of tokens rather than 128K;
- causal pooling helps most under stronger compression pressure and can be slightly
  harmful when capacity is abundant;
- derived/geodesic reach, which is attractive on passkey, loses on the language arms;
- proportional allocation is essential so the score actually receives heavy slots.

Therefore the live claim is **short natural-language cache-quality preservation under
compression**, not 128K prose reasoning.

### 8.6 Mechanism identity: O11

The wrong-query control changes only the question used for the scoring pass; the model
is still asked the real question at answer time. Correct query scoring succeeds, while
two unrelated scoring questions and the no-query condition collapse the passkey result.

The mature interpretation is explicit:

> Given a question, the mechanism identifies cached positions that the question reaches
> for. It does not establish a question-independent notion of "important text."

### 8.7 Runtime: O12

C2 measures rather than assumes cost:

- directional pooling is approximately parity with the centered window it replaces;
- score accumulation during decode has a real overhead;
- `grow_spans` costs more than fixed pooling;
- the query-aware scoring pass is much more expensive than the naive
  `question_tokens / chunk_tokens` estimate because short forwards still stream model
  weights;
- nevertheless, bounded-cache compressed prefill can be faster end to end than no
  compression because later chunks attend to a capped cache.

At the snapshot, the C2 draft reports **252/252 pinned claims reproducing** and
`check_results_integrity.py` reports **0 problems**.

---

## 9. Passkey-to-reasoning program

The long-reasoning program exists to stop a single-span retrieval result from being
misreported as arbitrary reasoning at long context.

### Stage 1 — distractor competition: established

Adding neutral numeric competitors costs little at the measured thin budget. Giving
the same competitors query-shaped framing causes a much larger drop. Failure
classification and the budget-rescue ladder show that the dominant issue is **capacity
consumed by plausible spans**, not simply choosing the wrong decoy. Increasing the
budget restores the result.

### Stage 2 — word phrases: withdrawn

The word-phrase task attempted to extend the evidence-span axis beyond digit keys. It
changed the type of the payload to the same type as the filler. The model can then
recite filler and fail even with a full cache. Because the instrument no longer isolates
cache survival, the reported word-span curve is withdrawn.

### Long digit spans — endpoint ceiling

A corrected digit-only extension preserves the unique-type property but hits another
limit: the model itself cannot reliably reproduce sufficiently long digit strings with a
full cache. The program records roughly an 11-digit verbatim-recall ceiling on this
instrument. This makes longer generation-based span retention **not measurable on R2**;
it does not show that C2 necessarily fails on longer spans.

### Stage 3 — 2/3-hop chains: partial

The model first clears a short uncompressed capability gate. Corrected comparisons then
show:

- two-hop: the score beats random in the valid matched analysis and the compression cost
  itself is not detectable;
- three-hop: the score advantage is **not detectable** at the available n, with the
  nominal direction unfavorable;
- moving to much longer context damages chain reasoning even when the cache policy is
  not the causal difference.

Thus the mature domain is **retrieval/preservation of evidence spans**. The repository
does not establish general long-context multi-hop reasoning under extreme compression.

---

## 10. State-sufficiency program

`Rineya_state_experiment_bundle/` is a controlled approximate-Markov-sufficiency
experiment with three training conditions:

- full-context trajectory supervision;
- ordered compact-state transitions;
- the same stateful transitions globally shuffled.

Its evaluation separates local transition competence, state sufficiency when oracle
state is supplied, rollout error propagation, and autonomous end-to-end capability.
The bundle includes a generated dataset, checks, configs, and GPU/CPU runners.

**Canonical status: `BUILT — UNRUN`.** The bundle's README explicitly says that no
adapter has been trained on this dataset with a real checkpoint and that prior runs were
smokes/probes/calibration rather than results. Future audits must not convert its large
test count into a scientific outcome.

---

## 11. PCSM: Peer-Conditioned Stabilizer Matrix

PCSM is a **training-time** sparse native-weight method and is separate from the failed
inference-time FFN stabilizer.

For target matrix `t`, compatible peers contribute structural descriptors/property maps
to generate:

- exact binary support `M_t` — where native weights may move;
- bounded positive scale `A_t` — how strongly selected coordinates may move.

The real task gradient retains the update direction. `SparseNativeAdamW` masks gradient,
optimizer moments, realized update, and decoupled weight decay so undeclared native
coordinates cannot drift.

### 11.1 Synthetic feasibility

The committed five-seed constructed study reports mean true loss:

| Arm | Mean true loss | Support recall |
|---|---:|---:|
| **real peer-conditioned** | **0.022917** | **0.932552** |
| coordinate-shuffled peers | 0.035245 | 0.102344 |
| random support | 0.039252 | 0.091146 |
| gradient top-K | 0.069804 | 0.142969 |
| dense noisy update | 0.127059 | 1.000000 |
| no update | 0.030312 | 0.000000 |
| oracle | 0.000000 | 1.000000 |

The hidden optimum is intentionally constructed from peer structural properties. The
valid conclusion is therefore only:

> PCSM's implementation can learn and exploit peer-coordinate structure **when that
> structure is truly predictive**.

It does not establish that real transformer peer coordinates contain the same useful
signal.

### 11.2 Phase C real-gradient screen

Commit `647c60ea` adds the first executable real-model causal gate. It uses:

- `mlp.down_proj` across retained layers;
- primary 0.1% exact support per candidate matrix;
- leakage-separated `META_TRAIN`, `MASK_CAL`, `ONE_STEP_DEV`, `SEALED` partitions;
- random, gradient-top-K, target-only, real-peer bootstrap, calibrated real-peer,
  coordinate-shuffled and layer-identity controls;
- five preregistered seeds;
- an actual one-step held-out loss comparison from a byte-identical base checkpoint.

The decisive coworker comparison is **calibrated real peer coordinates versus the same
peer information with coordinate identity destroyed**. Full SFT is allowed only if the
five-seed advancement gate in `EXPERIMENT_PROTOCOL.md` passes.

**Canonical status: `BUILT — UNRUN`. No Rineya PCSM quality result is claimed.**

---

## 12. Negative and withdrawn knowledge that must remain visible

| Item | Status | Killing/correcting observation |
|---|---|---|
| 19L LoRA-soup repair | **RETIRED** | Degenerate output after deep structural deletion; confounded multi-change design. |
| `len <= 4` tokenizer selection | **CLOSED — NEGATIVE** | Token inflation crosses its own compute break-even. |
| SWA as long-context memory strategy | **CLOSED — NEGATIVE** | Peak/reach/quality all contradict the intended memory benefit on this stack. |
| Geometric/NMS KV eviction | **CLOSED — NEGATIVE** | Suppression loses to top-k and matched random. |
| Raw accumulated score | **CLOSED — NEGATIVE** | Persistent-survivor ratchet; late arrivals lose unfairly. |
| Query-blind chunk scoring | **CLOSED — NEGATIVE** | Evidence is evicted before the final question can score it. |
| Uniform int4 KV | **CLOSED — NEGATIVE** | Attention error reaches the damaging calibration regime. |
| Naive tokenwise K int8 | **CLOSED — NEGATIVE** | Same bit width, much worse error than per-channel K. |
| Cross-layer KV adapter / CLA | **CLOSED — NEGATIVE** | High held-out error and unfavorable parameter arithmetic. |
| Unhealed GQA as deployable quality point | **CLOSED — NEGATIVE** | 4-KV RAG collapses; healing/uptraining is required. |
| FFN stabilizer speed claim | **CLOSED — NEGATIVE** | Dynamic sparse gather is slower. |
| FFN stabilizer persistent-memory claim | **CLOSED — NEGATIVE** | Firing sets turn over too quickly. |
| Stabilizer low-rank compression anchor | **CLOSED — NEGATIVE** | Oracle required rank exceeds the preregistered threshold and rises with data. |
| C2 `sqrt(L)` budget law | **WITHDRAWN / SUPERSEDED** | It describes the centered-pool configuration and ceases to apply after backward reach changes the mechanism. |
| Exact `back >= d-1` law | **WITHDRAWN** | Does not extrapolate cleanly to d=7. |
| "Derived reach beats best tuned reach" | **WITHDRAWN** | Per-span tuned back=8 beats derived reach at d=7. |
| Word-phrase Stage 2 | **WITHDRAWN** | Filler and payload type are confounded; uncompressed controls fail by filler recitation. |
| "Multi-hop breaks because of compression" | **WITHDRAWN** | Matched-length controls show length/model capability is the larger breaker. |
| "Significance is intrinsic importance" | **REFRAMED** | O11 wrong-query controls show query-conditioned retrieval. |
| 12-30 digit compressed failures as cache evidence | **INVALID INTERPRETATION** | Full-cache model already fails the endpoint. |

`docs/NEGATIVE_RESULTS.md`, `docs/OPEN_QUESTIONS.md`, and the retraction appendices in
the paper drafts remain the detailed records. This table is the status index, not a
replacement for their evidence.

---

## 13. Paper, register and verification status at the snapshot

| Artifact | Role | Snapshot status |
|---|---|---|
| `docs/RESEARCH_PROGRAM.md` | Program objective and workstreams | Current program charter; some historical predictions are marked superseded in place. |
| `docs/MINI_MODEL_ROADMAP.md` | R2 surgery/healing provenance | Canonical R2 construction and B0 record. |
| `docs/FINDINGS_KV_PROGRAM.md` | C1 raw record | Canonical C1 mechanism/findings/retractions record. |
| `docs/PAPER_DRAFT_SIGNIFICANCE.md` | C1 paper-shaped draft | Current C1 draft; LaTeX manuscript also exists under `paper/significance_matrix/`. |
| `docs/CORNERSTONE_LEDGER.md` | C1 <-> C2 map | Historical/generated ledger with an explicit top correction section; read corrections before old body text. |
| `experiments/stabilizer/results/FINDINGS*.md` | Stabilizer/C2 mechanism record | P0, P1 and P2 detailed evidence. |
| `docs/PAPER_DRAFT_CORNERSTONE2.md` | C2 paper | Submission draft revised through 2026-08-29. |
| `docs/OPEN_QUESTIONS.md` | Anomaly register | **13 distinct O1-O13 entries, 13 closed, 0 open** as of 2026-08-29; O1/O8 include supersession rather than causal answers. |
| `scripts/design/verify_claims.py` | Numerical/prose drift guard | C2 draft reports **252/252** pinned claims reproducing at the snapshot. |
| `scripts/design/check_results_integrity.py` | Result-tree completeness guard | C2 draft reports **0 problems** at the snapshot. |
| `experiments/stabilizer/peer_optimizer/` | PCSM | Synthetic feasibility complete; Phase C executable; no real-model quality result yet. |

---

## 14. Branch/history handling

### Current remote state

At this snapshot the GitHub branches API returns **only `main`**. Historical names in
commits/docs such as `r2-healing-harness`, `exp/stabilizer-matrix`,
`exp/peer-stabilizer-matrix`, and `exp/geometric-significance` are provenance labels,
not live refs.

This is exactly why this audit exists: **branch cleanup must not erase program history,
and stale branch labels must not be mistaken for current development state.**

### Geometry branch

The geometric-significance code/result directory is not present on current `main`, but
the negative outcome is explicitly preserved in canonical documents. It is therefore a
valid **historical negative result**, not a live experiment directory.

### SGQR / CSGQR archival note

Older project summaries have referred to SGQR/CSGQR remnant-recovery work. At this
snapshot, the canonical GitHub remote exposes no live branch by those names and current
`main` contains no SGQR/CSGQR experiment directory. This audit therefore makes **no
numerical SGQR/CSGQR claim**. If a preserved commit SHA, report, or artifact is restored,
add it here as an archival track with evidence rather than reconstructing it from memory.

Status for this audit: **ARCHIVAL / NOT AUDITABLE FROM CURRENT CANONICAL REFS**.

---

## 15. Current scientific frontier

The highest-value unresolved questions are not the dead experiments above:

1. **PCSM Phase C:** does real peer-coordinate structure causally predict a better
   sparse native one-step update than coordinate-shuffled peers? This is the next
   training-time gate and must precede matched SFT.
2. **C1/C2 + mixed-precision store:** the store probe suggests a compositional memory
   win, but the combined end-to-end stack still needs a clean deployment experiment.
3. **C2 on a genuinely long-context-capable model / realistic long prose:** the current
   R2 instrument becomes the limitation on reasoning and long verbatim spans, so this is
   the main external-validity question for the mature inference-time mechanism.

By contrast, do not spend new compute repeating SWA on the same stack, geometry/NMS,
naive low-rank factorization, the original FFN stabilizer system, pure int4 KV, or the
withdrawn word-phrase task unless a specific underlying assumption has changed.

---

## 16. One-line program interpretation

The repository's results consistently favor:

> **remove capacity the model truly does not use, then allocate runtime/training
> resources conditionally to what the current task actually needs**

rather than assuming that lightly used, low-activation, geometrically similar, or
apparently redundant structure can be made permanently absent without cost.

That through-line connects the successful vocabulary trim, C1 query-conditioned cache
selection, C2 span coherence, and the still-unproven PCSM training-time hypothesis.
