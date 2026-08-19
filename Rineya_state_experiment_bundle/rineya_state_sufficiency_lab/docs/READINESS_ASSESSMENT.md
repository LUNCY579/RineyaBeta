# Readiness assessment — before the first real training run

Status of the lab against the experiment's hypothesis, measured on the actual
model and dataset in this workspace. Written before any real adapter has been
trained.

---

> **Checkpoint changed (2026-08-18).** The experiment now targets the official
> **Qwen/Qwen3-0.6B** (`../qwen3-0.6b`). The layer-dropped model soup originally in
> `../model` is retired: section 1 and the dress rehearsal in 7b describe *that*
> checkpoint and are kept as the record of why it was rejected. Section 1b
> describes the checkpoint actually under test.

## 1b. The checkpoint under test: official Qwen3-0.6B

Downloaded from `https://huggingface.co/Qwen/Qwen3-0.6B`.

| Property | Value | Consequence |
|---|---|---|
| Architecture | `Qwen3ForCausalLM`, **28 layers**, hidden 1024, 16 heads / 8 KV | Full-depth model, not a pruned one |
| Parameters | ~596 M (≈0.6 B); 10,092,544 trainable at LoRA r=16 | Adapter-only; base stays immutable |
| Stored dtype | bfloat16 (1.50 GB) | bf16 loads natively on any modern CUDA card |
| Vocab / embeddings | 151,936, tied | No `modules_to_save` needed |
| pad / eos / bos | `<|endoftext|>` (151643) / `<|im_end|>` (151645) / none | pad ≠ eos, so completion counting is exact |
| `generation_config.json` | `do_sample=true, T=0.6, top_k=20, top_p=0.95`, eos `[151645, 151643]` | Lab overrides to greedy and passes **both** stop ids |
| Chat template | bundled, loads correctly | `template_source = tokenizer_chat_template` |
| Thinking mode | `<think>\n\n</think>\n\n` is injected **only** when `enable_thinking=false` | See the decision below |

**Provenance is now clean.** This is a published, fully-trained instruct model with
no layer surgery and no adapter soup. Conclusions still apply to *this* checkpoint
rather than to language models generally, but the "already fine-tuned on unknown
data" caveat that applied to the soup no longer does.

### Decision: `enable_thinking` is pinned **false** for every run

Recorded in `configs/model_local.yaml` and every `gpu_*.yaml`, via
`model.chat_template_kwargs: {enable_thinking: false}`. Reasons:

1. **Train/eval prompt identity.** The training targets contain no reasoning
   blocks. Leaving thinking on at evaluation would score the adapters on a prompt
   shape they were never trained under.
2. **It would defeat the hypothesis.** The compact-state claim is that `s_t`
   suffices for the next decision. If the model may perform unbounded internal
   chain-of-thought at every step, it can silently reconstruct the history the
   experiment is trying to remove.
3. **Cost.** Thinking output is long, and the DEV suite is ~80,500 generations per
   adapter.

The setting is identical for both arms and for the untouched-base baseline, so it
cannot favour either side. The base model is additionally probed with thinking
**on** purely as context on how much the pin costs it (§1c).

## 1c. Capability probe of the untouched Qwen3-0.6B

Run on this host, CPU, bf16, `enable_thinking: false`, against the real harness
(`runs/probe_nothink/`, `runs/probe_fair/`).

| Protocol | n | strict | lenient | dominant extraction routes |
|---|---:|---:|---:|---|
| `full_direct` (200 new tokens) | 8 | 0% | 0% | truncated mid-derivation |
| `full_direct` (768 new tokens) | 6 | **16.7%** | 16.7% | `labeled_answer`, `boxed_latex` |
| `direct_answer` | 8 | 12.5% | 25% | `labeled_answer`, `boxed_latex` |
| `oracle_transition` | 34 | 0% | 0% | **`json_fenced` 29/34** |

Three things this settles:

1. **The model is coherent and instruction-following.** It reasons step by step,
   uses `\boxed{}`, and — untouched, with no adapter — **already emits well-formed
   fenced JSON for 29 of 34 transition prompts**. The SPLIT contract is not exotic
   for this checkpoint.
2. **There is large headroom and no floor-effect risk.** Baseline sits near
   0-17% depending on protocol, so Gate G9 should clear comfortably; the
   pathology that retired the previous checkpoint is gone.
3. **Two harness defects were exposed and fixed** (below). Both would have
   systematically *understated* accuracy for every arm.

### Defects the probe exposed

**Answer-label coverage.** The model writes `The final answer is: **Ivan**.` and
`### ✅ Final Answer: **False**.` Neither matched any label pattern, so a correct
answer fell through to the last-line fallback and lost strict credit. Re-scoring
the same stored responses with the fixed patterns moved `full_direct` 0/8 → 1/8
and `direct_answer` 0/8 → 1/8 strict. Fifteen decorated forms are now covered by
regression tests, with two guards against over-matching: a dash separates only
when whitespace-surrounded (so `The final answer is -6` keeps its minus sign), and
emphasis is skipped only when followed by whitespace.

**Structured compact states.** The model sometimes returns
`updated_compact_state` as a JSON *object* rather than a string. That is a
contract violation, but the old code returned `None`, which made the rollout treat
the step as unusable and abort. Aborting there would have converted the
error-propagation measurement into a formatting-compliance measurement. Structured
states are now serialised so the rollout continues, flagged `state_coerced`, and
denied strict credit.

**Token budget.** At 320 new tokens the untouched model was cut off mid-derivation
and scored `no_answer_found` for reasoning that was on its way to a correct
answer. The base-model baseline now runs at 768; the adapters stay at 320, which
is ample for the ~79-token targets they are trained on. Being generous to the
baseline is the conservative direction: it makes the adapters' improvement harder
to claim, not easier.

## 1. The retired checkpoint (`../model`) — kept for the record

Read from `model/config.json`, `model/soup_manifest.json`, `model/chat_template.jinja`.

| Property | Value | Consequence for the experiment |
|---|---|---|
| Architecture | `Qwen3ForCausalLM`, 19 layers, hidden 1024, 16 heads / 8 KV | Standard LoRA target modules apply unchanged |
| Parameters | 461,319,936 total; 6,848,512 trainable at LoRA r=16 | 1.5% trainable — adapter-only, base stays immutable |
| Stored dtype | float32 (1.82 GB on disk) | fp16/bf16 load halves resident memory |
| Vocab / tied embeddings | 151,936 / tied | No `modules_to_save` needed |
| pad / eos / bos | `<|endoftext|>` (151643) / `<|im_end|>` (151645) / none | pad ≠ eos, so completion-token counting is exact |
| `generation_config.json` | `do_sample=true, T=0.6, top_k=20, top_p=0.95`, **two** eos ids `[151645, 151643]` | The lab must override to greedy and must pass **both** stop ids |
| Chat template | present as `chat_template.jinja`, loads correctly | `template_source = tokenizer_chat_template`, not the portable fallback |
| Thinking mode | generation prompt is `<|im_start|>assistant\n`; the empty `<think></think>` block is emitted **only** when `enable_thinking=false` | Left undefined, the model may emit free-form `<think>…</think>` at inference |

**Provenance caveat.** `soup_manifest.json` shows this checkpoint is a *model soup*:
a layer-dropped Qwen3 base (`stage_09_drop_14_24_6_23_8_10_18_9_7`) merged with two
LoRA adapters at 0.5/0.5. It is therefore **already fine-tuned on unknown data**.
This does not invalidate the experiment — both arms start from the identical
checkpoint, which is what the design requires — but it does bound the claim:

> Any result is a statement about *this* checkpoint, not about Qwen3-0.6B and not
> about language models generally. Prior exposure to reasoning-style supervision
> cannot be ruled out from the manifest alone, so a high base-model score is not
> automatically evidence of contamination, and a low one is not automatically
> evidence of a hard task. Run `configs/eval_base_model.yaml` first and report it.

---

## 2. Measured token budgets (this tokenizer, this dataset)

Measured over 3,000 TRAIN rows per view with the real chat template:

| View | prompt μ | target μ | total μ | total p99 | total max |
|---|---:|---:|---:|---:|---:|
| FULL (`messages`) | 43.6 | 78.8 | 122.4 | 263 | 377 |
| SPLIT stateful (`stateful_rollout_messages`) | 102.3 | 52.9 | 155.2 | 254 | 305 |
| SPLIT next-step (`messages`) | 81.3 | 18.2 | 99.5 | 166 | 205 |

One native epoch:

| Arm | rows | target tokens | total tokens |
|---|---:|---:|---:|
| FULL | 43,200 | 3.40 M | 5.29 M |
| SPLIT stateful | 170,941 | 9.04 M | 26.53 M |
| **SPLIT / FULL ratio** | **3.96×** | **2.66×** | **5.02×** |

**This is the single most important number in the readiness check.** A native
one-epoch comparison gives SPLIT 2.66× the supervised tokens and 5.02× the total
tokens. Any SPLIT advantage under Regime N is uninterpretable on its own. The
matched-budget regimes are not optional extras here — at least one of Regime T
(matched target tokens) or Regime U (matched updates) must be the primary result.

`max_seq_len: 320` covers the p99 of both arms and the observed max of SPLIT;
FULL's max is 377, so 384 truncates nothing at all. Use **384** and expect
`truncated_targets = 0` in both arms.

---

## 3. Capability inventory against the specification

Legend: **✅** implemented and tested · **⚠️** implemented but not usable on this
machine · **🔧** gap that must be closed before the real run.

### Training pipeline

| Requirement | Status | Evidence |
|---|:--:|---|
| FULL-context LoRA training | ✅ | `configs/train_full.yaml`; real adapter produced in tests |
| SPLIT stateful-transition LoRA training | ✅ | `configs/train_split_stateful.yaml`; real adapter produced |
| SPLIT next-step-only ablation | ✅ | `configs/train_split_next_step.yaml`, with its own plain-text scoring contract |
| Ordered-transition ablation | ✅ | `configs/train_split_ordered.yaml`; identical record set proven by test |
| QLoRA configurations | ⚠️ | configs ship; `bitsandbytes` has no Apple-Silicon build — unusable here |
| Automatic BF16/FP16 selection | 🔧 | correct on CUDA; **MPS falls back to fp32** and thrashes 8 GB RAM |
| Assistant-only loss masking | ✅ | computed exactly from a separate prompt encode; test asserts the mask |
| Target-preserving truncation | ✅ | prompt head+tail retained, target never cut unless it alone overflows |
| Gradient checkpointing | ✅ | `training.gradient_checkpointing` |
| Checkpoint resume | ✅ | restores optimizer, scheduler and exact position; test asserts fewer records consumed |
| Deterministic seeding | ✅ | SHA-256 salted ordering, reproducible across machines |
| Per-problem loss weighting | ✅ | test asserts each problem's weights sum to exactly 1.0 |
| Exposure and token accounting | ✅ | records/problems/input/target/padded/updates/truncations |
| HF chat template + portable fallback | ✅ | this model uses its own template; fallback covered by tests |
| Accelerate launch configurations | ⚠️ | single-GPU / multi-GPU configs ship; no GPU here |
| FSDP and DeepSpeed ZeRO-2 | ⚠️ | configs ship; `deepspeed` not installable on macOS |
| Windows PowerShell + Unix launchers | ✅ | 10 `.sh` + 10 `.ps1`, syntax-checked in CI |
| Offline tiny-model training smoke tests | ✅ | `rineya-lab smoke --all` trains two real adapters, passes |

### Training controls

| Control | Status |
|---|:--:|
| Native epoch exposure | ✅ |
| Matched optimizer updates | ✅ |
| Matched target-token budgets | ✅ (`rineya-lab budget` computes the value for both arms) |
| Matched unique source-problem exposure | ✅ |
| One-transition-per-problem sampling | ✅ (test asserts exactly one row per problem, rotating across epochs) |
| Globally shuffled transitions | ✅ (test asserts both adjacency counts stay at the uniform-shuffle scale; zero is not the invariant — see `order_diagnostics`) |
| Trajectory-ordered transitions | ✅ (test asserts contiguous ascending steps) |
| Reverse-order and curriculum ablations | ✅ |

### Evaluation architecture

All ten protocols implemented and exercised end-to-end. The four questions are
kept structurally separate:

| Question | Protocols |
|---|---|
| 1. Local transition competence | `oracle_transition` |
| 2. State sufficiency | `oracle_transition` vs `full_history_transition`, floored by `no_state_transition` / `last_state_only` |
| 3. Error propagation | `oracle_trajectory` vs `generated_rollout` |
| 4. End-to-end capability | `full_direct` (FULL) vs `generated_rollout` (SPLIT) |

Perturbations (`shuffled_state`, `distractor_state`), loop / stagnation /
premature-completion / nontermination detection, first-error-depth analysis and
the teacher-forcing-versus-free-rollout gap are all implemented, and the smoke
suite proves the harness *detects* each failure mode rather than assuming it.

### Response handling, scoring, statistics, reproducibility

Every item in the specification is implemented and tested, with one exception
(`<think>` blocks, below). Notably:

- 2,400/2,400 DEV gold answers score correct under their declared `answer_type`;
  0/2,400 perturbed variants are falsely accepted.
- Unparseable, refused and empty responses stay in the denominator
  (`scored_items == total_items` asserted by test).
- Pairing is on `problem_id`; a test proves 9,460 transition rows collapse to the
  correct number of paired units.
- Full 48,000-problem / 189,922-transition strict validation **passes**, including
  the causal safeguard (which is itself tested to fire on injected leakage).

---

## 4. Gaps found — and their resolution

> **All eight are now closed.** The table below records what each one was, why it
> mattered, and what fixed it, so the history is auditable rather than erased.

None are design flaws; all are concrete and small.

| # | Gap | Why it blocked | Resolution | Status |
|---|---|---|---|---|
| 1 | `evaluation.generation.batch_size` declared but never used — every generation was a batch of 1 | The DEV suite is ~80,500 generations per adapter; unbatched decoding dominated total runtime | `Backend.generate_batch(batch, oracles=...)` with a parallel per-item oracle list; all six transition protocols and both answer protocols now batch. A test asserts batch size 1 and 8 produce **identical** records | ✅ closed |
| 2 | Training device came from `evaluation.backend.device`; `select_precision` returned fp32 on MPS | Training silently ran on CPU; fp32 on 8 GB unified memory drove swap to 6.9 GB | Added `training.device`; MPS now resolves to fp16; trainable LoRA parameters are promoted to fp32 (`training.fp32_trainable_params`) so low-precision bases cannot round away adapter updates | ✅ closed |
| 3 | No `<think>…</think>` handling | This checkpoint's template permits free-form thinking; scratch reasoning could be mined for answer candidates, inflating `ambiguous` and biasing the base-model baseline most | `strip_reasoning_blocks()` removes closed blocks before extraction and discards everything after an *unterminated* opener; controlled by `evaluation.strip_reasoning_blocks`; the removal is recorded as a note on every affected record | ✅ closed |
| 4 | Only one eos id passed to `generate` | This checkpoint declares two (`151645`, `151643`); missing one lets generation run past a real stop and inflates `nontermination` | `_stop_token_ids()` unions the checkpoint's `generation_config` with the tokenizer's; completion-token counting stops at the first of any of them; the resolved list is recorded in the report | ✅ closed |
| 5 | `training.eval_steps` was a dead option; no DEV-based checkpoint selection | The protocol requires a fixed, identical checkpoint-selection rule; it was implicitly 'last' | `training.checkpoint_selection: last | best_dev_loss` with a fixed DEV probe (`dev_eval_max_records`). The selected adapter is recorded, and the final-step adapter is preserved as `last_adapter` so the choice is auditable | ✅ closed |
| 6 | No cross-seed aggregation | The protocol asks for ≥3 seeds; each seed produced an isolated comparison | `rineya-lab aggregate` → seed-level t interval, between-seed SD, and an explicit **`inconclusive_at_this_power`** verdict when run-to-run spread exceeds the margin. A sign-disagreement warning fires when seeds point in different directions | ✅ closed |
| 7 | Rollouts were strictly sequential | `generated_rollout` is the primary SPLIT endpoint and the slowest protocol | Step-synchronous batching: depth-*t* calls of different problems are batched, each rollout still consuming only its own state. A test asserts no state leaks between problems, and finished rollouts drop out of the batch | ✅ closed |
| 8 | The DEV/TEST subset was keyed on `experiment.seed` | With `evaluation.limit` set, each training seed would score a *different* subset | Added `evaluation.subset_seed`, decoupled from the training seed. A test asserts seeds 1, 2 and 999 select an identical subset and that both arms record the same `problem_ids_sha256` | ✅ closed |

---

## 5. Does this strictly meet the hypothesis?

**The design does. The claim it can support is narrower than the hypothesis as
stated, and that gap is intrinsic, not fixable by more code.**

### What the design establishes cleanly

The hypothesis is `P(r_{t+1} | q, H_t) ≈ P(r_{t+1} | q, s_t)`, operationalised as
"can sequence-free compact-state training replace full-sequence training".
Three properties make the answer interpretable rather than suggestive:

1. **Condition B exists and is provably record-identical to condition C.** Without
   it, an A-vs-C difference has four candidate explanations. With it, order is
   isolated as a single variable. A test asserts the two conditions consume the
   same row set and differ only in adjacency.
2. **Budget matching is available and, given the measured 2.66×/5.02× token ratio,
   necessary.** The design can distinguish "SPLIT is better" from "SPLIT saw more".
3. **The four evaluation questions cannot be silently conflated**, so
   "compact states are sufficient *when supplied*" cannot be reported as
   "the model maintains sufficient states autonomously". The free-rollout gap
   makes the difference a reported number rather than a rhetorical choice.

### Threats to validity the design cannot remove

State these in any write-up; they are limits of the experiment, not bugs.

1. **Format asymmetry.** SPLIT is asked for JSON; FULL is asked for free text. The
   lab quantifies this (`format_attributable_pp`, strict-vs-lenient tracks) but
   cannot eliminate it. *Mitigation available:* the `split_next_step` arm is
   plain-text and can act as a format control.
2. **Cross-protocol primary pairing.** FULL is scored via `full_direct`, SPLIT via
   `generated_rollout`. Each arm is evaluated the way it was trained; that is the
   fairest available contrast, but it is not an identical prompt.
3. **One checkpoint, and a non-neutral one.** A souped, layer-dropped, already-fine-tuned
   model. Conclusions do not transfer to other model families without repetition.
4. **One dataset realization.** Templates and numeric ranges are shared between
   TRAIN and TEST. Cross-template generalization requires regenerating the dataset
   at a different seed (`scripts/generate_dataset.py --seed`) and *not* tuning on
   it. Until then, "generalization" claims are out of scope.
5. **Approximate, not exact, Markov sufficiency.** The experiment measures whether
   the *approximation* is good enough to preserve capability. It does not
   establish that `s_t` is information-theoretically sufficient.
6. **Statistical power.** 2,400 held-out problems support a ±2 pp decision only if
   the observed difference and its variance are small. If per-seed variance
   exceeds the margin, the honest report is "inconclusive at this power", not a
   directional claim.

### Verdict

Once gaps 1–4 and 8 are closed and 5–6 are declared or implemented, the pipeline is
strictly adequate to answer the hypothesis **for this checkpoint and this dataset**,
with the six caveats above stated explicitly in the write-up.

---

## 6. How this run will be monitored and supervised

The purpose of this section is that neither of us has to trust a summary. Every
claim below is backed by an artifact on disk that you can open.

### 6.1 Pre-flight gates — all must pass before any adapter is trained

Each is automated and produces a file. A red gate stops the run; I do not
"proceed anyway".

| Gate | Command | Pass condition | Artifact |
|---|---|---|---|
| G1 Dataset integrity | `rineya-lab validate-data` | `status: PASS`, 0 errors, 48,000 / 189,922 | `runs/validation_full.json` |
| G2 Order really is destroyed | `rineya-lab plan -c train_split_stateful.yaml` | both adjacency counts ≤ their `uniform_shuffle_reference` ceiling | `runs/plan_split.json` |
| G3 Order really is preserved (cond. B) | `rineya-lab plan -c train_split_ordered.yaml` | `adjacent_consecutive_steps ≥ 0.99 × consecutive_pairs_available` | `runs/plan_ordered.json` |
| G4 B and C are record-identical | pytest `test_ordered_and_unordered_conditions_use_the_same_record_set` | passes | test log |
| G5 No supervision is truncated | `rineya-lab train -c dry_run_*.yaml` | `truncated_targets == 0` in **both** arms | `runs/dry_run_*/run_manifest.json` |
| G6 Budgets measured | `rineya-lab budget` | matched values computed for both arms | `runs/budget.json` |
| G7 Base-model baseline | `rineya-lab evaluate -c eval_base_model.yaml` | completes; recorded as the zero point | `runs/eval_base_*/eval_report.json` |
| G8 Harness self-check | `rineya-lab smoke --all` | `status: PASS`, all 7 failure modes detected | stdout + smoke dir |
| **G9 Learnability / headroom** | train condition A at seed 1, then `full_direct` on DEV | strict accuracy clears the pre-declared floor **and** materially exceeds G7 | `runs/T_eval_full_s1/eval_report.json` |

G9 is the stop-the-experiment gate, added after the dress rehearsal in 7b. If the
FULL arm cannot learn the task on this checkpoint, both arms sit near zero and an
A-vs-C null becomes uninterpretable: "sequence-free training is just as good" and
"neither representation worked" produce the same number.

G7 deserves emphasis: **every adapter number will be reported as an absolute
score and as a delta from the untouched checkpoint.** If the base model already
solves these problems, the experiment has little headroom and we should know that
before spending the compute, not after.

### 6.2 Live monitoring during training

I watch a fixed set of signals and stop on a fixed set of thresholds, rather than
watching loss go down and declaring success.

| Signal | Source | I stop and report if |
|---|---|---|
| loss trajectory | `run_manifest.json → training.loss_history` | NaN/Inf, or no decrease over 200 updates |
| grad-norm behaviour | training log | repeated clipping at `max_grad_norm` (LR too high) |
| `truncated_targets` | exposure account | > 0 (supervision is being silently cut) |
| `truncated_prompts` | exposure account | > 1% of records |
| `dropped_records` | exposure account | > 0 unexpectedly (budget admission control firing) |
| exposure parity | both manifests | the arms' `target_tokens` differ by > 1% under a matched-token regime |
| throughput | `records_per_second` | drops > 30% from the first 100 updates (thermal/swap) |
| host memory | `vm.swapusage` | swap climbing steadily — on this machine that means the config is too large |
| stop reason | `training.stop_reason` | anything other than the expected budget/data-exhaustion reason |

I will report progress at each phase boundary, not continuously — and I will tell
you when something is *still running* rather than predicting how it will turn out.

### 6.3 Phase gates that require your decision

I will stop and hand back to you at these four points. These are the moments
where a wrong call quietly invalidates the result, so they are yours, not mine.

- **Gate A — after pre-flight.** You see G1–G8, the measured token ratio, and the
  realistic time estimate. You choose the scope (full 48 k vs a scaled pilot) and
  the primary budget regime.
- **Gate B — before TEST is ever touched.** You freeze the non-inferiority margin
  in `docs/PREREGISTRATION.md` and set `margin_frozen: true`, with a written
  justification. I will not run a TEST evaluation while that flag is `false`.
- **Gate C — after DEV results for all arms.** You see the full DEV comparison and
  decide whether the result is worth spending TEST on. TEST is opened exactly once.
- **Gate D — before any claim leaves this workspace.** You approve the wording,
  specifically which of the two conclusions the evidence supports:
  *"compact states are sufficient when supplied"* versus *"the model maintains
  sufficient states autonomously"*.

### 6.4 Guardrails against fooling ourselves

These are the ways an experiment like this usually goes wrong, and what the lab
does about each.

| Failure mode | Guardrail |
|---|---|
| Tuning on TEST | All tuning on DEV. TEST opened once, after Gate B. Split is recorded in every report. |
| Moving the margin after seeing results | `margin_frozen` defaults to `false` and every comparison report prints a warning until it is frozen with a justification. |
| Cherry-picking a budget regime | Every regime run is reported, including the ones that disagree. A result that appears under only one regime is reported as such. |
| Cherry-picking a seed | All seeds reported; cross-seed variance reported next to the effect. If variance > margin, the verdict is "inconclusive at this power". |
| Silently dropping bad outputs | `scored_items == total_items` is asserted; unparseable output scores as failure with a typed label. |
| Confusing format compliance with reasoning | Strict and lenient tracks both reported; `format_attributable_pp` isolates the formatting component. |
| Overclaiming from oracle-state results | The free-rollout gap is computed and printed in the same table. |
| Silent config drift between arms | Every run records a `config_fingerprint`; I diff the two resolved configs and report exactly which keys differ. |
| A "result" that is actually the reference backend | Reference-backend reports carry a warning field and comparisons raise a validity warning. |

### 6.5 What you will receive

Per training run: `run_manifest.json` (resolved config + fingerprint, dataset
hashes, seeds, environment, exposure and token accounting, loss history, stop
reason, throughput, peak memory) and `train.log`.

Per evaluation: `eval_report.json`, a human-readable `eval_report.md`, and
`records/<protocol>.jsonl` containing **every** prompt, raw response, extraction
route, parse status, scorer, normalised prediction and failure label.

Per comparison: `comparison_report.json` + `.md` with paired differences,
bootstrap CIs, exact McNemar, sign-flip randomization, the superiority /
non-inferiority / equivalence decision, per-category / per-depth / per-failure-type
strata with Holm-Bonferroni correction, strict-vs-lenient disagreement, and a
deterministic audit sample of disagreements for you to read by hand.

Plus one final `EXPERIMENT_REPORT.md` that states the result, the caveats from
§5, and — explicitly — which of the two conclusions is supported.

### 6.6 What I will not do without asking

- Run a TEST-split evaluation while the margin is unfrozen.
- Change any config that is supposed to be held constant across arms, mid-experiment.
- Drop a condition, a seed, or a protocol from the report because it was slow or
  inconvenient.
- Present a reference-backend number, a partial run, or a single-seed result as a
  finding.
- Reduce the scope of the experiment to fit the hardware without telling you
  exactly what was cut and what that costs in interpretation.

---

## 7. Hardware feasibility — measured, not estimated

Host: Apple Silicon (arm64), **8 GB unified memory**, 4 torch threads, no CUDA,
MPS available. Measured directly with this model and LoRA r=16.

| Configuration | Result |
|---|---|
| fp32 + MPS, bs=4, seq=320 | Model reached the device; **no optimizer step ever completed**. Host swap rose to **6.9 GB of 8 GB**. Killed after ~10 min. |
| fp16 + MPS, bs=2, seq=320 | Model on device at 44 s; **first optimizer step still unfinished after 8 min**. Killed. |
| fp16 + MPS, bs=1, seq=128 | Completed. Step times: **0.63 s, 10.1 s, 13.0 s, 41.2 s, 49.6 s, 114.0 s** — median **25.7 s** |

The 180× spread between the fastest and slowest identical step is the diagnosis:
**this machine is memory-bound, not compute-bound.** The 0.63 s step is what the
GPU can actually do; everything above it is swap.

Extrapolated one-epoch training time, at the two ends of that range and at the
sequence length the data actually needs (384, which truncates nothing):

| Arm | rows | at best-observed rate | at median observed rate |
|---|---:|---:|---:|
| FULL | 43,200 | ~23 h | ~308 h (13 days) |
| SPLIT stateful | 170,941 | ~90 h | ~1,219 h (51 days) |

And that is **one epoch of one arm at one seed.** The protocol calls for three
conditions × ≥3 seeds, plus ~80,500 generations per adapter for the DEV protocol
suite (~241,600 for three arms).

### 7b. Dress rehearsal on the RETIRED soup checkpoint (why it was rejected)

Before writing any plan I ran the real checkpoint through the real harness on four
DEV problems (`runs/dress_rehearsal/`). This is the most consequential finding in
this document.

The untouched model is **severely degraded**. Verbatim samples:

| Gold | Raw response (truncated) | Extraction |
|---|---|---|
| `Ivan` | `Ivan is not the order.` | last-line fallback, no answer marker |
| `-6` | `imony` repeated ~100 times | degenerate repetition loop |
| `False` | `untos ... Okay, I need to use the values ... I think the first two is the value of the first two.` (looping prose) | matched `the answer is ...`, scored wrong |

`full_direct` strict **0/4**, lenient **0/4**. It emitted no `<think>` blocks, no
JSON, and no coherent final answers. This is consistent with its provenance: a
Qwen3 base with **9 of 28 layers dropped**, then souped from two LoRA adapters.

**This checkpoint was retired as a result.** The user replaced it with official
Qwen3-0.6B. The reasoning below is retained because it is why Gate G9 exists, and
because the same failure mode could recur with any weak base.

**Implication - a floor-effect risk that would silently void the experiment.**
Large headroom is good: there is room for an adapter to show an effect. But if
LoRA cannot lift this checkpoint to competence, **both** arms land near zero and
"no difference between A and C" becomes uninterpretable - indistinguishable from
"neither representation worked". A null result under a floor effect is not
evidence for the hypothesis.

This adds a mandatory gate:

> **Gate G9 - learnability / headroom.** After training condition A (FULL) at the
> first seed, its `full_direct` strict accuracy on DEV must clear a floor declared
> in advance (suggest **>= 25%**, and materially above the base-model baseline from
> G7). If it does not, **stop**: the comparison cannot answer the hypothesis on
> this checkpoint. Remedies, in order of preference: (a) a healthier base
> checkpoint, (b) more LoRA capacity (r 32-64) and/or more epochs, (c) restrict to
> the categories FULL can learn, and report that restriction.

The floor must be declared in the preregistration **before** seeing the number, or
it is not a gate - it is a rationalisation.

### Verdict on this host

**The full 48,000-problem experiment cannot be run on this machine.** Not "slowly" —
the memory pressure makes even the pilot timing unpredictable, and unpredictable
timing is itself a threat to a controlled comparison (thermal and swap variation
would differ between the FULL and SPLIT runs).

Two viable paths:

**Path A — run it on a CUDA GPU (recommended).** This 0.46 B model with LoRA at
seq 384 is small. On a single mid-range GPU a FULL epoch is minutes, a SPLIT epoch
tens of minutes, and the whole three-condition × three-seed design plus evaluation
fits comfortably in a day. The repository already ships the accelerate, FSDP,
DeepSpeed ZeRO-2 and QLoRA configurations for exactly this, and `bitsandbytes`
(unavailable on Apple Silicon) becomes usable. **Nothing in the design changes.**

**Path B — a scaled pilot on this machine.** Reduce the source-problem count and
the DEV evaluation subset, close gaps 1–4 and 8 first (batched generation matters
most), and close other applications to reduce swap. Indicative sizing, to be
re-measured on a quiet machine before committing:

- TRAIN: 2,000 of 43,200 problems (`data_view.max_unique_problems: 2000`)
  → FULL 2,000 rows, SPLIT ≈ 7,900 rows
- DEV: 200 of 2,400 problems (`evaluation.limit: 200`) → ≈ 6,700 generations/adapter
- One seed, three conditions ≈ 12–16 h; three seeds ≈ 2 days

**The cost of Path B must be stated up front:** with ~200 paired problems, the
95% CI half-width on a paired accuracy difference is roughly **±7 percentage
points**. A 2 pp non-inferiority decision is **not attainable** at that scale —
approximately 900–2,400 paired problems are required, which is precisely why the
dataset ships 2,400 held-out problems. A pilot can therefore answer
*"is there a large effect, and does the pipeline behave?"* but it cannot answer
*"is SPLIT non-inferior within 2 pp?"* That question needs Path A.

### 7c. Decision: CPU pilot on this host (no GPU available)

No GPU is available, so the experiment runs here. MPS was re-tested with the
official checkpoint and **abandoned**: with ~1 GB of free RAM the model could not
be moved onto the Metal device within 30 minutes, and earlier MPS runs showed
step times varying 0.63 s -> 114 s. Unstable timing is not merely slow -- if the
FULL and SPLIT arms meet different swap conditions, the comparison acquires a
confound unrelated to training representation. The pilot therefore runs on **CPU
fp32**, which is slower but stable.

**The scope reduction is applied to TRAIN, never to EVAL.** This is the single
most important design choice in the pilot, and it follows from where statistical
power comes from:

| Quantity needed for a 95% CI half-width h | discordance d=0.10 | d=0.20 |
|---|---:|---:|
| h = 2 pp | 960 paired problems | 1,921 |
| h = 3 pp | 427 | 854 |
| h = 4 pp | 240 | 480 |

Power depends on the number of paired **problems evaluated**, not on how much
data the adapters were trained on. Shrinking EVAL would destroy the ability to
conclude anything; shrinking TRAIN only narrows the claim to "at this training
scale". So the pilot evaluates on the **full 2,400-problem DEV split** and
reduces only the training budget.

What the pilot can and cannot support:

* **Can**: a genuine, budget-matched A-vs-C (and B-vs-C) comparison at a reduced
  training scale, with full-power intervals on the primary endpoint, the
  free-rollout gap, state-sufficiency contrasts and the failure taxonomy.
* **Cannot**: any claim about behaviour at full training scale, and — from a
  single seed — any claim at all about direction, because one seed provides no
  estimate of run-to-run variance. Seeds 2 and 3 and `rineya-lab aggregate` are
  required before a verdict.

Configs: `configs/pilot_*.yaml`. Runner: `scripts/run_mac_pilot.sh`, which is
resumable stage-by-stage because a multi-day run on a swapping host will be
interrupted.

### Recommended sequencing either way

1. Close gaps 1–4 and 8; declare gap 5; implement gap 6 (cross-seed aggregation).
2. Run pre-flight gates G1–G8, including the untouched-base-model baseline.
3. Re-measure throughput on a quiet machine and fix the scope with you (Gate A).
4. Train the three conditions at seed 1; evaluate on DEV; report.
5. Repeat for seeds 2 and 3; aggregate across seeds.
6. Freeze the margin (Gate B), then open TEST once (Gate C).
7. Write the final report and agree the wording of the conclusion (Gate D).
