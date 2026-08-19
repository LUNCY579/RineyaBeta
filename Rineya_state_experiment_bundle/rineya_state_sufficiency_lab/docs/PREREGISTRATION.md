# Preregistration template

Fill this in **before** any TEST-split result is generated. Commit it, then do not
edit it — record deviations in the "Deviations" section at the bottom instead.

Nothing in this repository enforces preregistration. What it does enforce is
visibility: `comparison.margin_frozen` defaults to `false`, and every comparison
report carries a warning until you set it to `true` and supply a justification.

---

## 1. Hypothesis

**H1 (primary).** A model trained on independently sampled, globally shuffled
compact-state transitions attains end-to-end final-answer accuracy on held-out
problems that is non-inferior to a model trained on complete reasoning
trajectories from the same base checkpoint.

Formally, with `s_t = g(q, H_t)`:

```
P(r_{t+1} | q, H_t)  ≈  P(r_{t+1} | q, s_t)
```

**H2 (order).** Chronological presentation of transitions during training does
not materially change learned capability (condition B vs condition C).

**H3 (decomposition).** Decomposed state-transition supervision differs from
full-rationale supervision (condition A vs condition B).

State the *direction* you expect for each, and what result would change your mind.

- H1 expected direction: ______
- H2 expected direction: ______
- H3 expected direction: ______

## 2. Conditions

| Condition | Config | Training representation | Order |
|---|---|---|---|
| A — FULL | `configs/train_full.yaml` | problem → complete rationale + answer | shuffled problems |
| B — ordered transitions | `configs/train_split_ordered.yaml` | (problem, state) → transition | trajectory order |
| C — unordered transitions | `configs/train_split_stateful.yaml` | (problem, state) → transition | globally shuffled |

Condition B is not optional. Without it, an A-vs-C difference could come from any
of: removing the full rationale, changing the target to local transitions,
compacting the context, or destroying chronological order.

Additional planned ablations (mark which will run):

- [ ] next-step-only (`train_split_next_step.yaml`)
- [ ] reverse order (`train_split_reverse.yaml`)
- [ ] curriculum order (`train_split_curriculum.yaml`)
- [ ] base checkpoint with no adapter (`eval_base_model.yaml`)

## 3. Fixed controls

Declare these once; they must be identical across all conditions.

```
base checkpoint + revision : ../qwen3-0.6b  (official Qwen/Qwen3-0.6B, Qwen3ForCausalLM,
                             28 layers, hidden 1024, 16 heads / 8 KV, vocab 151936,
                             tied embeddings, bf16; record the exact commit sha here: ______)
tokenizer + chat template  : bundled Qwen3 tokenizer + its chat template
                             (template_source = tokenizer_chat_template;
                              chat_template_kwargs: {enable_thinking: false}, pinned
                              identically for both arms and the baseline)
LoRA r / alpha / dropout   : 16 / 32 / 0.05, bias none
target modules             : q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
                             (6,848,512 trainable of 461,319,936 = 1.48%)
optimizer / scheduler / LR : AdamW / cosine / 1e-4
warmup policy              : warmup_ratio 0.03
precision / quantization   : auto (bf16 on CUDA); quantization none;
                             trainable LoRA parameters kept in fp32
max_seq_len                : 384  (truncated_targets measured 0 in both arms)
gradient clipping          : max_grad_norm 1.0
checkpoint selection rule  : best_dev_loss  (lowest DEV target NLL, eval_steps 100,
                             512-record DEV probe, identical rule in both arms)
decoding parameters        : greedy (do_sample false), max_new_tokens 320 (answers)
                             / 224 (transitions), all declared stop ids passed
seed list                  : ______  (>= 3 seeds required; fill before the run)
```

## 4. Budget regime

A single comparison cannot match rows, problems, tokens and updates
simultaneously. Declare the **primary** regime and which others you will report.

- [x] **T — matched target tokens (`configs/budgets/matched_target_tokens.yaml`) — PRIMARY**
- [x] P — matched source-problem exposure (`configs/budgets/matched_problems.yaml`) — confirmatory
- [x] N — native epochs (`configs/budgets/native.yaml`) — context only
- [ ] U — matched optimizer updates (`configs/budgets/matched_updates.yaml`) — optional

**Primary regime: T (matched target tokens).**

*Justification.* The hypothesis is a capability claim, and the dominant confound is
supervision quantity: measured on this checkpoint's tokenizer, one native epoch
gives SPLIT **2.62×** FULL's target tokens and **4.93×** its total tokens. Matching
supervised tokens is the tightest available control on that confound, and at
FULL's token budget SPLIT still touches roughly 85% of the source problems, so the
control does not gut coverage. Regime P is reported as confirmatory because it
holds the *curriculum* constant instead (identical 43,200 problems per epoch),
which is the purest form of the state-sufficiency question. If T and P agree the
conclusion is robust; **if they disagree, that disagreement is the finding and is
reported as such.** Regime N is reported only as context, never as the headline.

Measured values (`rineya-lab budget --sample 4000`, checkpoint `../qwen3-0.6b`,
`enable_thinking: false`):

| Arm | rows | target tokens | total tokens | optimizer updates |
|---|---:|---:|---:|---:|
| FULL | 43,200 | 3,425,404 | 5,530,421 | 1,350 |
| SPLIT stateful | 170,941 | 9,028,676 | 27,192,653 | 5,341 |

Ratio SPLIT / FULL: **2.64×** target tokens, **4.92×** total tokens.
Matched target-token budget applied to **both** arms: **3,425,404**.
`truncated_targets` measured as **0** in both arms at `max_seq_len: 384`.

## 5. Primary endpoint

- Metric: strict final-answer accuracy on the shared held-out TEST problems.
- FULL is evaluated with `full_direct`; SPLIT with `generated_rollout` — each arm
  in the inference mode it was trained for.
- Pairing unit: `problem_id`. Never the expanded transition row.
- Test: paired cluster bootstrap CI (primary) + exact McNemar + sign-flip
  randomization (supporting).

## 6. Non-inferiority margin — FREEZE THIS

The shipped 2.0 pp is a **placeholder**, not a recommendation.

```
non_inferiority_margin_pp : ______
equivalence_margin_pp     : ______
```

Justification (required — what accuracy loss would change a practical decision
about adopting sequence-free compact-state training, and why?):

> ______

Then set in your comparison config:

```yaml
comparison:
  non_inferiority_margin_pp: <your value>
  margin_frozen: true
  margin_justification: "<your justification>"
```

## 7. Secondary endpoints

1. category-macro accuracy
2. oracle-state next-step accuracy
3. free-rollout final accuracy
4. free-rollout gap (oracle transition − free rollout)
5. accuracy by trajectory length
6. accuracy by step index
7. robustness to omitted history (`no_state_transition`, `last_state_only`)
8. robustness to corrupted state (`shuffled_state`, `distractor_state`)
9. strict-vs-lenient disagreement
10. failure-type distribution

## 8. Multiplicity

Stratified tests are corrected with Holm-Bonferroni within each stratification
family (`category`, `total_steps`, `failure_type`). The primary endpoint is a
single test and is not corrected.

## 8b. Learnability floor (declare BEFORE looking)

The untouched checkpoint scores ~0 on this task (dress rehearsal,
`docs/READINESS_ASSESSMENT.md` 7b). If the FULL arm also lands near zero, an
A-vs-C null result is uninterpretable.

```
minimum FULL full_direct strict accuracy on DEV : ______ %   (suggest 25)
minimum improvement over the base-model baseline: ______ pp  (suggest 15)
```

If either floor is missed the experiment **stops**, and is reported as
"the checkpoint could not learn the task; the hypothesis was not tested."
A floor-effect null must never be reinterpreted as support for the hypothesis.

## 9. Stopping and exclusion rules

- No item is excluded for being unparseable; it scores as a failure.
- Runs are excluded only for infrastructure failure (`backend_failure` > ___% of
  items, or a crashed training run). State the threshold: ______
- DEV is used for every tuning decision. TEST is opened exactly once, after this
  document is committed and `margin_frozen: true`.

## 10. Deviations

Record any departure from the above, with date and reason.

| Date | Deviation | Reason |
|---|---|---|
| | | |
