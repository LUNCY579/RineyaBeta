# DAST Phase-0 defect audit

**Branch:** `research/depth-aligned-semantic-transitions-v1` at `a343e1f`  
**Method:** seven read-only subsystem audits against the committed code, each required to
cite `path:line`. Produced 2026-08-31.

## Verification status - READ THIS FIRST

The audit ran as a two-stage workflow: audit, then one adversarial verifier per claimed
defect whose job was to *refute* it. **All seven audit agents completed; 55 of the
verifiers died on an API session limit.** So apart from the rows explicitly marked
**CONFIRMED by lead** or **PARTIALLY REFUTED by lead**, every entry below is
**UNVERIFIED** - a single-agent claim with a citation, not an established defect.

Per `EVIDENCE_STANDARD.md` an unverified claim is not evidence. Treat this file as a
triage queue: verify before acting, and do not quote a row as a known defect.

Nothing here is a model-quality claim. Program status remains **BUILT - UNRUN**.

---

## Dimension: `interventions`

### Scope-contract verdicts

| Verdict | Scope item |
|---|---|
| `IMPLEMENTED_BUT_INEFFECTIVE` | activation-patching interventions that affect downstream computation |

### Additional defects claimed (12)

**1. [BLOCKER] Zero-initialized feedback decoder makes every activation patch a bitwise no-op under the shipped probe config; the smoke test hides this by overwriting the init with values that exist in no config.**

> CONFIRMED by lead. model.py:143 `nn.init.zeros_(feedback_decoder.weight)` makes `delta = decoded * scale` identically zero, and smoke.py:_feedback_qualification overwrites BOTH the decoder (`normal_(std=0.02)`) and the gates (`fill_(2.0)`), plus raises feedback_scale to 0.2, before measuring. The PROGRESS.md 'non-zero logit delta' was therefore measured on a hand-modified model no training run produces.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:142-143 (`nn.Linear(..., bias=False)` then `nn.init.zeros_`) -> decoded is identically 0 -> delta is identically 0. MEASURED with configs/feedback_causal_probe.yaml:6-7 init (scale 0.05, gates -3.0): max|dlogits| = 0.0 and torch.equal(clean.logits, intervened.logits) is True for negate_state, zero_state, replace_with(ones), and all-four-stages-at-once. Same model with src/dast/smoke.py:132-134 + smoke.py:122 amplification: max|dlogits| = 0.628. The PASS recorded at PROGRESS.md:30 and asserted at smoke.py:209-210 is produced only by that amplification.
- Proposed minimal patch: Either (a) initialize feedback_decoder.weight with a small nonzero std (e.g. 1e-3) so the causal path is live at t=0, or (b) make _feedback_qualification build the model from the real config's feedback_scale and NOT reinitialize, and add a separate test asserting that the shipped init produces a nonzero logit delta. Additionally add a runtime guard in the hook that raises when feedback_decoder.weight is all-zero AND an intervention is active, so a probe can never silently report a null causal effect.

**2. [BLOCKER] No executable path applies any intervention to a real model: neither training nor teacher-forced evaluation nor the CLI passes interventions=, so the config named feedback_causal_probe.yaml produces zero causal measurements.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/training.py:357-362 and training.py:232-237 call model(...) without the `interventions` keyword. src/dast/cli.py:115-146 registers only validate-data, inspect-alignment, train, smoke, compare. scripts/run_dev_matrix.sh:7-10 does not include feedback_causal_probe. The only two call sites that ever pass interventions are src/dast/smoke.py:155 (toy backbone) and tests/test_model_feedback.py:62 / :69. `grep -rn 'direction|counterfactual|wrong.stage|norm.matched|preservation' src/` returns nothing, so none of docs/RUNBOOK.md:163-168, docs/PREREGISTRATION.md:113-116 (G6), or docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md:298-306 has an implementation.
- Proposed minimal patch: Add a `dast causal-probe --config configs/feedback_causal_probe.yaml` subcommand that runs the five RUNBOOK controls over the eval loader with a matched-pair sampler, and writes per-example (clean answer, intervened answer, expected direction, control arm) rows plus a directionality rate to a causal_probe.jsonl artifact. Until it exists, mark feedback_causal_probe.yaml as non-executable in docs/RUNBOOK.md section 9.

**3. [VALIDITY] The mandatory final checkpoint (fraction 1.0) injects into the output of the last decoder layer, so its patch changes logits at only one sequence position and can never alter any generated answer token.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/config.py:215-216 forces the last checkpoint fraction to be exactly 1.0, so every config (configs/base.yaml:20) has a stage on the final decoder layer. src/dast/model.py:156 hooks `layers[layer_number - 1]`, i.e. the last decoder layer for that stage; there are no subsequent layers to propagate the injection through attention. MEASURED on the toy 8-layer backbone with an amplified decoder: patching stage 3 changed logits at exactly positions [26] and [21] — the two rows' own last_prompt indices — versus 29 and 34 changed positions when patching stage 1.
- Proposed minimal patch: Document in docs/ALGORITHM.md section 8 that the final-stage feedback influences only the first predicted response token, and exclude the last stage from the causal-directionality probe (or add a config option to place the last feedback site at fraction < 1.0 while keeping the readout at 1.0).

**4. [VALIDITY] The feedback gate is initialized at tanh saturation, not "near zero": |tanh(-3)| = 0.995, and its gradient is exactly 0 at init and ~100x attenuated afterwards, so the scale is pinned at 0.995*feedback_scale for the whole run.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:146 sets gates to -3.0; the comment at model.py:144-145 claims "tanh(-3) starts near zero but not exactly zero", and docs/ALGORITHM.md:143 states "The gate starts near zero". MEASURED: effective scale at stage 0 = -0.04975 = 0.05 * -0.9951, i.e. 99.5% of the maximum magnitude, not near zero. MEASURED: after a backward pass at the shipped init, feedback_gates.grad = [0.0, 0.0, 0.0, 0.0] (because dL/dgate is proportional to the zero-valued `decoded`), and tanh'(-3) = 0.00987 thereafter. With 250 optimizer steps (base.yaml:29 / :66 / :68 / :65) at aux lr 2e-4 (base.yaml:70), the gate can move at most 0.05, from -3.0 to -2.95.
- Proposed minimal patch: Initialize feedback_gates to a genuinely small pre-activation (e.g. 0.0 gives tanh=0 exactly, or use a sigmoid-scaled gate initialized at logit(1e-2)) and fix the model.py:144-145 comment and docs/ALGORITHM.md:143 to state the actual initial gate value.

**5. [VALIDITY] An intervention keyed by an out-of-range stage or by a layer number instead of a stage index is silently ignored with no error, so a mis-specified causal probe reports a null effect indistinguishable from a real null.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:186 does `self._active_interventions.get(stage)` and model.py:187-188 returns the unmodified state when the lookup misses; nothing validates that every supplied key was consumed. MEASURED on an amplified model: `interventions={4: negate_state}` -> max|dlogits| = 0.0, no exception; `interventions={14: negate_state}` (a plausible layer-number typo, since checkpoint_layers for Qwen3-0.6B are 7/14/21/28) -> max|dlogits| = 0.0, no exception. By contrast model.py:245-246 does validate the feedback_enabled precondition, so the validation is half-done.
- Proposed minimal patch: In DASTModel.forward, after the base_model call, assert set(interventions) is a subset of range(self.num_stages) and raise ValueError naming the unmatched keys; ideally track consumed keys inside the hook and raise if any key never fired.

**6. [VALIDITY] Under the shipped precision:auto (bf16) path, the largest injectable delta after one epoch is below half a bf16 ulp for residual components with |h| >= 8, so the patch is expected to be rounded away for most of the residual stream.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:172 casts delta to `hidden.dtype` and model.py:180 adds it to the bf16 residual; configs/base.yaml:15 is `precision: auto`, and src/dast/hf.py:19-22 resolves that to bf16 on CUDA (hf.py:142-148 promotes only trainable params, not the frozen bf16 base). Derived bound: 250 optimizer steps (base.yaml:29/:66/:68/:65) x AdamW per-step magnitude <= lr 2e-4 (base.yaml:70) gives |dW_ij| <= 0.05; row norm <= 0.05*sqrt(128) = 0.566 (state_dim, base.yaml:36); |Wz| <= 0.566 since z is unit-norm (src/dast/semantic.py:28); |scale| = 0.0498 (gate pinned at -3, measured); so |delta_i| <= 0.0282. MEASURED bf16 spacing: half-ulp is 0.00781 at |h|=3.06, 0.03125 at |h|=8, 0.0625 at |h|=16, 0.125 at |h|=49. F6 measured elementwise RMS 49 at layers 7/14/21. Cosine LR decay (src/dast/training.py:186) roughly halves the bound again.
- Proposed minimal patch: Perform the injection in fp32: compute `modified = hidden.float()`, add delta in fp32, then cast back to hidden.dtype; or raise feedback_scale / open the gate so the delta clears the bf16 quantum. Add an assertion in the causal probe that `(modified != hidden).any()` at the injection row, so a rounded-away injection is detected rather than reported as a null causal effect.

**7. [VALIDITY] The counterfactual-swap control is an identity no-op at the shipped batch size, is not validated to be non-identity, and cannot express the required matched-problem pairing.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/interventions.py:16-25 permutes states along dim 0 of the current batch. configs/feedback_causal_probe.yaml:1 extends state_ordered.yaml which extends base.yaml, and no config in the chain overrides configs/base.yaml:66 `batch_size: 1`. MEASURED: swap_batch_states(torch.tensor([0])) on a [1, 12] state returns a tensor bitwise equal to its input — the only legal permutation of length 1 is the identity. interventions.py:21-22 validates only ndim and numel, never that the permutation moves anything. The batch composition is also uncontrolled and unrecorded (base.yaml:80 `dataloader_shuffle: true`; the collator at src/dast/tokenization.py:190 keeps problem_ids but nothing pairs them by differing conclusion), so docs/RUNBOOK.md:164 "correct counterfactual state swap" and docs/EVIDENCE_STANDARD.md:64-65 "a state from a matched problem whose correct conclusion is known to differ" are not constructible with this helper.
- Proposed minimal patch: Reject an identity permutation in swap_batch_states (raise when torch.equal(order, arange(n))), and add a matched-pair batch sampler plus a `swap_matched_problems(pair_index)` intervention that records the (source_problem_id, target_problem_id, expected_answer_change) triple into the run artifacts.

**8. [VALIDITY] The feedback hooks fail open: any call that bypasses DASTModel.forward silently runs the model with feedback disabled instead of refusing, so a feedback-trained model can be evaluated as if the causal path were absent.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:158-160 returns the layer output unchanged whenever `_active_prompt_mask is None`, and model.py:183 registers the handles permanently on base_model (only removable via the never-called model.py:201-204). MEASURED: on an amplified feedback model, `model.base_model(input_ids=..., attention_mask=..., labels=...)` diverges from DASTModel.forward by max|dlogits| = 1.152 with all 4 handles still registered. The "cached generation is blocked" contract of docs/ALGORITHM.md:146 and configs/feedback_causal_probe.yaml:10-11 is enforced at exactly one call site, src/dast/evaluation.py:112-116; and because feedback_causal_probe.yaml:14 sets do_generation false, src/dast/training.py:435 never even reaches that guard.
- Proposed minimal patch: Make the hook fail closed: raise RuntimeError when `_active_prompt_mask is None` and the module is in a feedback-enabled state, with an explicit `model.feedback_disabled()` context manager for the intentionally-unhooked case. Also assert in evaluate_generation that no handles remain registered rather than relying on the config flag.

**9. [VALIDITY] The feedback hook and the auxiliary readout sample two different tensors at the final checkpoint: the hook sees the raw pre-final-RMSNorm residual while the auxiliary branch sees the post-norm state, so the shared projector is fed inputs ~16x apart in magnitude depending on feedback_enabled.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:263 reads `hidden_states[layer_number]`, and for layer_number == num_hidden_layers Hugging Face returns the post-final-norm state (F6 measured lm_head(h[28]) reproducing logits to 0.0, elementwise RMS 3.06). src/dast/model.py:156 instead hooks `layers[layer_number - 1]`, whose output is the pre-norm residual (F6 measured RMS 49 / 49 / 49.5 at layers 7/14/21). src/dast/alignment.py:26-28 documents only the hidden_states indexing convention and does not note that the hook uses a different tensor at the last stage. Consequence: a state_projector trained in auxiliary mode (configs/base.yaml:22 feedback_enabled false) and then reused under configs/feedback_causal_probe.yaml:6 receives a final-stage input ~16x larger in norm than it was trained on, and DASTOutput.states is not comparable across the two branches.
- Proposed minimal patch: In _install_feedback_hooks, when layer_number == num_hidden_layers, hook the final norm module (base_model.model.norm) instead of the last decoder layer, or apply the base model's final norm to the hooked hidden before pooling — and inject the delta on the pre-norm residual either way. Add a test asserting states from the feedback branch match the auxiliary branch to within tolerance when feedback_scale == 0.

**10. [HYGIENE] With device_map set, the feedback path adds a delta living on the projector device to a residual living on a shard device, so feedback_enabled plus device_map crashes; nothing in validate() blocks the combination.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:163-164 computes raw_state on `projector_device`, model.py:170 decodes there, and model.py:172 casts delta to `hidden.dtype` only — there is no `.to(hidden.device)`. model.py:180 then adds it to `modified`, which lives on hidden.device. src/dast/training.py:294-303 is precisely the branch that creates this layout: it explicitly moves state_projector, feedback_decoder and feedback_gates to the input device while leaving base shards elsewhere, and the comment at training.py:295-296 claims "Hidden states from other shards are copied to this small head explicitly in DASTModel" — the copy of the result back to the shard is missing. src/dast/config.py:217-220 validates only gradient_checkpointing and feedback_scale, never device_map. configs/base.yaml:17 sets device_map: null, so the shipped config does not trigger it.
- Proposed minimal patch: Change model.py:172 to `delta = decoded.to(device=hidden.device, dtype=hidden.dtype) * scale.to(device=hidden.device, dtype=hidden.dtype)`, and add a config.validate() error for feedback_enabled together with a non-null device_map until that path is tested.

**11. [HYGIENE] The unconditional re-normalization of intervention outputs silently cancels any magnitude-scaling intervention, and its correctness depends on an undocumented, untested invariant that the projector already emits unit vectors.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: src/dast/model.py:199 returns `F.normalize(result, dim=-1)` for every intervention, while the no-intervention path at model.py:187-188 returns the state unchanged. MEASURED: normalize(0.5*s) != 0.5*s, so an intervention scaling a state by 0.5 is a no-op; normalize(-s) == -s and normalize(zeros) == zeros only because src/dast/semantic.py:28 already normalizes the projector output. No test pins that invariant; if the normalize in SharedStateProjector.forward were removed, negate_state would silently become -s/||s|| compared against an unnormalized clean state, confounding negation with renormalization.
- Proposed minimal patch: Assert inside _apply_intervention that the incoming state is unit-norm (or drop the re-normalize and require interventions to preserve norm themselves), and document in docs/ALGORITHM.md section 8 that the intervention family is direction-only so magnitude sensitivity cannot be probed.

**12. [HYGIENE] With mean prompt pooling, a stage-k injection reaches the stage-(k+1) state readout diluted by roughly 1/num_prompt_tokens, so the intended cross-depth semantic recurrence is negligible at the shipped 512-token prompt budget.**

> Status: UNVERIFIED.

- Breaks: 1 (activation-patching interventions that affect downstream computation)
- Evidence: configs/base.yaml:21 sets `pooling: mean`; src/dast/pooling.py:20-22 averages over all prompt positions. src/dast/model.py:180 modifies exactly one position (the last prompt token). Hooks fire in increasing layer order because src/dast/alignment.py:36-37 forces checkpoint_layers to be strictly increasing, so stage k+1's pooled read does see stage k's injection — but the direct contribution is delta/num_prompt_tokens, i.e. ~1/512 at configs/base.yaml:19 max_length 512, on top of the bf16 rounding bound above.
- Proposed minimal patch: Use pooling: last_prompt for the feedback probe config (src/dast/pooling.py:23-27 already supports it), which makes the stage-k injection enter the stage-(k+1) read at full weight, and record the pooling mode in the causal-probe artifact.

### Challenges to the lead's own findings (4)

| Claim | Position | Reasoning |
|---|---|---|
| F1 | `CONFIRMED` | Confirmed on the code path, with an interventions-specific consequence the claim does not state: because apply_chat_template_ids is the only tokenization entry point for real models, the feedback/intervention hook has never been executed against a real tokenizer or a real model at all. src/dast/tokenization.py:50-53 hands the result of apply_chat_template(tokenize=True) straight to _as_int_list; for a dict, `hasattr( |
| F6 | `CONFIRMED` | Confirmed and extendable in a way that matters for interventions: the two branches of DASTModel read DIFFERENT tensors at the final stage, so the representation-class split F6 identified is not merely a projector-input heterogeneity problem, it is also a feedback-vs-auxiliary inconsistency. The auxiliary branch reads hidden_states[28] (post-final-RMSNorm, RMS 3.06 per F6) while the feedback hook reads the output of d |
| F7 | `CONFIRMED` | The premise (configs/base.yaml:66 batch_size: 1) is confirmed and it damages the interventions dimension too, which F7 does not mention. configs/feedback_causal_probe.yaml extends state_ordered.yaml -> base.yaml and overrides only gradient_checkpointing (line 9) and do_generation (line 14), so batch_size stays 1. swap_batch_states permutes along dim 0, and the only legal permutation of length 1 is the identity — MEAS |
| F2 | `CONFIRMED` | Confirmed and worth narrowing: the CPU-snapshot-vs-CUDA-params comparison is the projector-updated check, not the feedback check, so the false PASS of the intervention claim is device-independent. src/dast/smoke.py:180-182 snapshots and smoke.py:189-192 compares projector params of the model that train() moved to CUDA (src/dast/training.py:286-293). _feedback_qualification builds a separate model at smoke.py:130 via  |

---

## Dimension: `operators-and-losses`

### Scope-contract verdicts

| Verdict | Scope item |
|---|---|
| `PARTIAL` | low-rank reusable semantic transformation operators |
| `PARTIAL` | state, transition, final-answer, and operator-balance losses (overall) |
| `IMPLEMENTED` |   2a. state loss |
| `IMPLEMENTED` |   2b. transition loss |
| `ABSENT` |   2c. final-answer loss |
| `IMPLEMENTED_BUT_INEFFECTIVE` |   2d. operator-balance loss |
| `IMPLEMENTED_BUT_INEFFECTIVE` |   extra term: contrastive anti-collapse (not in contract; documented in ALGORITHM section 6) |
| `IMPLEMENTED` |   extra term: direct-state consistency with stop-gradient (ALGORITHM section 7) |
| `IMPLEMENTED_BUT_INEFFECTIVE` |   extra term: operator diversity (ALGORITHM section 7) |

### Additional defects claimed (10)

**1. [VALIDITY] diversity_loss penalizes the gram DIAGONAL as well as the off-diagonal, contradicting ALGORITHM.md section 7 ('squared off-diagonal cosine similarity of flattened operators'); the variable is even named `off_diagonal` but is `gram - I` over all entries.**

> Status: UNVERIFIED.

- Breaks: state, transition, final-answer, and operator-balance losses (the operator-diversity term, and the loss/operator_diversity metric reported for it)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/operators.py:60-65: `off_diagonal = gram - torch.eye(...)` then `.square().sum()` over ALL entries. Doc: experiments/depth_aligned_semantic_transitions/docs/ALGORITHM.md:7 bullet 3. MEASURED on a freshly constructed bank (D=128, M=8, R=8, i.e. right==0 per operators.py:41): shipped diversity_loss = 0.1428571 while the documented off-diagonal-only quantity = 0.0; gram diagonal is all zeros because F.normalize(0, eps=1e-8) = 0. Backward at that point gives grad|max| 0.0 on both left and right.
- Proposed minimal patch: In operators.py:64, mask the diagonal instead of subtracting the identity: `off_diagonal = gram.masked_fill(torch.eye(self.num_operators, dtype=torch.bool, device=gram.device), 0.0)`; keep the /(M*(M-1)) divisor.

**2. [VALIDITY] multi_positive_contrastive_loss returns a DETACHED constant (not merely a zero) when batch < 2, and every shipped config runs batch_size 1, so the documented anti-collapse term is inert in all eight arms; the only path with batch_size >= 2 is the toy smoke, so the test suite structurally cannot catch it.**

> Status: UNVERIFIED.

- Breaks: state, transition, final-answer, and operator-balance losses (the anti-collapse term ALGORITHM.md section 6 relies on, and the reported loss/contrastive metric)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/losses.py:70-72 returns `predictions.new_zeros(())`. MEASURED at batch 1: value 0.0, requires_grad False, grad_fn None, and `.backward()` raises 'element 0 of tensors does not require grad'; at batch 2 the same inputs give 1.0217. configs/base.yaml:66 sets batch_size 1 and `grep -rn batch_size configs/*.yaml` shows NO config overrides it (only eval_batch_size 2 at base.yaml:67 and generation_batch_size 4 at :86). experiments/depth_aligned_semantic_transitions/src/dast/smoke.py:82 sets batch_size 2. Consequences: (i) losses.py:120-125 adds a constant 0.10*0 for every training step in every semantic arm, so metrics.jsonl loss/contrastive is always exactly 0.0 (training.py:395-404); (ii) because evaluate_teacher_forced runs at eval_batch_size 2, dev_summary.json REPORTS a nonzero loss/contrastive (training.py:238-246) for a quantity that was never optimized; (iii) no other mechanism guards collapse - depth_alignment_metrics is strictly within-example (losses.py:170, einsum 'bkd,bjd->bkj' pairs each z only with its own targets), so cross-problem state collapse is invisible to every logged diagnostic.
- Proposed minimal patch: Two options: (a) set training.batch_size >= 2 in configs/base.yaml:66 and reduce gradient_accumulation_steps to keep the update budget matched across arms; or (b) accumulate states/targets across the accumulation window and compute the contrastive term once per optimizer step. Independently, drop loss/contrastive from dev_summary.json or label it 'not-optimized' so the reported number is not read as a trained objective.

**3. [VALIDITY] Given F6, the third of the three transitions the single shared operator bank must model crosses a representation-class boundary (raw residual -> post-final-RMSNorm) that the other two do not, so 'one reusable dictionary' is fitted to a heterogeneous transition set.**

> Status: UNVERIFIED.

- Breaks: low-rank reusable semantic transformation operators (shared/reusable across depths)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/alignment.py:23-44 maps fraction 1.0 to layer == num_hidden_layers, i.e. the LAST entry of hidden_states, which per F6 is post-final-RMSNorm (elementwise RMS 3.06) while indices 7/14/21 are raw residuals (RMS 49-49.5); configs/base.yaml:20 sets checkpoint_fractions [0.25, 0.50, 0.75, 1.0]; src/dast/model.py:271-272 applies ONE bank and ONE router to all K-1 transitions with no depth conditioning; the shared projector (src/dast/semantic.py:12-28) is likewise depth-agnostic. Result: transitions z1->z2 and z2->z3 are within-class while z3->z4 crosses classes, and the router must absorb that with 8 atoms and no depth feature.
- Proposed minimal patch: Either read the pre-norm residual for the final checkpoint (hook the last decoder layer output rather than hidden_states[-1]) so all four checkpoints are the same representation class, or report per-transition transition-cosine and per-transition router load so the z3->z4 anomaly is visible rather than pooled into one scalar.

**4. [POWER] The router receives EXACTLY ZERO gradient from the transition and consistency losses at initialization, because B is zero-initialized and the router only reaches the loss through the operator delta; the only signal it gets on step 0 is the 1e-3-weighted balance term.**

> Status: UNVERIFIED.

- Breaks: low-rank reusable semantic transformation operators (the router that selects which reusable operator applies)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/operators.py:41 (`self.right = nn.Parameter(torch.zeros(...))`) and :55-58 (delta = einsum(weights, low @ right), so d(loss)/d(weights) = 0 whenever right == 0). MEASURED on the shipped geometry with 0.50*transition backward at init: grad|max| right = 1.5e-03, left = 0.0, router.weight = 0.0, router.bias = 0.0. After right is perturbed to N(0, 0.05): router.weight grad|max| = 1.2e-04. Same zero-gradient result for `left`. The competing signal from 0.001*balance gives router.weight grad|max| 1.5e-06.
- Proposed minimal patch: Initialize `right` with a small nonzero std (e.g. `scale/math.sqrt(rank)` as operators.py:45 already does for the frozen_random branch) so the router and A factors are live from step 0, or accept a one-step lag explicitly and document it in ALGORITHM.md section 7. If exact identity-at-init must be preserved, add a learned per-operator output gate initialized near zero so the router still gets gradient.

**5. [POWER] Because the bank is the exact identity at init, the transition term is numerically identical to an off-by-one state loss, so 0.50/1.00 of the depth-alignment gradient on stages 1..K-1 initially pulls z_k toward the WRONG (next-stage) target. This is undocumented.**

> Status: UNVERIFIED.

- Breaks: state, transition, final-answer, and operator-balance losses (the transition term, and its interaction with the state term)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/operators.py:41 (right = 0) and src/dast/losses.py:131 (target = training_targets[:, 1:]). MEASURED: identity-at-init max deviation 2.98e-08; at init, cosine_alignment_loss(bank(z)[0], targets[:,1:]) = 0.9207251667976379 and cosine_alignment_loss(z[:,:-1], targets[:,1:]) = 0.9207251667976379, equal to < 1e-6. Weights: configs/base.yaml:40 state_weight 1.0 vs configs/operator_ordered.yaml:8 transition_weight 0.50, and the transition gradient reaches the shared projector through `states` (measured states grad|max| 4.3e-02 from 0.50*transition at init).
- Proposed minimal patch: Warm up operators.transition_weight from 0 over the first few hundred optimizer steps, or gate the transition loss behind a minimum ||B|| so it only applies once the bank is non-trivial. At minimum, document in ALGORITHM.md section 7 that the transition term degenerates to an off-by-one state loss while the bank is near-identity.

**6. [POWER] repeated_stage_mask is computed and then dropped entirely - it never reaches the losses, the collator, the CLI inspector, or the run manifest - so identity-target transitions are neither masked nor recorded, contradicting ALGORITHM.md section 4's claim that repeats 'are explicit and recorded'.**

> Status: UNVERIFIED.

- Breaks: low-rank reusable semantic transformation operators (the amount of non-degenerate signal the reusable bank actually receives) and the transition loss
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/alignment.py:109,116 computes and stores it on AlignedTrajectory; `grep -rn repeated_stage_mask src/ tests/` shows the ONLY other reference is tests/test_alignment.py:30. It is absent from TokenizedExample (src/dast/tokenization.py:20-32), absent from the collator output dict (:183-203), absent from `inspect-alignment` output (src/dast/cli.py:42-55), and absent from build_run_manifest (src/dast/artifacts.py:67-98). Doc: docs/ALGORITHM.md:4. Consequence for the operator arms: when training_source_indices[k] == training_source_indices[k+1] the transition target equals the current target, so the loss-optimal operator is the identity and the term pulls B back toward zero. MEASURED on the shipped fixture (fixtures/tiny_reasoning.jsonl, 13 records): 11/39 = 28.2% of operator transitions have identity targets (every N=3 record maps to source indices (0,1,1,2)). Structural: _progress_indices(N, 4) gives 3/3 identity transitions at N=1, 2/3 at N=2, 1/3 at N=3, 0/3 at N>=4 (src/dast/alignment.py:47-57). Directional check on a repeated slot: <grad_right, right> = +2.6e-03, i.e. gradient descent shrinks ||B|| toward the identity.
- Proposed minimal patch: Carry repeated_stage_mask through TokenizedExample/the collator, log its per-arm rate into run_manifest.json, and either mask identity-target pairs out of the transition loss or report the effective non-repeated transition count alongside loss/transition so the operator signal strength is auditable.

**7. [HYGIENE] tests/test_operators_losses.py never calls compute_loss and asserts none of the properties that actually failed: no test for the batch<2 contrastive short-circuit, no test for the consistency stop-gradient, no test that balance_loss is 0 at uniform load and maximal at collapse, no test that diversity decreases for orthogonal versus identical banks, and no rank test.**

> Status: UNVERIFIED.

- Breaks: both scope items (the operators and every loss term are effectively untested at the compute_loss level)
- Evidence: experiments/depth_aligned_semantic_targets is not involved; see experiments/depth_aligned_semantic_transitions/tests/test_operators_losses.py:1-56. `grep -rn 'compute_loss|detach|balance|matrix_rank' tests/` matches only tests/test_operators_losses.py:14 (`bank.diversity_loss() + bank.balance_loss(weights)` summed purely for a finiteness/backward check) - compute_loss and LossBreakdown are never exercised by any test in the suite. tests/test_operators_losses.py:34-39 (test_contrastive_handles_duplicate_targets_as_multi_positive) only asserts torch.isfinite(loss) and never compares against the eye-positive baseline, so it would still pass if multi-positive grouping were removed entirely.
- Proposed minimal patch: Add: (1) `assert multi_positive_contrastive_loss(p[:1], t[:1], group_ids=g[:1], temperature=0.07) == 0` with an xfail/regression marker; (2) a stop-gradient test asserting states.grad[:, -1] is exactly zero for the consistency term; (3) balance_loss(uniform) == 0 and balance_loss(one-hot collapse) > balance_loss(spread); (4) diversity_loss(orthogonal bank) < diversity_loss(identical bank); (5) matrix_rank(A_j B_j) <= rank; (6) one compute_loss test asserting each configured weight appears in total.

**8. [HYGIENE] operators.frozen_random freezes the ROUTER along with A and B, so the intended random-frozen-operator control cannot route; nothing validates it against nonzero diversity/balance weights, and no config uses it.**

> Status: UNVERIFIED.

- Breaks: low-rank reusable semantic transformation operators (the frozen-random matched-capacity control)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/operators.py:44-47 (`for parameter in self.parameters(): parameter.requires_grad_(False)` - MEASURED requires_grad False for left, right, router.weight, router.bias); plumbed at src/dast/cli.py:70 but NOT at src/dast/smoke.py:108-113 (the smoke path silently ignores it); configs/base.yaml:54 sets it false and no other config sets it; src/dast/config.py:243-249 validates num_operators/rank/router_temperature under operators.enabled but never frozen_random, so frozen_random with diversity_weight > 0 would add a non-optimizable constant to the total loss.
- Proposed minimal patch: Freeze only `left` and `right` in operators.py:46-47 (leave self.router trainable) so the control is 'random fixed atoms, learned routing'; add a config.validate() rule rejecting frozen_random together with nonzero diversity_weight; and plumb frozen_random through smoke.py:108-113.

**9. [HYGIENE] AdamW weight decay 0.01 is applied to every trainable tensor including the operator bank, the router bias, and the projector LayerNorm, adding a second pull of B toward zero (the identity operator) on top of the identity-target transitions.**

> Status: UNVERIFIED.

- Breaks: low-rank reusable semantic transformation operators (whether the learned atoms stay non-trivial)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/training.py:164-176 builds two param groups that set only `lr`; weight_decay is passed once globally at :176 with no per-group override and no no-decay list. configs/base.yaml:71 sets weight_decay 0.01. The auxiliary group (training.py:151-157) explicitly includes model.operator_bank, whose `right` starts at exactly zero (operators.py:41), and model.state_projector, whose nn.LayerNorm weight/bias (semantic.py:20) should normally be excluded from decay.
- Proposed minimal patch: Split the auxiliary group into decay / no-decay by parameter name (exclude *.bias and LayerNorm weights), and set weight_decay 0.0 for the operator bank so the diversity term (weight 0.001) is not competing with a decay term.

**10. [HYGIENE] The router has a trainable bias that the documented parameterization alpha_k = softmax(W_r z_k / tau) does not include.**

> Status: UNVERIFIED.

- Breaks: low-rank reusable semantic transformation operators (router specification)
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/operators.py:42-43 (`nn.Linear(state_dim, num_operators)` with only `nn.init.zeros_(self.router.bias)` - the bias exists and remains trainable); docs/ALGORITHM.md:7 and docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md:234-238 write the router as a pure linear map of z_k. MEASURED: router.bias appears in named_parameters with requires_grad True.
- Proposed minimal patch: Either `nn.Linear(state_dim, num_operators, bias=False)` at operators.py:42, or amend ALGORITHM.md section 7 to alpha_k = softmax((W_r z_k + b_r)/tau). A state-independent bias term lets the router acquire a fixed prior over atoms that is not a function of the semantic state, which weakens the 'operator selected by the state' reading.

### Challenges to the lead's own findings (2)

| Claim | Position | Reasoning |
|---|---|---|
| F7 | `CONFIRMED` | Confirmed and stronger than stated in three ways. (1) The function does not return 'exactly 0' as a differentiable zero - it returns `predictions.new_zeros(())`, a fresh leaf with requires_grad False and grad_fn None; calling .backward() on it raises. So the term is a hard constant, not a zero-valued expression. (2) The inertness is not limited to configs/base.yaml: `grep -rn batch_size configs/*.yaml` shows batch_si |
| F8 | `CONFIRMED` | The mechanism is confirmed, but two qualifications matter. (a) I could not verify the 42% figure from the shipped configuration: per F4 the corpus that configs/base.yaml:26 points at is a metadata-only manifest with no compact_state_trace, so the distinct-state distribution is not reachable from the shipped data path. On the shipped 13-record fixture the corresponding rate is 11/39 = 28.2% of transitions. The 42% num |

---

## Dimension: `governance`

### Scope-contract verdicts

| Verdict | Scope item |
|---|---|
| `IMPLEMENTED_BUT_INEFFECTIVE` | 1. DEV/TEST separation with a sealed-test unlock guard |
| `PARTIAL` | 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status |
| `CONTRADICTED_BY_CODE` | 3. explicit BUILT-UNRUN status until a qualifying real Rineya/Qwen checkpoint run exists |
| `PARTIAL` | 4. staged gates before any stop-gradient or independently trained block claim |

### Additional defects claimed (21)

**1. [BLOCKER] `dast inspect-alignment --split test` dumps sealed TEST record content (compact states including the final answer) with allow_sealed_test:false and exit code 0, because command_inspect_alignment never calls assert_split_allowed.**

> CONFIRMED by lead. cli.py:31 calls load_records with no assert_split_allowed, so `--split test` prints canonical_targets, which contain strings like 'Final answer: 246 bolts'. Sealed answers reach stdout with allow_sealed_test:false.

- Breaks: 1. DEV/TEST separation with a sealed-test unlock guard
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/cli.py:29-35 calls load_records(config.resolved_path(config.data.path), split=args.split or config.data.train_split, limit=args.limit) with no guard; cli.py:128 `inspect.add_argument("--split")` has no choices. Executed on fixtures/tiny_reasoning.jsonl (which contains one split="test" record, sealed_001) with a config carrying allow_sealed_test:false, preregistration_frozen:false, preregistration_sha256:"": output included canonical_targets/training_targets ["Known counts: white=10, black=2.","Total beads: 12.","Total beads: 12.","Final answer: 12."] and returned 0. docs/RUNBOOK.md:54-56 instructs the operator to run this exact command.
- Proposed minimal patch: In cli.py:command_inspect_alignment, insert `assert_split_allowed(config, args.split or config.data.train_split)` immediately after load_config, and import assert_split_allowed from dast.config.

**2. [VALIDITY] The sealed-test unlock never verifies the preregistration hash against any document: any 64-hex string opens TEST, so the "machine-enforced preregistration hash gate" is a format check.**

> CONFIRMED by lead. config.py:354-356 validates only the FORMAT of preregistration_sha256 and never compares it against any document.

- Breaks: 1. DEV/TEST separation with a sealed-test unlock guard
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/config.py:354-356 and config.py:264-265 only apply re.fullmatch(r'[0-9a-f]{64}'). Nothing in the repo hashes docs/PREREGISTRATION.md; artifacts.py:157 hash_text() has zero callers. Executed: preregistration_sha256='0'*64 with preregistration_frozen=True and allow_sealed_test=True passes validate() and assert_split_allowed(cfg,'test'). Contradicts docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md:449 ("machine-enforced preregistration hash gate"), docs/RUNBOOK.md:194,198, and README.md:47. tests/test_config.py:43-48 pins the weak behaviour with 'a'*64.
- Proposed minimal patch: Add a preregistration_path field to ExperimentConfig and, in assert_split_allowed, require sha256_file(config.resolved_path(preregistration_path)) == preregistration_sha256 and that the file's freeze block reads status: FROZEN; update tests/test_config.py:43-48 to use a real file digest.

**3. [VALIDITY] The toy smoke (and CI) emit a run_manifest.json whose machine-readable status is DEV_COMPLETE for a random 8-layer toy model, and CI publishes it as a downloadable artifact.**

> Status: UNVERIFIED.

- Breaks: 3. explicit BUILT-UNRUN status until a qualifying real Rineya/Qwen checkpoint run exists
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/smoke.py:183-188 calls dast.training.train with ToyCausalLM(vocab_size=512, hidden_size=48, num_hidden_layers=8) (smoke.py:102) and eval_split="dev" (smoke.py:56); training.py:333 writes status="RUNNING_DEV" and training.py:460 rewrites it to "DEV_COMPLETE". model_info={'kind':'offline_toy','quality_claim_allowed':False} (smoke.py:187) is stored but read by no code (repo-wide grep). .github/workflows/dast-ci.yml:45 runs `smoke --output-dir runs/ci-smoke`; lines 46-52 upload that directory as artifact dast-ci-smoke-report.
- Proposed minimal patch: Pass an explicit status through train() (e.g. train(..., status_prefix)) and have smoke.py request status="ENGINEERING_PASS_TOY"; alternatively make build_run_manifest refuse any status other than BUILT_UNRUN/ENGINEERING_PASS when model_info.get('quality_claim_allowed') is False.

**4. [VALIDITY] A run that TRAINS on the sealed TEST split is recorded as a DEV run: sealed_test_opened, status, and the summary filename are all derived from eval_split only, ignoring train_split.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/artifacts.py:75 `split = config.data.eval_split.lower()` and artifacts.py:89 `"sealed_test_opened": split == "test"`; training.py:333 and training.py:460 branch on config.data.eval_split only; training.py:455 picks the summary name from eval_split only. Executed with train_split='test', eval_split='dev' (both passing assert_split_allowed because allow_sealed_test was set): manifest sealed_test_opened=False, status would be RUNNING_DEV then DEV_COMPLETE, summary file dev_summary.json. Only the raw data.train_split string (artifacts.py:87) reveals the truth.
- Proposed minimal patch: Compute `opened = 'test' in {config.data.train_split.lower(), config.data.eval_split.lower()}` and use that single boolean for sealed_test_opened, the status branch, and the summary filename.

**5. [VALIDITY] Status labelling is case-sensitive while the sealing flag and summary filename are not, so eval_split: "TEST" produces a manifest that says DEV_COMPLETE next to sealed_test_opened: true and a test_summary.json.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/training.py:333 and 460 compare `config.data.eval_split != "test"` without .lower(), whereas training.py:455 uses .lower(), artifacts.py:75 uses .lower(), and data.py:148 filters on split.lower(). Executed with eval_split='TEST': sealed_test_opened=True, training.py:333 status='RUNNING_DEV', training.py:460 status='DEV_COMPLETE', summary='test_summary.json'.
- Proposed minimal patch: Use config.data.eval_split.lower() in the two status expressions at training.py:333 and training.py:460 (or normalise splits to lowercase once in DASTConfig.validate).

**6. [VALIDITY] The training path never validates the dataset and never counts sealed records in the file it hashes, so a concatenated train+dev+test file yields a manifest asserting sealed_test_opened: false.**

> Status: UNVERIFIED.

- Breaks: 1. DEV/TEST separation with a sealed-test unlock guard
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/training.py:66-78 calls load_records twice and never validate_dataset; data.py:139-153 load_records never calls record.validate(). validate_dataset is called only from cli.py:24, smoke.py:174 and tests. The real corpus ships data/full_context/{train,dev,test}.jsonl (Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1/data/manifests/sha256sums.txt lines for full_context) while DataConfig has a single `path` (config.py:78), so operators must concatenate files; docs/PHASE0_READINESS.md:133-140 recommends exactly that (option B4-a). Nothing detects or records that a concatenated file also contained TEST rows, and docs/RUNBOOK.md:100 ("no TEST record appears in a manifest") is therefore not machine-checkable.
- Proposed minimal patch: In build_dataloaders, run validate_dataset(data_path) once and copy its split_counts into the manifest as data.split_counts_in_file; raise if split_counts.get('test') and not config.experiment.allow_sealed_test.

**7. [VALIDITY] run_manifest.json contains no metric values and no checksum of the summary file — only a filename — so the manifest cannot bind a result to a run.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/training.py:461 `manifest["final_metrics_file"] = summary_name` followed by training.py:462 atomic_write_json; build_run_manifest (artifacts.py:67-98) has no metrics key. Executed build_run_manifest: top-level keys are arm, checkpoint_layers, claim_boundary, config, config_sha256, data, environment, experiment, model, schema_version, status. dev_summary.json (training.py:456) and generation_predictions.jsonl (evaluation.py:169) are never hashed: artifacts.py:150-153 writes checksums.json only inside checkpoint dirs. Contradicts README.md:48 ("deterministic manifests, dataset hashes, metrics, checkpoints") and docs/EVIDENCE_STANDARD.md:16-19 (evidence hierarchy items 1-2).
- Proposed minimal patch: After writing the summary, add manifest['final_metrics'] = final_metrics and manifest['artifact_sha256'] = {name: sha256_file(output_dir/name) for name in (summary_name,'generation_predictions.jsonl','metrics.jsonl') if (output_dir/name).exists()} before the final atomic_write_json.

**8. [VALIDITY] Artifact checksums are entirely inert in the shipped config: checksums.json is written only inside checkpoint directories, and configs/base.yaml sets save_every_steps: 0 so no checkpoint is ever written.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/artifacts.py:150-153 builds checksums.json from root.rglob inside checkpoint-NNNNNNNN/; the only call site is training.py:416-426, gated on `config.training.save_every_steps`; configs/base.yaml:79 sets save_every_steps: 0 (and base.yaml:78 eval_every_steps: 0, so evaluations.jsonl is never written either). The end-of-run artifacts written at training.py:456-462 get no checksums. docs/RUNBOOK.md:175-182 advertises "file checksums" as part of every checkpoint.
- Proposed minimal patch: Set save_every_steps to a nonzero value in configs/base.yaml, and/or write a run-level checksums.json over output_dir after training.py:462.

**9. [VALIDITY] No peak-VRAM or memory instrumentation exists anywhere, so a required reporting field can never be populated and the plan's own "false efficiency claim" mitigation is absent.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: grep -rn 'memory|vram|VRAM|max_memory_allocated' over experiments/depth_aligned_semantic_transitions/src and /tests returns zero matches. docs/EVIDENCE_STANDARD.md:43 requires "peak allocated/reserved VRAM and elapsed time"; docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md:389-390 requires the same; PLAN:451 names VRAM/FLOP gates as the mitigation for a false efficiency claim. Only elapsed_seconds is recorded (training.py:452).
- Proposed minimal patch: Call torch.cuda.reset_peak_memory_stats() before the epoch loop and record torch.cuda.max_memory_allocated()/max_memory_reserved() into final_metrics and the manifest before training.py:462.

**10. [VALIDITY] Run directories are not immutable: a re-run silently overwrites run_manifest.json, dev_summary.json and generation_predictions.jsonl while APPENDING to metrics.jsonl, and the shipped configs hard-code _seed17 output dirs with no seed loop in the matrix launcher.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/training.py:326-327 output_dir.mkdir(parents=True, exist_ok=True) with no emptiness check; atomic_write_json replaces (artifacts.py:22-27) at training.py:338, 456, 462; append_jsonl opens mode 'a' (artifacts.py:33) at training.py:404, 414; evaluation.py:169 opens generation_predictions.jsonl with mode 'w'. configs/base.yaml:5 and every arm config set output_dir ../runs/<arm>_seed17 while seed lives only in base.yaml:4; scripts/run_dev_matrix.sh:7-10 and scripts/run_dev_matrix.ps1:4-7 loop over arms only, no seed. Contradicts docs/RUNBOOK.md:123 ("Never overwrite a run"), RUNBOOK.md:108 ("for every frozen seed") and PLAN:450 ("immutable run directories and all-seed aggregation").
- Proposed minimal patch: In train(), refuse to start if output_dir already contains run_manifest.json unless an explicit allow_overwrite flag is set; make run_dev_matrix take a seed argument and derive output_dir as ../runs/<arm>_seed<seed>.

**11. [VALIDITY] dev_summary.json will report a nonzero loss/contrastive for a run whose training contrastive gradient was identically zero, because base.yaml uses batch_size 1 for training but eval_batch_size 2 for the evaluation that produces the reported metric.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/losses.py:70-72 returns new_zeros(()) when batch < 2; configs/base.yaml:66 batch_size: 1 (training loop training.py:348-370) versus configs/base.yaml:67 eval_batch_size: 2, which is what training.py:111-117 gives evaluate_teacher_forced; evaluate_teacher_forced calls the same compute_loss (training.py:238-244) and its per-key averages land in dev_summary.json via training.py:428-456. semantic.contrastive_weight is 0.10 (base.yaml:41).
- Proposed minimal patch: Record the training-time contrastive activity explicitly, e.g. manifest['contrastive_active_in_training'] = config.training.batch_size >= 2, and prefix the eval-only value as eval_only/loss/contrastive.

**12. [VALIDITY] Cross-split leakage detection is limited to problem_id equality even though the corpus supplies source_pair_sha256, which the record parser reads and then discards.**

> Status: UNVERIFIED.

- Breaks: 1. DEV/TEST separation with a sealed-test unlock guard
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/data.py:159,177-183 keys seen_ids on record.problem_id only; data.py:51 parses source_pair_sha256 and data.py:28 stores it, and a repo-wide grep shows source_pair_sha256 (like generator and domain) has no consumer outside data.py. The corpus manifest does carry it: Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1/data/manifests/problem_manifest.jsonl rows include source_pair_sha256 alongside split. docs/EVIDENCE_STANDARD.md:54 makes "a problem/source template crosses split boundaries" a leakage invalidator.
- Proposed minimal patch: In validate_dataset, add a second dict keyed on record.source_pair_sha256 (when non-empty) and emit an error when the same source pair appears under two different splits.

**13. [VALIDITY] The manifest status vocabulary emitted by code is disjoint from the authoritative status vocabulary, and no code path can emit ENGINEERING PASS, EXPLORATORY DEV, CONFIRMATORY DEV, or WITHDRAWN.**

> Status: UNVERIFIED.

- Breaks: 3. explicit BUILT-UNRUN status until a qualifying real Rineya/Qwen checkpoint run exists
- Evidence: Code emits exactly BUILT_UNRUN (artifacts.py:73, unreachable), RUNNING_DEV / RUNNING_SEALED_TEST (training.py:333) and DEV_COMPLETE / SEALED_TEST_COMPLETE (training.py:460). docs/EVIDENCE_STANDARD.md:5-12 defines BUILT — UNRUN, ENGINEERING PASS, EXPLORATORY DEV, CONFIRMATORY DEV, TEST OPENED, POSITIVE — BOUNDED, CLOSED — NEGATIVE, WITHDRAWN. Only BUILT — UNRUN overlaps, and it is the one string no run can produce.
- Proposed minimal patch: Define the EVIDENCE_STANDARD vocabulary as a Literal in config.py or artifacts.py, validate build_run_manifest's status against it, and map RUNNING/COMPLETE to ENGINEERING PASS or EXPLORATORY DEV based on whether the base model is a real checkpoint.

**14. [HYGIENE] environment_manifest records git HEAD without detecting an uncommitted working tree, so a run from a dirty tree is provenanced as a clean commit.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/artifacts.py:37-44 _git_revision() returns os.environ GIT_COMMIT/GITHUB_SHA or `git rev-parse HEAD`; there is no `git status --porcelain` or diff check, and the value is stored as environment.git_revision at artifacts.py:53. docs/EVIDENCE_STANDARD.md:16-19 places the run manifest at hierarchy level 2 and RUNBOOK.md:203-209 requires branch/commit reporting.
- Proposed minimal patch: Add result['git_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()) to environment_manifest, guarded by the same try/except.

**15. [HYGIENE] evaluate_generation's split guard inspects only records[0].split, so a mixed record sequence beginning with a DEV record passes the seal.**

> Status: UNVERIFIED.

- Breaks: 1. DEV/TEST separation with a sealed-test unlock guard
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/evaluation.py:117-118 `if records: assert_split_allowed(config, records[0].split)`. The shipped caller (training.py:436-440) supplies a split-filtered homogeneous list, so this is latent rather than live, but the function is public and the sealed record is only one line away in a concatenated corpus.
- Proposed minimal patch: Replace with `for split in {record.split for record in records}: assert_split_allowed(config, split)`.

**16. [HYGIENE] The per-example depth assignment and repeated_stage_mask — the only thing that distinguishes the treatment from its decisive controls — are never persisted in any run artifact.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/alignment.py:110-117 returns assignment and repeated_stage_mask; the only reader is cli.py:50-52 (inspect-alignment, whose output is printed to stdout and not written into the run directory). tokenization.py:175-199 puts only training_group_ids in the batch, so nothing could log them per step; GenerationRecord (evaluation.py) carries problem_id/category/answer_type/expected/extracted/raw_response/correct/extraction_route only. Contradicts PLAN:178 ("Repeated states are retained and recorded") and PLAN:280,350 (stratify by repeated checkpoint mapping, generator and domain — generator/domain are parsed at data.py:49-50 and never recorded).
- Proposed minimal patch: Write an assignment_manifest.jsonl (problem_id, split, assignment, canonical_source_indices, repeated_stage_mask) from build_dataloaders, and add generator/domain to GenerationRecord.

**17. [HYGIENE] The paired-comparison report carries no provenance: it omits the input paths, arm names, config hashes, and whether the compared runs were DEV or TEST.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/compare.py:80-96 builds the report from paired_problems, accuracies, bootstrap CI, discordance, mcnemar p, bootstrap_samples, seed and a claim_boundary string; baseline_path and candidate_path (compare.py:49-58) are never recorded, and _read_predictions (compare.py:23-34) reads only problem_id and correct. A TEST-vs-TEST comparison is indistinguishable from a DEV one. Contradicts docs/EVIDENCE_STANDARD.md:16-19 hierarchy item 3 and PLAN:409-411.
- Proposed minimal patch: Add baseline_path/candidate_path (resolved), their sha256, and the arm+status+eval_split read from each run's sibling run_manifest.json to the report dict.

**18. [HYGIENE] artifacts.py has zero test coverage: no test asserts manifest contents, status semantics, or sealed_test_opened, so every governance regression above is invisible to CI.**

> Status: UNVERIFIED.

- Breaks: 3. explicit BUILT-UNRUN status until a qualifying real Rineya/Qwen checkpoint run exists
- Evidence: 37 tests across experiments/depth_aligned_semantic_transitions/tests/*.py; a repo-wide grep for build_run_manifest, environment_manifest, save_checkpoint and run_manifest finds hits only in src/dast/artifacts.py and src/dast/training.py — no test file references them. tests/test_smoke.py:11 asserts only report['status'] == 'PASS' (the smoke report, not the run manifest). .github/workflows/dast-ci.yml:43-45 runs pytest and the smoke, so nothing in CI would catch a status or sealing regression.
- Proposed minimal patch: Add tests/test_artifacts.py asserting: default status is BUILT_UNRUN; sealed_test_opened is true when either split is test in any case; a toy-model run does not produce a DEV_COMPLETE status; and the manifest contains metrics and artifact checksums.

**19. [HYGIENE] Nothing enforces the preregistered protocol validity condition that resolved checkpoints be 7/14/21/28, so an off-protocol depth schedule produces an artifact labelled the same as a valid one.**

> Status: UNVERIFIED.

- Breaks: 3. explicit BUILT-UNRUN status until a qualifying real Rineya/Qwen checkpoint run exists
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/alignment.py:23-44 resolve_checkpoint_layers accepts any num_hidden_layers and only rejects collapse to duplicates; config.py:206-214 validates the fractions but never the resolved layers; artifacts.py:91 records checkpoint_layers without comparing them to an expected tuple. docs/PREREGISTRATION.md:16-17 states "runtime-resolved checkpoints must be 7, 14, 21, 28 or the run is invalid for the initial protocol" and docs/RUNBOOK.md:66 makes it a manual check.
- Proposed minimal patch: Add an optional experiment.expected_checkpoint_layers tuple to the config and have train() raise if model.checkpoint_layers differs, recording the expectation in the manifest.

**20. [HYGIENE] The repository-wide experiment audit does not list DAST at all, so the BUILT — UNRUN classification exists only inside the branch package.**

> Status: UNVERIFIED.

- Breaks: 3. explicit BUILT-UNRUN status until a qualifying real Rineya/Qwen checkpoint run exists
- Evidence: docs/EXPERIMENT_PROGRAM_AUDIT.md contains no reference to DAST, depth_aligned, or Depth-Aligned (grep -rln over docs/, manifests/ and paper/ matches only docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md), yet the pipeline was already committed in 91f2e1c. docs/RUNBOOK.md:214 requires "update the repository-wide experiment audit in the same merge commit".
- Proposed minimal patch: Add a DAST row to docs/EXPERIMENT_PROGRAM_AUDIT.md recording branch, commit a343e1f, status BUILT — UNRUN, and sealed evaluation CLOSED.

**21. [HYGIENE] The environment manifest omits bitsandbytes and numpy although quantization: 4bit/8bit is a supported and documented path, so a QLoRA run cannot be attributed to a quantization stack version.**

> Status: UNVERIFIED.

- Breaks: 2. manifests recording configuration, hashes, environment, assignment, metrics, and evidence status
- Evidence: experiments/depth_aligned_semantic_transitions/src/dast/artifacts.py:58 iterates only ('transformers','peft','accelerate','safetensors'); hf.py:28-44 imports BitsAndBytesConfig for config.model.quantization in {'4bit','8bit'} (config.py:66) and PLAN:419 offers QLoRA as a fallback. docs/PHASE0_READINESS.md:40 records bitsandbytes as absent on the intended host, so this omission would silently hide the reason a QLoRA run failed or fell back.
- Proposed minimal patch: Extend the package tuple at artifacts.py:58 to include 'bitsandbytes' and 'numpy'.

### Challenges to the lead's own findings (2)

| Claim | Position | Reasoning |
|---|---|---|
| F4 | `CONFIRMED` | Both halves verified, and the second half has a governance consequence the claim does not state: because DataConfig exposes a single `path` and build_dataloaders derives both splits by filtering it, the manifest can only ever record ONE data path and ONE file hash. Following the recommended concatenation workaround therefore produces a manifest whose data.sha256 covers a file whose split composition is never recorded |
| F7 | `CONFIRMED` | Confirmed as stated for the training objective, but understated in one governance-relevant respect: the reported metric will NOT be zero. configs/base.yaml sets eval_batch_size: 2, and evaluate_teacher_forced runs the identical compute_loss over those 2-example batches, so dev_summary.json will contain a nonzero loss/contrastive for a run in which the contrastive gradient was identically zero at every step. A reader  |

---

## Dimension: `assignments-and-matching`

### Scope-contract verdicts

| Verdict | Scope item |
|---|---|
| `IMPLEMENTED` | 1a. deterministic ORDERED depth assignment |
| `IMPLEMENTED` | 1b. deterministic REVERSED depth assignment |
| `IMPLEMENTED` | 1c. shuffled assignment is DETERMINISTIC |
| `CONTRADICTED_BY_CODE` | 1d. shuffled assignment is GUARANTEED NON-IDENTITY |
| `IMPLEMENTED_BUT_INEFFECTIVE` | 1e. shuffled assignment is a CLEAN negative control (disjoint from the other controls) |
| `IMPLEMENTED` | 1f. final_repeated / none assignments |
| `PARTIAL` | 1g. repeated-slot bookkeeping is recorded |
| `IMPLEMENTED` | 2a. matched TRAINABLE-PARAMETER capacity across the six Phase-1 arms |
| `IMPLEMENTED` | 2b. cot_baseline is parameter-matched AND still builds the projector |
| `IMPLEMENTED` | 2c. matched number of depth losses |
| `PARTIAL` | 2d. matched TOKEN exposure |
| `IMPLEMENTED` | 2e. matched OPTIMIZER-STEP budget |
| `CONTRADICTED_BY_CODE` | 2f. matched DATA exposure |
| `IMPLEMENTED` | 2g. SEED controls cover python/numpy/torch/cuda |
| `IMPLEMENTED_BUT_INEFFECTIVE` | 2h. SEED controls actually govern the run (init + arm matching) |
| `IMPLEMENTED` | 2i. matched DATALOADER shuffling across arms |
| `ABSENT` | 2j. per-seed run isolation (three frozen seeds) |

### Additional defects claimed (10)

**1. [BLOCKER] Auxiliary head and LoRA adapter are initialized from an unseeded global RNG because set_global_seed runs only inside train(), after the model is built.**

> CONFIRMED by lead. cli.py:62 build_hf_components and cli.py:72 DASTModel(...) both run before train() reaches training.py:280 set_global_seed, so LoRA and projector init are unseeded and the same config+seed yields different initial weights each run.

- Breaks: 2. matched trainable-parameter, data, token, step, and seed controls (seed control)
- Evidence: src/dast/cli.py:62 (build_hf_components), :72-81 (DASTModel), :85 (train) vs src/dast/training.py:280 (set_global_seed); src/dast/model.py:129; src/dast/hf.py:130; src/dast/operators.py:40,42. Three identical CPU-only processes (PYTHONHASHSEED=0, same config/seed) produced state_projector W1 sha256 prefixes 6aee5ac3f572d8dc, cc9289967ad0dc01, b65cfad309038b3f and torch.initial_seed() 3926570476000, 3930038553400, 3933259228700.
- Proposed minimal patch: In src/dast/cli.py:command_train, call dast.training.set_global_seed(config.experiment.seed, deterministic_algorithms=config.training.deterministic_algorithms) immediately after load_config (before line 62), and keep the existing call in train() so the library entry point is also safe. Optionally reseed just before DASTModel(...) so the projector init is independent of how many draws the base-model load consumed.

**2. [BLOCKER] configs/base.yaml data.path is a metadata-only manifest, and build_dataloaders derives train and eval from one path while the corpus ships one file per split, so no arm can load data at all.**

> Status: UNVERIFIED.

- Breaks: 2. matched trainable-parameter, data, token, step, and seed controls (data control)
- Evidence: configs/base.yaml:26; src/dast/data.py:35-38 required-field check; executed load_records raises 'record is missing required fields: problem, answer, full_response, reasoning_steps, compact_state_trace'; src/dast/training.py:66-78 uses one data_path for both splits; Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1/data/manifests/sha256sums.txt lists data/full_context/train.jsonl, dev.jsonl, test.jsonl separately (that directory is absent from the bundle).
- Proposed minimal patch: Add DataConfig.eval_path (defaulting to data.path when empty) and use it in build_dataloaders/evaluate_generation; point data.path at data/full_context/train.jsonl and eval_path at data/full_context/dev.jsonl once the corpus is regenerated; keep the split filter as a belt-and-braces assertion. Add a test that load_config('configs/base.yaml') resolves to an existing file whose first record passes ReasoningRecord.from_mapping.

**3. [VALIDITY] The shuffled control's non-identity guard checks the permutation, not the target sequence, so 2.0-3.3% of problems make state_shuffled byte-identical to state_ordered.**

> Status: UNVERIFIED.

- Breaks: 1. deterministic ordered, reversed, and shuffled-depth assignments
- Evidence: src/dast/alignment.py:66-69 vs docs/ALGORITHM.md:84-86 and README.md:110. Measured over the 48k corpus at seed 17: 867/43,200 train (2.01%), 53/2,400 dev (2.21%); over the shipped 4,000/800 slice (configs/base.yaml:29-30): 93/4,000 (2.33%) and 26/800 (3.25%). Example rineya_si_v1_biology_0005 (step_count=3): canonical_src=(0,1,1,2), perm=(0,2,1,3) (non-identity, passes the guard), shuffled_src=(0,1,1,2).
- Proposed minimal patch: Move the guard after target construction in align_trajectory: compute `training_source_indices`, and while `training_source_indices == source_indices` draw the next permutation from the same seeded rng (or rotate) until the sequence differs; raise only when num_stages distinct source indices < 2 (i.e. a genuinely degenerate trace). Add a regression test with raw_targets of length 2 and 3 asserting training_texts != canonical_texts for the shuffled arm.

**4. [VALIDITY] max_train_records=4000 with break-on-limit over a category-sorted manifest gives a train slice covering 3 of 30 categories while the dev slice covers 10, so 7 of 10 DEV categories are never trained on.**

> PARTIALLY REFUTED by lead. True of data/manifests/problem_manifest.jsonl, which is category-sorted. FALSE of data/full_context/train.jsonl, the file the config should point at: it is pre-shuffled, and the first 800 records already cover all 30 categories (measured). Both statements hold; they concern different files.

- Breaks: 2. matched trainable-parameter, data, token, step, and seed controls (data control)
- Evidence: configs/base.yaml:29-30; src/dast/data.py:146-153 (`break` on limit, no shuffling or stratification); measured on the shipped manifest: first 4,000 train records = {biology, boolean_logic, chemistry}; first 800 dev records = {biology, boolean_logic, chemistry, code_trace, cooking, data_table, decimals, elapsed_time, finance, fractions}. Repeated-depth-slot rate also jumps from 42.03% corpus-wide to 56.00% in that train slice and 51.88% in the dev slice.
- Proposed minimal patch: Make record selection stratified and seeded: read all records for the split, then subsample with a Random(config.experiment.seed) stratified by category (or at minimum shuffle before truncating), and record the selected problem_id list (or its SHA-256) in run_manifest.json. This is matched across arms today, so it is not an arm asymmetry - but it silently turns 70% of DEV into out-of-category evaluation and voids PLAN §7 step 5 / §8 stratification.

**5. [POWER] The identity-rotation fallback makes permutation (1,2,3,0) twice as likely as any other, and state_shuffled coincides with state_reverse on ~6% of problems, so the two decisive controls are not independent.**

> Status: UNVERIFIED.

- Breaks: 1. deterministic ordered, reversed, and shuffled-depth assignments
- Evidence: src/dast/alignment.py:67-69; measured at seed 17 over 48,000 problem_ids: identity branch fires 2,033 times (4.24%), so post-guard (1,2,3,0) holds 8.45% mass vs ~4.15% for the other 22; shuffled target sequence equals reverse target sequence for 2,624/43,200 train (6.07%) and 159/2,400 dev (6.62%). PREREGISTRATION.md:64-66 requires ordered-vs-shuffled and ordered-vs-reverse as jointly reported independent contrasts.
- Proposed minimal patch: Replace the rotation at alignment.py:67-69 with rejection sampling from the same seeded rng (redraw until the resulting target sequence differs from both the ordered and the reversed sequence), and record per-problem assignment overlap counts in the run manifest so the shuffled/reverse contrast overlap is auditable rather than latent.

**6. [HYGIENE] run_manifest.json omits every budget-exposure field the evidence standard requires: no token counts, no train/eval problem IDs or slice hash, no update count, no truncation count, no repeated-slot statistics.**

> Status: UNVERIFIED.

- Breaks: 2. matched trainable-parameter, data, token, step, and seed controls (token/step/data accounting)
- Evidence: src/dast/artifacts.py:76-98 (manifest body) plus src/dast/training.py:335-338 (only runtime_precision + trainable/total parameter counts added); required by PLAN §9.3:390 and docs/EVIDENCE_STANDARD.md:38-40; tokenization.py:117,194 computes truncated_prompt_tokens and nothing consumes it; alignment.py:109 computes repeated_stage_mask and nothing persists it; data.sha256 (artifacts.py:86) hashes the whole file, not the used slice.
- Proposed minimal patch: Accumulate prompt-token, assistant-token, truncated-prompt-token, example, and repeated-stage counts in the training loop, and write them plus sha256 of the sorted used problem_id list into run_manifest.json before the final atomic_write_json at training.py:462. Also copy optimizer_steps/total_updates into the manifest, not only dev_summary.json.

**7. [HYGIENE] No test loads any shipped config, and the offline smoke trains only operator_ordered, so the ordered/shuffled/reverse/final_repeated/literal/cot_baseline end-to-end paths are never executed even on CPU.**

> Status: UNVERIFIED.

- Breaks: 1 and 2 (both scope items are untested at the shipped-config level)
- Evidence: grep over tests/ finds no reference to configs/ (tests/test_config.py:8-26 builds its own tmp_path YAML); src/dast/smoke.py:177 build_smoke_config(..., arm='operator_ordered') is the only arm passed to train(), with state_ordered used solely for the feedback probe at :128; scripts/run_smoke.sh:5-8; tests/test_alignment.py has no case for literal_ordered, operator_*, or cot_baseline, and no case that exercises the alignment.py:67-69 rotation branch.
- Proposed minimal patch: Add a parametrized test over all six Phase-1 arms that runs load_config('configs/<arm>.yaml'), asserts the model/data/lora/training/evaluation sections equal base.yaml's except the arm-mandated keys, and (against the toy model + fixture) runs one train() step; add an alignment test whose (problem_id, seed) hits the identity branch.

**8. [HYGIENE] The cot_baseline guard tests only the SUM of state_weight and contrastive_weight, so a baseline with active, cancelling auxiliary weights passes validation.**

> Status: UNVERIFIED.

- Breaks: 2. matched trainable-parameter, data, token, step, and seed controls (parameter/loss matching for the baseline arm)
- Evidence: src/dast/config.py:234-236 (`auxiliary_weight = state_weight + contrastive_weight` then `!= 0`); contrastive_weight has no non-negativity check anywhere in validate() (only semantic.stage_weights is checked at config.py:222-223).
- Proposed minimal patch: Replace the sum check with per-field checks (`state_weight != 0 or contrastive_weight != 0`) and add `if self.semantic.contrastive_weight < 0` / `state_weight < 0` to the error list.

**9. [HYGIENE] Gradient clipping is applied jointly over backbone and auxiliary parameters, so the LM update magnitude becomes a function of the arm's auxiliary loss.**

> Status: UNVERIFIED.

- Breaks: 2. matched trainable-parameter, data, token, step, and seed controls (optimizer-schedule matching)
- Evidence: src/dast/training.py:382-385 clips over every requires_grad parameter with configs/base.yaml:73 max_grad_norm: 1.0, while _make_optimizer (training.py:148-176) deliberately separates the groups for the learning rate. MEASURED on the CPU toy at identical init: pre-clip total grad norm 79.5379 (cot_baseline), 79.5815 (ordered), 79.5586 (shuffled), 79.5969 (reverse), 79.3792 (final_repeated), 79.6144 (literal) -> clip coefficient spread ~0.3%, projector-only norm 0.9482-1.3038.
- Proposed minimal patch: Clip the backbone and auxiliary parameter groups separately (two clip_grad_norm_ calls over the same lists _make_optimizer already builds) so the LM step size is independent of the auxiliary loss, and log both norms. Honest caveat: on the toy the asymmetry is only ~0.3% because the toy backbone dominates the norm; with LoRA-only trainables the auxiliary share (and hence the asymmetry) can be far larger, but I could not measure that without loading a real model.

**10. [HYGIENE] configs/feedback_causal_probe.yaml is not parameter-matched or numerics-matched to any control arm, and no matched feedback control config exists.**

> Status: UNVERIFIED.

- Breaks: 2. matched trainable-parameter, data, token, step, and seed controls
- Evidence: configs/feedback_causal_probe.yaml:5-14 sets feedback_enabled: true (adding feedback_decoder + feedback_gates at src/dast/model.py:142-146), gradient_checkpointing: false, and do_generation: false, while extending state_ordered.yaml; no feedback_shuffled/feedback_reverse config exists in configs/ and config.py:19-27 ARM_SPECS offers no feedback-specific arm.
- Proposed minimal patch: Either add configs/feedback_shuffled.yaml (same feedback settings, arm: state_shuffled) so the Phase-3 probe has a matched-capacity control, or state explicitly in configs/feedback_causal_probe.yaml and docs/RUNBOOK.md §9 that this probe is intra-run (intervention-vs-unmodified) only and is never to be compared across arms.

### Challenges to the lead's own findings (3)

| Claim | Position | Reasoning |
|---|---|---|
| F4 | `CONFIRMED` | Both halves hold, and the second half is stronger than stated. (a) The manifest is metadata-only: I executed load_records on configs/base.yaml's path and got ValueError listing exactly the five missing fields. (b) build_dataloaders really does derive train AND eval from one path, and the corpus really does ship one file per split - I confirmed this from the dataset's own checksum file rather than by inference, so wit |
| F7 | `CONFIRMED` | Confirmed, with two refinements that matter for the matching dimension. (i) The inertness is SYMMETRIC across all five semantic arms, so it does not create an arm asymmetry - it simply deletes the anti-collapse term that ALGORITHM.md:96-97 and PLAN §6.3 rely on. (ii) It is NOT uniformly inert: configs/base.yaml:67 sets eval_batch_size: 2, so multi_positive_contrastive_loss is ACTIVE during evaluate_teacher_forced and |
| F8 | `CONFIRMED` | Confirmed and quantifiable exactly, but the 42% figure UNDERSTATES the shipped configuration and the 'partial no-ops' framing understates the failure. Exact corpus rate: 20,171/48,000 = 42.02% have step_count < 4 (1,614 at N=2, 18,557 at N=3); per split 42.03% train / 42.54% dev / 41.29% test. Within the records the shipped configs actually consume (first 4,000 train, first 800 dev) the rate is 56.00% and 51.88%, bec |

---

## Dimension: `depth-mapping`

### Scope-contract verdicts

| Verdict | Scope item |
|---|---|
| `PARTIAL` | hidden-state checkpoints mapped from relative reasoning progress |
| `IMPLEMENTED_BUT_INEFFECTIVE` | one shared semantic projection interface across depths |

### Additional defects claimed (8)

**1. [VALIDITY] DASTModel's two forward paths read a different tensor at the final checkpoint: the auxiliary path takes post-final-RMSNorm hidden_states[L], the feedback path takes the pre-norm output of decoder layer L. A projector trained in one mode is fed an out-of-distribution stage-K vector in the other.**

> Status: UNVERIFIED.

- Breaks: one shared semantic projection interface across depths
- Evidence: model.py:263 `pool_prompt_hidden(hidden_states[layer_number], ...)` vs model.py:155-162 `layer = layers[layer_number - 1]` ... `hidden = _extract_hidden(output)`. transformers/utils/output_capturing.py:270-277 replaces hidden_states[-1] with last_hidden_state; modeling_qwen3.py:423 applies self.norm before returning it. Constructive check on a 4-layer random Qwen3 (CPU, no checkpoint loaded): hidden_states[L] == model.norm(hook-captured raw last-layer output) -> True; == raw -> False; max|diff| = 2.80. PHASE0_READINESS.md:326-330 measures cos(h27,h28)=0.104 and RMS 50.93 vs 3.06 on the real checkpoint, so the two tensors are near-orthogonal in practice.
- Proposed minimal patch: In DASTModel, make the final-checkpoint representation explicit and identical in both paths. Either (a) apply the base model's final norm to every pooled checkpoint before the projector in both the aux path (model.py:262-265) and the hook (model.py:161-162), or (b) resolve the last checkpoint to a pre-norm index in both paths. Then add a test that runs the same batch with feedback_enabled True/False and asserts output.states agree to tolerance when feedback_scale=0.

**2. [VALIDITY] The offline toy backbone omits the final norm from hidden_states, so every unit test and the smoke path exercise a homogeneous depth interface that the real model does not have. The mixed pre/post-norm defect is structurally invisible to CI.**

> Status: UNVERIFIED.

- Breaks: one shared semantic projection interface across depths
- Evidence: toy.py:164-168 appends only post-layer residual states to all_hidden; toy.py:169 uses self.final_norm only inside the logits computation, so hidden_states[num_hidden_layers] is pre-norm. Real path: transformers/utils/output_capturing.py:270-277 + modeling_qwen3.py:423 make hidden_states[L] post-norm. tests/test_model_feedback.py:20-39 and smoke.py:100-124 use ToyCausalLM exclusively; RUNBOOK.md:44 accepts '2/4/6/8 for the toy 8-layer model' as qualification.
- Proposed minimal patch: Change ToyCausalLM.forward to append self.final_norm(hidden) as the last hidden_states entry (mirroring HF), and add a test asserting hidden_states[-1] is the normed state while hidden_states[L-1] is not. This makes the existing toy tests reproduce the real representation split.

**3. [VALIDITY] _progress_indices distributes reasoning progress uniformly over stage INDEX and never sees checkpoint_fractions, so 'mapped from relative reasoning progress' holds only while the depth schedule is equally spaced.**

> Status: UNVERIFIED.

- Breaks: hidden-state checkpoints mapped from relative reasoning progress
- Evidence: alignment.py:57 `round(stage * (num_targets - 1) / (num_stages - 1))` uses only stage ordinals; data.py:220-226 and training.py:83-94 pass num_stages=len(fractions) and never the fractions themselves. Measured: fractions (0.25,0.5,0.75,1.0) and (0.05,0.10,0.15,1.0) both produce source indices (0,3,5,8) for N=9. config.py:206-214 permits any strictly-increasing tuple ending at 1.0, and PLAN:180-182 explicitly contemplates a frozen alternative (non-uniform) schedule.
- Proposed minimal patch: Pass checkpoint_fractions (or the resolved layers normalized by L) into align_trajectory and interpolate progress against the actual normalized depth of each checkpoint, rescaled so the endpoints stay pinned; keep the current behaviour as the special case of equally spaced fractions and assert equality in a regression test.

**4. [VALIDITY] With feedback_enabled, generation silently bypasses the shared semantic interface, and no config validation prevents a run from reporting generated accuracy for a causal arm whose causal mechanism was off.**

> Status: UNVERIFIED.

- Breaks: one shared semantic projection interface across depths
- Evidence: evaluation.py:150 calls model.base_model.generate(**generation_kwargs), never DASTModel.forward, so _active_prompt_mask stays None and every feedback hook returns the unmodified layer output at model.py:159-160. config.py:199-268 blocks only feedback + gradient_checkpointing (config.py:217-218). ALGORITHM.md:148 and RUNBOOK.md:165 say cached generation is 'blocked'/'unsupported'; the only thing that actually prevents it is configs/feedback_causal_probe.yaml:13-14 setting do_generation: false.
- Proposed minimal patch: Add to DASTConfig.validate: `if self.model.feedback_enabled and self.evaluation.do_generation: errors.append('feedback_enabled is incompatible with cached generation in V1')`.

**5. [HYGIENE] resolve_checkpoint_layers silently REPAIRS schedules that collapse to duplicate layers instead of rejecting them, contradicting the SSOT, and it also silently monotonizes non-monotone fraction tuples.**

> Status: UNVERIFIED.

- Breaks: hidden-state checkpoints mapped from relative reasoning progress
- Evidence: alignment.py:36-37 `if resolved and layer <= resolved[-1]: layer = resolved[-1] + 1`; the raise at alignment.py:38-42 fires only if the repair overflows L. PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md:175 states 'The code rejects schedules that collapse to duplicate layers.' Measured on L=28: (0.01,0.02,0.03,1.0) -> (1,2,3,28) with no error, and that tuple passes config.py:206-214 unchanged (in (0,1], strictly increasing, last 1.0); (0.5,0.25,1.0) -> (14,15,28), i.e. a request for 25% depth is served by layer 15.
- Proposed minimal patch: Raise on the collision itself rather than bumping: replace alignment.py:36-37 with `if resolved and layer <= resolved[-1]: raise ValueError(...)`, and add an explicit strictly-increasing check on the incoming fractions inside resolve_checkpoint_layers so the guard does not depend on config.validate having run.

**6. [HYGIENE] repeated_stage_mask detects only ADJACENT duplicate source indices and is never written to any run artifact, so the SSOT requirement that repeated states be 'recorded' is unmet; it captures about half the duplicates actually present in the shuffled arm.**

> Status: UNVERIFIED.

- Breaks: hidden-state checkpoints mapped from relative reasoning progress
- Evidence: alignment.py:109 compares training_source_indices[index] with [index-1] only. Nothing persists it: artifacts.py:67-97 records checkpoint_layers only, training.py writes metrics/summaries with no trajectory field, and cli.py:42-53 emits assignment plus source indices but not the mask. Contract: PLAN:177-178 and ALGORITHM.md:66-68. Census over the real manifest at seed 17, K=4: for state_shuffled the mask records 9,817 of the 19,597 duplicates actually present in train (50.1%) and under-reports on 9,306/43,200 records (21.5%); dev 550/1,118 (49.2%), under-reporting on 22.0%. Example: source indices (0,0,1,1) with assignment (2,1,3,0) gives training indices (1,0,1,0), mask (False,False,False,False), two real duplicates.
- Proposed minimal patch: Compute the mask from the multiset (e.g. flag any stage whose training target text already appeared at another stage, not just the previous one) and write per-example {assignment, canonical/training source indices, repeat mask} plus corpus-level repeat rates into the run directory alongside run_manifest.json.

**7. [HYGIENE] Python's banker's rounding in _progress_indices diverges from the round() in the algorithm spec whenever (K-1) is even; inert at the shipped K=4 but wrong for K=3 or K=5.**

> Status: UNVERIFIED.

- Breaks: hidden-state checkpoints mapped from relative reasoning progress
- Evidence: alignment.py:57 uses the builtin round (round-half-to-even). Measured N=6, K=3: exact progress [0.0, 2.5, 5.0]; code yields (0,2,5) while ALGORITHM.md:64's round() read as half-up yields (0,3,5). With K=4 the quantity k*(N-1)/3 can never land on .5, so all shipped configs (configs/base.yaml:20, inherited by every arm config) are unaffected.
- Proposed minimal patch: Use explicit half-up rounding, e.g. `int(math.floor(stage * (num_targets - 1) / (num_stages - 1) + 0.5))`, or state the tie-breaking rule in ALGORITHM.md section 4.

**8. [HYGIENE] The preregistration's run-invalidating condition on resolved checkpoints (must be 7/14/21/28) is not machine-enforced anywhere, so a different-depth checkpoint silently produces a schedule outside the frozen protocol.**

> Status: UNVERIFIED.

- Breaks: hidden-state checkpoints mapped from relative reasoning progress
- Evidence: PREREGISTRATION.md:16-17 ('runtime-resolved checkpoints must be 7, 14, 21, 28 or the run is invalid for the initial protocol'); RUNBOOK.md:66 makes it a manual check. Code: model.py:127 resolves silently, artifacts.py:91 and cli.py:82 only record the result, and config.py:199-268 has no assertion. Measured: L=27 -> (7,14,20,27), L=29 -> (7,14,22,29), both accepted without warning.
- Proposed minimal patch: Add an optional `model.expected_checkpoint_layers` to ModelConfig and have DASTModel (or train()) raise when the runtime-resolved tuple differs, so the protocol constraint fails closed instead of relying on a human reading the manifest.

### Challenges to the lead's own findings (2)

| Claim | Position | Reasoning |
|---|---|---|
| F6 | `CONFIRMED` | The factual core is right and I reproduced its mechanism from first principles without the real checkpoint. Two caveats sharpen it rather than weaken it. (1) Half of the cited evidence is not load-bearing: the elementwise RMS gap (49 vs 3.06) is absorbed by the projector's own input LayerNorm at semantic.py:20, so the operative harm is the directional/near-orthogonality result, not the scale. The PHASE0 report itself |
| F8 | `CONFIRMED` | Confirmed and strengthened. I recomputed the census independently from the shipped manifest's step_count field (N = step_count, since data.py:64 requires len(compact_state_trace) == len(reasoning_steps)+1 and data.py:71-74 drops the initial state) and reproduced the lead's figures exactly: 42.0% of train and 42.5% of dev traces receive fewer than K=4 distinct depth targets. Two additions make the finding sharper than |

---

## Dimension: `targets-and-pooling`

### Scope-contract verdicts

| Verdict | Scope item |
|---|---|
| `IMPLEMENTED_BUT_INEFFECTIVE` | cumulative semantic states rather than literal sentence fragments |
| `IMPLEMENTED` | prompt-only pooling so answer tokens cannot leak into intermediate targets |

### Additional defects claimed (9)

**1. [VALIDITY] No runtime guard that prompt_mask excludes assistant/label positions; only the padding half of the 'hard validity contract' is machine-enforced.**

> Status: UNVERIFIED.

- Breaks: prompt-only pooling so answer tokens cannot leak into intermediate targets
- Evidence: src/dast/model.py:215-218 checks only `prompt_mask & ~attention_mask.bool()`. `labels` is already a parameter of the same call (model.py:212) but is never cross-checked. PLAN:132-138 calls prompt-only pooling 'a hard validity contract with unit tests'; EVIDENCE_STANDARD.md lists 'semantic pooling includes assistant target tokens' as a withdrawal-triggering invalidator. MEASURED (CPU toy model, fixture batch): passing prompt_mask = attention_mask.bool() (covering all 30 assistant tokens as well as the 29 prompt tokens) through DASTModel.forward is ACCEPTED with no error, and the resulting states differ from the honest states by up to 0.217 in max-abs. The padding guard did not and cannot catch it because assistant tokens have attention_mask=1.
- Proposed minimal patch: In DASTModel.forward, after the padding check at src/dast/model.py:217, add: `if labels is not None and torch.any(prompt_mask & (labels != IGNORE_INDEX)): raise ValueError("prompt_mask cannot include assistant/target positions")` (import IGNORE_INDEX from dast.tokenization).

**2. [VALIDITY] The 'exclude the initial state' contract is a positional slice with no enforcement on the training path, and the only enforcement mechanism cannot run on the configured data path.**

> Status: UNVERIFIED.

- Breaks: cumulative semantic states rather than literal sentence fragments
- Evidence: src/dast/data.py:75 unconditionally returns compact_state_trace[1:]. The length contract that makes that slice correct is src/dast/data.py:64-68, inside ReasoningRecord.validate(), which is called ONLY from validate_dataset (src/dast/data.py:175); grep for `.validate()` in src/dast shows config.py:341, data.py:175, smoke.py:96, training.py:279 - none of them is the record-level check on the train/eval path. load_records (data.py:139-153) and build_examples (data.py:209-228) never validate. MEASURED: a record with reasoning_steps=['s1','s2','s3'] and compact_state_trace=['A=1.','B=2.','Final answer: 7'] (no initial state) is reported bad by record.validate() ('3 != 4') yet build_examples accepts it and silently drops the first REAL milestone: raw_targets -> ('B=2.', 'Final answer: 7'), canonical_texts -> ('B=2.','B=2.','Final answer: 7','Final answer: 7'). MEASURED on the shipped configs/base.yaml:26 data.path: validate_dataset returns status FAIL with 'record 1: record is missing required fields: problem, answer, full_response, reasoning_steps, compact_state_trace' and split_counts {} - so RUNBOOK.md:3 step 'dast validate-data <manifest>' never reaches the trace-length check for a single real record, and load_records on the same path raises ValueError immediately.
- Proposed minimal patch: Call `errors = record.validate()` inside load_records (src/dast/data.py:147) and raise on non-empty, or assert in raw_targets that `len(self.compact_state_trace) == len(self.reasoning_steps) + 1` before slicing at data.py:75.

**3. [VALIDITY] Docs describe the depth targets as 'cumulative' semantic states, but the corpus supplies non-cumulative replacement states by explicit design and no code accumulates them.**

> Status: UNVERIFIED.

- Breaks: cumulative semantic states rather than literal sentence fragments
- Evidence: Claim sites: docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md:30 ('Assign **cumulative semantic reasoning states**'), :194 ('cumulative states in canonical order'), :512; experiments/depth_aligned_semantic_transitions/docs/ALGORITHM.md:7,40 ('treats post-step compact states as cumulative milestones'); README.md:18. Contradicting sources: Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1/scripts/common.py:112-131, where make_steps sets `state = next_state` each iteration so every state REPLACES its predecessor; DATA_CARD.md:76 'It should not replay all completed reasoning prose.' Real record rineya_si_v1_biology_0001 has states 'Growth model: 30 x 2^6.' / 'Doubling factor after 6 intervals: 64.' / 'Computed population: 1920.' / 'Final answer: 1920 cells' - disjoint snapshots. src/dast/data.py:71-80 and src/dast/alignment.py:88-109 pass them through verbatim; there is no concatenation, union, or running-state construction anywhere in src/dast.
- Proposed minimal patch: Either replace 'cumulative' with 'successive/post-step Markov-sufficient' in PLAN:30/194, ALGORITHM:7/40 and README:18, or add a target_kind='cumulative_state' in src/dast/data.py:71-80 that joins compact_state_trace[1:k+1] before encoding.

**4. [VALIDITY] preserve_target truncation slices the prompt from the right and drops the chat-template head, so the model would be trained on a sequence starting mid-system-message.**

> Status: UNVERIFIED.

- Breaks: prompt-only pooling so answer tokens cannot leak into intermediate targets
- Evidence: src/dast/tokenization.py:95-98: `keep_prompt = max_length - len(response_ids); prompt_ids = prompt_ids[-keep_prompt:]`. Because len(full_ids) > max_length implies len(prompt_ids) > keep_prompt, this ALWAYS removes leading tokens when it fires. MEASURED with dast.toy.HashTokenizer at max_length=64: untruncated input_ids start (2, 13, 136) with bos=2; truncated input_ids start (136, 136, 136), truncated_prompt_tokens=94, BOS/system header retained = False. MEASURED with the real Qwen3-0.6B template at max_length=128: keep_prompt=56, 3 tokens dropped, and the retained prompt begins 'Solve the problem carefully. ...' - the '<|im_start|>system\n' role header is gone. The shipped test tests/test_tokenization.py:39-48 only asserts truncated_prompt_tokens > 0 and that some label survives; it never checks that the template head survives. MITIGATION (important): with configs/base.yaml:19 max_length=512 this never fires on this corpus - measured full-sequence token lengths over the 60 real samples are mean 149.5, max 307, 0/60 above 512. The defect is latent, not active in the shipped config.
- Proposed minimal patch: Preserve the template head in src/dast/tokenization.py:97, e.g. keep a fixed number of leading prompt tokens and truncate from the middle of the user turn, or fail closed with TokenizationContractError instead of silently reshaping the prompt.

**5. [POWER] The advertised leakage test is an arithmetic tautology over a hand-written mask and exercises neither the tokenizer, the collator, nor a real chat template.**

> Status: UNVERIFIED.

- Breaks: prompt-only pooling so answer tokens cannot leak into intermediate targets
- Evidence: tests/test_pooling_semantic.py:8-14 builds `hidden = torch.randn(2,6,8)` and a literal mask, sets `changed[~mask] = 1_000_000`, and asserts the masked mean is unchanged - which is true for any masked-mean implementation regardless of whether the real prompt_mask is correct. README.md:119-121 presents this as the safety boundary ('A test changes every assistant hidden state by a huge amount and verifies that the pooled state is unchanged'). The mask-construction test tests/test_tokenization.py:15-27 uses dast.toy.HashTokenizer, whose apply_chat_template (src/dast/toy.py:50-64) appends to a growing id list so the prefix property holds by construction and cannot fail. PREREGISTRATION.md:8 G0 requires a 'real tokenizer prefix and prompt-mask seam test'; grep for AutoTokenizer/from_pretrained/Qwen/seam across tests/, src/dast/smoke.py, scripts/ and src/dast/cli.py returns nothing, and src/dast/cli.py:115-148 exposes no such subcommand. The test also covers only mode='mean'; mode='last_prompt' (pooling.py:23-27) has no leakage assertion.
- Proposed minimal patch: Add a test that tokenizes a fixture record with the real tokenizer (or a mock that emits a non-prefix template), asserts prompt_mask == (labels == IGNORE_INDEX) & attention_mask, and asserts DASTModel.forward raises when the mask is widened to assistant positions; add the same perturbation assertion for mode='last_prompt'.

**6. [POWER] The V1 target geometry leaves a maximum canonical order margin of ~0.17, and the fixed random projection destroys ~20% of even that.**

> Status: UNVERIFIED.

- Breaks: cumulative semantic states rather than literal sentence fragments
- Evidence: MEASURED by exactly reproducing src/dast/semantic.py:63-105 (real Qwen3-0.6B embed_tokens rows, real tokenizer, state_dim=128, projection_seed=1701) on the 60 real records: within-record cosine between DIFFERENT stage targets is mean 0.8325 / median 0.8600 / p90 0.9815 (adjacent stages 0.8771); cross-record targets from different problems already sit at 0.7120 mean. Consequently, with a PERFECT ordered predictor the preregistered secondary outcome state/order_margin (src/dast/losses.py:169-172, PREREGISTRATION.md:7, RUNBOOK.md:7) tops out at 0.1675, and the cosine state loss (src/dast/losses.py:43-56) separates ordered from reverse by only 0.1711 and from final_repeated by 0.1456 - that 0.17 is the entire dynamic range the depth-assignment manipulation can inject into a loss added at weight 1.0 (configs/base.yaml:40) next to an LM loss of order 1. The fixed projection at semantic.py:65-70 makes this worse, not better: off-diagonal stage cosine is 0.7920 BEFORE the 1024->128 QR projection and 0.8325 after, i.e. the achievable margin drops 0.2080 -> 0.1675 (-19.5%), contradicting the stated rationale at ALGORITHM.md:45-52 / PLAN:3.3 that the projection is a neutral width-independence device.
- Proposed minimal patch: Before any confirmatory run, compute the perfect-predictor order margin on the frozen train split and record it as the ceiling in the run manifest; and either raise semantic.state_dim toward the embedding width or drop the projection (semantic.py:65-70) so the target cone is not further compressed.

**7. [HYGIENE] truncated_prompt_tokens is computed and collated but never reported anywhere, so the RUNBOOK check 'prompt truncation is rare and balanced' has no supporting metric.**

> Status: UNVERIFIED.

- Breaks: prompt-only pooling so answer tokens cannot leak into intermediate targets
- Evidence: Computed at src/dast/tokenization.py:96, stored at :117, emitted into the batch at :194 as 'truncated_prompt_tokens'. grep for 'truncated_prompt_tokens' across src/dast returns only those tokenization.py sites - training.py, artifacts.py and evaluation.py never read the key, so it is dropped by _move_batch (src/dast/training.py:118-123) and never reaches metrics or run_manifest.json. RUNBOOK.md:5 nonetheless instructs the operator to 'Check after both runs: ... prompt truncation is rare and balanced'.
- Proposed minimal patch: Accumulate sum/max of batch['truncated_prompt_tokens'] in the training loop and write 'data/truncated_prompt_tokens_total' and '_max' into the metrics stream and run manifest.

**8. [HYGIENE] Masked mean pooling multiplies rather than selects, so a NaN or Inf at any assistant or padding position propagates into the pooled semantic state.**

> Status: UNVERIFIED.

- Breaks: prompt-only pooling so answer tokens cannot leak into intermediate targets
- Evidence: src/dast/pooling.py:21-22 computes `(hidden * weights).sum(dim=1) / counts`, and 0.0 * NaN = NaN, 0.0 * Inf = NaN. MEASURED: with mask [[1,1,1,0,0,0]] and NaN (or Inf) written into the three masked positions, pool_prompt_hidden(..., 'mean') returns non-finite output; the same perturbation leaves mode='last_prompt' exactly unchanged. The leakage test at tests/test_pooling_semantic.py:13 uses the finite value 1_000_000 and therefore cannot detect this. Under bf16 training (configs/base.yaml:15 precision: auto) an overflow anywhere in the assistant region of hidden_states would silently poison the state and contrastive losses (src/dast/losses.py:112-125) rather than being isolated by the mask, contradicting the module docstring 'Leakage-resistant pooling of hidden states over prompt tokens only' (pooling.py:1) and ALGORITHM.md:19-24.
- Proposed minimal patch: Replace the multiply with a select in src/dast/pooling.py:21-22: `masked = torch.where(mask.unsqueeze(-1), hidden, hidden.new_zeros(()))` then sum/divide.

**9. [HYGIENE] model.pooling is not validated by DASTConfig.validate(), so a typo silently reaches the first forward pass instead of failing at config load.**

> Status: UNVERIFIED.

- Breaks: prompt-only pooling so answer tokens cannot leak into intermediate targets
- Evidence: src/dast/config.py:71 declares `pooling: Literal["mean", "last_prompt"] = "mean"` as a type annotation only; DASTConfig.validate() (src/dast/config.py:199-260) checks checkpoint_fractions, max_length, feedback_scale, state_dim, projector_hidden_dim, stage_weights, arm/operator consistency and training scalars, but never `self.model.pooling`. The value is passed straight through src/dast/cli.py:77 -> DASTModel.__init__ (src/dast/model.py:128) and is only rejected at src/dast/pooling.py:28, i.e. after the model and dataloaders are built and the run manifest has been written (src/dast/training.py:334).
- Proposed minimal patch: Add to src/dast/config.py:validate(): `if self.model.pooling not in {"mean", "last_prompt"}: errors.append(f"unsupported model.pooling: {self.model.pooling}")`.

### Challenges to the lead's own findings (4)

| Claim | Position | Reasoning |
|---|---|---|
| F1 | `CONFIRMED` | Reproduced exactly on this venv. transformers 5.15.0 returns a BatchEncoding (a Mapping) from apply_chat_template(tokenize=True); iterating it yields the string keys 'input_ids'/'attention_mask'. tokenization.py:34-41 has no Mapping branch: BatchEncoding has no .tolist, `value[0]` does not produce a list, and the final comprehension `[int(item) for item in value]` iterates the keys. Because tokenize_example (tokeniza |
| F4 | `CONFIRMED` | Confirmed the metadata-only half with a direct run, and it has a consequence the lead engineer did not list: this path is also the sole enforcement point for the 'exclude the initial state' contract. configs/base.yaml:26 points at problem_manifest.jsonl, whose rows carry only category/difficulty/domain/*_sha256/problem_id/split/step_count. ReasoningRecord.from_mapping (data.py:35-38) therefore rejects every row, so ` |
| F8 | `CONFIRMED` | Independently reproduced on the 60 real full-context records at the two levels that matter for target construction. Distinct RAW post-step states per record: 2 states x2, 3 states x24, 4 x20, 5 x8, 6 x3, 7 x1, 8 x2 -> 26/60 = 43.3% have fewer than K=4, matching the stated 42%. After alignment.py:47-57 maps them onto 4 stages, distinct stage-target TEXTS per record are: 4 x34, 3 x24, 2 x2 -> the same 26/60 = 43.3% of  |
| F6 | `OVERSTATED` | The direction half of F6 stands, but the magnitude half is neutralized by code the finding does not account for. SharedStateProjector's first layer is nn.LayerNorm(hidden_size) (semantic.py:20), applied to the pooled vector after an explicit upcast to fp32 (semantic.py:28). LayerNorm normalizes across the hidden dimension, so the reported elementwise-RMS gap (49/49/49.5 for layers 7/14/21 versus 3.06 for the post-fin |

---

## Dimension: `backends`

### Scope-contract verdicts

| Verdict | Scope item |
|---|---|
| `IMPLEMENTED_BUT_INEFFECTIVE` | toy/offline smoke backend and Qwen/LoRA real-model backend — toy/offline smoke backend |
| `IMPLEMENTED_BUT_INEFFECTIVE` | toy/offline smoke backend and Qwen/LoRA real-model backend — Qwen/LoRA real-model backend |

### Additional defects claimed (10)

**1. [BLOCKER] pyproject pins are inadequate in three ways: transformers has no upper bound (so the shipped pin admits the version that breaks the only real tokenization entry point), torch is not a core dependency despite being imported at module scope by dast.cli's transitive imports, and ruff is unbounded (so the lint gate resolves to a version that fails).**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: pyproject.toml:14-17 `dependencies = ["numpy>=1.24","pyyaml>=6.0"]`; pyproject.toml:20-26 `transformers>=4.51`, `peft>=0.10`; pyproject.toml:28 `ruff>=0.4`; pyproject.toml:29-32 `all` extra. Torch is imported at module scope by src/dast/model.py:8, src/dast/smoke.py:9, src/dast/tokenization.py:9, src/dast/toy.py:10, src/dast/training.py:13, src/dast/artifacts.py:15, src/dast/evaluation.py:13 — all reachable from src/dast/cli.py:11-16. Installed resolution in this environment: transformers 5.15.0, peft 0.20.0, ruff 0.16.5.
- Proposed minimal patch: In pyproject.toml pin `transformers>=4.51,<5` and `peft>=0.10,<0.21` in both [train] and [all]; move `torch>=2.1` (and numpy) into [project] dependencies or make dast.cli import torch-free for the validate-data path; pin `ruff>=0.4,<0.17` (or fix the import order).

**2. [BLOCKER] CI is gated red at the Lint step, so the compile/pytest/smoke steps that the package advertises as its qualification never execute in CI at all.**

> Status: UNVERIFIED.

- Breaks: toy/offline smoke backend (CI qualification)
- Evidence: .github/workflows/dast-ci.yml:41 `python -m ruff check src tests` runs before :43 `pytest -q` and :45 `python -m dast.cli smoke`. Reproduced: `ruff check src tests` with ruff 0.16.5 exits 1 with I001 at tests/test_tokenization.py:1. There is no `continue-on-error`, so steps 42-45 are skipped. PROGRESS.md:66 makes "Ruff, unit/integration tests, and toy smoke all pass" the precondition for the first real rehearsal, which is therefore unreachable without a code change.
- Proposed minimal patch: Reorder `from dast.tokenization import IGNORE_INDEX, CausalBatchCollator, tokenize_example` in tests/test_tokenization.py:6 (ruff --fix), or add `--select=E,F,B` / an isort exemption. Optionally move Lint after the tests so a style failure cannot mask functional status.

**3. [BLOCKER] The one content-bearing corpus file present in the audited worktree has no `dev` split, so even a correctly-pointed data.path cannot satisfy the shipped eval_split under the split-sealing policy.**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: The audited worktree contains exactly two files with compact_state_trace: experiments/depth_aligned_semantic_transitions/fixtures/tiny_reasoning.jsonl (13 records) and Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1/samples/paired_examples.jsonl (measured 60 records; splits train=30, test=30, dev=0). configs/base.yaml:28 sets eval_split: dev, so training.py:82 raises ValueError('no records found for eval split \'dev\''); switching to test hits PermissionError from config.py:348-351 via training.py:68. The real per-split corpus (data/full_context/{train,dev,test}.jsonl, measured 43200/2400/2400 with content fields) exists only in the main working tree C:\Users\clclc\RineyaExperiment, not in this worktree.
- Proposed minimal patch: Point configs/base.yaml:26 at data/full_context/, and split DataConfig.path into train_path/eval_path (or accept a directory + per-split filename convention) in src/dast/training.py:66-78 so one path per split is expressible.

**4. [VALIDITY] With quantization enabled, hf.py loads a bnb-quantized model without a device_map, and training.py then calls model.to(device) on it, which transformers blocks for bitsandbytes models — so the QLoRA path is not merely untested, it is wired to fail.**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: src/dast/hf.py:77-82 sets kwargs['quantization_config'] and only sets device_map if config.model.device_map is truthy (configs/base.yaml:17 device_map: null). src/dast/training.py:286-293 then takes the `device_map is None` branch and calls model.to(device). transformers/modeling_utils.py:3661-3686 guards cuda()/to() for QuantizationMethod.BITS_AND_BYTES. base.yaml:16 sets quantization: none so this is currently inert, but PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md:419 offers QLoRA as the 8 GiB fallback.
- Proposed minimal patch: In hf.py:77-82, when quantization != 'none' and model_cfg.device_map is falsy, force kwargs['device_map'] = {'': 0} (or set config.model.device_map) so training.py:294-303 takes the shard-aware branch instead of model.to().

**5. [HYGIENE] The documented install command cannot produce the QLoRA capability the docs claim, and the friendly error that was supposed to catch this is unreachable.**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: README.md:46 claims "LoRA/QLoRA-capable Hugging Face path" and PROGRESS.md:16 claims a "real Hugging Face/PEFT/QLoRA loader"; README.md:58 and docs/RUNBOOK.md:17 both instruct `pip install -e ".[all]"`, but pyproject.toml:29-32's `all` extra omits bitsandbytes (it lives only in the separate `quant` extra, pyproject.toml:27). src/dast/hf.py:31-35 only raises the "quantization requires transformers and bitsandbytes" message on ImportError of transformers.BitsAndBytesConfig, which is importable even with bitsandbytes absent (verified: `hasattr(transformers,'BitsAndBytesConfig')` is True while `import bitsandbytes` raises ModuleNotFoundError in this environment), so the failure surfaces later and less clearly from from_pretrained.
- Proposed minimal patch: Add "bitsandbytes>=0.43" to the `all` extra, and in hf.py:31-35 also probe `import bitsandbytes` (or check transformers.utils.is_bitsandbytes_available()) before returning a BitsAndBytesConfig.

**6. [HYGIENE] cli.py loads the full base model plus LoRA adapter (onto the GPU) before any dataset record is read, so the shipped-config data failure only surfaces after a complete model load, once per arm.**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: src/dast/cli.py:61-62 calls build_hf_components(config, for_training=True) immediately after load_config; the first dataset access is src/dast/training.py:284 -> :69 load_records. scripts/run_dev_matrix.sh:8-10 loops this over six arms.
- Proposed minimal patch: In cli.py:command_train, call dast.data.load_records / validate_dataset (or a build_dataloaders dry run) on config.resolved_path(config.data.path) before build_hf_components.

**7. [HYGIENE] hf.py's fp32_trainable_params promotion is a no-op for the shipped LoRA config because peft already upcasts adapters to fp32, so the manifest records fp32_trainable_tensors_promoted: 0 — which reads as 'promotion did not happen'.**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: src/dast/hf.py:142-148 promotes only parameters whose dtype is float16/bfloat16 and requires_grad; get_peft_model at hf.py:130 defaults autocast_adapter_dtype=True (peft/peft_model.py:144, :166-168) which calls _cast_adapter_dtype -> cast_adapter_dtype, documented at peft/tuners/tuners_utils.py:590-603 as upcasting float16/bfloat16 to float32. configs/base.yaml:75 sets fp32_trainable_params: true.
- Proposed minimal patch: Either record the resolved trainable dtypes instead of a promotion count in hf.py:148, or note in the manifest that peft performed the cast.

**8. [HYGIENE] The generation attention mask is derived by value comparison against pad_token_id, which is unsafe under hf.py's own pad=eos fallback, and is inconsistent with the length-based mask the training collator builds.**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: src/dast/evaluation.py:136 `attention_mask = input_ids.ne(pad_id).long()` versus src/dast/tokenization.py:170 which builds the mask from the true sequence length. src/dast/hf.py:64-67 aliases pad_token to eos_token when pad is None; README.md:63 explicitly invites swapping in "an audited local Rineya/R2 checkpoint". Under that fallback every in-prompt EOS/turn-end token would be masked out of the prompt during generation only. Not triggered by Qwen/Qwen3-0.6B (measured pad 151643 != eos 151645) nor by artifacts/attn_shape/r2-gqa1kv (pad 32742 != eos 32744).
- Proposed minimal patch: Build the mask from the per-row prompt lengths already available at evaluation.py:133-135 instead of ne(pad_id).

**9. [HYGIENE] CI covers only ubuntu-latest and never installs the [train] extra, so neither the Windows launcher nor any transformers/peft import is exercised, yet README lists Windows launchers and CI as delivered qualification.**

> Status: UNVERIFIED.

- Breaks: toy/offline smoke backend (CI qualification)
- Evidence: .github/workflows/dast-ci.yml:22 `runs-on: ubuntu-latest` (single job), :34-37 installs CPU torch and `-e ".[dev]"` only. scripts/run_smoke.ps1 and scripts/run_dev_matrix.ps1 are never invoked. README.md:49 lists "Linux/Windows launchers, and CI" under "What is implemented"; README.md:72 documents scripts\run_smoke.ps1.
- Proposed minimal patch: Add a windows-latest matrix leg running scripts/run_smoke.ps1, and add an import-only job that installs `.[train]` and asserts build_hf_components' library seam (e.g. a tokenizer-only chat-template contract test) so the transformers pin is actually resolved in CI.

**10. [HYGIENE] hf.py passes the deprecated torch_dtype kwarg rather than dtype; it still works on transformers 5.15 via a back-compat shim, so this is hygiene only — recording it explicitly to close off a plausible-but-wrong hypothesis.**

> Status: UNVERIFIED.

- Breaks: Qwen/LoRA real-model backend
- Evidence: src/dast/hf.py:80 `kwargs["torch_dtype"] = dtype`. transformers/modeling_utils.py:4108 `torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC` and :4133-4136 merge it into `dtype` with no exception; the deprecation warning at modeling_utils.py:1489-1492 is on the _from_config path, not from_pretrained. So this is not a break.
- Proposed minimal patch: Rename to kwargs["dtype"] in hf.py:80 once the transformers floor is >=4.56.

### Challenges to the lead's own findings (9)

| Claim | Position | Reasoning |
|---|---|---|
| F1 | `OVERSTATED` | The mechanism is fully confirmed, but the scope clause "every real-model path dies in _as_int_list" is false for the shipped configs: no real-model path ever REACHES _as_int_list, because configs/base.yaml:26 points at a metadata-only manifest and data.py:35-38 raises on the very first record inside build_dataloaders (training.py:69), which runs before the collator ever tokenizes anything (training.py:284 precedes th |
| F2 | `CONFIRMED` | The device mismatch is real and I located it exactly. smoke.py:179-182 snapshots model.state_projector parameters while the DASTModel is still on CPU (smoke.py:100-124 never moves it); train() then unconditionally moves it, because ModelConfig.device_map defaults to None (config.py:67) and build_smoke_config never sets it, so training.py:286-293 takes the `model.to(device)` branch with device=cuda whenever torch.cuda |
| F3 | `CONFIRMED` | Reproduced exactly, and the consequence is larger than the claim states: because the Lint step precedes the test and smoke steps in the workflow with no continue-on-error, the entire advertised offline qualification is currently unreachable in CI. |
| F4 | `CONFIRMED` | Both halves hold, and the second half is stronger than stated. Half one: the configured file is metadata-only, so load_records fails on record 1. Half two: the canonical corpus really is one file per split, and build_dataloaders structurally cannot address it — a single data.path plus split filtering means whichever of train.jsonl / dev.jsonl / test.jsonl you name, the other split's loader gets zero records and train |
| F5 | `OVERSTATED` | The right-padding defect is real, but 'the misaligned decode window' is the wrong diagnosis. With right padding, generate() returns [prompt_i / pad*(width-L_i) / new tokens], so slicing at the batch-max width recovers exactly the newly generated tokens — the decode window is correct and self-consistent. The actual defect is upstream: the model is asked to predict from a PAD position for every row shorter than the bat |
| F6 | `CONFIRMED` | Confirmed independently, and I can now name the exact library mechanism rather than just the symptom, which matters because it is version-specific to the transformers 5.x output-capture rewrite. The hidden_states tuple is assembled by forward hooks on Qwen3DecoderLayer, so indices 1..N are raw post-block residual states, and then the last entry is deliberately overwritten with last_hidden_state, which is the post-fin |
| F7 | `CONFIRMED` | Confirmed both halves, and the backend-relevant addition is why it was never caught: the offline smoke deliberately uses batch_size 2, so the only automated path that ever executes compute_loss is the one configuration in which the anti-collapse term is NOT inert. Green CI therefore cannot certify the shipped setting. |
| F8 | `CANNOT_VERIFY` | This is a data-dimension statistic and I deliberately did not re-measure it, but I can flag a provenance problem the parent should resolve: the file configs/base.yaml names cannot support the measurement at all, and the audited worktree contains no 48k file with compact_state_trace. The only two content-bearing files in this worktree are the 13-record fixture and a 60-record sample; the 43200/2400/2400 per-split corp |
| F9 | `CANNOT_VERIFY` | The truncation and extraction measurements require a real generation run, which I did not perform. The code-level preconditions are all confirmed, so the claim is mechanically plausible: max_new_tokens is 128 with no min_new_tokens and no stopping criterion other than a single eos_token_id, and the extractor's third route is an unguarded last-nonempty-line fallback that will happily return a truncated fragment. Two c |

---

Total additional defects claimed across all dimensions: **80**.
