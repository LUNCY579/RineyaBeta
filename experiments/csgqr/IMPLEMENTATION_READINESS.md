# CSGQR — Implementation Readiness Gate (Stage II)

Assessed against the campaign brief §26. Every requirement is marked `PASS`,
`PARTIAL` or `FAIL`, with the evidence that decides it. Evidence names a check in
`src/verify_csgqr.py` (**144 checks, all passing**), a committed result file, or a
measurement recorded in this session.

---

## Architecture

| Requirement | Status | Evidence |
| --- | --- | --- |
| Correct MLP location | `PASS` | `layers[j].mlp` replaced. Verified against the **installed** `transformers 5.15.0` `Qwen3DecoderLayer.forward` source this session, not against memory of it. |
| Correct norm and residual ordering | `PASS` | The remnant consumes the post-norm tensor; the gate and router read the pre-norm residual. Two distinct tensors from two distinct points; a shape mismatch on capture raises. The module performs no residual arithmetic — the decoder layer still does it, once. |
| Separate modes | `PASS` | 25 arms in `variants.CATALOGUE`, each a `RoutingPolicy`. Capability is derived from the policy (`needs_lens`, `needs_probe`, `needs_group_utility`) and cross-checked against the `NEEDS_ROUTER`/`NEEDS_PROBE` lists, so an arm cannot silently be a different arm. |
| Valid generation | `PASS` | `run_eval.py --self-check` reproduces `artifacts/gate_runs/r2_base_chat_dev` at **\|Δ\| = 0.000e+00** over 510 cases. |

## Quantization

| Requirement | Status | Evidence |
| --- | --- | --- |
| Real INT8 | `PASS` | pack/unpack bijection; byte accounting exact. |
| **Packed** INT4 | `PASS` | Two values per byte, low nibble first. Measured **4,939,872 B** against 18,874,464 B bf16 = **3.821×**, and the analytic byte model predicts the measured figure exactly. Bijection verified at even *and odd* widths. |
| Storage accounting | `PARTIAL` | `packed_bytes()` itemises weight / scales / zeros; `budget.py` gives the analytic model; the two agree. Optimizer-state bytes are structurally zero (buffers, not parameters). **Dequantization workspace and GPU-resident deltas are not yet measured** — that is `bench_gpu.py`, which has not run. |
| Frozen donor | `PASS` | Packed storage is registered as **buffers**, and integer dtypes cannot carry gradients at all — a stronger guarantee than a flag. `no gradient can be created for packed storage`; `quantized matrix retains no fp32/bf16 weight buffer and no parameters`. |
| No hidden dense donor | `PASS` | `keep_donor_resident=False` deletes the three projections; `remove()` then refuses rather than reinstalling a hollow module. The **deployment artifact** was tested this session: it reloads into a module built from an *unrelated* MLP and reproduces full and partial outputs bit-exactly, all three packed matrices are uint8, and the file is the accounted packed bytes plus only container overhead. |

## Groups

| Requirement | Status | Evidence |
| --- | --- | --- |
| Additive decomposition | `PASS` | `F = Σ_g F_g` exact in float64 to **< 1e-14**. A biased `down_proj` is **refused**, because such a bias would be counted once per active group and would scale with `k_t` — looking exactly like a routing effect. |
| Selected execution (executed, not masked) | `PASS` | `PackedMatrix.dequantize(lo, hi)` slices packed bytes, scales and zero points *before* any arithmetic; `select` returns ids and nothing downstream sees an inactive group. `row-slice dequantization equals the full dequantization on that slice`, bit-identical. |
| Non-divisible groups | `PASS` | Remainder folded into the last group; `group_flops(tokens, group)` and `active_flops` report true sizes. Verified on 100 = 6×14 + 16. |
| Measured cost | `PARTIAL` | Analytic `active_flops` verified. **Wall-clock per-group cost not yet measured on an exclusive card.** One incidental measurement exists and is informative: a per-token-varying selection (`b5-random-25`) took **622 s** against **39 s** for a fixed group set at the same budget on the same prompts — a **16×** reference-backend penalty for per-token bucketing. |

## Confidence

| Requirement | Status | Evidence |
| --- | --- | --- |
| Correct state | `PASS` | Pre-norm post-attention residual via `forward_pre_hook` on `post_attention_layernorm`, whose input *is* that tensor. `CSGQRMlp` raises rather than falling back to the post-norm tensor. Arm-independence of that state measured at max \|Δ\| = **0.000e+00** over 798 positions. |
| Correct projection | `PASS` | Model's own final norm and LM head, **borrowed not copied** (a duplicated 32768×1024 head would be 64 MiB against a 4.7 MiB remnant). |
| Calibration separation | `PASS` | `fit_temperature` on held-out data; the module never reads a generation config. Confidence is computed from raw lens logits before any sampling transform. |
| Current-token-only at runtime | `PARTIAL` | Prefill **is** gated by design — skipping a prefill position changes its KV for every layer above, so gating decode only would be a different method — so every gated position genuinely needs a signal. The transient is bounded by chunking at 256 positions rather than by sequence length. Unchunked it was ~1.6 GiB at 2048 tokens on an 8 GiB card (BUG_LEDGER **S8**). |
| No retained vocabulary tensors | `PASS` | `features_from_hidden` returns scalars only, under `@torch.no_grad()`. The chunked reduction is **bit-exact** against the unchunked one; end-to-end agreement is at GEMM epsilon with **zero** argmax changes. |

## Utility

| Requirement | Status | Evidence |
| --- | --- | --- |
| Paired traces | `PASS` | Installed once, policy swapped between forwards, so every group is measured against **the same packed weights**. Identical prompts, positions, gold tokens, weights and context. |
| Documented divergence direction | `PASS` | `U = KL(p^T‖p^0) − KL(p^T‖p^cand)`, stated once, never reversed. Larger is better. |
| Synthetic validation | `PASS` | Sign convention, exposure normalisation and ranking controls tested on distributions with known utility. |
| Reproducible group utilities | `PASS` | 798 positions × 24 groups committed with a manifest. `subset_policy` now **asserts** that the policy selects exactly the requested groups, closing the `fraction = k/n` rounding hole (**S7**) that would have mislabelled subset utilities. |

## Router

| Requirement | Status | Evidence |
| --- | --- | --- |
| Hard skip | `PASS` | `empty group set returns exact zeros without touching any weight`. |
| Correct full mode | `PASS` | All-group adaptive equals the full quantized remnant. |
| Correct partial mode | `PASS` | Selected groups equal explicit manual slicing; coalesced run-merged gather is **bit-identical** to a per-group gather across six group sets. |
| Independent baselines | `PASS` | random / magnitude / shuffled / reversed / contiguous, plus permuted-label and prompt-mean-oracle probe controls. All written before any CSGQR result was inspected. |
| Deterministic behaviour | `PASS` | `stable=True` sort — ties resolve to ascending group id (**S6**). `fraction = k/n` recovers exactly `k` for all 91 tested cases (**S7**). Random selection is reproducible per token. Dynamic significance selects the same groups at every confidence, so the removed `(1−C)^γ` degeneracy cannot recur (**S5**). |

## Evaluation

| Requirement | Status | Evidence |
| --- | --- | --- |
| Original protocol preserved | `PASS` | Protocol read from `manifests/eval_assets.json`; nothing about decoding is configurable. Self-check exact. |
| Data roles separated | `PASS` | Probes fitted on the trace's prompts; `eval_forced` **refuses** a probe-gated run whose prompts overlap the probe's training trace, naming the `--skip-prompts` value. Demonstrated this session. All held-out results use `--skip-prompts 96`. |
| Sealed test untouched | `PASS` | `rag_test_SEALED`, `puzzle_test_SEALED`, `puzzle_challenge_SEALED` never opened; both eval scripts refuse a path containing `SEALED`. |

## Systems

| Requirement | Status | Evidence |
| --- | --- | --- |
| Synchronized CUDA timing | `FAIL` | `bench_gpu.py` implements it correctly — warmup, `torch.cuda.synchronize()` around every timed region, P50/P90/P95 — and **has never been executed**. |
| Allocated and reserved VRAM | `PARTIAL` | Both recorded in all ten Tier-1 manifests. Not yet isolated per component. |
| Confidence overhead separated | `PASS` (analytic) / `FAIL` (measured) | Closed form `\|V\|/(3·d_ff) = 3.556×` asserted to exceed 1 on this checkpoint; probe alternatives 0.0109% and 0.2604% of the gated MLP. The measured comparison is implemented in `bench_gpu.py` and not run. |
| Dequantization overhead separated | `FAIL` | Implemented in the microbenchmark, not run. |
| No host spill | `FAIL` | Not verified. Windows/WDDM spills past VRAM into host RAM instead of raising, turning an over-budget run into a ~100× slowdown rather than an error. |

**Why the systems block fails, and why it is not blocking the science.** During
this session a *different* experiment (`experiments/kv_significance/src/passkey.py
--lengths 131072`) ran concurrently on the same 8 GiB card, contending for VRAM
and SMs. It caused a `CUBLAS_STATUS_EXECUTION_FAILED` mid-sweep. Latency and
memory measured under contention are meaningless, so taking them opportunistically
would have produced a number that looked like evidence and was not. Every systems
requirement above is gated on an exclusive card.

This does **not** gate the mechanism experiments, for a reason fixed in advance:
one MLP layer is 1.99% of this model, INT4 on it saves 1.47% of model VRAM, and
even `ρ = 0` removes only 1.99% of model MACs. No end-to-end systems claim is
available at this scope whatever the measurement says, and none is made.

## Reproducibility

| Requirement | Status | Evidence |
| --- | --- | --- |
| Hashes | `PASS` | Model weights sha256 and config sha256 in every result file's `environment` block; dataset sha256 in run manifests. |
| Environment | `PASS` | `envinfo.collect()` embedded in every result: python, platform, torch **and its CUDA build**, transformers, peft, GPU name/capability/VRAM, driver, and an `nvidia-smi` cross-check. `require_cuda()` refuses rather than falling back to the CPU-only ambient torch. |
| Seeds | `PASS` | Fixed and recorded. Trainable components are closed-form (ridge) or deterministic full-batch L-BFGS from a fixed zero init, so there is no optimiser seed to vary. |
| Configs | `PASS` | Every arm is a serialized `RoutingPolicy`; `to_dict` now includes `group_costs` (**S9**). |
| Git SHA | `PASS` | SHA **and dirty flag** in every result file. |
| Clean deployment reload | `PASS` | Verified this session — see "No hidden dense donor". |

---

## Implementation confidence grade

Reported separately, as the brief requires.

| Aspect | Grade | Justification |
| --- | --- | --- |
| **Reference backend** | **A — high confidence** | 144 deterministic checks pass. The load-bearing invariants are *exact*, not approximate: additivity < 1e-14 in float64, row-slice dequantization bit-identical, coalesced gather bit-identical, deployment reload bit-exact, and on the primary endpoint the FP-passthrough arm is numerically identical to dense (KL −0.00000, CI ±1e-5, argmax agreement 1.0000) — a **zero** noise floor. Eight rows that were `INCORRECT` at the start of this session are fixed with a regression test each. |
| **Optimized backend** | **C — partial** | There is no optimized backend. Contiguous-run coalescing is the only optimisation and it is bit-identical, so it is not a second numerical path. No Triton, no fused dequantize-GEMM, no CUDA graphs. The one measured cost — 16× for per-token bucketing versus a fixed group set — is a reference-backend number and must not be read as the algorithm's limit. |
| **Experimental interpretability** | **B — substantial, with one bounded limitation** | The primary endpoint is paired at the reply position with prompt-clustered intervals, and its power is demonstrated rather than assumed: it separates INT8 (KL 0.00088 [0.00085, 0.00090]) from INT4 (0.01049 [0.01009, 0.01090]) with non-overlapping intervals, where the generation scorer could not separate deletion from dense. The bounded limitation is that the *generation* endpoint has no power here and its Tier-1 table cannot be used as evidence about routing — established with the arithmetic, not asserted. |

Grade for a strong algorithmic verdict must be at least `B`. The reference
backend is `A` and interpretability is `B`, so a verdict is admissible on the
mechanism. It is **not** admissible on systems performance, where the grade is
`C` and the measurement is absent.

---

## One caveat that limits every grade above

**This is not an independent audit.** The branch contained no CSGQR
implementation when the campaign began — `origin/research/csgqr-quantized-remnant`
was byte-identical to `origin/r2-healing-harness`, re-verified this session at 0/0
ahead-behind with an empty diff. The implementation graded here was written as
part of the campaign. No amount of testing repairs that; see
`LIMITATIONS.md` §4 for the structural compensations, of which the load-bearing
one is that several of this session's fixes were found by a **test failing**
rather than by reading code. The per-group α bug is the clearest case: it is
invisible in source review and only shows up when you demand that a *bit-exact*
remnant return exactly 1.

---

## Verdict

Mechanism experiments: every architecture, quantization, group, confidence,
utility, router, evaluation and reproducibility requirement is `PASS` or a
`PARTIAL` whose gap is measurement-not-yet-taken rather than
behaviour-not-yet-correct. No core row is `MISSING`, `INCORRECT` or unexplained.

Systems experiments: the timing and memory block is `FAIL`, blocked on an
exclusive GPU, and no systems claim is made.

# `READY_FOR_SCIENTIFIC_EXPERIMENTS`

with the explicit carve-out that **Gate F (systems value) is not assessable from
committed evidence** and remains gated on `bench_gpu.py` running on an
uncontended card.
