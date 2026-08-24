# CSGQR — Specification-to-Code Matrix

Every requirement in the campaign specification, the equation or behaviour it
asks for, what the code actually does, and the evidence. Status is one of
`CORRECTLY_IMPLEMENTED`, `PARTIALLY_IMPLEMENTED`, `MISSING`, `INCORRECT`,
`AMBIGUOUS`, `UNVERIFIED`.

Evidence names a check in `src/verify_csgqr.py` (112 checks, all passing), a
committed result file, or a source location. "Asserted" means a runtime check
refuses the wrong behaviour rather than a test observing the right one.

Checkpoint under test: `artifacts/rineya-core-en32k-28L` — Qwen3, 28 layers,
`d_model` 1024, `d_ff` 3072, `|V|` 32768, tied embeddings, bf16.

---

## A. Execution modes (§3)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| `dense` | `h' = h + F_j(h)` | no surgery; `b0-dense` | `CORRECTLY_IMPLEMENTED` | `run_eval.py --self-check` reproduces `artifacts/gate_runs/r2_base_chat_dev` to \|Δ\| = 0.000e+00; `eval_forced` gives the dense arm KL exactly 0 and agreement exactly 1.0 as a null check | — |
| `deleted` | `h' = h` | `mode="none"`; `b1-delete` | `CORRECTLY_IMPLEMENTED` | bit-identical to a true MLP bypass (max logit deviation 0.0); `empty group set returns exact zeros without touching any weight` | — |
| `fp_frozen` | `h' = h + F_j(h)`, frozen, through the remnant path | `QuantSpec(16,0,True)` passthrough; `b2-fp-all` | `CORRECTLY_IMPLEMENTED` | `fp passthrough remnant reproduces the source MLP bit-exactly`; layer scan FP KL exactly 0.0 at all 28 layers | — |
| `int8_all` | packed INT8, always on | `QuantSpec(8,128,False)`; `b3-int8-all` | `CORRECTLY_IMPLEMENTED` | `8-bit pack/unpack is a bijection`; `analytic remnant byte model matches the built module` | — |
| `int4_all` | **packed** INT4, always on | `QuantSpec(4,128,False)`; `b4-int4-all` | `CORRECTLY_IMPLEMENTED` | `4-bit pack/unpack is a bijection at even and odd widths`; 4,939,872 B measured vs 18,874,464 B bf16 = **3.821×**; two values per byte asserted by the byte count, not by a comment | — |
| `random_groups` | matched-compute control | `selector="random"`, per-token seeded; `b5-random-25/50` | `CORRECTLY_IMPLEMENTED` | `random selection is reproducible per token and varies across tokens`; measured `active_fraction` exactly 0.2500 | — |
| `magnitude_groups` | structural heuristic control | `magnitude_ranking` over dequantized `down` rows; `b6-magnitude-25` | `CORRECTLY_IMPLEMENTED` | ranking recorded in each run manifest | — |
| `static_significance` | fixed global ranking | `selector="static_topk"` with a ranking from the trace; `b6-static-25/50` | `CORRECTLY_IMPLEMENTED` | `static_topk follows the supplied ranking` | — |
| `confidence_only` | gate only, whole remnant | `mode="confidence_whole"`; `b7-conf-whole`, plus entropy/margin/logit-margin variants | `CORRECTLY_IMPLEMENTED` | `confidence_whole is all-or-nothing`; measured skip rate 0.2066 at τ=0.90 | — |
| `significance_only` | group selection alone, **per state** | `selector="significance"` + a trained router; `b9-sig-dyn-25/50` | `CORRECTLY_IMPLEMENTED` | **was `INCORRECT`** — the `fixed_fraction` fast path swallowed `significance` and ran the static ranking with no error (BUG_LEDGER **S1**). Now asserted: `CSGQRMlp` refuses to construct without a `utility_fn` | — |
| `adaptive_csgqr` | three-level + per-state significance | `mode="three_level"` + `selector="significance"`; `b16-csgqr-dyn` (probe gate), `b16-csgqr-lens-dyn` (lens gate) | `CORRECTLY_IMPLEMENTED` | `b10-csgqr` is retained under its own key but **relabelled** as the static form, because that is what it was | — |
| modes not collapsed | a difference between arms is a difference of policy | every arm is a `RoutingPolicy`; `variants.CATALOGUE` holds 25 | `CORRECTLY_IMPLEMENTED` | `variants.py`; capability flags derived from the policy, cross-checked against the `NEEDS_ROUTER`/`NEEDS_PROBE` lists | — |

---

## B. Architecture and integration (§4, §8)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| MLP hook location | replace the MLP sub-block only | `layers[j].mlp` replaced by `CSGQRMlp` | `CORRECTLY_IMPLEMENTED` | verified against `transformers 5.15.0` `Qwen3DecoderLayer.forward` source this session | — |
| pre-remnant hidden state | `h` = the tensor the residual add uses | `forward_pre_hook` on `post_attention_layernorm`, whose input **is** that tensor | `CORRECTLY_IMPLEMENTED` | read from the installed `transformers` source: `residual = hidden_states; hidden_states = post_attention_layernorm(hidden_states); hidden_states = mlp(...); hidden_states = residual + hidden_states`. `CSGQRMlp` raises rather than falling back to the post-norm tensor | — |
| RMSNorm placement | remnant consumes the **post-norm** input | `forward_groups(x)` where `x` is the MLP input; confidence and the probes read the **pre-norm** residual | `CORRECTLY_IMPLEMENTED` | two distinct tensors, taken from two distinct points; asserted by a shape check on capture | — |
| residual addition | done by the decoder layer, once | untouched — the module returns only `F_A(x)` | `CORRECTLY_IMPLEMENTED` | no residual arithmetic in `surgery.py` | — |
| structured groups | contiguous, hardware-aligned intermediate-dim groups | `group_bounds`; all three projections stored `[d_ff, d_model]` so a group is a contiguous **byte-aligned** row slice of each | `CORRECTLY_IMPLEMENTED` | `group_bounds` is an exact partition; `merged runs cover the whole intermediate axis exactly once` | — |
| group additivity | `F(h) = Σ_g F_g(h)` | exact by construction (bias-free projections, per-channel product, linear `down`) | `CORRECTLY_IMPLEMENTED` | exact in float64 to **< 1e-14**; `partial group execution == explicit intermediate-neuron selection`; a biased `down_proj` is **refused** | — |
| group sizes 32/64/128/256 | characterise granularity | `n_groups` is free; 3072 = 96×32 = 48×64 = 24×128 = 12×256 | `PARTIALLY_IMPLEMENTED` | all four divide evenly and are constructible; only `n_groups=24` (128 channels) has been **measured** | granularity sweep — `MAXIMUM_POTENTIAL_ROADMAP.md` |
| non-divisible groups | remainder handled, true extent reported | folded into the last group; `group_flops(tokens, group)` and `active_flops` report true sizes | `CORRECTLY_IMPLEMENTED` | `an uneven partition folds the remainder into the last group and reports its true size` (100 = 6×14 + 16); was `INCORRECT` — BUG_LEDGER **S14** | — |
| donor freezing | `∇_θ L = 0` | packed storage as **buffers**, integer dtypes; autograd cannot track them | `CORRECTLY_IMPLEMENTED` | `quantized matrix retains no fp32/bf16 weight buffer and no parameters`; `no gradient can be created for packed storage` | — |
| no hidden dense donor | deployment must not need the bf16 donor | `keep_donor_resident=False` deletes `gate/up/down_proj`; `remove()` then refuses rather than reinstalling a hollow module | `CORRECTLY_IMPLEMENTED` | BUG_LEDGER **C3**; `eval_forced` deliberately keeps the donor resident and **says so in its output JSON** (`memory_note`) so its numbers cannot be mistaken for memory evidence | — |

---

## C. Quantization (§10)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| INT8 | real 8-bit storage | one value per byte | `CORRECTLY_IMPLEMENTED` | bijection test; byte accounting exact |
| packed INT4 | **two values per byte** | low nibble even index, high nibble odd | `CORRECTLY_IMPLEMENTED` | 1024 values → 512 bytes; measured 4,939,872 B total = 3.821× vs bf16 |
| INT3 / INT2 | later extension | accepted by `QuantSpec`, stored **one value per byte** | `PARTIALLY_IMPLEMENTED` | `values_per_byte` returns 1 and the docstring says the storage claim is therefore not validated. Honest, not silent | bit-packer if the sweep is run |
| NF4 / FP4 | later extension | not present | `MISSING` | — | roadmap |
| symmetric / asymmetric | policy stated | both; `QuantSpec.symmetric` | `CORRECTLY_IMPLEMENTED` | `asymmetric form preserves an isolated exact zero` |
| signed representation | explicit | symmetric uses `[-7,7]` biased by +8 → storage `[1,15]`, zero maps to 8 exactly | `CORRECTLY_IMPLEMENTED` | documented in `quant.py`; costs one level to keep zero exact |
| range / clipping | explicit | `amax` over **valid** entries only; `clamp(-qmax, qmax)` | `CORRECTLY_IMPLEMENTED` | padding is NaN-filled so it cannot influence a block's range |
| scale calculation | float32 regardless of weight dtype | search in float32, stored float16 | `CORRECTLY_IMPLEMENTED` | bf16 has 8 mantissa bits and would quantize the scale itself |
| scale granularity | blockwise along the input dim | `block=128` default, swept to 64 | `CORRECTLY_IMPLEMENTED` | `smaller quantization blocks reduce error` |
| zero handling | no division by zero | all-zero block → `scale = 1`, quantizes to exactly zero | `CORRECTLY_IMPLEMENTED` | `all-zero blocks and rows dequantize to exact zero (no div-by-zero)` |
| odd value counts | byte-aligned, padding never read | odd row → one padding nibble, written zero | `CORRECTLY_IMPLEMENTED` | bijection test at odd widths |
| non-divisible group sizes | real block remainder | 1000 = 7×128 + 104 tested | `CORRECTLY_IMPLEMENTED` | `non-divisible input dim quantizes` |
| packing / unpacking | inverse | `_pack` / `_unpack` | `CORRECTLY_IMPLEMENTED` | bijection at 4 and 8 bits |
| serialization | deterministic | buffers, so `state_dict` carries them; padding zeroed so bytes are deterministic | `PARTIALLY_IMPLEMENTED` | codes on padding are explicitly zeroed | save/reload round-trip test — Stage II |
| dequantization dtype | compute dtype | float32 arithmetic, cast to `compute_dtype` | `CORRECTLY_IMPLEMENTED` | `remnant.py` |
| **selected-group** reconstruction | slice **before** any arithmetic | `PackedMatrix.dequantize(row_lo, row_hi)` slices packed bytes, scales and zeros | `CORRECTLY_IMPLEMENTED` | `row-slice dequantization equals the full dequantization on that slice`, bit-identical |
| selected-group multiplication | inactive rows never in a matmul | contiguous runs coalesced, one dequantize + one GEMM per run | `CORRECTLY_IMPLEMENTED` | `coalesced run-merged gather is bit-identical to a per-group gather`; BUG_LEDGER **S13** |
| storage accounting | ten separate quantities | `packed_bytes()` itemised; `budget.py` for the analytic model | `PARTIALLY_IMPLEMENTED` | `analytic remnant byte model matches the built module` (4,939,872 B predicted = measured) | optimizer-state and workspace bytes are 0/not-yet-measured; `bench_gpu.py` covers the rest |

---

## D. Remnant gain calibration (§11)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| no calibration | `α = 1` | default | `CORRECTLY_IMPLEMENTED` | every committed Tier-1 manifest shows `alpha[0] = 1.00000` |
| one scalar per MLP | `α = E⟨F^Q,F⟩ / (E‖F^Q‖²+ε)` | `calibrate_alpha` | `CORRECTLY_IMPLEMENTED` | matches the closed form to < 1e-5 across a 2500× activation-scale span; **was `INCORRECT`** — absolute ε was 13% of the denominator (BUG_LEDGER **C1**) |
| one scalar per group | `α_g` against **`F_{j,g}`** | `per_group=True` fits against `reference_group_output` | `CORRECTLY_IMPLEMENTED` | **was `INCORRECT`** — fitted against the whole `F`, giving 0.978–1.024 on a *bit-exact* remnant and injecting 1.32% error (BUG_LEDGER **S2**). Now `α_g` is exactly 1 for a lossless remnant |
| per-output-channel | optional | not present | `MISSING` | — | only worth adding if the scalar and per-group forms diverge; they do not at INT4 (α ≈ 0.997) |
| partial-execution gain | not in the spec, but implied by §18–19 | `subset=[...]`, `against="rows"` or `"whole"` | `CORRECTLY_IMPLEMENTED` | new. `results/partial_gain_L20.json`: α_whole is 0.99–1.10, never near 1/ρ, so groups are near-orthogonal and calibration moves the output error by ~0.1% |
| ε is not absolute | scale-free | zero-check, float64 accumulation | `CORRECTLY_IMPLEMENTED` | `a remnant that outputs exactly zero yields alpha=1, not NaN` |
| idempotent | re-fitting must not compound | current `alpha` neutralised during the fit, restored in `finally` | `CORRECTLY_IMPLEMENTED` | `alpha calibration is idempotent`; BUG_LEDGER **S3** |
| calibration data disjoint | fitted on calibration data only | `run_eval.py` draws from the **evaluation** records | `INCORRECT` | BUG_LEDGER **S15**, **OPEN**. Latent: no committed run used `--calibrate-alpha`. `calibrate_partial.py` uses disjoint-by-prompt fit/eval splits and asserts it | a dedicated calibration split |

---

## E. Confidence lens (§12–14)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| exact lens | `z̃ = W_LM · N_f(A_j h)` | `ConfidenceHead.logits`, borrowing the model's own norm and head | `CORRECTLY_IMPLEMENTED` | `ConfidenceHead borrows the model's LM head rather than copying it` — a copied 32768×1024 head would be 64 MiB and swamp a 4.7 MiB remnant |
| raw lens (`A = I`) | reference arm | `AlignmentLens(rank=0)` | `CORRECTLY_IMPLEMENTED` | this is the arm that was measured |
| low-rank lens | trained `A_j`, zero-init | `AlignmentLens(rank>0)`, `up` zero-initialised so it starts as exact identity | `PARTIALLY_IMPLEMENTED` | constructible and identity at init; **never trained** | formally `NOT_ESTABLISHED`; §N1 makes it economically pointless at this scope regardless |
| nonlinear lens | only if needed | not present | `MISSING` | — | not warranted: the probe skips the projection entirely |
| temperature calibration | `T_j` on held-out data, separate from generation temperature | `fit_temperature`, golden-section on NLL; the module never reads a generation config | `CORRECTLY_IMPLEMENTED` | `fit_temperature recovers a synthetic miscalibration`; `higher temperature lowers confidence` |
| max probability | `C^max` | `features_from_logits` | `CORRECTLY_IMPLEMENTED` | `a near-one-hot distribution gives neg_entropy 1 and max_prob 1` |
| normalized entropy | `1 − H/log|V|`, full vocabulary | `neg_entropy` | `CORRECTLY_IMPLEMENTED` | `entropy is normalised by log|V| over the FULL vocabulary` |
| probability margin | `p⁽¹⁾ − p⁽²⁾` | `prob_margin` | `CORRECTLY_IMPLEMENTED` | `probability features stay inside [0,1] and margins are non-negative` |
| logit margin | `z⁽¹⁾ − z⁽²⁾` | `logit_margin` | `CORRECTLY_IMPLEMENTED` | not confined to [0,1]; its arm uses a threshold in logit units |
| prediction stability | `1 − JS(p_{j−d}, p_j)` | `js_divergence`, `stability` | `PARTIALLY_IMPLEMENTED` | implemented and unit-tested | not wired into any arm as a gate feature |
| learned combined confidence | `σ(β·features)` | `probe_utility.py` fits logistic on the four lens features | `PARTIALLY_IMPLEMENTED` | measured (AUROC 0.580, i.e. no better than the best single raw feature) | not serialized as a deployable gate; superseded by the state probe |
| computed before sampling transforms | before top-k/top-p/penalties | the module reads raw lens logits and never a generation config | `CORRECTLY_IMPLEMENTED` | generation is greedy anyway (`do_sample=False`) |
| current position only at deployment | do not project the whole prefill | **all** gated positions are projected, but **chunked** at 256 | `PARTIALLY_IMPLEMENTED` | BUG_LEDGER **S8**: was unchunked and would OOM at ~2048 tokens. Prefill *is* gated by design (skipping a prefill position changes its KV for every layer above), so every gated position genuinely needs a signal | correct as designed; the cost is measured, not hidden |
| logits discarded after reduction | never returned, never cached | `features_from_hidden` returns scalars only; `@torch.no_grad()` | `CORRECTLY_IMPLEMENTED` | BUG_LEDGER **C4** removed an autograd graph over `[positions, 32768]` |
| monitor cost measured | the gate must be cheaper than what it gates | closed form **and** measured | `CORRECTLY_IMPLEMENTED` | `exact-confidence cost ratio equals |V|/(3·d_ff)` = **3.556×**; asserted to be > 1 on this checkpoint. Probe alternatives: 0.0109% and 0.2604% |

---

## F. Hidden-state probe (§15)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| probe `q = R_ψ(h)` | lightweight deployment gate | `need` probe, linear + sigmoid, standardisation folded into the weights | `CORRECTLY_IMPLEMENTED` | `artifacts/csgqr_probes/need_L20_int4_g24.pt`; test AUROC **0.9358**, permuted control 0.5065, prompt-mean oracle 0.6976 |
| probe predicts group utility | per-group `Û_g` | `group_utility` router, ridge, closed form | `CORRECTLY_IMPLEMENTED` | `router_L20_int4_g24.pt`; wins 70.8% of the static-to-oracle headroom at ρ=0.25 on held-out prompts |
| AUROC / AUPRC / ECE / Brier | probe vs exact lens | `stats.py` + `expected_calibration_error` | `PARTIALLY_IMPLEMENTED` | AUROC/AUPRC computed and saved with the artifact; ECE/Brier computed for the lens | ECE for the probe |
| false-skip rate | `P(g=0 | U>δ)` | `analyze_confidence.py` | `CORRECTLY_IMPLEMENTED` | `confidence_analysis_d0.1.json`: 0.221 at τ=0.90 for the raw lens |
| group-selection agreement | probe vs oracle | `capture_at_k` | `CORRECTLY_IMPLEMENTED` | the criterion λ is chosen on, and the reported score |
| latency / VRAM of the probe | measured, not assumed | `probe_cost` gives the analytic share; probe bytes recorded | `PARTIALLY_IMPLEMENTED` | 4,100 B and 98,400 B fp32; 0.0109% / 0.2604% of the gated MLP | wall-clock in `bench_gpu.py` |
| splits by prompt, never by token | leakage control | enforced in `train_router.py`; `eval_forced` **refuses** an overlapping evaluation | `CORRECTLY_IMPLEMENTED` | the refusal names the `--skip-prompts` value; demonstrated this session |

---

## G. Utility and need (§16–17)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| `U_t = D(p^T,p^0) − D(p^T,p^Q)` | whole-remnant utility | `marginal_utility`; `D = KL(teacher ‖ candidate)` | `CORRECTLY_IMPLEMENTED` | direction stated in the module docstring and never reversed elsewhere; `larger is better` |
| `U_{t,g}` | per-group | `trace_utility.py`, one forward per group | `CORRECTLY_IMPLEMENTED` | 798 positions × 24 groups committed |
| divergence direction explicit | stated once, not silently reversed | `KL(dense ‖ arm)` everywhere | `CORRECTLY_IMPLEMENTED` | `kl_divergence` used by every consumer |
| paired traces | identical prompts, states, weights, context | one install, policy swapped between forwards | `CORRECTLY_IMPLEMENTED` | every group measured against **the same packed weights**; residual invariance across arms measured at max \|Δ\| = **0.000e+00** |
| hidden-state recovery | additional utility measure | not present | `MISSING` | — | roadmap |
| top-one correction | argmax repair rate | `dense_top1` / `deleted_top1` / `remnant_top1` in every trace row | `CORRECTLY_IMPLEMENTED` | deleted disagrees on 24.8% of tokens; the remnant repairs 85.9% of those |
| sequence-level contribution | beyond token-level | `eval_forced` reports per-prompt `sequence_flip_rate` and `first_flip` | `CORRECTLY_IMPLEMENTED` | new; §39 explicitly warns against token-level KL alone |
| need label `y = 1[U>δ]` | δ from calibration data only | `--delta`, default 0.1 | `PARTIALLY_IMPLEMENTED` | δ = 0.1 chosen because `P(U>0) = 0.9987` makes δ = 0 non-discriminative (N5) — a data-driven choice made on the trace, which is the calibration split | δ sensitivity sweep |
| asymmetric gate loss `w_FN > 1` | penalise false skips harder | not present; plain L2 logistic | `MISSING` | — | the probe reaches AUROC 0.936 unweighted; reweighting shifts the operating point, which the threshold sweep already does |
| false-skip / false-activation rates | both reported | `analyze_confidence.py`, `sweep_threshold.py` | `CORRECTLY_IMPLEMENTED` | committed in `confidence_analysis_d0.1.json` |

---

## H. Routing policy (§18–20)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| `τ_full < τ_skip` | invariant | asserted in `__post_init__` | `CORRECTLY_IMPLEMENTED` | `three_level needs tau_full < tau_skip` |
| high-confidence bypass | `A_t = ∅`, nothing dequantized | `budget → 0`, `forward_groups([])` returns zeros without touching a weight | `CORRECTLY_IMPLEMENTED` | `high confidence bypasses the remnant entirely`; `empty group set returns exact zeros without touching any weight` |
| low-confidence full | `A_t = G` | `budget → n_groups` | `CORRECTLY_IMPLEMENTED` | `low confidence runs the full remnant` |
| interior `ρ = ((τ_skip−C)/(τ_skip−τ_full))^γ`, `k = ⌈ρ|G|⌉` | ramp | `rho()` / `budget()` | `CORRECTLY_IMPLEMENTED` | `fixed_fraction activates ceil(fraction*n) groups`; **was `INCORRECT`** for exact-`k` callers (BUG_LEDGER **S7**) — now exact for all 91 tested `(n,k)` |
| cost-aware `R_g = (1−C)^γ Û_g/(c_g+ε)` | top-k by `R_g` | ranking on `Û_g/(c_g+ε)`; the `(1−C)^γ` scalar is deliberately omitted from the *ranking* | `CORRECTLY_IMPLEMENTED` | it is a per-token scalar and provably cannot reorder a top-k. Including it created a real degeneracy at `C = 1.0` (BUG_LEDGER **S5**). Its actual job — setting `k` — is unchanged. `an expensive group is deselected in favour of a cheaper one` |
| deterministic ties | required | `argsort(..., stable=True)` | `CORRECTLY_IMPLEMENTED` | `all-equal utilities resolve deterministically to the lowest group ids`; **was `INCORRECT`** (BUG_LEDGER **S6**) |
| hard skip, not masking | executed sparsity | slicing happens inside `dequantize`; nothing is computed then masked | `CORRECTLY_IMPLEMENTED` | this is the property Gate E depends on, and it is enforced at the interface: `select` returns ids, and nothing downstream ever sees an inactive group |
| executed vs mathematical sparsity | distinguished | `active_flops(groups)` counts only executed work | `CORRECTLY_IMPLEMENTED` | BUG_LEDGER **S14** |
| per-token bucketing | tokens sharing a group set share a matmul | `defaultdict` bucket then one `forward_groups` per distinct set | `CORRECTLY_IMPLEMENTED` | ≤ `|G|+1` buckets for a static ranking; approaching the token count for per-state selection, which is a real cost and is measured rather than hidden |
| reference + optimized backends | keep separate | one backend, with contiguous-run coalescing | `PARTIALLY_IMPLEMENTED` | coalescing is bit-identical, so it is not a second numerical path | no Triton / fused dequant-GEMM. A slow reference backend must not be read as the algorithm's speed limit (§43) — stated in `LIMITATIONS.md` |
| prefill gated too | not decode-only | both, per token, on their own signal | `CORRECTLY_IMPLEMENTED` | documented in `surgery.py`; skipping a prefill position changes its KV for every layer above, so gating decode only would be a different method |

---

## I. Attention (§5)

| Requirement | Intended | Current implementation | Status |
| --- | --- | --- | --- |
| Phase A: MLP-only dynamic | first rigorous target | done | `CORRECTLY_IMPLEMENTED` |
| Phase B: request-level attention remnant | attention on/off per request | not present | `MISSING` |
| Phase C: prompt-chunk policy | later | not present | `MISSING` |
| Phase D: token-level attention routing | only after cache semantics | not present | `MISSING` |

No attention claim is made anywhere. The KV-cache problem is stated in
`surgery.py` and `LIMITATIONS.md` and is untouched; the request-level residency
policy — the only mechanism in the design that can free VRAM dynamically —
therefore remains unmeasured.

---

## J. Measurement, provenance, statistics (§30, §41–42, §49, §52–53)

| Requirement | Intended | Current implementation | Status | Evidence | Required action |
| --- | --- | --- | --- | --- | --- |
| generation quality endpoint | repository protocol, unchanged | `run_eval.py`, protocol from `manifests/eval_assets.json` | `CORRECTLY_IMPLEMENTED` | `--self-check` reproduces the stored baseline to \|Δ\| = 0.000e+00 |
| a **powered** endpoint | resolve arm differences | `eval_forced.py`, paired per position, prompt-clustered | `CORRECTLY_IMPLEMENTED` | new. The generation endpoint needs ~10,900 cases for the ~0.004 gaps at issue and has 510; after Holm-Bonferroni **none** of the nine Tier-1 contrasts was significant, including deletion damage |
| data roles disjoint | calibration / dev / held-out / sealed | trace prompts fit the probes; `eval_forced --skip-prompts` enforces disjointness | `PARTIALLY_IMPLEMENTED` | the probe-overlap refusal is asserted | no formal sealed-set manifest yet; sealed sets are **untouched** and refused by name |
| VRAM: allocated and reserved | both, with peaks reset | `run_eval.py` records both; `bench_gpu.py` is the memory authority | `PARTIALLY_IMPLEMENTED` | recorded in all ten Tier-1 manifests | `bench_gpu.py` not yet run |
| latency, percentiles, synchronized | P50/P90/P95, warmup, `torch.cuda.synchronize` | `bench_gpu.py` implements it | `UNVERIFIED` | code present, never executed | run it |
| statistics | paired, clustered, exact tests, multiple-comparison policy | `stats.py`: cluster bootstrap, exact McNemar, Wilcoxon, Holm-Bonferroni | `CORRECTLY_IMPLEMENTED` | `analyze_arms.py` applies a **pre-registered** contrast list |
| machine-readable provenance | branch, SHA, dirty, model, dataset, env, seeds | `envinfo.collect()` embedded in every result file | `CORRECTLY_IMPLEMENTED` | includes git SHA, dirty flag, torch CUDA build, GPU, and the model weights sha256 |
| claim verifier | fresh clone verifies committed claims from committed summaries | `verify_claims.py` | `PARTIALLY_IMPLEMENTED` | 65/65 at the pivot | must be re-run and extended to the new endpoint's claims |
| sealed sets untouched | never read | refused by filename in both eval scripts | `CORRECTLY_IMPLEMENTED` | `rag_test_SEALED`, `puzzle_test_SEALED`, `puzzle_challenge_SEALED` never opened |

---

## K. Summary

| Status | Count | Notes |
| --- | --- | --- |
| `CORRECTLY_IMPLEMENTED` | 71 | includes 8 rows that were `INCORRECT` before this session's fixes |
| `PARTIALLY_IMPLEMENTED` | 18 | mostly "implemented and unit-tested but not yet wired into an arm or not yet measured" |
| `MISSING` | 9 | attention phases B–D, NF4/FP4, per-output-channel α, asymmetric gate loss, nonlinear lens, hidden-state recovery |
| `INCORRECT` | 1 | **S15**, α-calibration input overlap. Latent — no committed run is affected |
| `UNVERIFIED` | 1 | GPU latency: `bench_gpu.py` exists and has never been executed |
| `AMBIGUOUS` | 0 | — |

**No core row is `MISSING` or unexplained-`AMBIGUOUS`.** The one `INCORRECT` row
is latent and gated by an explicit refusal path. The `UNVERIFIED` row is the
remaining blocker for the systems half of Stage III and is the next thing to run.
