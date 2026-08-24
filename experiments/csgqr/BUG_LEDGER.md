# CSGQR — Bug Ledger

Every material defect found in the CSGQR implementation, what it would have done
to a result, how it was fixed, and which regression test now holds it down.

Severity scale: `BLOCKER` (no valid result possible) > `CRITICAL` (a headline
claim would have been wrong) > `MAJOR` (a reported number would have been wrong)
> `MODERATE` (a latent wrong number, not yet triggered) > `MINOR`.

Rows C1–C6 were found and fixed during the first campaign (before the remote
push) and are carried forward here from `METHOD_AUDIT.md` and
`NEGATIVE_RESULTS.md` N10 so that one file holds the whole ledger. Rows S1
onward were found in the Stage I review of this session.

All regression tests live in `src/verify_csgqr.py` unless stated otherwise.

---

## Stage I review of this session

| ID | Severity | Component | Symptom | Root cause | Scientific impact | Fix | Regression test | Status |
| -- | -------- | --------- | ------- | ---------- | ----------------- | --- | --------------- | ------ |
| **S1** | `CRITICAL` | `surgery.py` `CSGQRMlp.forward` | An arm configured with `selector="significance"` under `mode="fixed_fraction"` silently executed the **static** ranking instead of per-state dynamic selection. No error, no warning, correct-looking activation accounting. | The `fixed_fraction` fast path was guarded by `selector != "random"`, so `significance` fell into it and called `policy.select(0.0)` with `utility=None`; `select` then took its documented static fallback. | The matched-budget dynamic-vs-static contrast (brief §35, H3) was **unrunnable** — the dynamic arm would have produced numbers identical to the static arm and been reported as dynamic. This is the contrast the whole significance half of the method rests on. | Fast path now excludes both `random` (varies per token) and `significance` (varies per state). `CSGQRMlp.__init__` **refuses to construct** a policy that needs per-state utility without a `utility_fn`, naming `static_topk` as the arm the caller probably meant. Capability is read from new explicit `RoutingPolicy` properties (`needs_gate_signal`, `needs_lens`, `needs_probe`, `needs_group_utility`) rather than inferred from mode strings at the call site. | `policy capability flags identify which monitors an arm must pay for`; construction-refusal covered by the readiness run in `verify_readiness.py` | **FIXED** |
| **S2** | `MAJOR` | `remnant.py` `calibrate_alpha` | Per-group gain `alpha_g` did not converge to 1 for a **bit-exact** remnant; it returned values spanning 0.978–1.024 and injected **1.32% relative error** into an otherwise exact reconstruction. | Every group was fitted against the *whole* MLP output `F(h)` rather than against its own donor group output `F_g(h)`. Since `<F_g, F> = ‖F_g‖² + <F_g, Σ_{h≠g} F_h>`, the estimator is inflated by the cross-group inner product. A different estimator, not a looser one. | Brief §11 specifies `alpha_g` against `F_{j,g}`. Any per-group-calibrated arm would have been systematically mis-scaled, and the recorded finding "per-group α is worse than scalar α" (N7) was measured with the wrong estimator and is therefore not evidence about per-group α. | Added `reference_group_output(mlp, x, lo, hi)`, the donor's own group output. `per_group=True` now fits each group against it. | `per-group alpha is EXACTLY 1 for a bit-exact remnant (target is F_g, not F)`; `donor group outputs sum to the donor MLP exactly (float64)` | **FIXED** |
| **S3** | `MAJOR` | `remnant.py` `calibrate_alpha` | Calibrating twice compounded the gain, because the fit ran through `forward_groups`, which already applies `alpha`. | The current `alpha` buffer was live during the fit. | A second calibration pass — or a calibrate-then-recalibrate workflow, which is exactly what a per-budget sweep needs — would square the gain and silently degrade every quantized arm. | The fit neutralises `alpha` for its duration and restores it in a `finally`. | `alpha calibration is idempotent`; `calibration restores the alpha buffer it was called with` | **FIXED** |
| **S4** | `MAJOR` | `remnant.py` | No way to calibrate the gain for a **partial** group set, which is what every routed arm at `rho < 1` actually executes. All routed arms ran with `alpha = 1`. | Missing feature: only whole-remnant and per-group forms existed. | Brief §11 and §18–19 both assume a calibrated gain is available for the active set. Its absence is a live candidate explanation for `b6-static-25` scoring below deletion — a partial subset sum is not the whole MLP and rescaling it is not optional. | `calibrate_alpha(..., subset=[...])` fits a scalar against the donor output restricted to the **same rows**. Fitting a partial sum against the whole donor output would inflate the gain by ~`1/rho` and is explicitly not offered. | `subset alpha does not increase the partial reconstruction error` | **FIXED** |
| **S5** | `MODERATE` | `router.py` `select` | At `C = 1.0` the dynamic-significance selector silently became the **contiguous** selector, returning groups `0..k-1` regardless of utility. | `value = (1-C)^gamma * U / c` made every entry exactly `0.0`, and `argsort` on an all-zero vector returns ascending indices. | An arm reported as significance-routed would have been contiguous-routed for its most-confident tokens. Reachable through `confidence_whole`/`fixed_fraction` combinations even though `three_level` skips at `C ≥ tau_skip`. | The `(1-C)^gamma` factor is dropped from the *ranking* score. It is a per-token scalar and provably cannot reorder a top-k, so this is mathematically identical wherever the factor is positive and removes the degeneracy where it is not. The factor's real job — setting *how many* groups run — is unchanged in `rho()`/`budget()`. | `dynamic significance selects the same groups at every confidence` | **FIXED** |
| **S6** | `MODERATE` | `router.py` `select` | Tie-breaking among equal-utility groups was implementation-defined. | `torch.argsort` is not stable by default. | Blockwise quantization creates exact ties in group utility far more often than continuous data would. An unstable sort makes an arm irreproducible for precisely the tokens where the choice is arbitrary — and the brief requires deterministic tie behaviour (§22). | `torch.argsort(-value, stable=True)`, so ties resolve to ascending group id. | `all-equal utilities resolve deterministically to the lowest group ids`; `tied winners are ordered by group id, and losers are excluded` | **FIXED** |
| **S7** | `MODERATE` | `router.py` `budget` | `fraction = k / n_groups` did not recover `k` for every `k`: `5/24 * 24 == 5.000000000000001`, so `ceil` returned **6**. | Binary floating-point round-trip, then a bare `ceil`. | `trace_utility.py` builds every single-group and pair configuration this way. Any subset size whose fraction does not round-trip exactly would have been executed one group too wide while being **labelled** with the smaller size, corrupting the utility labels that the entire significance analysis is built on. Singles and pairs happen to round-trip cleanly, so the live traces are unaffected — but the next subset size used would have been silently wrong. | `ceil(r * n - 1e-9)`, a tolerance 1e-9 of a group wide, far below any meaningful `rho` resolution. `subset_policy` in `trace_utility.py` now also **asserts** that the policy selects exactly the requested groups. | `fraction = k/n recovers exactly k groups for every k (n = 7, 12, 24, 48)` — 91 cases; `a genuine fractional budget still rounds up` | **FIXED** |
| **S8** | `MAJOR` | `confidence.py` `features_from_hidden` | The gate materialised `[T, |V|]` and then three float32 tensors of that shape for the **whole prefill at once**. | No chunking over the position axis. | At `|V| = 32768` a 2048-token prompt is ~1.6 GiB of transient on an 8 GiB card already holding the model. The gate would have worked on the short dev prompts and OOM'd on long ones — the worst failure shape, because it makes the method look viable exactly where it was tested. Brief §14 requires that vocabulary tensors are not retained and that the monitor's own cost is bounded. | Chunked over positions (`CONFIDENCE_CHUNK = 256`), bounding the transient by the chunk instead of by sequence length. | `the confidence reduction is bit-exact when split over positions`; `chunking does not change which token the lens predicts`; `chunked features agree with unchunked to float32 GEMM epsilon` | **FIXED** |
| **S9** | `MINOR` | `router.py` `to_dict` | `group_costs` was omitted from the serialized policy. | Oversight. | A serialized policy that omits a field affecting selection cannot be replayed. Inert while all groups cost the same, and it would first bite when mixed-size groups arrive — the worst moment to find out. | Included, along with `feature`. | `to_dict serializes every field that changes selection, group_costs included` | **FIXED** |
| **S10** | `MINOR` | `router.py` `__post_init__` | An unknown `feature` string was accepted and only failed later as a `KeyError` inside the forward pass. | No validation. | A typo'd feature name would abort a long GPU run at the first token instead of at construction. | Validated against `FEATURES`. | `an unknown gate feature is refused rather than KeyError'd later` | **FIXED** |
| **S11** | `MINOR` | `surgery.py` | The forward-pre-hook fires on every forward regardless of policy, so arms that never read the residual held a reference to one activation tensor for the module's lifetime. | The residual was only cleared on paths that consumed it. | Resident bytes appearing in exactly the always-on and deleted arms whose memory numbers are meant to be the clean references. | `_drop_residual()` on every fast path. | covered by the memory checks in `bench_gpu.py` | **FIXED** |
| **S12** | `MODERATE` | `router.py` `RoutingTrace` | Arms with no gate signal recorded `confidence = 0.0`, so their manifests reported `mean_confidence 0.000`. | A sentinel that collides with a real value. | Reads as "this arm measured a maximally uncertain model", not "this arm has no such measurement". Misleading in exactly the baseline manifests a reader compares against. | Records `NaN`; `summary()` excludes non-finite entries and reports `gate_signal_tokens` alongside. `mean_confidence` is `null` when no signal exists. | `trace` checks in section F | **FIXED** |
| **S13** | `MODERATE` | `remnant.py` `_gather` | The reference backend issued `3 × len(groups)` dequantization calls per distinct group set per token bucket — 72 launches at `rho = 1` to reconstruct three matrices that are one slice each. | No coalescing of contiguous group runs. | Pure launch overhead, but it inflates the reference backend's latency and would have made the algorithm look slower than it is. The brief is explicit (§43) that a slow reference backend must not be mistaken for an algorithmic speed limit. | Contiguous groups merged into maximal runs; one dequantization per run. Bit-identical — the concatenation order is unchanged because runs are built from sorted groups over ascending row ranges. | `coalesced run-merged gather is bit-identical to a per-group gather`; `contiguous groups merge into one run; non-contiguous do not` | **FIXED** |
| **S14** | `MINOR` | `remnant.py` `group_flops` | Reported the *nominal* group size, under-counting the last group when `n_groups` does not divide the intermediate dimension. | `intermediate // n_groups` ignores the folded remainder. | Inert on this checkpoint (3072 = 24 × 128). Would silently under-report executed FLOPs for any uneven partition, and FLOP accounting is what the compute claim rests on. | `group_flops(tokens, group)` reports the true size; new `active_flops(groups, tokens)` sums true per-group costs. | `an uneven partition folds the remainder into the last group and reports its true size`; `active_flops over an uneven partition equals the whole layer` | **FIXED** |
| **S15** | `MAJOR` | `run_eval.py` `_collect_mlp_inputs` | The α-calibration inputs were taken from `records[:n]` — the **same development records being scored**. | Calibration and evaluation drew from one list. | Calibration leakage (brief §30 requires disjoint calibration data). Latent rather than realised: `--calibrate-alpha` defaults to 0, so no committed run used it. It would have activated the moment anyone calibrated. | See S15 disposition below. | — | **OPEN — see below** |

| **S16** | `MAJOR` | `router.py` `select` | `RoutingPolicy.select` raised `Expected all tensors to be on the same device` — but **only for the dynamic arms**, i.e. after every static arm in a sweep had already run. | The cost vector was built on CPU (`torch.tensor(self.group_costs)`) while a trained `utility_fn` returns CUDA tensors. Introduced by the S1 fix, which is what first made a GPU-resident utility vector reach this line. | Killed the first full powered sweep at arm 13 of 16, and because `eval_forced` wrote its JSON once at the end, the whole run's GPU time was discarded. | `torch.as_tensor(..., dtype=flat_u.dtype, device=flat_u.device)`. The per-token loop also moves the utility batch to CPU once rather than syncing per token. | covered by the dynamic-arm smoke run; the device-agnostic path is exercised by the `select` checks | **FIXED** |
| **S17** | `MAJOR` | `eval_forced.py` | One arm's failure cascaded into **21** consecutive failures with a misleading message (`'CSGQRMlp' object has no attribute 'gate_proj'`), making a single card-level fault look like a code fault. | A mid-arm exception left the `CSGQRMlp` installed at `layers[j].mlp`, so the next `build_arm` read the installed module instead of the real MLP. The exception handler's comment claimed it restored the model; it did not. | A resource failure on arm 3 produced 20 further "failures" that said nothing about the arms they were attached to. Any reader of that output would have concluded the implementation was broken. | The pristine MLP is captured once before any arm installs anything, and a `finally` restores it and removes every registered hook after each arm. `_is_cuda_context_error` additionally **aborts** the sweep on a context-poisoning CUDA error and records `_sweep_aborted`, because continuing past one of those fabricates failures; a plain allocator OOM is excluded from that class and only skips its own arm. | the sweep now records `failures` and `complete` in its output, so a missing arm is visibly a failure rather than silently absent | **FIXED** |
| **S18** | `MODERATE` | `stats.py` / `eval_forced.py` | The primary endpoint's bootstrap dominated its wall time. | `cluster_bootstrap` resamples index tensors: 4000 draws x 416 clusters = 1.66M small tensor operations **per metric per arm**. | Not a correctness problem, but it made the powered endpoint look expensive when its GPU work is a few minutes, which discourages using the endpoint that actually has power. | `cluster_bootstrap_mean` computes the same distribution in closed form from per-cluster sums and counts. 15x faster and **bit-identical** — point, lo and hi all agree to 0.0. | `vectorized clustered bootstrap matches the general one exactly`; `vectorized paired clustered bootstrap matches the general one`; `clustering by prompt widens the interval versus treating positions as independent` | **FIXED** |

| **D1** | `MAJOR` | `GPU_BENCHMARKS.md`, `FINDINGS.md`, `NEGATIVE_RESULTS.md`, `RESEARCH_VERDICT.md` | The microbenchmark table **mixed numbers from two different benchmark runs**: the rho<1 rows came from run 1 while the rho=1 row came from the committed run 3. Three derived claims were consequently wrong -- "FP at rho=0.25 runs in 55% of dense" (committed: **70%**), "the INT4 column is visibly non-monotone" (committed: **monotone**, 2.39 -> 2.45 -> 2.60 -> 2.77), and "the closed form predicts the projection-only cost well, measured 3.04x" (committed: **2.20x**, and the closed form *over*states it). | I wrote the prose from console output of an earlier run and did not re-derive it from the committed JSON after re-running the benchmark twice more. | A documentation defect, not a code defect -- but it is exactly the failure the brief names: "every number quoted in committed documentation must have a committed compact supporting summary". Two of the three wrong numbers made the method look **better** than the committed data supports. | All four documents corrected against `results/gpu_bench.json`, with an explicit provenance-correction note in `GPU_BENCHMARKS.md` §1.1 rather than a silent edit. | `verify_claims.py` now pins **13 microbenchmark ratios** to the JSON and asserts monotonicity in rho for all three formats, so any future drift between prose and data fails the verifier | **FIXED** |

### S16/S17 note on the diagnosis itself

The failure that triggered S17 was `CUBLAS_STATUS_EXECUTION_FAILED` on the
FP-passthrough arm. My first hypothesis was that the S13 coalescing change was at
fault: it returns a **view** (`self.dense[lo:hi]`) instead of a fresh `torch.cat`
when the selection is a single contiguous run, and `.t()` on that view could
plausibly have driven cuBLAS down a bad path. **That hypothesis was wrong.**
Tested directly on the GPU across all three specs (`fp`, `int8`, `int4`), five
prompt lengths including the known-awkward `T=22`, and 2-D and 3-D inputs: every
case passes and the gathered gate matrix is contiguous with stride `(1024, 1)`.

The remaining explanation is resource contention — a **different** experiment
(`experiments/kv_significance/src/passkey.py --lengths 131072`) was running
concurrently on the same 8 GiB card, and CUDA errors surface asynchronously at an
arbitrary later launch, so the reported call site need not be the faulting one.
Recorded because the wrong hypothesis was the plausible one, and because it is the
reason `GPU_BENCHMARKS.md` is gated on an exclusive card rather than taken
opportunistically.

### S15 disposition

`--calibrate-alpha` was never used by any committed run (`alpha[0] = 1.00000` in
every one of the ten Tier-1 manifests), so no existing number is affected. The
fix is not a one-line change — it needs a calibration split that is disjoint from
both dev splits, which is a data-roles question (brief §30) rather than a code
question. It is therefore tracked as open here and resolved in the calibration
work rather than patched in place. **No arm may use `--calibrate-alpha` against
its own evaluation set; the flag will require an explicit separate calibration
source.**

---

## Carried forward from the first campaign

| ID | Severity | Component | Symptom | Root cause | Scientific impact | Fix | Regression test | Status |
| -- | -------- | --------- | ------- | ---------- | ----------------- | --- | --------------- | ------ |
| **C1** | `CRITICAL` | `remnant.py` `calibrate_alpha` | A "calibrated" remnant was **worse** than an uncalibrated one (relative error 0.1148 → 0.1644); a true α of 0.9979 came out as 0.8797. | `E‖F^Q‖² + ε` implemented with an **absolute** `ε = 1e-8`. MLP output magnitudes are checkpoint-dependent; on a small-activation layer the sums were ~1e-7, making ε **13% of the denominator**. | Would have silently degraded every quantized arm. Invisible in source review — a fixed epsilon looks unimpeachable. | Degenerate case handled as the zero-check it actually is, with no additive term; sums accumulate in float64. | `alpha matches the closed-form least-squares solution at every activation scale` (2500× scale span) | **FIXED** |
| **C2** | `MAJOR` | `run_eval.py` | α fitted against the quantized remnant's **own** output, which returns exactly 1.0 by construction and looks like a working calibration. | Wrong reference tensor. | A no-op calibration reported as a calibration. | Fitted against the unquantized donor while it is still resident. | covered by C1's closed-form check | **FIXED** |
| **C3** | `MODERATE` | `surgery.py` `InstalledCSGQR.remove` | Reinstalled an MLP whose `gate_proj`/`up_proj`/`down_proj` had been freed, failing later and elsewhere. | `keep_donor_resident=False` deletes the projections; `remove()` did not know. | A confusing failure far from its cause, mid-run. | Refuses with an explicit message naming the reload as the remedy. | — | **FIXED** |
| **C4** | `MAJOR` | `confidence.py` | The confidence projection built an autograd graph over `[positions, 32768]` and held it for the whole prompt. | `lm_head.weight.requires_grad` is True and the call was not wrapped. | Memory growth attributable to the gate that was really attributable to autograd — it would have made the monitor look even more expensive than it is. | `@torch.no_grad()` on `features_from_hidden`. | — | **FIXED** |
| **C5** | `MINOR` | `analyze_significance.py`, `budget.py` | `UnicodeEncodeError` on the Windows cp1252 console, **after** the expensive computation. | Non-ASCII characters in print output. | Lost work at the end of a long run. | Sources are ASCII-only. | — | **FIXED** |
| **C6** | `MAJOR` | `run_eval.py` | A 24-group significance ranking passed to `b1-delete` (which runs `n_groups=1`) raised a permutation error on the first arm of the sweep. | No group-count check on `--ranking-from`. | Aborted the sweep. Silently truncating the ranking would have been worse — it would have invented an ordering the significance run never measured. | A ranking is used only when its length matches the arm's group count; otherwise it is ignored with a printed notice. | — | **FIXED** |

---

## Not bugs, but they change how results must be read

These are recorded because each one looks like a bug and is not, and mistaking
any of them for one would send the next person down a dead end.

**N8 generalised: shape-dependent GEMM is not a cuBLAS quirk.** The campaign
recorded that the remnant path is not bit-identical to `nn.Linear` at all shapes
(3.9e-3 max abs at `[1,22,1024]`) because cuBLAS selects a different bf16 tile.
This session reproduced the same effect **on the CPU path in float32**: the
confidence projection differs by 4.8e-07 max between a `[133,64]×[64,512]` GEMM
and the same work in blocks of 16. The *reduction* is bit-exact; only the matmul
moves. So this is a generic property of batched matrix multiplication on this
stack, not a GPU-specific confound, and `|b0 − b2|` remains the floor below which
no quality difference is attributable to CSGQR.

**`k = ceil(rho·|G|)` never activates zero groups in the interior region.**
Deliberate. Any confidence strictly below `tau_skip` activates at least one
group, so `k = 0` is reachable only through the `tau_skip` branch — which is what
the activation-rate accounting assumes.

**2- and 3-bit are stored one value per byte.** Only 4- and 8-bit are genuinely
bit-packed. `QuantSpec(2)` and `QuantSpec(3)` exist for the bitwidth sweep and
their *storage* claim is not validated; `values_per_byte` says so in the source.

**The `(1-C)^gamma` factor in `R_g` cannot affect group selection.** It is a
per-token scalar multiplying every group's value equally. This is a property of
the specification as written, not an implementation shortcut — the factor's real
job is setting `k`, which it does in `rho()`.

---

## Open

| ID | Severity | Item | Why it is still open |
| -- | -------- | ---- | -------------------- |
| **S15** | `MAJOR` | α-calibration inputs overlap the evaluation set | Needs a disjoint calibration split, which is a data-roles decision, not a patch. No committed run is affected (α = 1 everywhere). |
