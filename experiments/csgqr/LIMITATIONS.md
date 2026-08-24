# CSGQR — Limitations

Ordered by how much each one constrains what may be concluded. Every entry says
what is limited, why, and what would lift it.

---

## 1. Structural limits — these cannot be engineered away at this scope

### 1.1 One MLP layer is 1.99% of the model, so no systems claim is available

9,437,184 of 474,021,888 parameters. Consequences, all exact arithmetic fixed
before any measurement (`results/budget.json`):

| remnant format | packed | saved vs bf16 | share of model VRAM |
| --- | ---: | ---: | ---: |
| int8-asym | 9.211 MiB | 8.79 MiB | 0.97% |
| int4-asym | 4.711 MiB | 13.29 MiB | **1.47%** |
| int2-asym | 2.461 MiB | 15.54 MiB | 1.72% |

Routing is capped just as low: even `ρ = 0` — never running the remnant at all —
removes only **1.99%** of model MACs per token, and `ρ = 0.25` removes 1.49%.
These are below run-to-run allocator and clock variation on a laptop GPU.

**So "CSGQR is faster" is not a claim this campaign can make, whatever the
routing does.** Gate E is assessed against the per-layer microbenchmark and the
analytic ceiling. Lifting this requires many retired layers, which was not run.

### 1.2 The exact-confidence gate cannot pay for itself, by arithmetic

`|V| / (3·d_ff) = 32768 / 9216 = 3.556×`. A gate that skipped the remnant on
every single token would still cost 3.56× what it saved. The viability condition
is `|V| < 3·d_ff`; on this checkpoint `d_ff` would have to exceed 10,923 instead
of 3,072, and for current production models the inequality is worse, not better.

No threshold, calibration, lens or feature choice changes this. The
exact-probability form of the gate is therefore usable only as an *offline
labelling teacher*, never as a deployed gate at single-layer scope.

### 1.3 The attention half of the method is unimplemented

Only the MLP is retired. Token-wise attention-remnant gating introduces a
KV-cache problem that is not solved here: if attention is skipped for an earlier
token and activated later, later attention needs donor-layer keys and values that
were never generated. CALM names this same problem as one of its three core
challenges.

Consequence: **the request-level donor-residency policy — the only mechanism in
the design that can free VRAM dynamically — is unimplemented and unmeasured.**
Phases B, C and D are all `MISSING`. Nothing in this campaign says anything about
attention remnants, KV-cache behaviour, or dynamic residency.

---

## 2. Measurement limits

### 2.1 The generation-quality endpoint has no power here, and this invalidates the interpretation of the Tier-1 sweep

Measured on `chat_dev`, 510 cases, layer 20:

| quantity | value |
| --- | ---: |
| deletion damage (mean quality) | 0.01135 |
| paired sd of the per-case delta | **0.14901** |
| t statistic | −1.72 |
| replies changed by deletion | 94.5% |
| cases scoring **exactly** the same under deletion | **55.5%** |
| n for 80% power on deletion damage | 1,352 |
| n for 80% power on a 0.004 arm-to-arm gap | **10,892** |

After Holm–Bonferroni across the nine pre-registered contrasts, **not one is
significant — including deletion damage itself.** The quality score is a coarse
saturated rubric (p10 = median = 0.720).

The Tier-1 sweep was executed correctly and its numbers are real; what they are
not is *evidence about routing policies*. Every ordering in that table is inside
the noise, which is why `b6-static-25` appearing below `b1-delete` and
`b4-int4-all` appearing above dense are recorded as instrument artefacts and not
as findings.

`eval_forced.py` exists to replace it as the primary endpoint. That does not
retroactively power the generation numbers.

### 2.2 The GPU is shared, so no timing or memory number may be taken opportunistically

During this session a **different** experiment
(`experiments/kv_significance/src/passkey.py --lengths 131072`) was running
concurrently on the same 8 GiB card, contending for both VRAM and SMs. Latency
and memory measured under contention are meaningless.

This is why `bench_gpu.py` had not been run at the time of writing and why
`GPU_BENCHMARKS.md` is gated on an exclusive card. It is also why `eval_forced.py`
reports **no** timing at all and records `donor_resident: true` plus an explicit
`memory_note` in its output JSON: its numbers are quality numbers and must not be
mistaken for systems numbers.

### 2.3 Single layer, single checkpoint, one dev split, short replies

Layer 20 was chosen **by rule** (median deletion damage over a 28-layer scan),
not by preference, and the scan is committed. But it is one layer of one 0.47B
checkpoint. `max_new_tokens = 32` with a median reply of 7 tokens; out-of-domain
and long-generation behaviour is untested. `rag_dev` exists and was not swept.

### 2.4 The probes are fitted on 394 training positions

394 train / 150 val / 254 test positions from 96 prompts, split **by prompt**. The
need probe has 1,025 parameters against 394 training rows — it is
overparameterised, and the result rests on its controls rather than on its sample
size: permuted labels give AUROC 0.5065 (no leakage), the prompt-mean *oracle*
gives 0.6976 (so the probe's 0.9358 clears the ceiling on any purely
prompt-level signal), and λ was chosen on a disjoint validation split.

Those controls are load-bearing. Confirming at larger `n` and on `rag_dev` is the
first thing to do, and `eval_forced --skip-prompts` refuses to score a probe on
its own training prompts so the confirmation cannot be faked.

### 2.5 Additivity of utility is measured at the endpoints, interpolated in between

Group *outputs* are exactly additive (float64, < 1e-14). Group *utility* is a
difference of KL divergences and need not be. Measured: the sum of 24 singleton
utilities is within **1.7%** of the whole-remnant utility, and pairwise
interactions average **2.1%** of `|U(A)|+|U(B)|` over 40 sampled pairs.

So additivity is confirmed at `k = 2` and `k = 24`. Intermediate `k` is
**assumed**, not measured. A top-k rule is sound to the extent that assumption
holds.

### 2.6 The static-ranking capture metric is a ranking diagnostic, not a prediction

`capture_curve` sums *singleton* marginal utilities, which is what a top-k rule
optimises under additivity. Because per-token utilities have both signs, the
per-token **oracle** in that metric exceeds the whole-remnant utility (0.634 at
`k = 12` against 0.298 for all 24 groups). That is internally consistent under
per-token additivity — dropping harmful groups per token really does beat running
everything — but it means the oracle is not a bound on any *static* policy, and
the numbers are not directly comparable to a routed arm's measured utility. The
source says so; a reader skimming the table might not.

---

## 3. Implementation limits

### 3.1 One backend, and it is a reference backend

There is no Triton kernel, no fused dequantize-GEMM, no CUDA graph, no
route-based request batching. Contiguous selected groups are coalesced into
maximal runs (bit-identical, launch-overhead only), which is the one optimisation
present.

The brief is explicit that a slow reference backend must not be read as the
algorithm's speed limit, and it is not read that way here. But the converse also
holds: **a hypothetical optimised backend is not demonstrated performance**, and
nothing in this campaign shows that fused kernels would preserve the reference
frontier (H12 is `NOT_ESTABLISHED`, not supported).

### 3.2 Per-token bucketing is a real cost of the dynamic branch

Tokens sharing a selected group set share a matmul. For a static ranking that is
at most `|G|+1` buckets at any sequence length. For per-state dynamic
significance the sets genuinely differ per token and the bucket count approaches
the token count. Per-token policy evaluation is also a Python loop with a
GPU→CPU sync on the gate signal.

### 3.3 α calibration draws its inputs from the evaluation records

`BUG_LEDGER` **S15**, still **OPEN**. Latent rather than realised: no committed
run used `--calibrate-alpha`, and `alpha[0] = 1.00000` in all ten Tier-1
manifests. The fix needs a calibration split disjoint from both dev splits, which
is a data-roles decision. `calibrate_partial.py` does use disjoint-by-prompt
fit/eval splits.

### 3.4 2- and 3-bit storage claims are not validated

Only 4- and 8-bit are genuinely bit-packed. `QuantSpec(2)` and `QuantSpec(3)` are
stored one value per byte, `values_per_byte` says so, and their *storage* numbers
are therefore not claimable. NF4 and FP4 are absent entirely.

### 3.5 Components implemented but never exercised as an arm

* **Trained alignment lens** (`rank > 0`). Constructible, zero-initialised to
  exact identity, never trained. H7's lens variant is `NOT_ESTABLISHED`, not
  falsified — though §1.2 makes it economically pointless at this scope
  regardless of its accuracy.
* **Prediction-stability feature** `C^S = 1 − JS/log 2`. Implemented and
  unit-tested; not wired to any gate.
* **Learned combined confidence** over the four lens features. Measured offline
  (AUROC 0.580, no better than the best single raw feature); not serialized as a
  deployable gate.
* **Per-output-channel α**, **asymmetric gate loss** `w_FN > 1`,
  **hidden-state-recovery utility**: `MISSING`.

### 3.6 Survivor adaptation was not run

Donor-frozen, router-only. No LoRA on neighbouring blocks, no direct survivor
tuning. So the comparison against "deletion + the same LoRA" and "INT4-all + the
same LoRA" — the arms that would separate donor identity from generic capacity —
is not available, and H9 is `NOT_ESTABLISHED`.

---

## 4. Audit-independence limit

**This is not an independent audit.** The branch contained no CSGQR
implementation when the campaign began: `origin/research/csgqr-quantized-remnant`
was byte-identical to `origin/r2-healing-harness` (verified again this session at
0/0 ahead-behind with an empty diff). The implementation reviewed here was written
as part of the campaign, and no amount of testing repairs that.

The compensations are structural rather than promissory, and they are what the
credibility rests on:

1. The adversarial controls — random routing, shuffled and reversed significance,
   magnitude ranking, wrong-layer donor, permuted-label probe, prompt-mean oracle
   — were written before any CSGQR result was inspected.
2. The analytic ceilings in `results/budget.json` were computed before any GPU
   measurement, so the systems verdict is read against a number fixed in advance.
3. The contrast list in `analyze_arms.py` is pre-registered in source rather than
   chosen after seeing the numbers.
4. `verify_csgqr.py` (112 checks) tests properties the *claims* depend on, and
   several of this session's fixes were found by a test failing rather than by
   reading code — the per-group α bug in particular is invisible in source
   review.

---

## 5. Statistical caveats

* Token states within a prompt are **not** independent. Every interval in
  `eval_forced` and `analyze_significance` is a prompt-clustered bootstrap.
  Treating positions as independent would shrink intervals by roughly the square
  root of the mean reply length.
* Multiple comparisons are handled by Holm–Bonferroni over a **fixed** contrast
  list. Adding a contrast after seeing results would invalidate that.
* **Only one seed.** The trainable components here are closed-form (ridge) or
  deterministic full-batch L-BFGS from a fixed zero initialisation, so there is no
  optimiser seed to vary. But the brief asks for three seeds where feasible, and
  for anything with genuine stochasticity that has not been done.
* `δ = 0.1` for the need label was chosen because `P(U > 0) = 0.9987` makes
  `δ = 0` non-discriminative. That is a data-driven choice made on the
  calibration trace. Its sensitivity has not been swept.

---

## 6. What is explicitly not claimed

* Not claimed: that CSGQR improves measured VRAM or latency end-to-end (§1.1).
* Not claimed: that confidence gating works. Falsified for the raw lens.
* Not claimed: that a trained lens would fail. It was not run (§3.5).
* Not claimed: anything about attention remnants, KV-cache behaviour, or
  request-level residency (§1.3).
* Not claimed: that the group router beats an activation-magnitude predictor of
  the Deja Vu kind. That head-to-head was not run, and it is the comparison that
  would decide whether the interventional utility label earns its cost.
* Not claimed: anything about the sealed test sets. `rag_test_SEALED`,
  `puzzle_test_SEALED` and `puzzle_challenge_SEALED` were never read; both eval
  scripts refuse a path containing `SEALED`.
