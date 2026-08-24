# CSGQR — Experiment Log

Chronological. Every entry records the command, the git SHA it ran under, the
purpose, the result, and where the artifact landed. Timestamps are UTC.

**Machine (all runs).** Windows 11 26200, RTX 4070 Laptop (cc8.9, 8188 MiB,
driver 610.62), torch 2.13.0+cu130, transformers 5.15.0, peft 0.20.0,
Python 3.11.15 via `.venv/Scripts/python.exe`.
Manifest: `results/environment.json`.

**Interpreter warning that shaped every run.** The *ambient* `python` on this
machine is `torch 2.13.0+cpu` — CUDA unavailable. Only the repo venv has
`+cu130`. A run on the ambient interpreter would have produced quality numbers
indistinguishable from GPU numbers in a summary file, so
`envinfo.require_cuda()` refuses rather than warns, and every GPU entry below
used `.venv/Scripts/python.exe` explicitly.

**Model (all runs).** `artifacts/rineya-core-en32k-28L`, Qwen3, 28L, hidden 1024,
d_ff 3072, 16 heads / 8 KV, |V| 32768, tied embeddings, bf16.
`model.safetensors` = 948,078,864 bytes, sha256 `a6bc04ca357d446d…`.

**Sealed sets.** `rag_test_SEALED`, `puzzle_test_SEALED`,
`puzzle_challenge_SEALED` were never read. `run_eval.py` and `run_gates.py` both
refuse a path containing `SEALED`. Development used `chat_dev` (510) and the
`chat_dev` prompt prefix for tracing.

---

## 0 — Branch reconnaissance

**SHA at session start:** `f375908` on `exp/stabilizer-matrix` (not this branch).

Established that `research/csgqr-quantized-remnant` contained **no CSGQR work**:
tip `1664f54`, a significance-matrix paper commit shared with
`research/pre-deletion-recovery-routing-v0.1` and `-v0.2`, eleven commits behind
`main`. Zero files matching `*csgqr*` on any ref; zero occurrences of `remnant`
in any tracked file on `main`; nothing in `git stash`, the reflog, or the two
dangling commits.

`scripts/common/response_control.py` exists only on `main` — which identified the
branch label as stale rather than the brief as wrong.

**Action:** created the local branch at the remote tip and merged `origin/main`
into it. No history rewritten; no other branch touched. `exp/stabilizer-matrix`
advanced to `f375908` during the session via a concurrent process, not by me
(`d820797` is an ancestor of `f375908`; nothing lost).

**SHA after merge:** `3c051f1` — the base for all work below.

---

## 1 — Environment and analytic ceilings *(before any GPU experiment)*

```bash
.venv/Scripts/python.exe experiments/csgqr/src/envinfo.py
.venv/Scripts/python.exe experiments/csgqr/src/budget.py
```

**Purpose.** Fix the ceiling on what the mechanism could possibly save *before*
measuring anything, so a measured result is read against a pre-registered number.

**Result.** One MLP layer is 1.99% of the model. INT4 saves 13.29 MiB = 1.47% of
model VRAM; `ρ = 0` removes 1.99% of model MACs. The exact-confidence monitor
costs `|V|/(3·d_ff) = 3.556×` the MLP it gates.

**Interpretation.** The single-layer systems case is dead on arrival (N2), and
the exact-confidence gate is unaffordable by construction (N1). Reframed the
campaign as a mechanism study with the systems ceiling reported analytically.

**Artifacts:** `results/environment.json`, `results/budget.json`.

---

## 2 — Verification suite

```bash
.venv/Scripts/python.exe experiments/csgqr/src/verify_csgqr.py \
    --json experiments/csgqr/results/verify_csgqr.json
```

**Result.** 86/86 checks pass (0.5 s, CPU, deterministic).

Four failed on the first run. Three were over-tight thresholds against float32
epsilon — additivity measured `6.3e-8` on order-1 values against a `1e-9` bound.
Fixed by testing the *exactness* claim in float64 (`<1e-14`) and float32 on a
relative tolerance, not by loosening a bound. The fourth was a test that could
not detect what it claimed: cost-aware selection was checked with `k=2` when only
two groups had non-zero utility, so both were selected regardless of cost; `k=1`
makes it a real test.

**Key properties established:** exact-zero handling with no division by zero;
byte-aligned odd-width 4-bit packing; non-divisible block widths; row-slice
dequantization bit-identical to the full dequantization; no fp32/bf16 residue and
no parameters in packed storage; exact byte accounting; group_bounds an exact
partition; **additivity exact in float64**; a biased `down_proj` refused.

**Artifact:** `results/verify_csgqr.json`.

---

## 3 — Implementation bug: `calibrate_alpha`

Found by testing, not review. `ε = 1e-8` in `E‖F^Q‖² + ε` is absolute, and MLP
output magnitudes are checkpoint-dependent: on a small-activation layer the sums
were ~1e-7, making ε **13% of the denominator**. True α 0.9979 → 0.8797, and the
"calibrated" remnant was *worse* than the uncalibrated one (rel err 0.1148 →
0.1644).

Fixed to a zero-check with float64 accumulation. α now matches the closed form to
`<1e-5` across activation scales spanning 2500× and never increases error.

Also fixed: `remove()` reinstalling a hollow MLP; α fitted against the quantized
remnant's own output (would return 1.0 by construction).

---

## 4 — Integration test: remnant path vs. the real model

**Result.** `mode="none"` is bit-identical to a true MLP bypass (max logit
deviation **0.0**). The FP passthrough remnant reconstructs all three weight
matrices bit-identically and reproduces `nn.Linear` bit-identically at
`[64,1024]`, `[1,1,1024]`, `[1,64,1024]`, `[2,8,1024]` — but differs by `3.9e-3`
at `[1,22,1024]`.

**Interpretation.** cuBLAS tile selection, not a bug: identical weights,
identical arithmetic, different accumulation order. This fixed the experimental
design — `b2-fp-all` is the reference for quantized arms and `|b0 − b2|` is the
noise floor (N8).

---

## 5 — Gate A: 28-layer deletion/recovery scan

```bash
.venv/Scripts/python.exe experiments/csgqr/src/measure_layers.py \
    --prompts 64 --layers all --arms fp,int8,int4
```

SHA `3c051f1`. 1021 s. 64 prompts, 520 reply positions, metric
mean KL(dense‖arm) at reply positions, token-weighted.

**Result — Gate A PASSES.** FP recovery exactly **1.000 at every layer**; INT8
median 0.997; INT4 median 0.967 (range 0.945–0.996).

Deletion damage: L1 9.44 > L2 8.48 > L0 1.72, then 0.16–0.44 for the rest.
Least damaging L25 (0.1615); most damaging L1.

**Interpretation.** INT4 retains donor function well — dynamic routing is worth
studying. Also: MLP-only deletion damage is *not* the block-level
angular-distance prior. L26 sits near the top of the prunable list by angular
distance but mid-range here (0.328).

**Layer selection.** The rule (median deletion damage) picked **L20**: delete KL
0.3194, deleted top-1 agreement 0.750 — meaningful damage, nontrivial recovery
headroom, INT4 recovery 0.970 sitting at the population median. Chosen by rule,
not by preference.

**Artifact:** `results/layer_scan.json`. An earlier 6-prompt smoke run to the
same path was superseded by this one.

---

## 6 — Per-token utility trace at L20

```bash
.venv/Scripts/python.exe experiments/csgqr/src/trace_utility.py \
    --layer 20 --prompts 96 --pairs 40 --verify-invariance
```

614 s, 107 forwards per prompt. 798 reply positions from 96 prompts.

**Efficiency fix made first:** the per-group loop originally re-installed (and so
re-quantized) the remnant for every configuration — 107 quantizations per prompt.
Restructured to install once and swap the policy, which is both faster and
removes a confound: every group is now measured against *exactly* the same packed
weights.

**Result.** mean KL(dense‖delete) 0.31259; KL(dense‖remnant) 0.00913; mean
utility 0.30347; **P(U > 0) = 0.9987**.

**Residual invariance: max |Δ| = 0.000e+00.** Confirms that the pre-remnant
residual is identical across dense, deleted and remnant arms, so confidence is a
property of the surviving prefix and can be computed once. This was checked, not
assumed.

**Interpretation.** The remnant helps on essentially every token (N5), so
"whether" is barely a question and the labelling threshold had to be raised to
`δ = 0.1` (base rate 0.62) for AUROC to mean anything.

**Artifact:** `results/trace_L20_int4/`.

---

## 7 — H1: confidence vs. utility

```bash
.venv/Scripts/python.exe experiments/csgqr/src/analyze_confidence.py \
    --trace experiments/csgqr/results/trace_L20_int4 --delta 0.1
```

**Result — H1 FALSIFIED for the raw lens.** `max_prob` gap **−0.00115** (wrong
sign); best gap +0.024 against mean utility 0.303; AUROC 0.537–0.565; Pearson
corr(1−C, U) ≈ 0.00–0.02; all cluster-bootstrap intervals overlapping.

**Mechanism, from the same trace:** raw lens top-1 agrees with dense top-1 only
**26.9%** of the time. ECE(max_prob) 0.356. Confidently-wrong skips: 37 at
τ=0.90, 15 even at τ=0.99.

**Artifact:** `results/trace_L20_int4/confidence_analysis_d0.1.json`.

---

## 8 — H3 and additivity

```bash
.venv/Scripts/python.exe experiments/csgqr/src/analyze_significance.py \
    --trace experiments/csgqr/results/trace_L20_int4
```

**Result — H3 SUPPORTED.** Significance beats the random 95th percentile at every
active fraction (ρ=0.25: 0.1610 vs random mean 0.0740, p95 0.1270 — 2.2× random).
Reversed ranking collapses to ≈0, so the *order* carries the information.

**Additivity CONFIRMED.** Interaction `|U(A+B) − U(A) − U(B)|` is 2.1% of
`|U(A)|+|U(B)|` on average, 8.4% at worst over 40 sampled pairs; 13/40 pairs have
a detectable interaction but all are small. Top-k is a sound selection rule.

**Negative:** static ranking captures only a third of the per-token oracle at
ρ=0.25 (66.7% headroom unclaimed); per-token orderings agree with the global
ranking at mean Spearman 0.070 (N6).

Fixed two Windows `cp1252` crashes from non-ASCII console output — both fired
*after* the expensive computation.

**Artifact:** `results/trace_L20_int4/significance_analysis.json`.

---

## 9 — Is the signal in the state? (the campaign's pivot)

```bash
.venv/Scripts/python.exe experiments/csgqr/src/probe_utility.py \
    --trace experiments/csgqr/results/trace_L20_int4 --delta 0.1
```

Three-way split **by prompt** (48/19/29 prompts → 394/150/254 positions), λ chosen
on validation, test touched once.

**Result.** Predicting "remnant needed": raw lens 0.579–0.600, fitted lens
features 0.580, **hidden-state probe 0.936** (val 0.931).

Controls: permuted labels **0.506** (no leakage); reply-position-only 0.601;
prompt-mean **oracle** 0.698 — the ceiling on any purely prompt-level signal. The
probe clears that ceiling, so it resolves individual tokens.

Group selection: probe wins **66–73%** of the static-to-oracle headroom on
held-out prompts (ρ=0.25: random 0.077, permuted-probe 0.155, static 0.161,
probe-on-lens-features 0.189, **probe-on-state 0.392**, oracle 0.488).

**Interpretation.** The information was in the state all along; the vocabulary
projection destroys it. Combined with N1 (the projection costs 3.56× what it
gates), the exact-confidence path is both unaffordable *and* uninformative.

**Caveat recorded:** 1025 parameters against 394 training positions. The controls
carry the result, not the sample size.

**Artifact:** `results/trace_L20_int4/probe_analysis.json`.

---

## 10 — Simulated threshold frontier

```bash
.venv/Scripts/python.exe experiments/csgqr/src/sweep_threshold.py \
    --trace experiments/csgqr/results/trace_L20_int4 --feature max_prob
```

Routing quantities are reproduced exactly from the trace; utilities assume
per-token independence and group additivity, so frontier points are hypotheses
until confirmed by generation.

**Result.** Interior points with `0 < ρ < 1` exist across ρ ≈ 0.14–0.77, and
static beats random at **every** matched active fraction.

**Artifact:** `results/trace_L20_int4/threshold_sweep_max_prob.json`.

---

## 11 — Harness self-check

```bash
.venv/Scripts/python.exe experiments/csgqr/src/run_eval.py --self-check --dataset chat_dev
```

610 s, 510 cases, 1.196 s/case. Peak allocated/reserved 951.3 / 978.0 MiB.

**Result: PASS.** mean_quality_score `0.8335958` vs stored
`artifacts/gate_runs/r2_base_chat_dev` `0.8335958`, **|Δ| = 0.000e+00**. The
harness reproduces the baseline exactly, so arm comparisons are on the same scale
as the repository's existing results.

**Artifact:** `results/eval_b0-dense__chat_dev/`.

---

## 12 — Tier-1 arm sweep

```bash
bash experiments/csgqr/run_tier1.sh
```

Nine arms at L20 on `chat_dev` (510 cases each, ~10 min per arm).

**First attempt failed immediately** — a real bug: the 24-group significance
ranking was passed to `b1-delete`, which runs with `n_groups=1`, raising a
permutation error. Fixed by ignoring a ranking whose length does not match the
arm's group count; silently truncating it would have invented an ordering the
significance run never measured. All nine arms were then smoke-tested at
`--limit 4` for validity (routing accounting: delete 0.000, always-on 1.000,
matched arms exactly 0.250, `b7-conf-whole` 0.853 active / 0.147 skip,
`b10-csgqr` 0.598 active / 0.155 skip) before the full sweep was launched.

**Result:** see `FINDINGS.md` and `results/tier1.log`.

---

## 13 — Claim verification

```bash
.venv/Scripts/python.exe experiments/csgqr/src/verify_claims.py
```

**Result:** 65/65 claims OK, 0 failed, 0 missing, at the point of the pivot
(entry 9). Every headline number in the prose is derived from a committed result
file; a claim whose source is absent reports MISSING rather than passing.
