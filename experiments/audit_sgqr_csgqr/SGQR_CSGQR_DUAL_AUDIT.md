# SGQR / CSGQR — Independent Dual Audit

**Audit branch:** `audit/sgqr-csgqr-dual`, created from
`research/sgqr-v2-remnant-recovery` at `d527c19`. No historical file, artifact or
commit was modified. Every new artifact lives under
`experiments/audit_sgqr_csgqr/`.

**What this audit is.** A reproduction and re-analysis of the SGQR / CSGQR record
by code written for the audit, followed by a new controlled experiment that
isolates the confidence gate from everything else the two algorithms differ by.

**What it is not.** Independent of the implementation. The remnant, quantizer and
routing primitives are the campaign's own; re-implementing them would have
discarded the evidence they carry. What *is* independent here: the verification
(33 CPU checks that recompute invariants from raw tensors rather than calling the
campaign's verifier), the claim re-derivation (194 claims re-derived from raw JSON
by audit code), and the experimental design (one execution engine for both
algorithms, an exactly matched-compute allocator, three disjoint splits, and an
oracle ceiling the original design did not contain).

---

## 0. Artifacts produced by this audit

| file | what it is |
| --- | --- |
| `evidence/evidence_manifest.json` | sha256 + git provenance for 132 historical artifacts |
| `results/claim_evidence_table.{json,csv}` | 194 historical claims, each re-derived |
| `results/verify_mechanism.json` | 33 independent CPU checks of the mechanism |
| `results/repro_A_baselines.json` | bit-identical reproduction of CSGQR-v1 batch A |
| `results/repro_B_fast.json` | bit-identical reproduction of the matched-budget arms |
| `results/significance_96.json` | interventional significance over all 96 four-layer groups, with rank stability |
| `results/probes.pt`, `results/probe_report.json` | need and confidence probes: fitted on calib, ridge selected on val |
| `results/trace_{calib,val,test}_int4.jsonl` | per-position gate signal, deletion damage, remnant benefit |
| `results/frontier_test_b*.json` | matched-compute arms, one file per budget |
| `results/analysis_test.json` | factorial, gate and significance tables with Holm correction |
| `results/gate_signal_audit_test.json` | is confidence a proxy for need? |
| `results/systems_audit.json`, `results/timing_samples.csv` | latency, VRAM, and the contention that bounds them |
| `results/per_example_results_*.jsonl` | one row per arm x position |
| `results/results_*.csv`, `results/routing_statistics_*.csv` | machine-readable summaries |
| `RUN_MANIFEST.json` | git commit, config, splits, seeds, hardware for every run |

---

# A. Historical audit

## A.1 Provenance is clean, and the record is genuinely reproducible

`src/preserve_evidence.py` hashed **132** SGQR/CSGQR artifacts and mapped each to
the commit that carries it and to the script that produced it.

| finding | count | reading |
| --- | ---: | --- |
| artifacts hashed | 132 | — |
| worktree content differs from its carrying commit | **0** | (the two rows reported are files this audit itself patched) |
| untracked | 22 | 18 console `.log` files + 3 more logs, all matched by `.gitignore` `experiments/**/results/*.log`, plus one uncommitted script |
| result file predates a change to its producing script | 4 | examined individually below; none invalidates a result |

**The four "producer changed after the result" cases, resolved.**

| artifact | later change to its producer | effect |
| --- | --- | --- |
| `budget.json` | `budget.py` — a mojibake fix in a docstring | none |
| `layer_scan.json` | `measure_layers.py` — added an optional `skip=0` parameter | none; default behaviour unchanged |
| `forced_L20_chat_dev_heldout.json` | `eval_forced.py` — superseded run, not cited by any finding | none |
| `trace_manifest.json` | `trace_utility.py` — added the S7 subset assertion | **none, and the reason matters (§A.4)** |

**One uncommitted script.** `experiments/sgqr_v2/src/calibrate_significance.py`
exists in the working tree, is not committed, and has never been run — no
`significance_4layer.json` exists. It is work in progress, not evidence.

**One provenance limitation, minor.** The console logs that record the exact
command line for each run are gitignored by policy. The mitigation is real: every
result JSON embeds its own configuration (`prompts`, `skip_prompts`,
`ranking_from`, `router`, `probe`, `overrides`, `environment` with git SHA and
dirty flag), so the command is recoverable from the artifact rather than from the
log. `EXPERIMENT_LOG.md` additionally records the commands in prose.

## A.2 Reproduction: bit-identical

Re-running the committed sweeps with the committed code reproduces the committed
numbers **exactly** — point estimates and both bootstrap interval endpoints agree
to 0.0, not to a tolerance.

| batch | arms | agreement |
| --- | ---: | --- |
| A (baselines, `eval_forced`) | 6 | bit-identical on `kl`, `gold_top1`, and both CI endpoints |
| B (matched-budget selection, fixed-group arms) | 5 | bit-identical |

`b1-delete` 0.2805529, `b4-int4-all` 0.0104917, `b13-wrong-layer` 0.7864919,
`b6-static-25` 0.17822, `b6-magnitude-25` 0.23719, `b6-reversed-25` 0.24338,
`b6-static-50` 0.13565 — all recovered to the last stored digit.

The per-token-varying arms (`b5-random-25`, `b9-sig-dyn-*`) were not re-run: they
cost ~10 minutes each in the reference backend, and the fixed-group arms already
establish that the harness is deterministic.

**So the CSGQR-v1 record is `REPRODUCED`, not merely traceable.** That is the
strongest provenance status available and most of the historical record earns it.

## A.3 Claim-evidence table

`src/audit_claims.py` re-derives **194** numerical and structural claims from the
raw JSON by arithmetic written for this audit. It does not call the campaign's own
`verify_claims.py`, so a bug shared between producer and verifier could not hide.

| status | n |
| --- | ---: |
| `VERIFIED` (re-derives from the committed artifact) | **188** |
| `INVALID` | 3 |
| `AMBIGUOUS` | 2 |
| `MISSING_PROVENANCE` | 1 |

Of the 188 verified, the 11 covered by §A.2 are additionally `REPRODUCED`.

The full table is `results/claim_evidence_table.csv`, one row per claim with
claim, algorithm, doc, artifact, carrying commit, configuration, dataset, seed,
derived metric, doc value, audit value, tolerance, status, confound and note.

### The six claims that did not verify

**1. `INVALID` — `RESEARCH_VERDICT.md` names the wrong layers for its own
recommended next experiment.** It proposes retiring "L25, L21, L20, L18 **by the
committed layer scan**". Sorted ascending by `delete_kl`, the committed
`layer_scan.json` gives **L25, L13, L14, L24**. L18 is rank 13 of 28, L20 rank 15,
L21 rank 18. *Consequence: none.* SGQR-v2's preregistration applied the rule
correctly and retired {13, 14, 24, 25}. But the campaign's headline
recommendation, as written, does not follow from the artifact it cites.

**2. `INVALID` — the dense+LoRA control is not byte-matched.** Commit `d527c19`
introduces `--no-retire` as the control that "disambiguates recovery from
fine-tuning". As launched, the pruned arm attaches LoRA to the 24 surviving layers
(168 modules, 9,732,096 parameters, 98.5% of the remnant budget) while the dense
control attaches to all 28 (196 modules, 11,354,112 parameters, **114.9%**). The
control has 16.7% more trainable capacity than the arm it is meant to bound, so
"dense+LoRA reaches the same loss" could not have been read as "the adapter is not
recovering anything". This audit adds `--match-target-layers` and runs the
corrected control (§C.5).

**3. `MISSING_PROVENANCE` — that control never finished.** `lora_r18_DENSE.log`
stops at step 280 of 300 and no adapter artifact exists under
`artifacts/sgqr_v2_adapters/`. No conclusion rests on it either way, but the
commit message describes it as the control the campaign needs.

**4. `INVALID` (cosmetic) — a mismatched denominator in `bench_gpu.py`.** The
JSON field `measured_confidence_over_mlp_ratio` = 2.7945 divides the full monitor
cost by the **INT4 remnant** p50 (2.3221 ms), and the console line prints it next
to "closed form predicts 3.556x" — but the closed form `|V|/(3·d_ff)` is a ratio
against the **dense MLP**. `GPU_BENCHMARKS.md` uses the correct dense denominators
(2.20x projection-only, 13.96x full monitor) and both re-derive exactly, so no
published number is wrong. The console line and the unused JSON field are.

**5. `AMBIGUOUS` — the "damage is slightly superlinear" reading.**
`sgqr_v2/README.md` reports joint four-layer deletion at 0.84821 against 0.7301
for the sum of four single-layer damages, ratio 1.16. The arithmetic re-derives
exactly (1.1617). The *inference* does not follow: **the numerator and denominator
come from different prompt sets**. 0.84821 is 40 held-out prompts (96–135, 343
positions); 0.7301 sums a scan over 64 **calibration** prompts (0–63, 520
positions). No single-layer damage was ever measured on the held-out slice. The
README flags layer adjacency as a confound for this reading; it does not flag this
one. Superlinearity is `NOT_ESTABLISHED`, not refuted.

**6. `AMBIGUOUS` — two AUROC ranges for the same-sounding quantity.**
`FINDINGS.md` F4 reports lens features as need classifiers at AUROC
**0.5374–0.5649** (from `confidence_analysis_d0.1.json`, 798 positions).
`probe_analysis.json` reports the same four features at **0.5791–0.6003** (254
positions). Both are calibration data; neither is on the evaluation slice. The
difference is within sampling noise at these n, but FINDINGS quotes only the lower
range — the one that best supports "confidence fails as a classifier" — without
stating the choice.

### One artifact-hygiene finding

The campaign retracts the additive-proxy headroom result thoroughly in prose
(`FINDINGS.md` F7.1, `NEGATIVE_RESULTS.md` N6 revised, N14). **The machine-readable
artifact was not marked.** `trace_L20_int4/probe_analysis.json` still contains
`topk_capture` with `headroom_fraction_won` 0.66–0.73 and `probe_beats_static:
true`, with no retraction field. Any consumer reading artifacts rather than prose
— which is exactly what the campaign's own `verify_claims.py` does — recovers the
withdrawn claim intact.

## A.4 One bug-ledger entry does not hold up

`BUG_LEDGER.md` **S7** records a `MODERATE` defect: `fraction = k / n_groups` did
not recover `k`, because "`5/24 * 24 == 5.000000000000001`, so `ceil` returned
**6**", which "would have corrupted the utility labels that the entire
significance analysis is built on".

Re-derived (`verify_mechanism.json`, section 7):

* `5/24 * 24 == 5.0` **exactly** in IEEE-754 double. The stated example is false.
* Exhaustively, **no** `k` in [1, 24) triggers the defect at `n_groups = 24`, and
  none in [1, 96) at `n_groups = 96`. Those are the only group counts the campaign
  ever used.
* The named regression test sweeps `n ∈ {7, 12, 24, 48}` — **none of which has any
  affected `k` either**. The test passes identically on pre-fix and post-fix code,
  so it does not pin the defect it is attached to.
* The defect **is** real for other group counts: 158 of the 255 values of `n` in
  [2, 256] have at least one failing `k`, the smallest being `n = 25`.

**Net effect on the record: none, and in the method's favour.** The ledger's
conclusion "the live traces are unaffected" is correct. Its stated reason, its
worked example, and its regression test are not. The fix is still worth keeping,
because the defect is genuine at other group counts.

## A.5 Independent mechanism verification: 33/33

`src/verify_mechanism.py`, CPU only, written for this audit. It re-derives from
raw tensors rather than calling the campaign's suites.

* Group decomposition is exact: 24 group outputs sum to the whole MLP with max
  relative error **1.5e-15** in float64; no projection carries a bias.
* INT4 packing: hand-unpacking the nibbles (low nibble = even index) and applying
  the stored scales and zero points reproduces `dequantize()` with **max absolute
  difference 0.0**.
* Byte accounting recomputed from shape: qweight `3·d_ff·⌈d_model/2⌉` = 4,718,592 B,
  scales `3·d_ff·n_blocks·2` = 147,456 B, zeros `3·d_ff·n_blocks·1` = 73,728 B,
  total **4,939,872 B**, compression against bf16 **3.82082x**, four-layer budget
  **19,759,488 B**. All match.
* Executed sparsity, by instrumenting `PackedMatrix.dequantize` from outside: an
  empty group set issues **zero** dequantize calls; any selection touches exactly
  `3 × |sel| × group_size` rows and no row outside the selection.
* The FP passthrough remnant **copies** rather than aliases the donor (mutating the
  donor after construction moves the remnant output by exactly 0.0) — confirming
  the S19 fix.
* Precision is monotone on real post-norm activations: INT8 output error 0.0099 <
  INT4 0.1713.
* A wrong-layer donor at **identical bytes** is 12.3x further from the correct
  donor's output than INT4 quantization is.

## A.6 Historical status by algorithm

**SGQR** — as an empirical record, the confidence-free arms of the CSGQR-v1
campaign plus the SGQR-v2 four-layer arms. Both are `REPRODUCED` or `VERIFIED`
throughout, with the two documentation defects above.

**The `research/sgqr-core-quantized-remnant` branch contains no empirical record at
all.** It is a single commit (`fc800b8`) of 4,016 lines: a design document, a
reference implementation, and 14 synthetic checks. Its own `FINDINGS.md` says so:
"No model-quality or GPU-performance finding is claimed at branch initialization."
Nothing on that branch is cited by any other document. It is a design track, and
there is nothing there to audit.

**CSGQR** — the confidence-gated arms of CSGQR-v1. `REPRODUCED` for the fixed-group
arms, `VERIFIED` for the gated ones. The historical CSGQR record is honest about
its own weaknesses; §C shows that the one positive gate result in it does not
survive a tighter control.

---

# B. What distinguishes SGQR from CSGQR

## B.1 SGQR, as the repository implements it

Four components, all present in `experiments/sgqr_v2/src/multilayer.py` and the
CSGQR-v1 primitives it builds on:

1. **Structurally retired computation.** `install_sgqr` replaces
   `model.model.layers[j].mlp` with a module that does not contain the original
   projections. With `keep_donor_resident=False` the bf16 projections are deleted
   and the memory is freed. Deletion is the same code path with an empty group set,
   so the damage floor cannot differ from the routed arms by anything except the
   group set.
2. **A frozen low-bit remnant of the exact retired weights.** Blockwise asymmetric
   INT4/INT8, genuinely bit-packed at 4 and 8 bits, all three projections stored
   `[d_ff, d_model]` (`down_proj` transposed) so a group is a contiguous,
   byte-aligned row slice. Never trained, never updated.
3. **Significance-based selection of retained groups.** The intermediate dimension
   is partitioned into `n_groups` blocks. Because `F(x) = Σ_g F_g(x)` exactly,
   executing a subset **is** intermediate-neuron selection. Groups are ranked by an
   *interventional* marginal utility `U_{j,g} = KL(dense‖deleted) −
   KL(dense‖+(j,g))`, and the budget is a global top-k over all layers' groups.
4. **A controlled compute budget.** `ρ` fixes how many groups run. In
   `multilayer.SGQRMlp` it is fixed for the whole run: every token takes the same
   path.

**SGQR contains no per-token decision of any kind.**

## B.2 CSGQR, as the repository implements it

CSGQR is SGQR plus a per-token gate. `router.RoutingPolicy` maps a per-token
confidence `C_t` to a budget:

```
mode="confidence_whole"   k_t = 0 if C_t >= tau_skip else n_groups
mode="three_level"        k_t = 0                 if C_t >= tau_skip
                          k_t = n_groups          if C_t <= tau_full
                          k_t = ceil(rho_t * n)   otherwise,
                          rho_t = ((tau_skip - C_t)/(tau_skip - tau_full))^gamma
```

`C_t` comes either from an exact vocabulary lens (project the pre-MLP residual
through the model's own final norm and LM head, then reduce to `max_prob`,
`neg_entropy`, `prob_margin` or `logit_margin`) or from a trained linear probe on
that residual.

## B.3 The differences that are **not** the confidence gate

These must be named, because the historical comparison carries all of them:

| difference | why it is not the gate |
| --- | --- |
| **Execution engine.** `SGQRMlp` runs one group set for the whole run: one slice, three matmuls. `CSGQRMlp` buckets tokens by group set inside a Python loop with a GPU→CPU sync per token. | CSGQR-v1 measured this at **16x** wall-clock (622 s vs 39 s over the same 414 prompts). Any latency comparison between the two modules measures the loop, not the gate. |
| **Scope.** CSGQR-v1 is single-layer (L20, 24 groups). SGQR-v2 is four-layer (96 groups, global cross-layer allocation). | The two published records are at different scales and are not directly comparable. |
| **Which groups.** The three-level policy couples "how much" to confidence *and* "which" to significance; SGQR fixes the budget and only chooses "which". | A three-level-vs-fixed comparison moves two factors at once. |
| **α calibration and cost-aware selection** exist on the CSGQR path only. | Inert on this checkpoint (α = 1 everywhere, uniform group costs), but present. |

**This audit removes all four.** Both algorithms run through one module
(`core.LevelledSGQRMlp`) which executes a per-position level index; a fixed-budget
SGQR arm is that module with every position on one level. Both run at four-layer
scope over the same 96 groups with the same ranking. Both are given the **same
total number of group-token products, exactly**. The gate signal is read from the
residual entering layer 13, which no arm modifies, so every arm is gated on
literally identical numbers.

What is left as the controlled difference is exactly one thing: **the integer
vector of per-token budgets.**

## B.4 The experimental design

| axis | value |
| --- | --- |
| base model | `artifacts/rineya-core-en32k-28L` (Qwen3, 28L, d_model 1024, d_ff 3072, \|V\| 32768, bf16) |
| retired layers | 13, 14, 24, 25 — the four lowest-damage MLPs in the committed 28-layer scan |
| remnant | INT4 asymmetric, block 128, 24 groups per layer, 96 total, 19,759,488 B |
| ranking | interventional significance over all 96 groups, conditioned on all four layers deleted, calibrated on prompts 0–95 |
| gate signals | need probe (ridge on the layer-13 residual), confidence probe, exact vocabulary lens, and a true-utility **oracle** |
| evaluation | teacher-forced paired, reply positions only, prompt-clustered bootstrap, 4000 draws |
| splits | calib 0–95 (96 prompts / 798 positions) · val 96–223 (128 / 1143) · **test 224–509 (286 / 2687)** |
| what is fitted where | significance and probes on calib; ridge strength on val; nothing on test |
| budgets | ρ ∈ {0.10, 0.20, 0.30, 0.50, 0.75}, plus ρ = 0 (deletion) and ρ = 1 (always-on) |
| seeds | 3 for every random control |

The historical campaign used a two-way split (calib 0–95, held-out 96–509) and
read at least one reported quantity — the probe/lens frontier crossing at ~0.78
activation — off the held-out set without labelling it exploratory. The three-way
split here removes that.

---

*(Sections C through H and the verdicts follow.)*

# C. Controlled results: SGQR against CSGQR under identical conditions

All numbers below are on the **test split** — `chat_dev` prompts 224–509, 286
prompts, 2,687 reply positions — with prompt-clustered bootstrap intervals
(4,000 draws) and paired per-position contrasts. Nothing reported here was
selected on this split.

## C.1 Compute is matched to the unit, not to a tolerance

| budget ρ | group-token products executed, **every arm** |
| ---: | ---: |
| 0.30 | 1,509,896 |
| 0.50 | 2,516,496 |
| 1.00 | 5,032,992 |

Twenty-one arms at ρ=0.30 and twenty-one at ρ=0.50 executed *exactly* those
totals. The historical campaign's sharpest gate comparison was made at 0.6772
against 0.6819 activation — a 0.7% compute difference inside a comparison whose
effect size was 2.4%.

## C.2 The frontier at ρ = 0.30

| arm | KL vs dense | 95% CI | gold top-1 |
| --- | ---: | --- | ---: |
| dense | 0.00000 | — | 0.2884 |
| always-on INT4 remnant (ρ=1) | 0.02415 | [0.02295, 0.02546] | 0.2873 |
| **oracle token gate** (ceiling) | **0.24913** | [0.23997, 0.25860] | 0.2877 |
| CSGQR, confidence-probe gate | 0.38694 | [0.36922, 0.40497] | 0.2795 |
| CSGQR, need-probe gate | 0.39662 | [0.37793, 0.41578] | 0.2761 |
| randtok (CSGQR budgets permuted), s3 | 0.39765 | [0.37882, 0.41721] | 0.2784 |
| randtok, s2 | 0.39863 | [0.37941, 0.41859] | 0.2780 |
| **SGQR, fixed budget** | **0.39964** | [0.38058, 0.41943] | 0.2776 |
| randtok, s1 | 0.40041 | [0.38120, 0.42029] | 0.2776 |
| CSGQR, exact vocabulary lens gate | 0.40217 | [0.38379, 0.42198] | 0.2758 |
| CSGQR, binary probe gate | 0.42495 | [0.39487, 0.45802] | 0.2631 |
| fixed budget, random groups (3 seeds) | 0.47218 / 0.58797 / 0.65679 | — | 0.2687 / 0.2475 / 0.2613 |
| fixed budget, magnitude groups | 0.60230 | [0.57760, 0.62834] | 0.2661 |
| fixed budget, reversed significance | 0.77932 | [0.73831, 0.82030] | 0.2445 |
| four layers deleted (ρ=0) | 0.84122 | [0.80022, 0.88357] | 0.2493 |

## C.3 The frontier at ρ = 0.50

| arm | KL vs dense | 95% CI | gold top-1 |
| --- | ---: | --- | ---: |
| always-on INT4 remnant (ρ=1) | 0.02415 | [0.02295, 0.02546] | 0.2873 |
| **oracle token gate** (ceiling) | **0.14617** | [0.14120, 0.15118] | 0.2858 |
| CSGQR, confidence-probe gate | 0.27715 | [0.26376, 0.29118] | 0.2791 |
| CSGQR, exact vocabulary lens gate | 0.28175 | [0.26628, 0.29767] | 0.2769 |
| CSGQR, binary probe gate | 0.29030 | [0.26722, 0.31569] | 0.2668 |
| randtok, s3 | 0.30631 | [0.29190, 0.32093] | 0.2821 |
| CSGQR, need-probe gate | 0.30647 | [0.29210, 0.32046] | 0.2840 |
| randtok, s2 | 0.30713 | [0.29280, 0.32146] | 0.2810 |
| **SGQR, fixed budget** | **0.30779** | [0.29345, 0.32199] | 0.2821 |
| randtok, s1 | 0.30979 | [0.29524, 0.32418] | 0.2825 |
| fixed budget, random groups (3 seeds) | 0.31194 / 0.34287 / 0.46519 | — | 0.2761 / 0.2709 / 0.2694 |
| randtok-binary (3 seeds) | 0.40964 / 0.42335 / 0.44315 | — | — |
| fixed budget, magnitude groups | 0.45166 | [0.43555, 0.46809] | 0.2706 |
| fixed budget, reversed significance | 0.68892 | [0.65418, 0.72401] | 0.2415 |

## C.4 The factorial (brief section 11)

Both random cells use the **same** random group ordering so the 2×2 is paired.
KL to dense; lower is better; every cell at identical compute.

| ρ | remnant only (flat + random) | + significance (flat + sig) | + confidence (gate + random) | + both (gate + sig) |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.58797 | 0.39964 | 0.58513 | 0.39662 |
| 0.50 | 0.31194 | 0.30779 | 0.31607 | 0.30647 |

| ρ | significance effect | confidence effect | interaction |
| ---: | --- | --- | --- |
| 0.30 | **−0.18833** [−0.21650, −0.16256], d = −0.374 | −0.00284 [−0.00738, +0.00168], d = −0.024 | −0.00019 [−0.00524, +0.00495] |
| 0.50 | −0.00415 (this seed) / **−0.06554** [−0.07750, −0.05334] vs pooled seeds | +0.00413 (this seed) | −0.00546 [−0.01128, +0.00046] |

The ρ=0.50 row shows why three seeds were necessary. Random seed 1 happened to
land at 0.31194, close to significance; the three seeds span **0.31194 to
0.46519**. Against the pooled random arms the significance effect is
−0.06554 with a non-overlapping interval. A single-seed random control — which is
what the historical campaign used — could have reported anything from "no effect"
to a large one depending on the draw.

**Reading.** At ρ=0.30, choosing *which groups* run is worth **66×** more than
choosing *which tokens* get more of them (0.18833 vs 0.00284), and the two do not
interact. At ρ=0.50 the significance effect shrinks (fewer groups are excluded, so
the choice matters less) and the confidence effect remains at zero.

## C.5 The oracle says the headroom is real and the gate does not reach it

An oracle that allocates the same total budget by **true per-token remnant
utility** measured on the same split reaches:

| ρ | SGQR | best deployable gate | oracle | fraction of oracle headroom captured |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.39964 | 0.38694 (confidence probe) | 0.24913 | **8.4%** |
| 0.50 | 0.30779 | 0.27715 (confidence probe) | 0.14617 | **19.0%** |

So a perfect token-level gate would be worth a great deal — 38% of the remaining
divergence at ρ=0.30, 53% at ρ=0.50 — and the best available signal captures
under a fifth of it. The gate is not useless because the headroom is absent; it is
nearly useless because the signal is weak.

# D. What the confidence gate contributes

## D.1 The gate signal, measured directly (brief section 6)

`results/gate_signal_audit_test.json` and `results/within_prompt_signal_test.json`,
on the 2,687 held-out reply positions. Every signal is oriented so that **higher =
more remnant needed**.

Context first, because it decides what a gate could possibly do:

| quantity | value |
| --- | ---: |
| mean deletion damage `D_t = KL(dense‖deleted)` | 0.84122 |
| mean residual with the full remnant | 0.02415 |
| KL recovery of the always-on INT4 remnant | **0.9713** |
| positions with positive remnant utility | **99.78%** |
| positions with utility above 0.1 | **85.08%** |
| share of total utility in the top decile of positions | 39.2% |
| share of utility variance **between** prompts | 12.0% |

So "does this token need the remnant?" is answered *yes* almost everywhere — the
useful question is *how much*, and 88% of what a token gate could exploit lives
**within** a prompt.

**Correlation with what the gate is supposed to track:**

| signal | ρ vs deletion damage | ρ vs utility | **ρ vs utility, within prompt** | AUROC (U>0.1) | utility captured in its top 20% |
| --- | ---: | ---: | ---: | ---: | ---: |
| lens `max_prob` | +0.0468 | +0.0438 | **−0.0016** | 0.5548 | 0.186 |
| lens `neg_entropy` | +0.0931 | +0.0892 | **+0.0364** | 0.5733 | 0.198 |
| lens `prob_margin` | +0.0305 | +0.0282 | **−0.0062** | 0.5439 | 0.185 |
| lens `logit_margin` | +0.0144 | +0.0127 | **−0.0164** | 0.5282 | 0.184 |
| trained **need probe** | +0.3013 | +0.3051 | **+0.3316** | 0.6149 | 0.319 |
| trained confidence probe | +0.0703 | +0.0755 | +0.0919 | **0.4169** | 0.227 |
| position in the reply (free) | — | +0.1786 | +0.1752 | — | 0.205 |
| oracle (true utility) | +0.9992 | +1.0000 | — | 1.0000 | 0.579 |

Three findings, and the third is the one that matters.

**1. Vocabulary-lens confidence carries essentially no token-level information
about remnant need.** Within a prompt, three of the four lens features correlate
*negatively* with utility and the fourth reaches +0.036. Their top-20% selections
capture 18.4–19.8% of the utility against 20% for a random selection — that is, no
better than chance. Pooled correlations of +0.01 to +0.09 are the prompt-level
component leaking in.

**2. A trained need probe does carry real signal**, and it is not a position
counter: within-prompt ρ = +0.332 against +0.175 for position alone. Its
quartile-gap `E[U | high] − E[U | low]` is **+0.675 [+0.568, +0.784]**, interval
excluding zero.

**3. That signal is not enough.** The probe captures 31.9% of the utility in its
top 20% against an oracle's 57.9% and a random 20%. Converted into the actual
routing decision (§C.5) it buys 8–19% of the oracle's headroom.

**The δ sweep the historical campaign did not run.** `LIMITATIONS.md` §5 records
that `δ = 0.1` was chosen from the data and its sensitivity never swept. Swept
here, the picture is stable: lens AUROC 0.53–0.59 for every δ ≥ 0.05, need probe
0.61–0.72, and the *confidence* probe **below 0.5 at every δ up to 0.25** — it is
anti-predictive as a need classifier while still having a positive quartile gap,
which is exactly the pattern that makes a single summary statistic unsafe here.

**Is the readout or the state the binding constraint?** `results/probe_ceiling.json`:
a one-hidden-layer MLP on the same residual, selected on validation, reaches test
ρ = **+0.147** against the linear probe's **+0.305** — it does not clear the
linear model. An unregularized linear fit reaches train ρ = 1.000 and test
ρ = **−0.083**: pure memorization. **The state, not the readout, is the limit.**
The residual entering the first retired layer does not contain much about how
valuable the remnant will be, and no better probe placed there will fix that.

**Failure quadrants (brief section 6).** At a quartile split, for every signal
including the probe, **81–94% of the positions the signal ranks lowest still have
utility above 0.1** — the "safe to skip" quartile is not safe. And 8–17% of the
positions each signal ranks highest have little benefit. The bottom quartile of
the probe's ranking still holds 16.7% of all positive utility.

## D.2 Does the remnant change answers, and are the changes right?

Running the whole remnant changes the deleted model's argmax on **40.5%** of
positions. Of those changes, 19.1% land on the gold token and 9.8% break a token
the deleted model had right. Net gold top-1: deletion 0.2493 → remnant 0.2873
against dense 0.2884, paired delta **+0.03796 [+0.02446, +0.05106]**.

So at four-layer scope the remnant recovers *capability*, not only fidelity. This
is the one place where the four-layer setting differs qualitatively from CSGQR-v1's
single-layer setting, where deletion cost exactly 0.00000 gold top-1 and there was
no capability damage to recover.

# E. What significance selection contributes

## E.1 The ranking itself

`results/significance_96.json`. Interventional utility for all 96 groups,
conditioned on all four layers being deleted, on the 96 calibration prompts
(98 forward passes per prompt, 919 s).

| quantity | value |
| --- | ---: |
| mean `KL(dense ‖ all four deleted)` on calibration | 0.80966 |
| groups with **negative** marginal utility | **38 of 96** |
| sum of the 96 singleton utilities | 0.75634 |
| per-layer utility mass (L13 / L14 / L24 / L25) | 0.069 / 0.162 / 0.282 / 0.243 |

Two things follow immediately. Nearly **40% of groups individually make the
compressed model worse**, so a budget spent uniformly is spent partly on harm —
which is why the random control is so weak. And the four layers are far from equal:
L13 contributes 9% of the recoverable mass and L24 contributes 37%, so a global
top-k across layers is not a formality. At ρ=0.25 the global top-24 draws 4 groups
from L13, 8 from L14, 7 from L24 and 5 from L25.

**Rank stability, measured rather than assumed** (400 prompt-bootstrap resamples
of the calibration set):

| statistic | mean | 5th percentile |
| --- | ---: | ---: |
| Spearman against the full-sample ranking | **0.9414** | 0.8969 |
| top-10 overlap | 0.7922 | 0.6000 |
| top-24 overlap (ρ=0.25) | 0.8514 | 0.7917 |
| top-48 overlap (ρ=0.50) | 0.9135 | 0.8542 |

The ranking is stable across calibration draws. This is worth contrasting with
CSGQR-v1's finding that *per-token* orderings agree with the global ranking at
mean Spearman 0.070: which groups matter is highly token-specific, but *on
average* it is a reproducible property of the layer.

## E.2 Against random and magnitude, at matched compute, with seeds

| ρ | contrast | ΔKL | 95% CI | Holm-adjusted p |
| ---: | --- | ---: | --- | ---: |
| 0.30 | significance vs **pooled** random (3 seeds) | **−0.17267** | [−0.19244, −0.15344] | 0.0020 **SIG** |
| 0.30 | significance vs magnitude | −0.20265 | [−0.22404, −0.18126] | 0.0020 **SIG** |
| 0.30 | significance vs reversed significance | −0.37967 | [−0.41610, −0.34326] | 0.0020 **SIG** |
| 0.30 | magnitude vs random seed 1 | +0.01432 | [−0.01639, +0.04356] | 0.3305 ns |
| 0.50 | significance vs **pooled** random (3 seeds) | **−0.06554** | [−0.07750, −0.05334] | 0.0020 **SIG** |
| 0.50 | significance vs magnitude | −0.14387 | [−0.16059, −0.12759] | 0.0020 **SIG** |
| 0.50 | significance vs reversed significance | −0.38113 | [−0.41613, −0.34678] | 0.0020 **SIG** |
| 0.50 | magnitude vs random seed 1 | **+0.13973** | [+0.12374, +0.15605] | 0.0020 **SIG** |

On capability, at ρ=0.30 significance beats random by **+0.03015 gold top-1**
[+0.02051, +0.04025], Holm-significant. At ρ=0.50 the capability gap narrows to
+0.00595 and is not significant, consistent with the shrinking KL gap.

**Three conclusions.**

1. **Interventional significance beats random selection at both budgets**, by a
   wide margin at ρ=0.30 and a clear one at ρ=0.50.
2. **The magnitude heuristic is worse than random**, reproducing CSGQR-v1's most
   surprising finding at four-layer scale, and at ρ=0.50 the gap is significant
   against even the *best* of three random seeds. Weight norm is not a proxy for
   interventional utility; it is worse than not choosing.
3. **The order carries the information, not the identity of the selected set**:
   reversing the ranking is worse than random at both budgets.

**One methodological warning the historical record needs.** The random control's
own seed spread is large — 0.47218 to 0.65679 at ρ=0.30 and 0.31194 to 0.46519 at
ρ=0.50. CSGQR-v1 ran **one** random seed. At ρ=0.50 the seed it drew would have
supported "significance barely beats random" (0.30779 vs 0.31194, a 1.3% gap);
against the pooled seeds the gap is 17.6%. The direction of the historical
conclusion survives; the magnitude of any single-seed estimate does not.

# F. Mechanistic recovery versus capability recovery

The brief asks that these are not conflated. At four-layer scope they mostly
agree, which is itself a finding — it did not hold at one layer.

Across the 39 arms of the two budget points: Spearman(KL, gold top-1) =
**−0.885**, Spearman(KL, perplexity) = **+0.803**. Divergence from dense and loss
of reference-prediction accuracy move together.

| arm | KL | KL recovery | gold top-1 | capability recovery | perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.00000 | — | 0.2884 | — | 117.49 |
| always-on INT4 remnant | 0.02415 | **0.9713** | 0.2873 | **0.971** | 117.17 |
| SGQR ρ=0.50 | 0.30779 | 0.634 | 0.2821 | 0.839 | 97.15 |
| SGQR ρ=0.30 | 0.39964 | 0.525 | 0.2776 | 0.723 | 99.52 |
| four layers deleted | 0.84122 | 0.000 | 0.2493 | 0.000 | 195.67 |

**The always-on INT4 remnant is statistically indistinguishable from dense on
capability** (Δ gold top-1 = −0.00112, CI [−0.00602, +0.00380]) while recovering
97.1% of the divergence. **SGQR at ρ=0.30 is also indistinguishable from dense**
after Holm correction (−0.01079, adjusted p = 0.259) while running 30% of the
remnant.

**Perplexity must not be used as the capability metric here, and this is a trap
the historical record already fell into once.** Every partial arm has *lower*
gold-token perplexity than dense — SGQR at ρ=0.30 scores 99.52 against dense's
117.49 — while diverging 0.40 nats and losing gold top-1. Removing part of an MLP
flattens an over-confident distribution, and on a model that is right 28.8% of the
time, flattening lowers average NLL. `sgqr_v2`'s commit `d527c19` reports the
same-byte LoRA arm at perplexity 69.50 "far BETTER than dense at 122.68"; the
commit correctly refuses to read that as recovery, and this measurement shows the
effect is not specific to LoRA — it appears in every partial remnant arm too.
Gold top-1 and KL are the metrics that behave.

# H. Limitations of this audit

Ordered by how much each constrains the conclusions.

**H.1 Implementation independence is partial.** The remnant, quantizer, router and
evaluation harness are the campaign's own code. This audit re-derived their
invariants from raw tensors (33 checks, §A.5) and instrumented them from outside,
but a defect in a primitive that all arms share would move every arm together and
would not show up in a contrast. What the design does protect against is a defect
that moves *one* arm: SGQR and CSGQR execute the same module with the same kernels
and differ only by an integer vector.

**H.2 One checkpoint, one dataset, one layer set, short replies.** Qwen3-shaped,
0.47 B parameters, 28 layers, `d_ff` 3072, `|V|` 32768. One split of `chat_dev`
(510 prompts total, median reply 7 tokens, `max_len` 256). Four retired layers
chosen by a rule from a committed scan. `rag_dev` (752 cases) was not touched, and
the three `*_SEALED` sets were never read by any script here. Nothing in this
report generalises to a larger model, a longer generation, or a different domain
without being re-run there. The vocabulary-lens result in particular is
shape-dependent: `|V|/(3·d_ff)` is 3.56 here and worse on production models.

**H.3 The gate is read at one placement.** All four retired layers are gated from
the residual entering the *first* of them (layer 13). That is what makes the signal
arm-invariant and therefore identical across arms, which is the property that makes
the comparison clean — but it handicaps the gate for layers 24 and 25, where a
deployed system could read a later, better-informed state. A per-layer gate would
break arm-invariance (the later residuals depend on what the earlier remnants did)
and would need a different, weaker control. **This is the most important untested
alternative**, and §D.1's finding that a non-linear readout does not beat a linear
one at layer 13 does not settle what a probe at layer 24 would do.

**H.4 The oracle is an oracle.** It allocates by utility measured on the same
positions it is scored on. It is an upper bound on any token-level gate, not an
achievable arm, and it is labelled as such everywhere it appears.

**H.5 Budget granularity.** Per-token budgets are multiples of 4 groups (of 96) in
the two full-fidelity budget points and 8 groups in the supplementary sweeps. A
finer grid would let a gate express a more extreme allocation. The direction of the
bias is against the gate, so the null result for the graded gate is conservative;
the binary gate is unaffected because {0, 96} is representable at every rung.

**H.6 Shared, thermally-throttled GPU.** Every measurement here ran on one
RTX 4070 Laptop (8 GiB) shared with a desktop compositor and a browser. During the
sweeps `nvidia-smi` reported 2.5–3.2 GiB held by other processes, 82 °C, and free
memory as low as 348 MiB. **Quality numbers are unaffected** — they are
deterministic and were confirmed bit-identical across separate runs of the same arm
— but every absolute latency number in §G carries the label the systems audit
assigns it, and no absolute latency conclusion should be carried out of this
report.

**H.7 Statistical scope.** 286 test prompts, 2,687 reply positions, clustered by
prompt. Three seeds for every random control; the trained probes are closed-form
ridge with no optimiser seed. Holm–Bonferroni is applied within each pre-declared
family, and the families were fixed in `src/analyze.py` before the numbers were
read. Contrasts *between* families are not jointly corrected.

**H.8 No independent replication.** One machine, one operator, one week. The
artifacts and scripts are committed so that a replication is possible; none has
happened.

