# DAST V1 — CLOSED — NEGATIVE (EXPLORATORY DEV)

**Status:** `CLOSED — NEGATIVE (EXPLORATORY DEV)`  
**Branch HEAD at audit:** `research/depth-aligned-semantic-transitions-v1`  
**Sealed TEST:** never opened, never read. No config names `test.jsonl`.  
**Preregistration:** [`PREREGISTRATION.md`](PREREGISTRATION.md) remains DRAFT and unsigned. These results are not confirmatory.

This is the claim-boundary document for the completed V1 training formulation. It does not reopen V1, does not move the 2.0 pp practical margin, and does not license operators, causal feedback, or TEST.

Canonical tables: [`FINDINGS_G3_SEEDS.md`](FINDINGS_G3_SEEDS.md). Chronology: [`../PROGRESS.md`](../PROGRESS.md).

---

## 1. What was tested

Aligning **cumulative compact reasoning states** to **increasing Transformer depth**, through one shared projector, while the full chain-of-thought LM objective continues.

| Protocol constant | Value verified on all 12 complete V1 manifests |
|---|---|
| model | `Qwen/Qwen3-0.6B` |
| LoRA | r=16, alpha=32 |
| budget | 4,000 train / 800 DEV, 250 optimizer steps |
| seeds | 17, 29, 43 |
| readout | `checkpoint_readout: final_norm` |
| fractions | `[0.25, 0.50, 0.75, 0.964]` |
| resolved checkpoints | **layers 7 / 14 / 21 / 27** |
| target geometry | `target_centering: within_problem` |
| eval | greedy generation, `max_new_tokens: 384`, split `dev` |
| TEST | `allow_sealed_test: false`, `preregistration_frozen: false` |

**Deepest DAST readout is not the LM head input.** Layer 27 is pooled after the model’s final RMSNorm. The language-model head still reads post-final-norm **layer 28** (`hidden_states[28]`). Those tensors are not the same.

---

## 2. Local run audit (2026-09-01)

Twelve complete `(arm, seed)` pairs. A run counts as complete only when `run_manifest.json`, `dev_summary.json`, and `generation_summary.json` exist. Incomplete/OOM directories are not in this table.

Seed 17 primary arms live under `g2_budget_*`; seed 17 `state_final_repeated` and seeds 29/43 live under `g3_*`. Grouping is by manifest `(arm, seed)`.

| arm | seed | directory | accuracy | `L_state` | config_sha256 (prefix) |
|---|---:|---|---:|---:|---|
| `cot_baseline` | 17 | `g2_budget_cot_baseline` | 0.89375 | 0 | `bcd8cf7b` |
| `state_ordered` | 17 | `g2_budget_state_ordered` | 0.8775 → **0.8788** | 0.2622 | `c554f750` |
| `state_reverse` | 17 | `g2_budget_state_reverse` | 0.8900 | 0.2520 | `8b3b86d5` |
| `state_final_repeated` | 17 | `g3_s17_state_final_repeated` | 0.8850 | 0.1410 | `149423c8` |
| `cot_baseline` | 29 | `g3_s29_cot_baseline` | 0.8950 | 0 | `3959037b` |
| `state_ordered` | 29 | `g3_s29_state_ordered` | 0.88625 | 0.2644 | `82565075` |
| `state_reverse` | 29 | `g3_s29_state_reverse` | 0.89125 | 0.2518 | `edc79a32` |
| `state_final_repeated` | 29 | `g3_s29_state_final_repeated` | 0.89125 | 0.1433 | `9ddaf9a1` |
| `cot_baseline` | 43 | `g3_s43_cot_baseline` | 0.8925 | 0 | `2035f699` |
| `state_ordered` | 43 | `g3_s43_state_ordered` | 0.88625 | 0.2621 | `45168fc7` |
| `state_reverse` | 43 | `g3_s43_state_reverse` | 0.8950 | 0.2563 | `7baa471f` |
| `state_final_repeated` | 43 | `g3_s43_state_final_repeated` | 0.9100 | 0.1435 | `34d8b26b` |

Every listed run: `eval_split: dev`, `allow_sealed_test: false`, 800 records, adapters present. G1 pilots (`g1_*`, `g1b_*`) are underpowered engineering/exploratory and are **not** in the V1 closure table.

---

## 3. Result (fixed)

All twelve runs re-scored with one scorer (see §3a). Corrected values are bold.

| arm | s17 | s29 | s43 | mean |
|---|---:|---:|---:|---:|
| `state_final_repeated` | 0.8850 | 0.8912 | 0.9100 | **0.8954** |
| `cot_baseline` | 0.8938 | 0.8950 | 0.8925 | **0.8937** |
| `state_reverse` | 0.8900 | 0.8912 | 0.8950 | 0.8921 |
| `state_ordered` | **0.8788** | 0.8862 | 0.8862 | **0.8838** |

Pooled, 2,400 paired problems, exact two-sided McNemar:

| contrast | Δ | p | as published |
|---|---:|---:|---|
| ordered − final_repeated | **−0.0117** | **0.0167** | −0.0121, p=0.0134 |
| ordered − baseline | **−0.0100** | **0.0449** | −0.0104, p=0.0370 |
| ordered − reverse | **−0.0083** | **0.1279** | −0.0088, p=0.1102 |
| reverse − final_repeated | −0.0033 | 0.5514 | unchanged |
| reverse − baseline | −0.0017 | 0.8000 | unchanged |
| final_repeated − baseline | +0.0017 | 0.7843 | unchanged |

## 3a. Why these differ from the published numbers

`FINDINGS_G3_SEEDS.md` states the scorer was "frozen at `f160f38`". It was not.
`compare.py` reads the `correct` field baked into each predictions file at
generation time, so a run generated before a scorer change keeps its old verdict
permanently. **All three seed-17 arms** were generated at `e3ca792` / `1dce3cd`,
both of which predate `f160f38` — the commit that made `state`-type answers
order-insensitive.

Re-applying the current scorer to all twelve saved prediction files
([`../scripts/rescore_consistency.py`](../scripts/rescore_consistency.py),
[`../artifacts/rescore_consistency.json`](../artifacts/rescore_consistency.json))
changes **verdicts only** — generated text is fixed and committed — and flips
exactly one problem: `rineya_si_v1_recursive_state_updates_1300` in
`g2_budget_state_ordered`, whose gold bag `A=ON, B=OFF, C=OFF, D=OFF` was emitted
as `D=OFF, B=OFF, C=OFF, A=ON`.

The direction of every finding is unchanged. Its **strength is lower than
published**: the deficit is 1.00 pp rather than 1.04, and its p-value sits just
under 0.05 rather than comfortably below it.

## 3b. What this endpoint can detect

[`../scripts/endpoint_power.py`](../scripts/endpoint_power.py),
[`../artifacts/endpoint_power.json`](../artifacts/endpoint_power.json). Exact
McNemar against the measured 5.54% discordance rate:

| seeds | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| MDE (pp) | 2.27 | 1.71 | **1.39** | 1.08 | 0.74 | 0.53 |

**The observed 1.00 pp deficit is below its own three-seed MDE of 1.39 pp.** It
was detected at p=0.045 with roughly **54% post-hoc power**. That is the
quantitative form of the caution in `FINDINGS_G3_SEEDS.md` §4a, and it is the
reason this result is reported as directional rather than firm.

---

## 4. What is supported

- The V1 thesis — that forcing chronological cumulative states onto increasing depth **improves** held-out reasoning — is **not supported**.
- `state_ordered` is below `cot_baseline` in **all three seeds**.
- G3 **fails as a directional efficacy gate**: ordered did not beat `state_final_repeated`; the contrast is **adverse** (−1.17 pp, p = 0.0167).
- Semantic-stage information is already **linearly decodable** from the untouched backbone, and the trained projector does not match it. On 600 DEV problems a ridge probe reached margin ≈ 0.9947 / top-1 ≈ 0.825 / `L_state` ≈ 0.2229; re-measured on all 2,400 with the committed pipeline it reaches margin **1.1216** / top-1 **0.8653** / `L_state` **0.1240**, against the trained ordered projector's 0.935 / 0.833 / **0.263**. High trained mechanism metrics are **not** evidence that DAST created a depth-wise reasoning trajectory.
- The supervised signal does not require the depth stack at all. It is present in the **token-embedding centroid at layer 0**, before any decoder layer, and the four *shallowest* states fit the objective slightly better than V1's spread-out sites (§7). This is the most likely reason `L_state` is easy to drive and uninformative about reasoning.

## 5. What is unsupported

- Reverse superiority. Ordered − reverse = −0.83 pp, p = 0.128. Direct contrast, not a comparison of two separate p-values against 0.05.
- G2. `state_shuffled` was never run at 800 DEV × 3 seeds.
- A practical effect at the **pre-declared 2.0 pp** margin. The ordered deficit is below that bar.
- Confirmatory status. Unsigned preregistration.
- Universal transformer behaviour. One 0.6B model, one procedurally generated corpus, one LoRA schedule.
- Formal equivalence of final-repeated to baseline. Mean Δ = +0.17 pp, p = 0.78, but a 2.5 pp seed range.

## 6. What is known to be non-evidence

- Order margin, checkpoint top-1, and monotonicity of the **trained** ordered projector, once compared to the frozen ridge.
- Reverse `L_state` being slightly easier than ordered `L_state`. That is optimization ease, not reverse reasoning, and not a capability result.
- Isolated McNemar p-values after ~13 examined contrasts (Bonferroni threshold ≈ 0.0038 admits neither headline).

## 7. Frozen-backbone permutation (C1, non-causal)

[`../artifacts/c1b_permutation_k4_dev.json`](../artifacts/c1b_permutation_k4_dev.json), 20 splits, 24 permutations, **n = 1379** after the distinctness filter: identity median rank **7 of 24** (4/20 first); reverse median rank **18** (0/20 first); mean `L_state` spread **0.00269**, about **1.89%** of identity `L_state`.

The filter matters and was broken until now. `DepthCache.trajectory_lengths` recorded a constant, so `--require-distinct` dropped nothing; the first pass therefore included the **1,021 of 2,400 DEV problems (42.5%)** whose four depth targets are not all distinct, for which a permutation swapping two equal targets is a no-op. Removing them makes the frozen model look **more** indifferent (spread 5.84% → 1.89%) and moves identity from rank 2 to rank **7**.

So: reverse-preference is **refuted** (reverse 18/24, never first, across all three passes), but forward-preference is **not supported** (identity mid-pack; the best assignment is odd/even interleave by 0.0003 `L_state`). Decodability, not computation.

**42.5% duplication is a fact about V1, not only about this analysis.** For those problems the "four distinct cumulative states" premise does not hold and `state_ordered` is partly a `state_final_repeated` arm.

K=8 exhaustive permutation is **not** licensed on this corpus: only ~4.6% of DEV traces have eight distinct compact states, so extra slots are repeated targets.

**C1c — the signal precedes the stack.** The C1a profile found every centred stage
best decoded at layer 1, which alone could not separate "shallow is best" from
"the targets live in input-embedding space" — `FrozenTextTargetEncoder` builds them
from the frozen input embedding. Adding **layer 0** settles it: the token-embedding
centroid scores within 0.003 of the best layer and beats every layer from 3 upward.
Matched controls ([`../artifacts/c1c_depth_information_controls.json`](../artifacts/c1c_depth_information_controls.json)):

| condition | `L_state` | margin | top-1 |
|---|---:|---:|---:|
| layers 7/14/21/27 (V1 sites) | 0.12397 | +1.1216 | 0.8653 |
| layers 0/1/2/3 | **0.11127** | +1.1371 | 0.8615 |
| layer 0 at all four depths | 0.84576 | +0.0000 | **0.2500** (chance) |
| right depths, wrong problem | 0.67474 | +0.4165 | 0.5813 |
| *trained* `state_ordered` | *0.26290* | *+0.9350* | *0.8332* |

The chance-level `embedding_only` row is the control that keeps the others honest:
a shared readout fed identical input at every depth cannot separate stages, so
0.865 elsewhere is real four-way discrimination. But it needs only four
*distinguishable* representations — not a chronological axis — and the bottom four
layers do it slightly better than V1's. See
[`FINDINGS_C1_FROZEN.md`](FINDINGS_C1_FROZEN.md) §C1c.

**§2b's rank/spread figures are withdrawn.** `FINDINGS_G3_SEEDS.md` §2b cites "20
random half/half splits", identity median rank 3 vs reverse 12, and spread
0.0096/4.3%. `scripts/permutation_preference.py` contains no RNG and no split loop,
and its committed artifact says identity rank 1, reverse rank 4, spread 0.00816
(3.66%). The direction replicates three times; those specific numbers should not be
cited.

## 8. Open mechanistic questions

H1–H7 are stated in [`DAST_V2_HYPOTHESIS_TREE.md`](DAST_V2_HYPOTHESIS_TREE.md). V1 does **not** decide among them. The next authorized work is cheap diagnosis (B1/B2/C1), not a more complicated architecture and not the unsigned 9-run [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md) matrix.

## 9. Stop rules still in force

- No additional V1 seeds to hunt p < 0.05.
- TEST stays sealed.
- Operators / causal feedback stay gated on a **positive capability** result that V1 did not produce.
- Do not rewrite this conclusion unless new properly controlled evidence requires it.
