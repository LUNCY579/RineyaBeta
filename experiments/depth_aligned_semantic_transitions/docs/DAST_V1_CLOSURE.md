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
| `state_ordered` | 17 | `g2_budget_state_ordered` | 0.8775 | 0.2622 | `c554f750` |
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

| arm | s17 | s29 | s43 | mean |
|---|---:|---:|---:|---:|
| `state_final_repeated` | 0.8850 | 0.8912 | 0.9100 | **0.8954** |
| `cot_baseline` | 0.8938 | 0.8950 | 0.8925 | **0.8937** |
| `state_reverse` | 0.8900 | 0.8912 | 0.8950 | 0.8921 |
| `state_ordered` | 0.8775 | 0.8862 | 0.8862 | **0.8833** |

Pooled, 2,400 paired problems:

| contrast | Δ | p |
|---|---:|---:|
| ordered − final_repeated | −0.0121 | 0.0134 |
| ordered − baseline | −0.0104 | 0.0370 |
| ordered − reverse | −0.0088 | 0.1102 |
| reverse − final_repeated | −0.0033 | 0.5514 |
| reverse − baseline | −0.0017 | 0.8000 |
| final_repeated − baseline | +0.0017 | 0.7843 |

---

## 4. What is supported

- The V1 thesis — that forcing chronological cumulative states onto increasing depth **improves** held-out reasoning — is **not supported**.
- `state_ordered` is below `cot_baseline` in **all three seeds**.
- G3 **fails as a directional efficacy gate**: ordered did not beat `state_final_repeated`; the contrast is **adverse** (−1.21 pp).
- Semantic-stage information is already **linearly decodable** from the untouched backbone. A fitted ridge probe reaches order margin ≈ 0.9947 / top-1 ≈ 0.825 / `L_state` ≈ 0.2229, matching or beating the trained ordered projector (≈ 0.935 / 0.833 / 0.263). High trained mechanism metrics are **not** evidence that DAST created a depth-wise reasoning trajectory.

## 5. What is unsupported

- Reverse superiority. Ordered − reverse = −0.88 pp, p = 0.110. Direct contrast, not a comparison of two separate p-values against 0.05.
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

The committed analysis [`../artifacts/c1b_permutation_k4_dev.json`](../artifacts/c1b_permutation_k4_dev.json) (20 splits, 24 permutations, `require_distinct`): identity median rank **2** (6/20 first); reverse median rank **10** (0/20 first); mean `L_state` spread **0.0072** (about 5.8% of identity `L_state`). An older write-up cited median rank 3 / spread ~4.3%; both descriptions agree the landscape is **nearly flat**. This is decodability, not computation. See [`FINDINGS_C1_FROZEN.md`](FINDINGS_C1_FROZEN.md).

K=8 exhaustive permutation is **not** licensed on this corpus: only ~4.6% of DEV traces have eight distinct compact states, so extra slots are repeated targets.

## 8. Open mechanistic questions

H1–H7 are stated in [`DAST_V2_HYPOTHESIS_TREE.md`](DAST_V2_HYPOTHESIS_TREE.md). V1 does **not** decide among them. The next authorized work is cheap diagnosis (B1/B2/C1), not a more complicated architecture and not the unsigned 9-run [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md) matrix.

## 9. Stop rules still in force

- No additional V1 seeds to hunt p < 0.05.
- TEST stays sealed.
- Operators / causal feedback stay gated on a **positive capability** result that V1 did not produce.
- Do not rewrite this conclusion unless new properly controlled evidence requires it.
