# DAST next experiments — frozen ranking

V1 is closed. This file freezes **what may be run next**, in GPU-hour order.
Nothing here is a license to search DEV until a variant wins. Sign the relevant
block before launching GPU work. TEST stays sealed.

Hypothesis mapping: [`DAST_V2_HYPOTHESIS_TREE.md`](DAST_V2_HYPOTHESIS_TREE.md).
V1 boundary: [`DAST_V1_CLOSURE.md`](DAST_V1_CLOSURE.md).

**Revision budget:** only **motion DAST (H6)** spends SSOT §13. B1/B2/B3/C1/C2
are diagnostics inside the existing mechanism.

---

## Ranked list

| Rank | ID | GPU-hours (est.) | Status this pass | Spends revision budget? |
|---|---|---|---|---|
| 1 | C1 frozen landscape K=4 (existing artifacts) | 0 (CPU) | **RUN** write-up | no |
| 2 | C1 optional K=5/6 `--require-distinct` | 0 if cache exists; **do not rebuild cache** if a trainer may need the GPU | run only if `artifacts/depth_cache_*.pt` present | no |
| 3 | B2 gradient cosine on existing V1 adapters | ~0.3–1.0 | **RUN only if GPU idle** | no |
| 4 | B1 single-checkpoint one-seed pilots | ~4–5 (5× ~50 min) | **FROZEN, do not launch** | no |
| 5 | C2 per-problem π* (CPU after cache) | 0–1 | listed, not this pass | no |
| 6 | B3 λ sweep | ~4 (5 λ × one seed ordered) | freeze grid only | no |
| 7 | Stage E teacher-forced surrogate | cheap once instrumented | design note only | no |
| 8 | Parked V2 9-run H2 matrix | ~9 | **PARKED** until B2 supports H2 | no |
| 9 | Localized / Adaptive / Bidirectional / Motion / Rineya | large | choose **one** after B/C | motion only |

K=8 exhaustive permutation is **rejected**: 4.6% of DEV traces have 8 distinct compact states ([`scripts/permutation_landscape.py`](../scripts/permutation_landscape.py)).

---

## C1 — signed for CPU write-up (this pass)

**Question:** Is state→depth assignment weak at K=4 on the frozen backbone?
**Inputs:** `artifacts/c1a_layer_stage_profile_dev.json`, `artifacts/c1b_permutation_k4_dev.json`.
**Not causal.** Output: [`FINDINGS_C1_FROZEN.md`](FINDINGS_C1_FROZEN.md).

Optional K=5/6 only with an existing depth cache and `--require-distinct`. No GPU cache rebuild this pass if a training job might need the device.

---

## B2 — signed for a short diagnostic if GPU idle

**Question:** Is ordered cosine(g_LM, g_state) more negative than reverse / final-repeated?
**Method:** separate `autograd.grad` on `L_LM` and `L_state`; **no optimizer step**.
**Arms:** existing V1 run dirs, seed 29 preferred (complete adapters).
**Refuse** if `dast.cli train` or `resume_runs.py` is running.
**Do not** change production loss.

Stop: if cosines are similar across arms, H2 is not the primary story; do not sign the V2 matrix.

---

## B1 — FROZEN. Do not launch until this section is explicitly signed in PROGRESS.

**Not signed for execution in this pass.** Configs exist only so the protocol cannot drift.

**Question:** Is the ordered deficit localized to one checkpoint?
**Seed:** 17 only. Do not scale to 3 seeds unless one seed shows a structured effect large enough to matter under the 2.0 pp practical bar (or a clean localization pattern: one site explains most of −1 pp).
**Matched V1 protocol** except `semantic.supervised_stages` (and mismatch remap). All four readouts remain; only one stage receives state **and** contrastive loss.

| config | supervised checkpoint (0-index) | target |
|---|---|---|
| `configs/diag_b1_ckpt1_s1.yaml` | 0 (layer 7) | state 1 |
| `configs/diag_b1_ckpt2_s2.yaml` | 1 (layer 14) | state 2 |
| `configs/diag_b1_ckpt3_s3.yaml` | 2 (layer 21) | state 3 |
| `configs/diag_b1_ckpt4_s4.yaml` | 3 (layer 27) | state 4 |
| `configs/diag_b1_ckpt4_s1.yaml` | 3 (layer 27) | state 1 (mismatch) |

Compare each to V1 `g2_budget_cot_baseline` seed 17 and to V1 ordered seed 17. Incomplete/OOM runs are reported and excluded.

**Stop:** unstructured noise below 2 pp and no site pattern → do not three-seed; move to B2/C2 or close training.

---

## B3 — grid frozen, not launched

λ ∈ {0, 0.1, 0.25, 0.5, 1.0} × V1 `state_weight` (1.0), ordered arm, seed 17 only.
Launch only if B1/B2 indicate interference. Do not add extra λ after seeing results.

Informative pattern: mechanism saturates early while generated accuracy worsens as λ increases → H3 “readable ≠ useful”.

---

## C2 — next CPU study after C1 write-up

Per-problem or per-family held-out permutation preference. Stratify by category, step count, answer_type, generator. Look for stable π*(x). **Adaptive DAST is forbidden until this fires.**

---

## Stage E — design note only

Teacher-forced paired NLL / answer-span NLL as a **companion** endpoint. Cluster by generator. Never promote a mechanism solely because the surrogate moves.

---

## Successors (choose one, later)

| Finding | Mechanism | Required evidence |
|---|---|---|
| Deepest forward checkpoint harmful | Localized DAST | B1 replicate + B2 |
| Ordered grads anti-align with LM | λ / soft constraint; maybe signed V2 | B2 yes, then B3 |
| Stable π*(x) | Soft/Adaptive DAST | C2 held-out routing |
| Reverse **accuracy** > baseline, replicated | Bidirectional | not V1 reverse `L_state` |
| ΔH wins capability **and** conflict | Motion DAST | revision budget |
| Mechanism localized, then domain check | Rineya / larger Qwen | after B/C |
| All null | Close training direction | keep probes |

---

## Hard stops

- TEST sealed.
- No extra V1 seeds.
- No V2 9-run matrix until B2 supports H2.
- No B1 GPU this pass.
- One GPU job at a time; `--min-free-mib`; never start B2 while a trainer is live.
