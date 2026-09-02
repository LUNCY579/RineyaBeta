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

**Re-ranked after the endpoint power analysis** ([`../artifacts/endpoint_power.json`](../artifacts/endpoint_power.json)).
The generated-accuracy endpoint has a three-seed MDE of **1.39 pp** and a one-seed
MDE of **2.27 pp**. Any experiment whose question is *"does capability change by
about a pp or less"* cannot be answered on it at an affordable seed count. That
demotes B1/B3 as capability tests and promotes Stage E from optional to
**precondition**.

| Rank | ID | GPU-hours (est.) | Status this pass | Spends revision budget? |
|---|---|---|---|---|
| 1 | C1 frozen landscape K=4 + C1c controls | 0 (CPU) | **RUN**, written up | no |
| 2 | **Stage E** paired teacher-forced NLL from the 12 existing adapters | ~0.2 (spent) | **RUN** — not a capability screen; keep as an interference monitor | no |
| 3 | B2 gradient cosine on existing V1 adapters | ~0.3 (spent) | **RUN** — H2 not supported; V2 stays parked | no |
| 4 | C2 per-problem π* (CPU) | 0 | **RUN** — reliable but 0.003-sized; Adaptive stays forbidden | no |
| 5 | C1 optional K=5/6 `--require-distinct` | 0 if cache exists | optional | no |
| 6 | B1 single-checkpoint one-seed pilots | ~4–5 (5× ~50 min) | **FROZEN — precondition failed.** Stage E did not deliver a capability endpoint | no |
| 7 | B3 λ sweep | ~4 (5 λ × one seed ordered) | freeze grid only | no |
| 8 | Parked V2 9-run H2 matrix | ~9 | **PARKED** until B2 supports H2 | no |
| 9 | Localized / Adaptive / Bidirectional / **Motion** / Rineya | large | all closed: Adaptive by C2, Bidirectional by reverse-accuracy, **Motion by the H6 layer-0 gate**, Rineya by the target defect | motion budget UNSPENT |

### Power gate on B1 and B3 — read before signing either

| seeds | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| MDE (pp) | **2.27** | 1.71 | **1.39** | 1.08 | 0.74 | 0.53 |

A one-seed B1 pilot cannot detect even the **full** 1.00 pp ordered deficit. If that
deficit is distributed across the four checkpoints, a per-site share of ~0.25 pp
needs **~88 seeds of 800**. So:

> **B1's generated-accuracy column cannot answer the localization question, at one
> seed or at any seed count this project can afford. Do not read it as capability
> evidence in either direction.**

B1 remains worth running — but as a **mechanism** measurement (`L_state` per site,
LM perplexity, and the B2 gradient cosine per arm), where the quantities are
continuous and per-batch rather than a 5.5%-discordance binary. Its stop rule must
be written on those, not on accuracy. The same applies to B3: the informative
pattern there ("mechanism saturates while accuracy falls") requires an accuracy
endpoint that can see a sub-2 pp fall, which this one cannot.

This is the fourth time this endpoint's power has bitten the project;
`FINDINGS_G3_SEEDS.md` §4a records the first three.

K=8 exhaustive permutation is **rejected**: 4.6% of DEV traces have 8 distinct compact states ([`scripts/permutation_landscape.py`](../scripts/permutation_landscape.py)).

---

## C1 — signed for CPU write-up (this pass)

**Question:** Is state→depth assignment weak at K=4 on the frozen backbone?
**Inputs:** `artifacts/c1a_layer_stage_profile_dev.json`, `artifacts/c1b_permutation_k4_dev.json`.
**Not causal.** Output: [`FINDINGS_C1_FROZEN.md`](FINDINGS_C1_FROZEN.md).

Optional K=5/6 only with an existing depth cache and `--require-distinct`. No GPU cache rebuild this pass if a training job might need the device.

---

## B2 — RUN. H2 not supported.

Seed 29, three arms, 32 matched minibatches each, separate `autograd.grad`, no
optimizer step, production loss untouched.
[`../artifacts/b2_verdict_s29.json`](../artifacts/b2_verdict_s29.json)

| arm | global cosine | LoRA-only cosine | frac modules < 0 |
|---|---:|---:|---:|
| ordered | −0.00167 ± 0.00156 | −0.00623 ± 0.00516 | 0.534 |
| reverse | +0.00168 ± 0.00177 | +0.00475 ± 0.00511 | 0.479 |
| final_repeated | −0.00063 ± 0.00121 | −0.00364 ± 0.00668 | 0.481 |

Every contrast inside 2σ (largest |t| = 1.51). Direction is consistently as H2
predicts; magnitude says it does not matter — |cosine| ≈ 0.006, i.e. essentially
orthogonal.

**The frozen stop rule fired: cosines are similar across arms, so H2 is not the
primary story and the V2 matrix is not signed.**

A LoRA-only cosine is reported because `global_cosine` includes the
`state_projector`, where `grad_lm` is identically zero — that block shrinks
|cosine| toward zero for every arm alike. Not excluded: a *transient* early-training
conflict, since this reads adapters at step 250 rather than a trajectory.

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

## B3 — grid frozen, trigger did NOT fire

λ ∈ {0, 0.1, 0.25, 0.5, 1.0} × V1 `state_weight` (1.0), ordered arm, seed 17 only.
Launch only if B1/B2 indicate interference. **B2 shows none, so B3's trigger has not
fired.** It also inherits the power gate: the "mechanism saturates while accuracy
falls" pattern needs an accuracy endpoint that can see a sub-2 pp fall, and this one
cannot. Do not add extra λ after seeing results.

Informative pattern: mechanism saturates early while generated accuracy worsens as λ increases → H3 “readable ≠ useful”.

---

## C2 — RUN. Reliable heterogeneity, far too small to act on.

Split-half reliability of family-level permutation preference, fitted on a disjoint
FIT set, against a label-shuffled null:

| stratum | r(A,B) | label null | argmin agree | chance |
|---|---:|---:|---:|---:|
| generator | +0.3701 | +0.0814 | 0.1363 | 0.0417 |
| answer_type | +0.3893 | +0.1024 | 0.1501 | 0.0417 |
| step_count | +0.3650 | +0.2076 | 0.1250 | 0.0417 |

Family preference replicates at roughly 4× its null, so the heterogeneity is real.
But the whole permutation spread is 1.89% of `L_state`, so the reliably-predicted
quantity is a ~0.003 `L_state` difference with no demonstrated link to capability.

**Adaptive DAST remains forbidden.** C2 was the gate, and it passes reliability
while failing magnitude. Reopening it requires evidence that the preferred
assignment changes *generated capability*, not `L_state`.

---

## Stage E — RUN. Powerful, and on the wrong axis.

All 12 adapters scored, 9600 paired problems, forward-only.
[`../artifacts/stage_e_verdict.json`](../artifacts/stage_e_verdict.json)

| arm | mean per-token gold NLL | accuracy |
|---|---:|---:|
| `cot_baseline` | 0.006296 | 0.8937 |
| `state_final_repeated` | 0.006492 | 0.8954 |
| `state_reverse` | 0.007025 | 0.8921 |
| `state_ordered` | 0.007034 | 0.8838 |

The paired sign test is enormously powerful, because it uses all 2400 pairs rather
than the 133 discordant ones McNemar is limited to: `ordered − baseline`
**p = 2.0e-48**, `reverse − baseline` p = 9.1e-45, `final_repeated − baseline`
p = 1.2e-20.

But three checks decide whether it can stand in for accuracy, and all three fail:

| check | result |
|---|---|
| magnitude tracks accuracy | Pearson **r = +0.188** — uncorrelated |
| resolves `ordered − reverse` (accuracy −0.83 pp) | **no**, p = 0.076 |
| stays quiet on a null contrast | **no**, fires at p = 9e-45 on `reverse − baseline` |

Every arm carrying a state loss shows higher LM NLL, and the size of that penalty
has little to do with the accuracy it costs. `reverse` is flagged as hard as
`ordered` while costing 0.17 pp instead of 1.00; `final_repeated` is flagged at
p = 1e-20 while being slightly *better* than baseline on accuracy.

> **Verdict: Stage E is an AUXILIARY-LOSS INTERFERENCE monitor, not a capability
> screen.** Keep it for that. It does **not** make B1/B3 answerable on capability,
> so B1's precondition has failed.

Two reasons this was predictable in hindsight and worth remembering: mean gold-token
probability is ~0.9998 while free generation fails ~11%, so teacher forcing and
generation occupy different regimes by construction; and the NLL distribution is
severely skewed (mean/median 19×), which is why the robust sign test is reported
next to the clustered-bootstrap mean.

## Original Stage E rationale (superseded by the result above)

**Why it moved.** It is the only route to an endpoint that can resolve the effects
B1/B3 are about, and it is nearly free: all twelve V1 runs saved `adapter/` and
`dast_auxiliary.pt`, so per-problem teacher-forced NLL needs **forward passes only,
no retraining** — roughly 0.5 GPU-hours for the whole matrix against ~4–5 for a
single B1 pilot that cannot answer its own question.

**Method.** Per problem on the 800 DEV items: gold-token NLL over the assistant
span, answer-region NLL, and mean gold-token probability. Pair by `problem_id`,
cluster by `generator` (30 clusters), exactly as `compare.py` already does for
accuracy. A continuous paired endpoint is not throttled by the 5.5% discordance
rate that caps the binary one.

**Validation gate, to be passed before the endpoint is used for anything.** The
answer is already known on this matrix: `ordered` below `baseline ≈ reverse ≈
final_repeated`. If the NLL endpoint does not reproduce that ordering on the same
twelve runs, it is not a usable screen and must be discarded rather than explained.

**Standing constraint.** Generated accuracy remains the capability endpoint. Stage E
ranks candidates cheaply; it never promotes one on its own. Never promote a
mechanism solely because the surrogate moves while generation does not.

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
- B1/B3 accuracy is not a capability endpoint at these seed counts (power gate above).
- One GPU job at a time; `--min-free-mib`; never start B2 while a trainer is live.
