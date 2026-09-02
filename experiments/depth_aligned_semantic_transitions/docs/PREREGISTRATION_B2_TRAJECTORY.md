# B2-T preregistration — LM/state gradient conflict as a TRAJECTORY

**Status: FROZEN before any B2-T data was collected.** Written and committed prior to
running. Nothing below may be changed after seeing results; deviations must be
recorded as deviations.

**Claim boundary:** diagnosis only, DEV only, TEST sealed. This does not measure
capability and its accuracy column must not be read (see §7).

---

## 1. Why this experiment exists

`artifacts/b2_verdict_s29.json` measured cosine(∇L_LM, ∇L_state) on adapters saved at
**step 250** and found all three arms within 2σ and within ~0.01 of orthogonal. That
refuted H2 *at convergence*. It did not refute H2 as a mechanism, because at step 250
the LM loss is ~7e-4 — essentially converged — and two objectives cannot fight over a
loss that is already zero. A conflict that occurs early and dissipates would produce
exactly the snapshot we observed.

This is the single branch of the diagnosis tree where the right quantity was measured
at arguably the wrong time. It is the last open question in the DAST programme.

## 2. Hypothesis

**H2-T.** Under forward chronological supervision, `L_state` gradients anti-align
with `L_LM` gradients **early in training**, more strongly than under reverse or
flat-target supervision, and this conflict decays as the LM converges.

**H2-T null.** The arms are indistinguishable at every point in training, as they are
at step 250.

## 3. Design

Three instrumented training runs, one per arm, **seed 29**, otherwise matched to the
V1 protocol exactly (Qwen3-0.6B, LoRA r=16, 4000 train / 250 optimizer steps,
`checkpoint_readout: final_norm` at 7/14/21/27, `target_centering: within_problem`).

Seed 29 is chosen deliberately: the completed V1 runs at that seed give the step-250
snapshot this trajectory must reproduce at its end (§6, validity check).

| arm | config |
|---|---|
| `state_ordered` | `configs/diag_b2traj_s29_state_ordered.yaml` |
| `state_reverse` | `configs/diag_b2traj_s29_state_reverse.yaml` |
| `state_final_repeated` | `configs/diag_b2traj_s29_state_final_repeated.yaml` |

Generation is **disabled** (`do_generation: false`). It is the expensive part of a V1
run and this experiment has no capability endpoint.

## 4. Measurement

At scheduled global steps, before the production `.backward()`, take
`torch.autograd.grad(L_LM, θ, retain_graph=True)` and
`torch.autograd.grad(L_state, θ, retain_graph=True)` and record:

- **`lora_only_cosine`** — the PRIMARY metric. Excludes the `state_projector`, on
  which ∇L_LM is identically zero; including it inflates ‖g_state‖ without touching
  the dot product and shrinks |cosine| toward 0 for every arm alike.
- `global_cosine` (all trainable parameters), for comparability with the snapshot.
- per-module cosines and norms, `grad_norm_ratio`, `fraction_modules_negative`.
- `lm_loss`, `state_loss`, learning rate.

**Schedule** (global steps; grad_accum = 8, so 8 global steps = 1 optimizer step):

| range | every | points |
|---|---|---|
| 1–40 | 1 | 40 |
| 41–400 | 4 | 90 |
| 401–2000 | 20 | 80 |

≈210 measured steps of 2000. Dense where the hypothesis lives.

## 5. Frozen analysis plan

Three windows, fixed now by step index, not by anything observed:

| window | global steps | optimizer steps |
|---|---|---|
| EARLY | 1–40 | 1–5 |
| MID | 41–400 | 6–50 |
| LATE | 1601–2000 | 201–250 |

**Primary contrast.** Mean `lora_only_cosine` in the **EARLY** window:
`ordered − reverse` and `ordered − final_repeated`, each with a standard error over
measured steps and a Welch t.

**Decision rule, fixed in advance:**

- **H2-T SUPPORTED** if ordered's EARLY mean is more negative than **both** controls
  by more than 2 SE.
- **H2-T REFUTED** if all three arms lie within 2 SE of one another in **all three**
  windows.
- **AMBIGUOUS** otherwise. Report as ambiguous; do not reinterpret into a story.

**Secondary, descriptive only, no decision attached:** the shape of the trajectory
(does |cosine| decay with step?), per-module structure (do particular LoRA sites
carry the conflict?), and the relationship between cosine and LM loss magnitude.

**Multiplicity.** Two primary contrasts. Nothing else is decision-bearing. Secondary
analyses are explicitly labelled exploratory and generate no conclusions.

## 6. Validity checks, both required

1. **Non-interference.** The probe must not perturb the optimizer trajectory. Pinned
   by a test that runs the toy model with and without the probe and asserts
   **bit-identical** final parameters. `torch.autograd.grad` returns gradients without
   accumulating into `.grad`, and backward consumes no RNG, so this should hold
   exactly.
2. **Snapshot reproduction.** The LATE window must reproduce the existing step-250
   snapshot within noise: ordered ≈ −0.0062, reverse ≈ +0.0048,
   final_repeated ≈ −0.0036 (`lora_only_cosine`). If it does not, the instrument is
   suspect and the trajectory is not interpretable.

## 7. What this experiment cannot do

- It has **no capability endpoint**. Generation is off; even if it were on, the
  one-seed MDE is 2.27 pp against a 1.00 pp effect (`artifacts/endpoint_power.json`).
  These runs' accuracy must not be read or reported.
- It is **one seed**. A positive result would need replication before it licensed
  anything, and specifically before the parked `PREREGISTRATION_V2.md` matrix.
- It measures gradients along **its own** optimizer path, which is not bit-identical
  to the V1 runs (`deterministic_algorithms: false`).

## 8. Consequences, fixed in advance

- **SUPPORTED** → H2 revives as an early-training phenomenon. That does **not** by
  itself reopen DAST training: it would license a λ-schedule or gradient-surgery test
  (B3), and would require replication at a second seed first. It does **not** unpark
  the V2 matrix on its own.
- **REFUTED** → H2 is closed at trajectory resolution as well as at convergence. The
  last open branch of the diagnosis tree shuts, and the DAST closure becomes complete
  with no remaining untested mechanism.
- **AMBIGUOUS** → recorded as ambiguous. No successor is licensed on an ambiguous
  result.

In none of these three cases does this experiment alter the V1 closure or open TEST.

---

## 9. Pre-data amendment (2026-09-01, before any B2-T run)

Made **before any B2-T data existed**, for statistical reasons stated below, not
because of anything observed. The original §4/§5 text is left intact above so the
change is auditable; where they conflict, this section governs.

### 9a. The original plan understated uncertainty

§5 specified "a standard error over measured steps". Consecutive global steps share
nearly identical weights and adjacent minibatches, so those values are strongly
autocorrelated and an i.i.d. standard error over 40 of them would be far too small.
Taken literally the original rule would have inflated the false-positive rate for
H2-T — the direction that manufactures a result.

### 9b. Per-step PAIRING is available and is used

Verified empirically before freezing this amendment: the three arms are built from
the same seed and the same `build_dataloaders` construction, and the arm changes only
each example's target texts, never the batch composition. At global step *t* all three
arms therefore process the **identical batch** (checked: identical `problem_ids` for
the first 20 batches across all three arms).

So the primary statistic becomes the **paired per-step difference**

    d_ordered-control(t) = lora_only_cosine_ordered(t) - lora_only_cosine_control(t)

which removes batch-difficulty variance entirely. Honest limitation: the batch is
shared but the *weights* are not — the arms diverge after the first optimizer step,
so the pairing is exact at t=1 and progressively only approximate. It still removes
the dominant nuisance term, and it never makes the comparison invalid, only less
perfectly matched later in training.

### 9c. Revised measurement schedule

Denser early, so the EARLY window has enough blocks to bootstrap:

| range | every | points |
|---|---|---|
| 1–120 | 1 | 120 |
| 121–400 | 4 | 70 |
| 401–2000 | 20 | 80 |

≈270 measured steps of 2000.

### 9d. Revised windows

| window | global steps | optimizer steps | measured points |
|---|---|---|---|
| EARLY | 1–120 | 1–15 | 120 |
| MID | 121–400 | 16–50 | 70 |
| LATE | 1601–2000 | 201–250 | 20 |

### 9e. Revised primary analysis and decision rule

**Primary statistic:** mean paired per-step difference of `lora_only_cosine` in the
**EARLY** window, for `ordered − reverse` and `ordered − final_repeated`.

**Interval:** 95% percentile **block bootstrap**, resampling whole optimizer steps
(blocks of 8 consecutive global steps) rather than individual steps, which is what
respects the autocorrelation. EARLY gives 15 blocks.

**Decision rule, replacing the "2 SE" rule in §5:**

- **H2-T SUPPORTED** if the paired mean is negative and its 95% block-bootstrap CI
  excludes zero for **both** controls.
- **H2-T REFUTED** if both CIs include zero in **all three** windows.
- **AMBIGUOUS** otherwise — reported as ambiguous, with no successor licensed.

**Also reported, descriptive only:** the unpaired per-arm means from the original
§5 plan, so the amendment's effect is visible rather than hidden.

### 9f. Validity check 1 strengthened

The toy-model non-interference test exercises neither gradient checkpointing nor
dropout, so it alone gives false assurance about the real run, where checkpointing is
ON and LoRA dropout is 0.05. A second test now drives the exact pattern — two
`autograd.grad(retain_graph=True)` calls followed by a production `backward()` —
through a checkpointed network with active dropout, and asserts **bit-identical**
parameters for both `use_reentrant=True` and `False`. It passes:
`torch.utils.checkpoint` defaults to `preserve_rng_state=True`, so recomputation
saves and restores the RNG and the production backward sees unchanged dropout masks.
