# B2-T — gradient conflict as a trajectory — AMBIGUOUS

**Verdict: H2-T AMBIGUOUS** — unchanged by anything in §5 — by the rule frozen in
[`PREREGISTRATION_B2_TRAJECTORY.md`](PREREGISTRATION_B2_TRAJECTORY.md) §10e before any
data. One contrast is significant; it is not the preregistered pattern. **No successor
is licensed and the V2 matrix stays parked.**

Artifact: [`../artifacts/b2t_verdict.json`](../artifacts/b2t_verdict.json).
15 runs: 5 init seeds × 3 arms, seed 29 full length (270 measured of 2000 steps), the
other four to the EARLY window (120 steps each). Diagnosis only, generation disabled,
TEST sealed. **These runs have no capability endpoint and their accuracy is not read.**

---

## 1. The premise was right

At step 250 the LM loss is ~7e-4 and all arms sat within ~0.01 of orthogonal. Early in
training the picture is genuinely different: `lora_only_cosine` swings across roughly
±0.09 per step with LM loss 0.9–2.2. So the objection to the snapshot — that two
objectives cannot fight over an already-zero loss — was well founded. Measuring the
trajectory was the right call.

## 2. Primary result

Unit of replication is the **initialisation**. Within a seed all three arms share one
init and one batch order (digests verified equal per seed), so the paired per-step
difference is attributable to the arm. `D_k` = mean paired difference over the EARLY
window; one-sample t on {D_k}, 4 df, t_crit 2.776.

| seed | ordered − reverse | ordered − final_repeated |
|---|---:|---:|
| 29 | −0.002959 | +0.004487 |
| 101 | −0.005134 | +0.011525 |
| 211 | −0.002448 | −0.018016 |
| 307 | −0.005658 | +0.000597 |
| 401 | −0.001634 | +0.006641 |

| contrast | mean D | sd | t | p | 95% CI | significant |
|---|---:|---:|---:|---:|---|---|
| **ordered − reverse** | **−0.003566** | 0.001746 | **−4.57** | **0.0103** | [−0.00573, −0.00140] | **yes** |
| ordered − final_repeated | +0.001047 | 0.011364 | +0.21 | 0.847 | [−0.01306, +0.01515] | no |

**`ordered − reverse` is a real, replicated effect** — negative in 5 of 5 seeds, tight
sd, t = −4.57. It is the first replicated non-null arm difference anywhere in the DAST
programme. And it is the *clean* test of order per se: ordered and reverse carry the
**same target multiset** and differ only in which state sits at which depth.

`ordered − final_repeated` is null with **6.5× the variance** (sd 0.0114 against
0.0017), driven by seed 211 at −0.018 against four positives.

## 3. Why this is AMBIGUOUS and not support for H2

The frozen rule required ordered to be more anti-aligned than **both** controls. It is
not. But the deeper reason to withhold support is a dissociation the data makes plain:

| contrast | capability (generated accuracy) | early gradient conflict |
|---|---|---|
| ordered − final_repeated | **−1.17 pp, p = 0.0167 (significant)** | +0.0010, t = +0.21 (null) |
| ordered − reverse | −0.83 pp, p = 0.128 (null) | **−0.0036, t = −4.57 (significant)** |

**The gradient difference appears exactly where accuracy shows none, and there is no
gradient difference where accuracy shows one.** H2 was a proposed *explanation for the
ordered accuracy deficit*. A candidate explanation that is absent in the contrast
carrying the deficit, and present in the contrast that has no deficit, does not explain
it. The pattern is anti-correlated with the H2 story rather than merely short of
significance.

So: a genuine gradient-geometry asymmetry between forward and reverse assignment
exists, is replicable, and is **not** the mechanism behind V1's capability result.

## 4. Amendment 2 changed both answers — demonstrably

Amendment 2 replaced a within-run block bootstrap with replication over
initialisations, on the argument that a bootstrap over steps inside one run is blind to
anything constant across the window. The secondary output shows what the original
design would have concluded from seed 29 alone:

| contrast, seed-29 within-run bootstrap, EARLY | interval | would have read as |
|---|---|---|
| ordered − reverse | [−0.01081, +0.00524] | **null — the real effect missed** |
| ordered − final_repeated | [+0.00029, +0.00876] | **significant, POSITIVE — spurious** |

Both are wrong, in opposite directions. The single-run design would have missed the one
replicated effect and manufactured one that five seeds show is noise. This is a
concrete instance of the general point, not a hypothetical.

## 5. Validity check 2 — the check failed; the check was mis-specified

**First, what happened as written.** §6 required the LATE window to reproduce the
step-250 snapshot, and made **sign agreement** the test. On that test `state_ordered`
failed: LATE +0.010358 against a snapshot of −0.006230.

**Second, and reported with the walk-back risk stated openly:** that check compares
two quantities that are not the same estimand, and the data shows it. I am revising a
validity check *after* seeing it fail, which is exactly the move that deserves
suspicion. So the evidence is laid out to be judged rather than asserted, and the
primary decision rule in §2 is **untouched** — the verdict remains AMBIGUOUS.

The snapshot was measured on the adapter saved **after** 250 optimizer steps, i.e. at
global step 2000 exactly. The LATE window averages global steps 1601–2000, a
50-optimizer-step span. Two facts make those different quantities:

- **Per-step noise is large relative to the effect.** The per-step sd of
  `lora_only_cosine` is **0.023–0.032**, while the snapshot values themselves are
  ±0.006. A sign test on a quantity whose mean is a quarter of its own per-step noise
  is close to a coin flip.
- **The window is not stationary.** First half vs second half of LATE:
  `state_ordered` shifts −0.0124, `state_reverse` shifts −0.0313. Averaging 400 global
  steps of a wandering quantity does not estimate its value at the endpoint.

Evaluated as a statistical comparison rather than a sign test — window mean ± its SE
against the snapshot ± its SE, combined in quadrature:

| arm | B2-T LATE ± SE (n=20) | snapshot ± SE | gap / combined SE |
|---|---:|---:|---:|
| `state_ordered` | +0.010358 ± 0.005848 | −0.006230 ± 0.00516 | **2.13** |
| `state_reverse` | +0.000959 ± 0.007172 | +0.004750 ± 0.00511 | 0.43 |
| `state_final_repeated` | −0.002416 ± 0.004785 | −0.003640 ± 0.00668 | 0.15 |

All three agree within 2.2 combined SE. The single nominal exceedance is at 2.13 SE,
and across three comparisons the chance of at least one |z| > 2 under the null is
about 14%. That is an unremarkable draw, not a discrepancy.

**Conclusion: the instrument is not suspect; the check was.** I specified sign
agreement between a 50-optimizer-step window mean and a point measurement at the
window's endpoint, on a quantity with per-step noise four times its own mean. That was
a poorly designed check, and it is a design error of mine rather than a fault in
either measurement. Both the B2 step-250 snapshot — which
[`DAST_V1_CLOSURE.md`](DAST_V1_CLOSURE.md) cites as refuting H2 at convergence — and
the B2-T trajectory survive it.

**What a correct version of this check would have been:** compare the snapshot against
the trajectory's measurements *near* step 2000 only, or better, re-run the snapshot
procedure against the B2-T run's own saved adapter so the weights are identical and any
gap is purely procedural. The second is still the cleaner test and remains unrun (the
accelerator was taken by an unrelated job); it is not needed to lift the caveat, but it
would settle the procedural question outright.

## 6. What follows

- **H2 is not revived.** The frozen consequence of AMBIGUOUS is that no successor is
  licensed; `PREREGISTRATION_V2.md` stays parked and unsigned. B3's trigger does not
  fire.
- **The V1 closure is unchanged.** Nothing here is a capability result.
- **One new, narrow, positive fact** enters the record: forward and reverse depth
  assignment differ reliably in early-training gradient geometry, at matched target
  multiset. It is a fact about optimisation geometry, not about reasoning, and it is
  dissociated from capability. It would become interesting only if some future
  mechanism made a capability prediction that tracked it.

## Claim boundary

One model (Qwen3-0.6B), one corpus, one LoRA schedule, 5 initialisations, EARLY window
of 120 global steps. Diagnosis only. The measured cosine covers `L_LM` against
`L_state` and **excludes** the 0.1-weighted contrastive term the optimizer also
applies (§10d). DEV only. TEST sealed, never opened.
