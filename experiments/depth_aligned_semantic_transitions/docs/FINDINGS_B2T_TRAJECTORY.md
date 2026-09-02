# B2-T — gradient conflict as a trajectory — AMBIGUOUS

**Verdict: H2-T AMBIGUOUS** by the rule frozen in
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

## 5. Validity check 2 partially FAILS — stated, not explained away

§6 required the LATE window to reproduce the step-250 snapshot, and said that if it did
not, "the instrument is suspect and the trajectory is not interpretable."

| arm | B2-T LATE | snapshot | sign |
|---|---:|---:|---|
| `state_ordered` | +0.010358 | −0.006230 | **DISAGREES** |
| `state_reverse` | +0.000959 | +0.004750 | agrees |
| `state_final_repeated` | −0.002416 | −0.003640 | agrees |

Two of three agree; **ordered does not**, and the gap is larger than the snapshot's own
uncertainty (−0.0062 ± 0.0052). By the standard I set, **the LATE window is not
interpretable** and every LATE row here is reported as descriptive only.

What it does and does not undermine. The primary test lives entirely in the EARLY
window and does not read the LATE window at all, so the §2 result stands on its own
terms. But a failed validity check is a failed validity check: it lowers confidence in
the instrument generally, and any future use of this trajectory at convergence needs
the discrepancy resolved first. Two contributing differences worth recording — the
snapshot re-loaded saved adapters while B2-T measures in flight, and both quantities
are close to zero relative to their own spread — are candidate explanations, not
excuses.

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
