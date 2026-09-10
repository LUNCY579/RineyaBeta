# A10 / A11: the sink effect is real; the global-layer question is still unmeasured

Run `hlga-a10a11-20260909-cuda-math`, 12 arms x 72 open-synthetic DEV cases,
gated by `hlga-a10-gate-20260909-cuda-math` at one code identity,
`checkpoint_unchanged` and `source_unchanged` both true. Primary metric
`first_content_exact`. Predictions were filed in
`AGENT_A10_PREREGISTRATION.md` before either arm existed in code.

## 1. Prediction check

| | registered | observed |
|---|---|---|
| A10 overall / reachable / evicted | 0.20-0.28 / 0.45-0.55 / 0.00-0.05 | **0.222 / 16 of 32 / 0 of 40** — all three inside |
| A11 overall / evicted | 0.20-0.28 / 0-2 of 40 | **0.167 / 0 of 40** — overall **missed low** |

## 2. Results

| arm | globals | sinks | overall | reachable (n=32) | evicted (n=40) |
|---|---:|---:|---:|---|---|
| A0_global | 28 | 0 | 0.653 | 20/32 | 27/40 |
| A7_sink1024 | 7 | 4 | 0.431 | — | 1/20 |
| A3_sink512 | 7 | 4 | 0.250 | 17/32 | 1/40 |
| **A10_all_local_sink512** | **0** | **4** | **0.222** | 16/32 | 0/40 |
| A1_local1024 | 7 | 0 | 0.181 | — | 0/20 |
| **A11_backloaded_sink512** | **7 (21-27)** | **4** | **0.167** | 12/32 | 0/40 |
| A8_all_local512 | 0 | 0 | 0.069 | 5/32 | 0/40 |
| A2_local512 | 7 | 0 | 0.056 | 4/32 | 0/40 |
| A9_frontloaded512 | 7 | 0 | 0.042 | 3/32 | 0/40 |

Paired bootstrap 95% CIs (20k resamples, seed 17) alongside the registered
McNemar with Holm over 13 contrasts:

| contrast | delta | 95% CI | McNemar | Holm |
|---|---:|---|---|---|
| PRIMARY A0 -> A3 | −0.403 | [−0.528, −0.264] | 3w/32l, p=4.2e-07 | survives |
| **A8 -> A10** sinks, no globals | **+0.153** | **[+0.069, +0.236]** | 11w/0l, p=9.8e-04 | **survives** |
| A3 -> A11 back-load globals | −0.083 | [−0.153, −0.028] | 0w/6l, p=0.031 | does **not** survive |
| A10 -> A3 add 7 globals | +0.028 | [+0.000, +0.069] | 2w/0l, p=0.5 | does **not** survive |

## 3. CORRECTION — a claim I made and have to withdraw

I reported that **"the seven periodic global layers contribute essentially
nothing."** That is not supported, and an adversarial audit was right to call it
a blocker.

**The A10 -> A3 contrast had two discordant pairs.** With d = 2 the minimum
attainable two-sided exact p is 2 x 0.5² = **0.5**. The contrast **could not have
reached α = 0.05 whatever the outcome** — it had zero power by construction, not
by result. Reporting its null as a finding is exactly the error I criticised in
the two earlier global-layer contrasts (A8 -> A2, A9 -> A2), committed again in a
new form. Worse, my own preregistration amendment had already stated that this
contrast is "uninformative in either direction", and I then reported it as
informative.

What the data *do* support is a **bound**, not a null:

* A10 and A3 return the **same verdict on 70 of 72 cases**, and generate
  token-identical output on 41 of 72.
* The estimated contribution of the seven global layers is **+0.028, 95% CI
  [0.000, +0.069]** — so on this probe any contribution is confined to roughly
  **0 to 5 cases out of 72**. The interval does not include negative values, so
  if anything the globals help slightly.

Correct statement: *on this probe the seven periodic global layers change the
outcome on at most about 5 of 72 cases, and the design cannot resolve whether
the true effect is zero or small-positive.* That is much weaker than "contribute
nothing", and it does not license removing them.

**Consequently I also withdraw "A10 equals A3 at 57% of the KV."**
Non-significance at two discordant pairs is not equivalence; no equivalence
margin was preregistered; and the 57% figure is an analytical cache count, which
`protocol.json` explicitly forbids treating as a measured efficiency result.

## 4. What does stand

**The attention-sink effect.** A8 -> A10 is +0.153, **11 wins / 0 losses**,
p = 9.8e-04, survives Holm, CI excluding zero — obtained with **zero global
layers present**. This is the third independent one-sided replication after
A2 -> A3 (14/0) and A1 -> A7 (18/0). Whatever is helping these arms, it is the
four prefix sink tokens, and it does not require the hybrid topology.

**Frozen substitution is badly damaging.** A0 -> A3 is −0.403, CI
[−0.528, −0.264]. Every windowed arm is far below the all-global reference; the
best (A7) reaches 0.431 against 0.653.

**Nothing recovers evicted evidence.** Across three global placements with
sinks — periodic 1/40, back-loaded 0/40, none 0/40 — against A0's 27/40 on the
same cases. Three placements and a no-globals control now all fail.

**Back-loading is directionally harmful.** A3 -> A11 is 0 wins / 5 losses on the
reachable stratum, CI [−0.153, −0.028] excluding zero, but raw p = 0.031 does
**not** survive Holm over 13 contrasts. Reported as directional only.

## 5. Standing limitation

Three separate attempts to measure the global-layer contribution (A8 -> A2,
A9 -> A2, A10 -> A3) have all returned zero-power results, each for the same
structural reason: the arms being compared agree on almost every case, either
because both fail (evicted stratum) or because both succeed (reachable stratum).
**This probe cannot answer the global-layer question at n = 72**, and adding
arms will not fix it. A design that could would need either many more cases
concentrated where the arms can disagree, or a task where partial context loss
produces graded rather than all-or-nothing failure.

## 6. Bearing on published Rineya cores

Nothing here supports it, and one thing argues against it. Every configuration
tested is substantially worse than the all-global baseline when frozen, so there
is no "cheaper at equal quality" option on the table — the cheapest arms are
also among the worst. The only claim with real support is that four sink tokens
materially reduce damage in a windowed model, which is a finding about *windowed
inference*, not about improving a full-attention production model.

P4 (training) and P5 (independent confirmation) remain closed and are the only
routes by which any of this could reach a shipped core.
