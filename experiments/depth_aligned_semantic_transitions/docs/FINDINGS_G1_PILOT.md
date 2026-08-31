# G1 learnability pilot — findings

**Status: EXPLORATORY_DEV. Underpowered by design. This does NOT close or support
the DAST hypothesis.**
**Branch:** `research/depth-aligned-semantic-transitions-v1` at `f5ae82f`
**Run date:** 2026-08-31
**Sealed evaluation:** CLOSED, never touched.

Four arms, one seed (17), 1000 train / 200 DEV problems, Qwen3-0.6B + LoRA r=16,
corrected geometry (`checkpoint_readout: final_norm` at layers 7/14/21/27,
`target_centering: within_problem`), generation at `max_new_tokens: 384`, scorer as
fixed in `98bc669`. 63 optimizer steps per arm. ~12–14 min per arm.

## 1. Results

| arm | generated accuracy | order margin | checkpoint top-1 | monotonic | LM ppl | `L_state` |
|---|---:|---:|---:|---:|---:|---:|
| `cot_baseline` | **0.7800** | 0.0132 | 0.2700 | 0.6550 | 1.0547 | — (weight 0) |
| `state_ordered` | **0.7800** | **0.5751** | **0.7100** | **0.9750** | 1.0536 | **0.5441** |
| `state_shuffled` | 0.7600 | −0.0908 | 0.1863 | 0.4100 | 1.0551 | 0.9745 |
| `state_reverse` | 0.7650 | −0.1571 | 0.1450 | 0.0000 | 1.0557 | **0.5403** |

Paired, by problem id, over the same 200 DEV problems:

| contrast | difference | bootstrap 95% CI | discordant | exact McNemar p |
|---|---:|---|---:|---:|
| **ordered − baseline** | **+0.0000** | [−0.030, +0.030] | **5 : 5** | **1.0000** |
| ordered − shuffled | +0.0200 | [−0.010, +0.055] | 8 : 4 | 0.388 |
| ordered − reverse | +0.0150 | [−0.020, +0.055] | 9 : 6 | 0.607 |
| shuffled − baseline | −0.0200 | [−0.050, +0.005] | 2 : 6 | 0.289 |
| reverse − baseline | −0.0150 | [−0.055, +0.020] | 6 : 9 | 0.607 |

## 2. Read this power table before reading anything else

Measured discordance rate between arms sharing a base model and LoRA capacity is
**6.0%**, so n=200 yields only 10–15 discordant pairs.

| contrast | n_d | power at a true 2:1 split | at 3:1 |
|---|---:|---:|---:|
| ordered vs shuffled | 12 | **0.182** | 0.391 |
| ordered vs reverse | 15 | 0.210 | 0.461 |
| ordered vs baseline | 10 | 0.104 | 0.244 |

**A real 2:1 effect would have been missed here with ~82% probability.** No null in
§1 can be read as absence of an effect. This pilot exists to decide whether the
15-hour matrix is worth running, not to answer the scientific question.

## 3. What the pilot does establish

### 3.1 The auxiliary objective trains a readout, and trains it well

`state_ordered` reaches order margin 0.575, checkpoint top-1 0.710 (chance 0.25) and
monotonic decoded progress 0.975, against 0.013 / 0.270 / 0.655 for a baseline whose
projector is untrained. LM perplexity is preserved across every arm (1.0536–1.0557),
so nothing is being bought by damaging the language model.

### 3.2 But depth shows NO directional preference — the load-bearing result

`L_state` is each arm's fit to **its own** targets, so it is the one number that
compares arms on equal footing:

```text
state_ordered   L_state = 0.5441
state_reverse   L_state = 0.5403     <- fits its REVERSED targets equally well
state_shuffled  L_state = 0.9745
```

Reverse fits a fully **reversed** state-to-depth assignment marginally *better* than
ordered fits the canonical one. `SSOT §5` gives reverse the job of testing
"directional compatibility of reasoning and depth". On this evidence there is none:
the depth axis hosts the trajectory backwards as readily as forwards.

This is a direct measurement of fit quality, not a paired capability test, so it does
**not** depend on the §2 power problem.

Note the margin and monotonic columns are computed against the **canonical** order,
so reverse necessarily scores badly on them by construction. Those two columns
cannot be used to compare ordered against reverse; `L_state` can.

### 3.3 The ordered-vs-shuffled gap is shuffled being harmed, not ordered being helped

The baseline anchors this. Ordered lands *exactly* on baseline (+0.0000, 5:5
discordant, p=1.0), while shuffled (−0.020) and reverse (−0.015) sit slightly
*below* it. So the +0.020 of ordered over shuffled is not a benefit of ordering; it
is the cost of an auxiliary loss that cannot be fitted.

That shuffled cannot be fitted is near-tautological and must not be reported as
evidence: `_deterministic_permutation` derives the permutation per problem from
SHA-256 of seed and problem id, so there is **no consistent function** from (depth,
prompt) to target for the shared projector to learn. `L_state` 0.9745 is that
impossibility, not a fact about reasoning. Any consistent-versus-random labelling
would show the same gap.

## 4. The most plausible alternative explanation

**Training budget.** Each arm saw 63 optimizer steps over 1000 problems. If ordered
depth supervision needs substantially more training before it can reorganise
computation rather than merely fit a readout, this pilot could not have seen it.

Two further caveats on the same axis: the trained MLP projector's margin of 0.575
remains **below** the 0.977 reached by a fitted *linear* probe on the **untrained**
backbone under the same geometry (`PHASE0_READINESS.md` §3.8), and the endpoint sits
at 0.78 rather than near 0.5, which is not the maximum-power operating point.

## 5. Recommendation

**Do not spend ~15 hours on the 18-run matrix to confirm this null, and do not
declare the hypothesis closed.** Neither is warranted by §2.

Escalate the single most plausible confound first, because it is cheap and it is
prerequisite to any confirmatory claim:

1. **Training-budget escalation** — 4000 train records (250 optimizer steps, 4× this
   pilot) for `cot_baseline`, `state_ordered`, `state_reverse`, one seed, 800 DEV.
   About 3 hours, roughly a fifth of the full matrix. This answers: does more
   training move capability, and does the ordered/reverse `L_state` gap open up?
   Drop `state_shuffled` from this step — §3.3 shows it measures an impossibility.
2. If ordered still equals baseline and reverse still fits equally well, that is
   `SSOT §13`'s explicit **revise** condition ("ordered improves mechanism metrics
   but not capability"), reached on evidence rather than on a guess. The recorded
   revision candidate is semantic **motion** rather than absolute state — supported
   by `PHASE0_READINESS.md` §3.8, where between-problem same-stage delta cosine is
   0.94–0.99 against within-problem cross-stage −0.32.
3. Only if the escalation shows movement does the frozen 18-run matrix become worth
   its cost, at which point 3 seeds × 800 DEV gives **97.8%** power for a 2:1 split
   and a **1.56 pp** minimum detectable effect.

## 6. Preregistration inputs this pilot resolves

Two `TBD` fields can now be filled with measured values rather than guesses:

- **practical minimum effect:** 1.56 pp at 3 seeds × 800 DEV (80% power, α=0.05,
  measured 6.0% discordance). Propose freezing at **2.0 pp**.
- **DEV size:** 800 per seed with **three** seeds is required; one seed at 800 gives
  only 56.7% power and must not be treated as confirmatory.

## 7. Claim boundary

One seed. 200 DEV problems. 63 optimizer steps. No preregistration freeze. TEST
never opened. Nothing here is a Rineya capability claim, and no arm comparison in §1
is confirmatory. The `L_state` comparison in §3.2 is the only result robust to the
power problem, and it points against the directional premise rather than against the
hypothesis as a whole.
