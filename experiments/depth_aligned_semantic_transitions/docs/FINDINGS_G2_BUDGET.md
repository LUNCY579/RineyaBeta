# Training-budget escalation — findings

**Status: EXPLORATORY_DEV. One seed. This does NOT close the DAST hypothesis, and it
is NOT a confirmatory result.**
**Branch:** `research/depth-aligned-semantic-transitions-v1` at `f160f38`
**Run date:** 2026-08-31
**Sealed evaluation:** CLOSED, never touched.

Three arms, seed 17, **4,000 train / 800 DEV**, 250 optimizer steps — 4× the G1
pilot on both axes. Same corrected geometry (`final_norm` readout at layers
7/14/21/27, `within_problem` target centering), `max_new_tokens: 384`. About 48–57
min per arm; 21.8 min of that is training, the rest generation.

Purpose: test the single most plausible alternative explanation for the pilot's
null — that 63 optimizer steps was simply too little training for depth supervision
to reorganise computation rather than merely fit a readout.

`state_shuffled` was deliberately excluded: its permutation is drawn per problem
from SHA-256 of seed and problem id, so no consistent function from (depth, prompt)
to target exists and its failure to fit measures an impossibility, not a fact about
reasoning.

## 1. Results

| arm | generated accuracy | order margin | checkpoint top-1 | monotonic | `L_state` | LM ppl |
|---|---:|---:|---:|---:|---:|---:|
| **`cot_baseline`** | **0.8938** | +0.0141 | 0.2750 | 0.6125 | — | 1.0067 |
| `state_ordered` | 0.8775 | **+0.9356** | **0.8338** | **0.9862** | 0.2622 | 1.0076 |
| `state_reverse` | 0.8900 | −0.1911 | 0.1047 | 0.0000 | **0.2520** | 1.0075 |

Paired by problem id over the same 800 DEV problems, clustered bootstrap by
`generator` (30 clusters):

| contrast | difference | clustered 95% CI | discordant | exact McNemar p |
|---|---:|---|---:|---:|
| **ordered − baseline** | **−0.0163** | **[−0.0317, −0.0025]** | 13 : 26 | **0.0533** |
| reverse − baseline | −0.0037 | [−0.0252, +0.0157] | 20 : 23 | 0.7608 |
| ordered − reverse | −0.0125 | [−0.0326, +0.0100] | 19 : 29 | 0.1934 |

## 2. The budget confound is resolved: more training helped the MECHANISM a great deal

| | 63 steps (pilot) | **250 steps** |
|---|---:|---:|
| `L_state` (ordered) | 0.5441 | **0.2622** |
| order margin (ordered) | 0.5751 | **0.9356** |
| checkpoint top-1 | 0.7100 | 0.8338 |
| monotonic progress | 0.9750 | 0.9862 |

The pilot's caveat — that the trained margin sat **below** the 0.977 reached by a
fitted linear probe on the untrained backbone — is now nearly closed: 0.9356 against
that 0.977 floor. Depth-aligned semantic states are clearly learnable, and four
times the training makes them much more so.

## 3. But capability moved the other way

**`cot_baseline` is the best arm.** Plain CoT SFT, with no depth supervision at all,
scores 0.8938 against 0.8775 for ordered.

`ordered − baseline = −0.0163`, and the clustered interval **excludes zero**. This is
the first contrast in the program whose interval does not span zero.

So the dissociation is now sharp, and it runs opposite to the hypothesis:

> The shared projector learns the depth-to-state alignment extremely well
> (margin 0.936, top-1 0.834, monotonic 0.986), and learning it **costs** roughly
> 1.6 points of reasoning accuracy relative to not doing it at all.

`SSOT §13` names this exactly: *"ordered improves mechanism metrics but not
capability"* is a **revise** condition.

### Trend across budgets

| | baseline | ordered | ordered − baseline |
|---|---:|---:|---:|
| 63 steps, 200 DEV | 0.7800 | 0.7800 | **+0.0000** (5:5, p=1.0000) |
| 250 steps, 800 DEV | 0.8938 | 0.8775 | **−0.0163** (13:26, p=0.0533) |

Absolute accuracies are not comparable across rows — different DEV sets, 200 versus
800 problems. The **within-run contrast** is, and it moves from a dead tie at low
budget to a deficit at higher budget. Consistent with the auxiliary objective
consuming adapter capacity that the LM objective would otherwise have used, though
one seed cannot establish a trend from two points.

## 4. Four reasons not to over-read this

1. **One seed.** The preregistration requires three precisely because one seed at
   n=800 gives only **56.7%** power.
2. **The two tests disagree at the boundary.** The clustered bootstrap interval
   excludes zero while exact McNemar gives p = 0.0533. They are different tests and
   this is exactly the region where they diverge. Neither is wrong; the honest
   reading is "borderline".
3. **−1.63 pp is BELOW the 2.0 pp practical margin** proposed for the freeze. Even
   if real and replicated, it would not clear the preregistered bar for a
   practically meaningful effect.
4. **Directionality remains undecided.** `ordered − reverse` is −0.0125 with the
   interval spanning zero (p = 0.193), and reverse fits its own reversed targets
   marginally better (`L_state` 0.2520 vs 0.2622). There is still no detectable
   forward-direction advantage, which is the pilot's finding at 4× the data.

## 5. Verification performed on this result

Because two instrument failures had already produced confident wrong numbers in this
program, the ordered-vs-reverse contrast was checked before being reported:

| check | result |
|---|---|
| Is the reverse arm genuinely reversed? | assignment `[3,2,1,0]`, targets exactly inverted |
| Are the runs distinct? | different config hashes; different adapter SHA-256 |
| Same data and protocol? | identical train/dev SHA-256, checkpoints 7/14/21/27, readout, centering, seed |
| Same problems? | 800, same order; 88.9% identical raw responses |
| Extraction artifact? | routes identical across arms — `labelled` 800/800 |
| Discordance plausible? | **6.000%**, matching the pilot's 6.0% estimate on 4× the data |
| Are the wins real reasoning? | mostly yes — **one was not** |

The exception was a genuine scorer gap: `answer_type: state` gold is a bag of
key=value bindings, and a model answered `D=OFF, B=OFF, C=OFF, A=ON` for gold
`A=ON, B=OFF, C=OFF, D=OFF` — identical content, scored wrong under exact string
equality. Measured effect: **1 problem in 800**, moving ordered − reverse from
−0.0125 to −0.0112. Real, fixed in `f160f38`, and far too small to explain anything.

## 6. Recommendation

**Do not commit the 18–19 hour matrix yet.** Two targeted runs answer the go/no-go
question for about 40% of that cost:

1. **Seed replication** — the same three arms at seeds 29 and 43. **~5 h.** Converts
   a borderline single-seed result (p = 0.0533, two tests disagreeing) into an
   answer. Three seeds × 800 DEV = 2,400 paired problems gives **97.8%** power for a
   2:1 discordant split and a **1.56 pp** minimum detectable effect.
2. **`state_final_repeated`** at three seeds. **~2.5 h.** This is **G3**, and it is
   the control that decides *what* the deficit means. If final-repeated also costs
   ~1.6 pp, the harm comes from carrying **any** auxiliary depth loss, not from
   aligning it to reasoning order — a materially different conclusion, and one that
   would make the ordered/reverse question moot.

Only after those does the full matrix become worth its cost, and by then its purpose
would be the **confirmatory record** — `literal_ordered` for G4, `state_shuffled`
because `SSOT §5` requires it for the primary claim — rather than discovery.

## 7. Claim boundary

One seed. 800 DEV problems. 250 optimizer steps. No preregistration freeze. TEST
never opened. Nothing here is a Rineya capability claim. The one contrast whose
interval excludes zero (ordered − baseline) is borderline by the companion test,
sits below the proposed practical margin, and has not been replicated across seeds.
