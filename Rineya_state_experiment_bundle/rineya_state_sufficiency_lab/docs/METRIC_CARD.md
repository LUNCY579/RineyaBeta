# Metric card

## Primary metric

**Strict final-answer accuracy**, paired by `problem_id` on the shared held-out split.

- **Numerator**: items where an unambiguous, contract-compliant answer was
  extracted *and* the typed deterministic scorer accepted it.
- **Denominator**: every evaluated item, including empty responses, refusals,
  malformed output and backend failures.
- **Not** included in the numerator: last-line-fallback extractions, ambiguous
  responses with conflicting candidates, rollouts that ended by loop, stagnation,
  premature completion or step-budget exhaustion.

For rollouts, strict credit additionally requires `termination_reason == "done"`.
A rollout that produced the right answer while looping is *not* strictly correct;
it is counted in the lenient track and shows up as a control failure.

## Secondary metrics

| Metric | Definition | Where |
|---|---|---|
| Lenient accuracy | any recoverable value the scorer accepts | `protocols.<p>.lenient` |
| Track disagreement | items correct in exactly one track | `protocols.<p>.track_disagreement` |
| Format / control / reasoning rate | share of items in each failure group | `protocols.<p>.failure_groups` |
| Oracle-state transition accuracy | step + state + `done` all correct given the gold state | `protocols.oracle_transition` |
| Step match / state validity / done accuracy | components of the above, separately | `protocols.<p>.transition_components` |
| Free-rollout gap | oracle transition accuracy − free-rollout final accuracy | `derived.free_rollout_gap` |
| Teacher-forcing gap | oracle-trajectory accuracy − free-rollout accuracy | `derived.free_rollout_gap` |
| State information gain | compact-state accuracy − no-state accuracy | `derived.state_sufficiency` |
| Compact − history | compact-state accuracy − full-history accuracy | `derived.state_sufficiency` |
| First-error depth | first rollout step whose emitted state diverges from canonical | `protocols.generated_rollout.rollout` |
| Completion / overrun rate | share terminating with `done` / exceeding canonical length | `protocols.<p>.rollout` |
| Category-macro accuracy | unweighted mean over the 30 categories | `protocols.<p>.by_category` |

## Scoring

Deterministic and typed. The scorer is selected from `answer_type`
(21 types in this dataset) and recorded on every record.

| Family | Scorers |
|---|---|
| numeric | `integer`, `float`, `scientific`, `percentage`, `quantity` |
| exact rational | `fraction` |
| logical | `boolean` |
| textual | `text_exact`, `text_normalized`, `multiple_choice` |
| structural | `ordered_collection`, `unordered_collection`, `json` |
| temporal | `datetime` |
| symbolic | `symbolic` (sympy), `relation` |
| configurable | `regex:<pattern>` |
| fallback | `auto` |

Tolerances (`numeric_tolerance_abs`, `numeric_tolerance_rel`,
`percent_tolerance_abs`), unit synonyms, case-sensitive types and the
answer-type → scorer map are all configurable under `evaluation.scoring`.

A general-purpose LLM judge is deliberately **not** the correctness measure:
these tasks have verifiable canonical answers, and a judge would add an
uncontrolled source of variance to the primary endpoint. A judge may be added as
a supplementary qualitative diagnostic.

## Statistics

- **Paired cluster bootstrap** over problems (primary interval).
- **Exact McNemar** on discordant pairs (binary outcomes only; reported as
  not-applicable for fractional per-problem scores).
- **Sign-flip randomization** with the +1 correction, so p is never exactly 0.
- **Decisions**: superiority (CI above 0), non-inferiority (CI lower bound above
  −margin), equivalence (CI inside ±margin).
- **Multiplicity**: Holm-Bonferroni within each stratification family.

## Known limitations

1. Synthetic, template-generated tasks. High verifiability, low linguistic
   diversity. Results do not transfer automatically to open-ended reasoning.
2. Compact states are constructed from known canonical solutions. Real deployment
   requires a model or harness that constructs and verifies its own states —
   which is exactly what `generated_rollout` measures, and why the free-rollout
   gap is reported prominently.
3. SPLIT is asked for structured output and FULL is not. Format-attributable
   difference is quantified but not eliminated.
4. The cross-protocol primary pairing (`full_direct` vs `generated_rollout`) is
   the fairest end-to-end contrast available, but the two arms are not answering
   an identically-worded prompt. Same-protocol pairings are reported alongside.
5. Explicit reasoning targets are synthetic supervision traces, not evidence
   about any model's internal computation.
