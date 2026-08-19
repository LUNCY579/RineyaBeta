# Failure taxonomy

Every scored item receives exactly one label. Implemented in
`src/rineya_lab/evaluation/failures.py`.

The taxonomy separates format failures from reasoning failures because the
hypothesis is about reasoning. A formatting regression in one arm that is not
labelled as such would be indistinguishable from a capability difference.

## Groups

### Format / protocol failures

| Label | Definition |
|---|---|
| `empty_output` | the response contained no non-whitespace characters |
| `refusal` | the response matched a refusal pattern and contained no structured payload |
| `malformed_structure` | JSON was requested and nothing parseable could be recovered, even after conservative repair |
| `missing_fields` | a parseable object omitted a required key, or declared `done: true` with no `final_answer` |
| `invalid_completion_flag` | `done` was absent, null, or not a boolean |
| `ambiguous_answer` | multiple conflicting answer candidates were present |
| `no_answer_found` | no route produced any candidate |
| `overlong_output` | the response exceeded the extraction length guard |
| `backend_failure` | generation itself raised (OOM, timeout, decode error) |

### Rollout-control failures

| Label | Definition |
|---|---|
| `nontermination` | the step budget was exhausted without `done: true` |
| `loop` | a compact state was revisited after changing at least once |
| `stagnation` | the state was unchanged for `stagnation_patience` consecutive steps |
| `premature_completion` | `done: true` with no usable `final_answer` |

### Reasoning failures

| Label | Definition |
|---|---|
| `wrong_answer` | a well-formed final answer that the typed scorer rejected |
| `wrong_transition` | a well-formed next step that does not match the canonical step |
| `state_drift` | the step matched but the emitted `updated_compact_state` did not |

`correct` is the only non-failure label.

## Assignment rules

1. A correct strict item is always `correct`.
2. Otherwise, if the parse status names a format problem, that label wins. A
   response that cannot be read cannot be diagnosed as a reasoning error.
3. Otherwise, for rollouts, the control failure (loop / stagnation /
   nontermination / premature completion) wins.
4. Otherwise, for transitions: `wrong_transition` → `state_drift` →
   `invalid_completion_flag`, in that order.
5. Otherwise `wrong_answer`.

Rule 2 matters for the comparison: SPLIT is asked for structured output and FULL
is not, so SPLIT has more opportunities to fail on format. Reports therefore
always show `format_rate`, `control_rate` and `reasoning_rate` separately, and
the paired comparison reports how much of the difference is format-attributable.

## Reading the distribution

- High `format` in one arm only → the contrast is partly about output-format
  compliance. Report the lenient track next to the strict track.
- High `control` in SPLIT with high `oracle_transition` accuracy → local
  transitions were learned but self-generated states derail the rollout. This is
  the free-rollout gap, and it is a finding, not a bug.
- High `reasoning` in both arms → a genuine capability comparison.
- Any `backend_failure` at all → investigate before reporting; these are
  infrastructure events, not model behaviour.
