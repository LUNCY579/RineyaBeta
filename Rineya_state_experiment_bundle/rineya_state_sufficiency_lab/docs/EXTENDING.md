# Extending the lab

## Add a scorer

```python
# src/rineya_lab/evaluation/scoring.py
def score_roman_numeral(prediction, gold, cfg):
    return ScoreResult(to_int(prediction) == to_int(gold), "roman_numeral",
                       str(prediction), str(gold))

SCORERS["roman_numeral"] = score_roman_numeral
```

Then map an answer type to it, without touching code:

```yaml
evaluation:
  scoring:
    answer_type_scorers:
      roman: roman_numeral
```

A regex scorer needs no Python at all:

```yaml
evaluation:
  scoring:
    answer_type_scorers:
      code_output: "regex:OUTPUT=(\\S+)"
```

Scorers must never raise; `score_answer` catches exceptions and records them as
an incorrect verdict with a note, but a scorer that throws on ordinary input is
still a bug.

## Add an extraction route

Add the route constant in `evaluation/parsing.py`, append it to `STRICT_ROUTES`
only if it is genuinely unambiguous, and slot it into the priority list inside
`extract_final_answer`. Then document it in
`docs/RESPONSE_PARSING_CONTRACT.md` — an undocumented route makes reported
numbers non-interpretable.

## Add a protocol

1. Add the name to `PROTOCOLS` in `config.py`.
2. Add a prompt builder in `evaluation/prompts.py`. Make it a **minimal edit** of
   the canonical transition prompt so a difference is attributable.
3. Add the branch in `run_transition_protocol` or `run_answer_protocol`, and list
   the protocol in `ANSWER_PROTOCOLS` or `TRANSITION_PROTOCOLS`.
4. Set `in_distribution_for` honestly.
5. Add a test asserting the perturbation is visible in the emitted records.

## Add a backend

Subclass `Backend`, implement `generate` (and `generate_batch` if it can batch)
and `describe`, then register it in `build_backend` and in the `backend.kind`
validation list. If it is a test fixture, set `is_reference = True` so its
reports are labelled and comparisons warn.

## Add a training-order ablation

Add the mode to `ORDERS` in `config.py` and a branch in `_order_rows`. Order
**must** be produced by sorting on salted SHA-256 keys, not `random.shuffle`,
otherwise the B-vs-C contrast stops being reproducible. Add a test in
`tests/test_views.py` that asserts the resulting adjacency structure and that the
record *set* is unchanged relative to the shuffled condition.

## Use a different dataset

Provide the fields in `docs/DATASET_ASSUMPTIONS.md`, or implement a loader with
`iter_full(split)`, `iter_transitions(split)`, `trajectories(split)`,
`index(variant, split)`, `path_for(variant, split)` and `provenance()`. Everything
downstream depends only on that surface.

If your trajectories are longer than this dataset's, raise
`evaluation.rollout.max_steps_absolute`; the per-trajectory bound adapts on its own.

## Add a longer-chain extrapolation study

```yaml
# train on short trajectories
data_view: {step_index_max: 6}
# evaluate on long ones
evaluation: {split: test}
```

Filter the eval set by `total_steps` in analysis using the `by_total_steps`
stratification already present in every report.

## Add a state-corruption study

`shuffled_state` and `distractor_state` already exist. For an injected *false
fact*, add a protocol that rewrites one number inside `current_state` and record
whether the model follows it, detects it, repairs it, or abstains — the failure
taxonomy has room for a new `detected_inconsistency` label.

## Add a model judge

Add it as a **supplementary** field on the record (`components["judge"]`), never
as the source of `strict_correct`. Document the judge model, revision, prompt and
temperature in the metric card, and report judge-vs-deterministic disagreement.
