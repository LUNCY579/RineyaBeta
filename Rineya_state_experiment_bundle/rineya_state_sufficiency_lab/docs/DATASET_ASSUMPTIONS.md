# Dataset assumptions

Everything this repository assumes about
`rineya_sequence_independence_dataset_v1`. `rineya-lab validate-data` checks each
assumption; the check that verifies it is named beside each item.

## Layout

```
<root>/
├── data/full_context/{train,dev,test}.jsonl
├── data/split_transitions/{train,dev,test}.jsonl
├── data/manifests/{dataset_stats.json,validation_report.json,sha256sums.txt,...}
└── schemas/{full_context.schema.json,split_transition.schema.json}
```

`schemas/` is optional: without it (or without `jsonschema` installed) the
validator falls back to structural field checks and says which mode it used.

## Required fields

**full_context**: `variant`, `problem_id`, `split`, `domain`, `category`,
`problem`, `answer`, `full_response`, `reasoning_steps`, `compact_state_trace`,
`step_count`, `messages`. Optional but used: `answer_type`, `difficulty`.

**split_transition**: `variant`, `transition_id`, `problem_id`, `split`,
`domain`, `category`, `step_index`, `total_steps`, `problem`, `current_state`,
`target_step`, `next_state`, `done`, `messages`, `stateful_rollout_messages`,
`sequence_free_training`. Optional but used: `final_answer`,
`stateful_rollout_target`, `answer_type`.

## Structural invariants

| Assumption | Checked by |
|---|---|
| `problem_id` unique within a split | duplicate-id check |
| normalised problem text unique across the corpus | uniqueness check |
| TRAIN / DEV / TEST disjoint **at the problem level** | cross-split disjointness |
| `step_index` runs 1..`total_steps` with no gaps | `Trajectory.check_chain` |
| `transitions[0].current_state == compact_state_trace[0]` | `Trajectory.check_chain` |
| `transitions[i].next_state == transitions[i+1].current_state` | `Trajectory.check_chain` |
| exactly the final transition has `done: true` | `Trajectory.check_chain` |
| the final transition's `final_answer == answer` | `Trajectory.check_chain` |
| `[t.target_step] == reasoning_steps` | `Trajectory.check_chain` |
| `len(transitions) == step_count` | `Trajectory.check_chain` |
| `sequence_free_training` is `true` | flag check |
| no two adjacent transition rows share a `problem_id` | shuffle check |
| file hashes match `sha256sums.txt` | `verify_checksums` |
| row counts match `dataset_stats.json` | manifest cross-check |

## The causal safeguard

The dataset builder may inspect the *finished* trajectory to construct `s_t`, but
`s_t` must not contain anything knowable only after step *t*. The validator's
cheap, high-precision test: the literal final answer must not appear in an
explicit "final answer" / "answer is" form in a state that precedes the step
deriving it. A bare numeric coincidence is not flagged, because intermediate
values legitimately equal the answer.

This test is verified to actually fire (`test_causal_safeguard_catches_injected_leakage`);
a safeguard that can never fail proves nothing.

## Message views

- `messages` on a full_context row → FULL supervision.
- `messages` on a transition row → next-step-only supervision.
- `stateful_rollout_messages` on a transition row → stateful supervision, with a
  JSON target carrying `next_reasoning_step`, `updated_compact_state`, `done`,
  `final_answer`.

Config validation forbids mixing views inside one adapter run.

## Scale (v1.0)

| Item | Value |
|---|---|
| unique problems | 48,000 |
| categories | 30 (1,600 each) |
| full TRAIN / DEV / TEST | 43,200 / 2,400 / 2,400 |
| transition TRAIN / DEV / TEST | 170,941 / 9,460 / 9,521 |
| mean transitions per problem | 3.9567 |
| min / max transitions | 2 / 9 |
| answer types | 21 |

The code does not hard-code these numbers; it reads them from
`dataset_stats.json` and cross-checks. A regenerated dataset with a different
`--per-category` works without code changes.

## What the code does *not* assume

- That the dataset is at any particular path (resolution is configurable).
- That `sha256sums.txt` exists (verification is skipped with a warning).
- That answer types outside the 21 known ones are absent (unknown types fall back
  to the `auto` scorer chain).
- That every trajectory has the same length (bounds are per-trajectory).

## Adapting another dataset

See `docs/EXTENDING.md`. The short version: provide the fields above, or supply
your own loader implementing `iter_full`, `iter_transitions` and `trajectories`.
