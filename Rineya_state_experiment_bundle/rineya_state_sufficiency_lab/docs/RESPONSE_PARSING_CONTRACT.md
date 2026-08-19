# Response-parsing contract

Normative description of how an arbitrary model response becomes a scored item.
Implemented in `src/rineya_lab/evaluation/parsing.py`; tested in
`tests/test_parsing.py`.

## Principles

1. **Nothing is discarded.** Every response is retained verbatim in
   `records/<protocol>.jsonl`. An item that cannot be interpreted stays in the
   denominator and scores as a failure with a failure type attached.
2. **The route is recorded.** Every record carries `extraction_route`, so
   "the model knew the answer but formatted it badly" is separable from
   "the model was wrong".
3. **Repair is conservative.** Only transformations that cannot change the
   meaning of already-valid JSON are applied. Anything more aggressive risks
   manufacturing an answer the model never produced.
4. **Two tracks.** Strict is primary; lenient is diagnostic. They are always
   reported together, along with their disagreement.

## Requested formats

**Final-answer protocols** (`full_direct`, `direct_answer`, `oracle_trajectory`,
`generated_rollout`) accept any format. The extractor searches for an answer.

**Transition protocols** in the stateful condition request exactly:

```json
{"next_reasoning_step": "...", "updated_compact_state": "...", "done": false, "final_answer": null}
```

This mirrors the `stateful_rollout_messages` training target verbatim.

## Extraction routes, in priority order

| Route | Trigger | Strict credit |
|---|---|:--:|
| `json_exact` | the whole response parses as a JSON object | yes |
| `json_fenced` | a fenced ```` ```json ```` block parses | yes |
| `json_embedded` | a balanced `{...}` span inside prose parses | yes |
| `json_repaired` | parses after conservative repair | yes |
| `tag_final_answer` | `<final_answer>…</final_answer>`, `[FINAL ANSWER] …`, `<answer>…</answer>` | yes |
| `boxed_latex` | `\boxed{…}` (brace-balanced) | yes |
| `labeled_answer` | `Final answer: …`, `Answer: …`, `The final answer is: …`, `### ✅ **Final Answer:** …` | yes |
| `plain_text_step` | next-step-only condition: bare prose *is* the contract | yes |
| `last_line_fallback` | the final non-empty line | **no** |
| `refusal` | matches a refusal pattern | no |
| `empty_output` | response is whitespace only | no |
| `no_answer_found` | nothing matched | no |

The balanced-brace scanner respects string literals and escapes, so a `}`
inside a JSON string never terminates a span early.

## Conservative repair

Applied in this order:

1. Drop prose before the first `{` and after the last `}`.
2. `True`/`False`/`None`/`NULL`/`undefined` → `true`/`false`/`null`.
3. Single quotes → double quotes **only** when the text contains no double quote.
4. Strip a dangling trailing comma at end of input.
5. Append missing `}` / `]` (never remove content).
6. Strip trailing commas before any closer, including ones just appended.

Repair returns `None` unless the result parses. It never edits string contents.

## Answer-label decoration

Real models decorate the answer line. All of these are strict-credit
`labeled_answer` matches, verified against observed Qwen3-0.6B output:

```
The final answer is: **Ivan**.
### ✅ Final Answer: **False**.
- **Final Answer:** `246 bolts`
**Answer:** 09:44
So the final answer is 42.
Answer - 42
```

Two rules keep this from over-matching:

* a dash counts as a separator **only when whitespace-surrounded**, so
  `The final answer is -6` keeps its minus sign;
* emphasis is skipped after the label only when the run is followed by
  whitespace, so the opening `**` of `**Ivan**` is left for the symmetric-wrapper
  stripper rather than swallowed.

Symmetric wrappers (`**`, `__`, `*`, `_`, backticks, quotes) and trailing
punctuation are removed iteratively, because the layers nest: `**Ivan**.` needs
the period gone before the bold markers become symmetric.

## Conflicting and multiple answers

If several candidates are found and they are not all identical:

- `ambiguous = true`, status `ambiguous`;
- **strict withholds credit** — an ambiguous response is not an unambiguous answer;
- lenient scores the **last stated** candidate, on the reading that a model's
  final statement is its answer;
- both the count and the candidate list are stored on the record.

## Transition-specific rules

| Situation | Status | Strict | Lenient |
|---|---|:--:|:--:|
| all three required keys present and well typed | `ok` | yes | yes |
| `done` is `"yes"`/`"true"` as a string | `invalid_completion_flag` | no | yes |
| `done` missing, null or non-boolean | `invalid_completion_flag` | no | yes |
| a required key is absent | `missing_fields` | no | if a step is recoverable |
| `done: true` with empty `final_answer` | `missing_fields` (premature completion) | no | yes |
| `done: false` with a `final_answer` | `ok` + note | yes | yes |
| no JSON recoverable, prose present | `malformed` | no | yes (prose = step) |
| empty / refusal / backend error | `empty`, `refusal`, `backend_failure` | no | no |

A response yielding no `updated_compact_state` and no `done` is *unusable for
continuing a rollout*, even when prose was salvaged; the rollout treats it as an
invalid step (see `docs/FAILURE_TAXONOMY.md`).

## Strict vs lenient

- **Strict** = contract-compliant route **and** unambiguous **and** the typed
  scorer accepts the value. This is the primary track for every reported number.
- **Lenient** = any recoverable value the typed scorer accepts, including
  last-line fallback and last-of-conflicting.

`strict − lenient` is reported per protocol and per comparison as
`format_attributable_pp`. A large value means the measured difference is partly
about contract compliance rather than reasoning, and must be reported as such.
