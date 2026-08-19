# Contributing

## Ground rules

This repository exists to make one comparison trustworthy. Contributions are
judged by whether they preserve that.

1. **Never change one arm without the other.** Anything that touches FULL's data
   path, prompts, parsing or scoring must be checked against SPLIT's, and the
   reverse. A change that helps one arm is a confound.
2. **Never silently drop an item.** Unparseable, refused and empty responses stay
   in the denominator. If you add a code path that can discard a record, it needs
   a counter and a place in the report.
3. **Strict stays primary.** New extraction routes belong in `STRICT_ROUTES` only
   if they are genuinely unambiguous.
4. **Ordering is SHA-256, not `random.shuffle`.** Reproducible ordering across
   machines and Python versions is what makes the B-vs-C ablation meaningful.
5. **Document any new number.** A metric that is not in
   `docs/METRIC_CARD.md` cannot be interpreted by a reader.

## Setup

```bash
pip install -e ".[all]"
export PYTHONHASHSEED=0
export RINEYA_DATASET_ROOT=/abs/path/to/rineya_sequence_independence_dataset_v1
```

## Before opening a pull request

```bash
ruff check src tests
python -m compileall -q src
pytest -q
rineya-lab smoke --all
```

The smoke suite must report `"status": "PASS"`. Tests that need the dataset skip
cleanly without it; CI runs both modes.

## Tests we expect with a change

| Change | Required test |
|---|---|
| new scorer | equivalent-inputs pass, different-inputs fail, hostile input does not raise |
| new extraction route | a strict case, a lenient case, and a case it must *not* claim |
| new protocol | the perturbation/edit is visible in the emitted records |
| new order mode | adjacency structure asserted **and** record set unchanged vs shuffled |
| new statistic | a hand-computable case with a known answer |
| new manifest field | the relevant schema in `schemas/` updated and re-verified |

## Style

- Line length 120; `ruff` config lives in `pyproject.toml`.
- Type annotations on public functions.
- Comments explain *why*, especially where a choice protects the experiment's
  validity. Do not narrate what the code already says.
- Heavy dependencies (torch, transformers, peft) stay lazily imported so the data,
  evaluation and analysis stack runs on a CPU-only machine.

## Reporting a result

If you run the real experiment, include the run manifests, the evaluation
reports, the comparison report, the seeds used, the budget regime, and your
frozen margin with its justification. A number without its manifest is not a
result.
