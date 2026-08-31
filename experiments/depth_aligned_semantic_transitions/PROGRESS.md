# DAST progress

## 2026-08-30 — V1 foundation

**Status: BUILT — UNRUN on a real checkpoint. TEST CLOSED. REMOTE CI BLOCKED BEFORE RUNNER ASSIGNMENT.**

Implemented on branch `research/depth-aligned-semantic-transitions-v1`:

- branch-local research contract and program SSOT;
- strict configs for six Phase-1 arms and matched operator controls;
- hidden-state checkpoint resolution and variable-trace alignment;
- prompt-only semantic pooling with answer-token leakage tests;
- shared semantic interface and frozen deterministic target encoder;
- low-rank operator dictionary and regularized router;
- optional causal feedback hooks and intervention utilities;
- real Hugging Face/PEFT/QLoRA loader with local-only defaults;
- deterministic training, DEV metrics, generation output, paired statistics,
  manifests, hashes, and checkpoint artifacts;
- offline toy causal model and 13-record fixture;
- Linux/macOS and Windows launchers;
- path-scoped GitHub Actions workflow.

Local engineering validation before commit:

```text
python -m compileall -q src                         PASS
pytest -q                                           PASS (37 tests)
python -m dast.cli smoke --output-dir <temp>        PASS
fixture validation                                 PASS (13 records)
feedback causal-path check                         PASS (non-zero logit delta)
```

The smoke trained four toy batches and verified that the shared state projector
updated. It is **not** evidence that Qwen or Rineya reasoning improved. No real
adapter was trained, no DEV model result was produced, and the sealed TEST record
was not used.

### Remote CI qualification attempt

GitHub Actions workflow run `33356304292` was triggered by the branch push and was
retried once without changing the commit:

```text
attempt 1 job 99379039772    FAILURE before runner assignment
attempt 2 job 99379413359    FAILURE before runner assignment
```

For both attempts, the Actions API reported no executed steps, no downloadable job
log, `runner_id: 0`, an empty runner name/group, and zero billable job milliseconds.
The observed failure therefore occurred before checkout or any repository command
could run. It is classified as:

```text
EXTERNAL BLOCKER — GITHUB-HOSTED RUNNER NOT ASSIGNED
```

This is **not** an application-test failure and is **not** a remote CI pass. The
specific platform-side cause cannot be established from the available run metadata,
so the workflow and tested code were left unchanged rather than patched
speculatively.

### Next permitted action

Re-run the unchanged offline qualification workflow once a GitHub-hosted runner is
actually assigned. Only after its checkout, installation, static compilation,
Ruff, unit/integration tests, and toy smoke all pass may the program perform a
bounded one-seed resource rehearsal for `cot_baseline` and `state_ordered` on an
audited local checkpoint. Do not run operator or feedback science phases until
ordered-state DEV controls pass the preregistered gates.
