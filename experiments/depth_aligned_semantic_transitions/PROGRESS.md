# DAST progress

## 2026-08-30 — V1 foundation

**Status: BUILT — UNRUN on a real checkpoint. TEST CLOSED.**

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

### Next permitted action

Run the offline workflow in remote CI, then perform a bounded one-seed resource
rehearsal for `cot_baseline` and `state_ordered` on an audited local checkpoint.
Do not run operator or feedback science phases until ordered-state DEV controls
pass the preregistered gates.
