# Native Qwen3 CPU integration — verified successful

**32 tests passed, zero skipped, zero failures/errors.** This is additional
implementation evidence, not a trained Rineya model result.

GitHub Actions run: [34267747587](https://github.com/LUNCY579/RineyaBeta/actions/runs/34267747587)  
Job: `102201298268`, `cpu-native`, conclusion `success`  
Tested commit: `c586dde373b4f1fb7f5368cd05d1a4337c80ff89`  
Execution: 2026-09-08, job completed at approximately 19:14:49 UTC.

The job installed torch 2.6.0+cpu and Transformers 4.57.1 on Python 3.11.16,
with safetensors 0.8.0 and tokenizers 0.22.2. Its command was:

```
python -m experiments.hybrid_attention_v1.verify --run-id ci-native --require-hf
```

Unlike the earlier local receipt, this run executed the actual native
`Qwen3ForCausalLM` random-model integration test. It passed native full-attention
parity, streamed/full equivalence, compact-cache checks, and head-gate identity
on a tiny four-layer random model. This validates that implementation path on
this CPU/library stack. It does not qualify R2 weights, GPU/BF16, other library
versions, other Rineya checkpoints, or model quality.

The job's exact receipt payload is preserved in `ci_native_qualification.json`.
It was reconstructed from the returned complete job-log JSON and verified
against the original receipt-integrity checksum:

```
1a21e062c611f1c2b9e4d5859b144e5a7500d696077834621c54efb4a59cff01
```

Its code fingerprint matches the local qualification and the committed sources:

```
892a355e31dbe3749e50c4ce8a70f093a06dad81fe38eb73291a9b72cfabecd0
```

The successful job also uploaded the original receipt and `pip freeze` output as
artifact `10072536278`, configured for 30-day retention. The committed receipt
is permanent repository evidence even after that workflow artifact expires.
The separate local receipt with one skipped test remains unchanged.

All source code and the protocol are unchanged by this evidence-only commit.
Training, real Rineya forward passes, open quality diagnostics, sealed evaluation
and causal speed measurements remain unrun. The Node deprecation warnings in
the runner log did not fail the job; they do not affect the CPU test verdict.
