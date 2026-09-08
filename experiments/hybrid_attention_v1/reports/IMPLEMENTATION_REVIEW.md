# HLGA-1 implementation review — 2026-09-08

## Evidence classification

**Implementation and CPU fixture qualification only. No measured improvement in
Rineya quality, long-context reasoning, training efficiency or GPU throughput.**

The initial checkpoint target is the repository's R2 artifact, not Supa or a
randomly initialized stand-in. Its configuration/tokenizer are tracked; real
weights were not available in this execution environment. No trained Rineya model
was loaded, no quality benchmark was run, and no model checkpoint was written.

## Executed local validation

Command, from the new subsystem's local repository-root layout:

```
python -m experiments.hybrid_attention_v1.verify --run-id cpu-qualification-v2
```

Result: **32 test methods discovered, 31 passed, 1 explicitly skipped, zero
failures/errors**. The skipped test requires native Transformers; that package
was unavailable. Random Qwen3-shaped fixture tests are not native HF integration.
The exact result is preserved in `cpu_qualification.json`, including environment,
code fingerprint and receipt-integrity checksum.

Covered: periodic placement; matched sink budgets; exact causal/window boundaries;
blocked-vs-dense and SDPA equivalence; future-token isolation; gradients; compact
resident storage versus views; absolute/gapped positions; full-vs-streamed
attention and decoder outputs; request resets; EOS and budgets; gate identity,
458,752-parameter R2 gate count and serialization; a toy gate-only optimizer step
that leaves its random backbone unchanged; unique open diagnostic generation;
format/abstention/loop scoring; paired membership; missing weights/LFS rejection;
immutable output paths; stale/fixture receipt rejection; closed training/sealed CLI.

The toy optimizer step is a unit test, not Rineya training or evidence that a gate
improves task accuracy. Gate-only training is not automatically enforced by merely
constructing HybridRuntime; a future trainer must explicitly freeze and verify
the backbone and optimizer parameter allowlist.

## Actual synthetic storage probe

Using 28 layers, 8 KV heads, head_dim128, FP32, 1,025 tokens, period-four globals,
and a 512 total local budget (4 sinks +508 recent), the retained KV allocations
measured **146,857,984 bytes**, exactly matching the registered formula. The
all-global formula at those dimensions is 235,110,400 bytes. These are synthetic
KV tensors, not a Rineya forward or a peak-process-memory benchmark.

The local attention probe observed at most **81,792 mask elements** per block
(128 queries ×639 union keys) in nine calls. This supports the implemented local
mask bound. It does not establish end-to-end speed, GPU peak memory, efficient
training, or the absence of other transient allocations.

## Deliberate boundaries

The committed local environment was Python 3.13.5, torch 2.10.0+cpu, one CPU
thread; CUDA and Transformers were unavailable. The optional native HF test must
be run on an installed supported stack; even passing that tiny random-model test
would not qualify real R2 weights or another version/device/backend.

A whole-repository regression suite was not run: only the isolated new subsystem
was materialized and tested. Existing files under scripts/train, historical
experiments, checkpoint assets, datasets and frozen suites were not edited.
No sealed evaluation content was read.

Native-versus-wrapper parity with actual R2, the 72-case open diagnostic, trained
gate recovery, independent quality confirmation, and isolated hardware benchmarks
remain unrun. The first three runnable stages are asset audit, mechanical
qualification, and receipt-gated open diagnostics. Training and sealed evaluation
are not implemented as executable stages and cannot be licensed by this receipt.

Future CI results, when available, must be recorded separately rather than
rewriting this local receipt into a different environment's evidence.
