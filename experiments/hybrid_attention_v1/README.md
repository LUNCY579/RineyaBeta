# HLGA-1: Rineya hybrid local/global attention

**Branch:** `research/hybrid-local-global-attention-v1`  
**Base:** `3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Status:** CPU fixture-qualified; real Rineya checkpoint qualification and model
quality are **unmeasured**. See [the evidence review](reports/IMPLEMENTATION_REVIEW.md)
and [machine-readable local receipt](reports/cpu_qualification.json).

## Research question and prior evidence

Can a Qwen3-derived Rineya backbone tolerate periodic local attention while
reducing **resident** KV storage, and do persistent initial tokens prevent the
sink-eviction failure observed by this repository's earlier experiment?
A later, separately controlled training question asks whether small head gates
repair damage beyond what the same training gives an all-global gate control.

Read [the old attention-shape experiment](../attention_shape/README.md) first.
Its catalogue already included every-fourth-layer global attention. Its measured
negative results include a window-boundary quality cliff, dense-mask prefill
cost, and apparent logical cache savings that did not immediately free the backing
allocation. Those results are not retracted here. An external model trained from
scratch with hybrid attention is motivation, not evidence that a retrofit works.

The new contribution is an **executable controlled re-test**: matched backends,
exact window accounting, sink-preserving controls, compact storage, absolute
positions, bounded-query-block masks, and independently tested head gating.

## Initial target and invariant architecture

The target is R2 at `artifacts/rineya-core-en32k-28L`: 28 layers, width 1024,
16 query heads / 8 KV heads, head dimension 128, intermediate width 3072,
vocabulary 32768. Read the checkpoint's own config and hash its real weights.
The remote folder contains configuration/tokenizer assets but no tracked model
weights; a checkout alone therefore cannot run real-model qualification.

The runtime retains the original QKV/output projections, QK-Norm, RMSNorm,
SwiGLU, RoPE, embedding and output head. It does not copy Spark dimensions,
change the tokenizer, merge heads, extend the context limit, or patch global
Transformers functions. It traverses existing Qwen3 modules explicitly; the
unchanged native forward remains an independent parity oracle.

A gate is applied per query head **after attention and before output projection**:

```
out[h] *= 2 * sigmoid(W(normalized_hidden)[h]); W starts at zero
```

It is exactly identity at initialization and has a nonzero first-order gradient.
The 28-layer R2 gate adds 458,752 parameters. The alternative parameterization
`1 + alpha*tanh(W*x)` must NOT initialize both alpha and W to zero: that would
kill both gradients. No trained gate is included or claimed in this branch.

## Registered ablations

| Arm | Attention | Total local budget | Persistent sinks | Head gate |
|---|---|---:|---:|---|
| A0_global | All global, matched backend | — | — | No |
| A1_local1024 | 3 local / 1 global | 1024 | 0 | No |
| A2_local512 | 3 local / 1 global | 512 | 0 | No |
| A3_sink512 | Same periodic placement | 512 | 4 | No |
| A4_global_gate | All global | — | — | Identity initially |
| A5_local512_gate | Same as A2 | 512 | 0 | Identity initially |
| A6_sink512_gate | Same as A3 | 512 | 4 | Identity initially |
| A7_sink1024 | Same periodic placement | 1024 | 4 | No |

At 28 layers the global indices are `3,7,11,15,19,23,27` (zero-based). Other
depths explicitly keep the final layer global, so non-multiples of four are not
exact Spark replications. They require their own real-checkpoint qualification.

**A 512 budget includes the current token and sinks.** A3/A6 retain four initial
tokens plus 508 recent tokens, with no duplicate positions. Keeping four sinks
in addition to 512 recent tokens would silently change the matched-budget test.
A0=A4, A2=A5 and A3=A6 must generate identical token IDs before gate training.

## Memory and attention implementation

`CompactKV` retains fresh compact allocations using index selection, not trailing
views into a full prefill buffer. Eviction occurs only after every query in the
current chunk has attended. Original token positions survive eviction; RoPE
positions never restart at zero or use the number of retained keys.

`blocked_attention` computes exact attention on each query block's relevant key
union. Pure-window and sink variants use one joint softmax, not independently
normalized attention streams. Explicit GQA expansion is identical across arms.
A small dense oracle is used only in tests.

This is a correctness-oriented PyTorch implementation, not FlashAttention or a
native sparse kernel. Global layers still require quadratic attention work.
Full non-streamed local execution also scans position metadata for each block;
do not infer a strict end-to-end O(NW) runtime guarantee from its attention shape.
Inference chunking bounds local transient history; it does not eliminate the
persistent global cache or full model activations. Training autograd retains
additional state. Resident KV bytes, allocated/reserved GPU peaks, masks, and
end-to-end memory are different measurements.

## Run from the repository root

Use the intended environment explicitly. The historical Windows GPU environment
is `.venv/Scripts/python.exe`; the examples below use `python` for portability.
Never interpret a CPU fallback as a GPU measurement. The committed local receipt
records Python 3.13.5 and torch 2.10.0+cpu; Transformers is absent there. Native
integration is a separately required test on a host with Transformers installed.
This is not a claim of compatibility with every repository environment.

```bash
# Read-only: show the frozen experiment plan.
python -m experiments.hybrid_attention_v1.run plan

# No weights or datasets: CPU tests plus a synthetic R2-shaped storage probe.
# Use a NEW run-id each time. --require-hf forbids the optional integration skip.
python -m experiments.hybrid_attention_v1.verify --run-id cpu-check-001
python -m experiments.hybrid_attention_v1.verify --run-id native-check-001 --require-hf

# Requires the actual local R2 safetensors and tokenizer assets.
python -m experiments.hybrid_attention_v1.run audit --run-id r2-audit-001
python -m experiments.hybrid_attention_v1.run qualify --run-id r2-qualify-001

# Only a passing REAL checkpoint receipt with matching code/assets/environment
# and backend/block size permits these newly generated open diagnostics.
python -m experiments.hybrid_attention_v1.run diagnose --run-id r2-dev-001 \
  --receipt experiments/hybrid_attention_v1/runs/r2-qualify-001/receipt.json
```

A custom **local** checkpoint directory can be supplied with `--model PATH`.
`trust_remote_code=False`, local-only loading and safetensors are required; missing
weights, unresolved LFS pointers and unsupported modules fail closed. Source
assets are hashed before and after execution. No Hub download is initiated.
Do not substitute random weights just to obtain a passing real-model receipt.

For CUDA, use `--device cuda --backend sdpa --cap-mib 6144` consistently in both
qualification and diagnosis; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process
start when required by deterministic CUDA operations. This path has not run in
the local receipt. The cap must be below currently free device memory; FP32
qualification is deliberate. A BF16 or other backend change needs a separate
registered tolerance/qualification, not a relabeled FP32 result.

`--max-cases` allows a bounded initial diagnostic (1–72); it does not confer
statistical validity. Output goes only to new `runs/<run-id>` directories.
Existing receipts cannot be overwritten. Generated runs are ignored by Git;
copy deliberately reviewed small evidence to `reports/` before committing it.

## Endpoint, isolation and evidence levels

The open DEV generator creates 72 raw-completion prompts: lengths
511/512/513 and 1023/1024/1025, three evidence depths, and retrieval, two-hop,
absent and conflicting-evidence tasks. This is a boundary/readout diagnostic,
not the historical chat/RAG score, a realistic long-document benchmark, or
independent confirmation. The simple two-hop fixture does not establish broad
multi-step reasoning. Prompts are token-length exact and never truncated.

Report strict and whitespace-trimmed exact match separately; do not strip
`code`/`vault` prefixes after seeing results. Whole `UNKNOWN` is an abstention,
not repeated `UNKNOWN` until the generation budget expires. Reports separate
answerable accuracy, false abstention, proper abstention, empty outputs,
repetition, and budget exhaustion by family/length/depth. Paired comparisons
bind IDs and prompt hashes. Descriptive McNemar values are unadjusted and these
synthetic examples are correlated: they are not a significance claim.

All timing in the diagnostic is marked **non-causal**. Fixed arm ordering,
shared devices, Python overhead and absent hardware isolation prevent a speed
claim. No clock/thermal/contention qualification has been run.

The implemented stages are fixture tests, asset audit, real-checkpoint mechanical
qualification, and receipt-gated open diagnostics. `train` and `sealed` are
explicitly unavailable. There is **no production training runner or model export**
in v1. A future training increment must first register reviewed data provenance,
license/derivation and hashes, duplicate/split audit, fixed optimization budgets,
all-global trained-gate controls, at least three seeds, and independent evaluation
criteria. Passing unit tests cannot license that increment or change another
branch's approvals. No existing corpora, sealed answer keys or frozen suites
are opened by the new CLI.

Supported first implementation: native unquantized Qwen3, FP32 qualification,
batch one, unpadded sequences, no packing/beams, original unscaled full RoPE,
no preexisting forward hooks. GGUF/llama.cpp/MLX, PEFT/quantized models, dual RoPE,
KV-head surgery, fused sparse kernels, and importance-selected global placement
are deferred. Shape-generic support is not family-wide checkpoint validation.

## Source map

| File | Responsibility |
|---|---|
| `config.py` | Immutable policies, eight arms, exact cache-budget arithmetic |
| `attention.py` | Causal masks, dense oracle, bounded-block attention, compact storage |
| `runtime.py` | Reversible explicit Qwen3 traversal and centered head gates |
| `provenance.py` | Checkpoint/code/environment hashes, immutable output and receipt guards |
| `diagnostics.py` | New open boundary probes and separate readout/abstention metrics |
| `run.py` | Audit, native parity qualification and receipt-gated diagnostics |
| `verify.py`, `tests/` | Executed CPU qualification; optional native HF random-model test |
| `protocol.json` | Hypotheses, arms, gates, closed stages and deferred work |

## External references and attribution

Architecture inspiration: [Spark-X2.5-4B model card](https://huggingface.co/XHToken/Spark-X2.5-4B),
[configuration](https://huggingface.co/XHToken/Spark-X2.5-4B/blob/ea14618d20e76b5b093d3ee20a5b9d733bb12410/config.json),
and [released implementation](https://huggingface.co/XHToken/Spark-X2.5-4B/blob/ea14618d20e76b5b093d3ee20a5b9d733bb12410/modeling_spark.py).
The source model repository declares Apache-2.0. No Spark weights or source files
are vendored, and no blanket dataset-license inference is made from that license.
This branch is an independently written Rineya-specific experiment, not Spark
training reproduction. The source benchmark table does not isolate architecture
from pretraining, data or post-training.
