# HLGA-1: bounded Rineya local/global attention compatibility

**Branch:** `research/hybrid-local-global-attention-v1`  
**Canonical audit:** `3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Revision 2:** extends the existing implementation at `6770e67b471f4821004175d5d39a228f079b28f1`.

**Infrastructure is not a model result.** Real R2 compatibility, quality, training
benefit, long-context reach and speed are unmeasured in this increment. Historical
[CPU evidence](reports/IMPLEMENTATION_REVIEW.md) and [native CI evidence](reports/CI_REVIEW.md)
remain valid for their recorded code, not authorization for this revision.

Read the [current bounded compatibility contract](COMPATIBILITY_CONTRACT.md) and
[machine-readable protocol](protocol.json). The new `study.py` planner reports
analytical work and operational bounds; it does not train or benchmark a model.

## Question and target

Can a frozen Qwen3-derived R2 checkpoint tolerate periodic local attention with
compact resident KV storage? **A3 versus A0 is the primary frozen comparison.**
Spark's 3-local/1-global design motivates the topology; its benchmark table does
not isolate architecture from pretraining or post-training. This is not a Spark
training reproduction or a claim that its performance transfers to Rineya.

The primary location is `artifacts/rineya-core-en32k-28L`: 28 layers, width 1024,
16 Q / 8 KV heads, head dimension 128, intermediate width 3072, vocabulary 32768.
The committed config has maximum positions 40960 and RoPE theta 1e6. Actual local
weight/tokenizer hashes are required. Shape compatibility does not establish
weight lineage or family-wide empirical support; the remote config alone is not
a real checkpoint qualification.

Keep QKV/O, QK-Norm, RMSNorm, SwiGLU, RoPE, embeddings, output head and tokenizer.
Batch-one unpadded native Qwen3, FP32 and local safetensors are the initial scope.
No quantization, packing, hooks, recurrence, KV-head surgery, context extension,
new kernel, remote-code loading or model export is introduced.

## Registered arms

| Arm | Attention | Total local budget | Prefix sinks | Identity gate |
|---|---|---:|---:|---|
| A0_global | All-global, matched research backend | — | 0 | No |
| A1_local1024 | Periodic 3-local/1-global | 1024 | 0 | No |
| A2_local512 | Periodic | 512 | 0 | No |
| A3_sink512 | Periodic; primary treatment | 512 | 4 | No |
| A4_global_gate | All-global training control, untrained now | — | 0 | Yes |
| A5_local512_gate | Periodic | 512 | 0 | Yes |
| A6_sink512_gate | Periodic | 512 | 4 | Yes |
| A7_sink1024 | Periodic | 1024 | 4 | No |
| A8_all_local512 | All layers local; resource-unmatched mechanism control | 512 | 0 | No |
| A9_frontloaded512 | Same global count, changed placement | 512 | 0 | No |

R2 periodic globals: `3,7,11,15,19,23,27`. A9: `0,1,2,3,4,5,27`.
Non-multiple-of-four fixtures preserve final-global placement except A8, which
has no globals. W512 includes the current token and sinks: four initial plus
508 recent positions, not 512 plus four. All positions remain absolute.

A gate is `2*sigmoid(W*x)` after attention and before the output projection,
with W=0. It is exactly identity with a live gradient and adds 458,752 parameters
on R2. A0=A4, A2=A5 and A3=A6 token-for-token before training or the diagnostic
fails. Gate training remains unavailable.

## Implemented protections

`CompactKV` retains independent compact storage, not a slice backed by the whole
prefill. `blocked_attention` gathers each query block's relevant key union and
uses one joint softmax for sink/recent keys. Global attention remains quadratic;
this is a correctness-oriented PyTorch path, not an optimized sparse kernel.
Read the [historical attention-shape negatives](../attention_shape/README.md):
logical cache savings, resident memory, peak memory and prefill speed are not
interchangeable. Those results are not retracted by this branch.

Revision 2 adds all-local and equal-count placement controls, direct strict/trimmed
contrasts, and accounting for actual query-head dimensions, masked score
rectangles, QKV/O/MLP/LM-head/gate matrix work and resident KV. Every actual DEV
case is checked against its generated-length accounting. Shape counters are not
a device profiler. Same output caps do not make EOS-stopped runs compute-matched.

The diagnostic is bounded to 10 arms, 72 open synthetic prompts, 24 output tokens,
block size at most 128 and a CUDA allocator cap at most 6144 MiB. The six prompt
lengths are 511/512/513 and 1023/1024/1025, across three depths and four task
families. It is a readout/window-boundary diagnostic, not broad reasoning or an
independent preservation benchmark. Low all-global accuracy or UNKNOWN loops
make the endpoint inconclusive, not a hybrid success. All DEV timing is non-causal.

## Commands

```bash
python -m experiments.hybrid_attention_v1.run plan
python -m experiments.hybrid_attention_v1.study
python -m unittest experiments.hybrid_attention_v1.tests.test_study -v
python -m experiments.hybrid_attention_v1.verify --run-id hlga-v2-native-001 --require-hf
python -m experiments.hybrid_attention_v1.run audit --run-id hlga-v2-r2-audit-001
python -m experiments.hybrid_attention_v1.run qualify --run-id hlga-v2-r2-qualify-001
python -m experiments.hybrid_attention_v1.run diagnose --run-id hlga-v2-r2-dev-001 \
  --receipt experiments/hybrid_attention_v1/runs/hlga-v2-r2-qualify-001/receipt.json
```

Use a new run ID. `--model PATH` must name actual local primary-shape weights;
missing weights or unresolved LFS pointers fail. P2 requires seven checks on the
current code/assets/environment/backend/block size, including both new topology
controls and an unchanged-checkpoint receipt. Neither old receipts nor random
fixtures authorize real-model diagnosis. No remote model download is initiated.

For CUDA use consistent `--device cuda --backend sdpa --block-size 128 --cap-mib
6144` options in qualification and diagnosis, with an appropriate lower cap when
necessary and deterministic CUDA prerequisites configured before launch. CPU
fallback is forbidden. FP32 qualification does not qualify BF16 automatically.
The default CPU path may be expensive for real R2; it is not a resource guarantee.

`train` and `sealed` explicitly refuse. Future gate training requires reviewed
data source/license/derivation and hashes, duplicate/split audits, an optimizer
registration, identical gate allowlists, three seeds and separate token-matched
and compute-matched budgets. The contract defines those planning bounds without
opening training. Historical and external sealed suites remain untouched.

## Source and evidence map

| File | Responsibility |
|---|---|
| `config.py` | Ten stable arms, topology and exact cache budget |
| `attention.py`, `runtime.py` | Explicit attention, compact cache, traversal and gates |
| `study.py` | Target lock, work planner, direct contrasts, receipt and execution guards, future budget arithmetic |
| `provenance.py`, `run.py` | Local hashes, immutable receipts and gated CLI |
| `diagnostics.py` | Existing deterministic open DEV and separate readout metrics |
| `tests/`, `verify.py` | Infrastructure tests; full native-HF integration in CI |
| `protocol.json`, `COMPATIBILITY_CONTRACT.md` | Current registration, closed gates and interpretation boundaries |

Preserve reviewed receipts in separate evidence-only commits; do not turn passing
CI into a model-performance headline. No shared trainer, model asset, corpus,
canonical audit, main branch or other research branch is modified by this study.

External architecture source: https://huggingface.co/XHToken/Spark-X2.5-4B .
The inherited source pointers for configuration and implementation remain pinned
to `ea14618d20e76b5b093d3ee20a5b9d733bb12410` in that model repository. No source
files or weights are vendored and no dataset-license claim follows from the
model repository's Apache-2.0 declaration.
