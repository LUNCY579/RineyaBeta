# HLGA-1 bounded compatibility contract, revision 2

**Branch:** `research/hybrid-local-global-attention-v1`  
**Canonical audit:** `3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Implementation inherited, not authored by this increment:** `6770e67b471f4821004175d5d39a228f079b28f1`  
**Evidence boundary:** infrastructure and protocol changes; no Rineya quality,
training, context-extension, or speed result is established here.

The branch had already advanced three commits beyond the audit when this work
started. Preserve those commits and their receipts. Their code-specific results
are historical evidence, not authorization for this changed revision. Do not
reset the branch, change main, or alter other branches/tags to reconcile an older
snapshot. `protocol.json` and this contract supersede the eight-arm registration
for future runs, without relabeling old results as ten-arm experiments.

## 1. Question, target and exclusions

The first question is **frozen compatibility**, not whether Spark's benchmark
scores transfer: can one particular R2 checkpoint tolerate periodic local
attention with actual compact KV storage? The primary comparison is A3 versus A0.
Spark's model card describes three sliding-window layers per full-attention layer
and substantial pretraining/post-training. That is motivation for a topology,
not an architecture-only causal result or a training recipe reproduced here.
Source: https://huggingface.co/XHToken/Spark-X2.5-4B (reviewed 2026-09-08).
The source card's Apache-2.0 declaration does not license unrelated training data.
No Spark source files or weights are copied into this experiment.

Primary asset location: `artifacts/rineya-core-en32k-28L`. The committed config
specifies 28 layers, width 1024, 16 Q heads, 8 KV heads, head dimension 128,
MLP width 3072, vocabulary 32768, maximum positions 40960, and RoPE theta 1e6.
The Q projection width is **2048, not 1024**. The CLI now rejects a different
primary shape before loading a model for qualification or diagnosis. An arbitrary
same-shaped checkpoint is not thereby proven to be R2: local asset hashes bind
runs, and source-weight lineage must be reviewed before any model identity claim.
No real pretrained weights were available to the local implementation tests.

Keep the original QKV/O projections, QK-Norm, RMSNorm, SwiGLU, RoPE, embeddings,
output head and tokenizer. The current scope is native, unquantized Qwen3,
FP32, batch one, unpadded sequences, no packing/beams/hooks, and local-only
safetensors loading with remote code disabled. A different family checkpoint,
BF16, quantization, another backend or block size requires its own qualification;
shape-generic Python support is not family-wide empirical validation.

Exclude KV-head surgery, dual/partial RoPE, head-dimension changes, activation
replacement, recurrence, importance-selected placements, learned routing,
pre/post-processing changes, fused sparse kernels, export and 1M context.
Sink retention and identity gates are separate Rineya controls, not assumptions
that Spark's entire architecture has been replicated.

## 2. Baselines and controls

| Arm | Purpose | Local policy / global placement |
|---|---|---|
| Native oracle | Independent implementation-parity baseline | Original native eager full attention; not replaced by the research wrapper |
| A0_global | Frozen causal research baseline | All-global, same explicit backend/blocking as research arms |
| A1_local1024 | Secondary window-budget control | W1024, periodic globals |
| A2_local512 | Pure-window damage control | W512, periodic globals |
| A3_sink512 | **Prespecified primary treatment** | W512 including four persistent prefix slots, periodic globals |
| A4_global_gate | Future trained-capacity control; identity now | A0 with the same head-gate architecture used by A5/A6 |
| A5_local512_gate | Future local repair arm; identity now | A2 plus gate |
| A6_sink512_gate | Future sink/local repair arm; identity now | A3 plus gate |
| A7_sink1024 | Secondary sink control | A1 with four prefix slots inside W1024 |
| A8_all_local512 | Isolate contribution of global access | W512 everywhere; **not compute-matched** to A2 |
| A9_frontloaded512 | Placement control at equal modeled work | W512; same seven global layers and final-global constraint, different locations |

For R2, periodic global indices are `3,7,11,15,19,23,27`; A9 uses
`0,1,2,3,4,5,27`. A9 holds global count, total parameters, resident KV budget,
visible-pair count and modeled matrix work fixed. It is not importance-guided.
On very shallow fixtures the two schedules can coincide; those fixtures do not
establish a placement effect. The new eight-layer tests exercise distinct layouts.
A8 deliberately removes all global layers, including the last one.

A local budget includes the current token. W512 with four sinks is four initial
positions plus 508 recent positions, with duplicates removed. Sinks cannot attend
to the future. Absolute positions must survive eviction. Multiple stacked local
layers can convey information farther than a single window, so an A8 success at
1025 tokens does not establish unrestricted long-range access or a broad context
limit. The co-located two-hop fixture is not a distributed multi-hop benchmark.

The gate is `2*sigmoid(W*x)` after attention and before the output projection,
with W initialized to zero. It is exactly identity with a live gradient. R2 has
458,752 gate parameters. A0/A4, A2/A5 and A3/A6 must generate identical token IDs
before training. Gate parameters and projection/multiply overhead are reported;
identity behavior is not zero compute. No gate has been trained in this increment.

Named contrasts are recorded in both directions explicitly as reference/treatment:
A3-A0 (primary), A2-A0 (damage), A3-A2 and A7-A1 (sinks at equal budgets),
A2-A1 (window budget), A2-A8 (global access), A2-A9 (placement), plus the three
identity controls. Positive deltas favor the treatment. Strict and trimmed exact
match and answerability subsets remain separate; empty subsets are unmeasured,
not zero. No post-hoc `code`/`vault` prefix normalization is allowed.

## 3. What compute matching means here

**Frozen inference:** hold weights, prompt IDs/order, tokenization, decoding,
precision, backend, chunk size, context and output cap fixed. Every complete DEV
run contains at most 10 x 72 = 720 generations, with at most 24 output tokens each
(17,280 emitted-token ceiling). Block size is 1-128 and the CUDA allocator cap
cannot exceed 6144 MiB or current free device memory. This is an operational
bound, not a promise that FP32 R2 will fit a particular host or finish quickly.

EOS termination produces different actual decode lengths. Consequently the
quality run is **not an iso-FLOP or causal latency experiment**, even with a common
24-token cap. `workload_envelope.json` records the fixed-24-token analytical
comparison. Each actual generation separately records its real generated length,
attention-call shapes and resident cache measurements. The last predicted token
is not forwarded; decoder/cache occupancy is prompt length + generated length - 1.

For a contiguous causal sequence of n tokens, a local budget w has
`C(n,w) = m(m+1)/2 + (n-m)w`, where `m=min(n,w)`. Global attention has
`C(n)=n(n+1)/2`. Sink replacement preserves row cardinality, not the identities of
visible keys. **These visible pairs are not the work issued by a block-dense
implementation.** The planner also counts the query-block x gathered-key
rectangles, including masked positions. Runtime counters check submitted shapes
against that plan, without adding a tensor reduction/device synchronization.
They do not measure physical GPU instructions or prove which SDPA kernel ran.

Report QKV/O and all three MLP matrix products, LM-head calls and gate projections,
not only attention. Matrix FLOPs use multiply-add = two operations. Norms, RoPE,
softmax/sigmoid, gathers, GQA expansion, allocation/cache copies, host overhead,
backward and optimizer work remain explicitly excluded. Model-matmul estimates
are not whole-program timings. Global layers remain quadratic. Persistent KV,
transient cache allocations, logical bytes, resident backing storage and peak
allocated/reserved memory are different quantities.

**Future token-matched training, CLOSED:** A4/A5/A6, seeds 17/29/43, 64 optimizer
updates per seed, microbatch one, accumulation eight, sequence length 1024
(524,288 input tokens per seed). Freeze the backbone and use exactly the same
gate parameter allowlist, optimizer, LR, examples/order, loss masks, supervised
tokens, checkpoint rule and decoding. Padding or ignored labels do not count as
supervised tokens. The data review and optimizer/LR registration are still missing;
these numbers do not authorize a trainer. No production training CLI is added.

**Future compute-matched training, separately CLOSED:** use A4's 64-update budget
and whole-update profiled FLOPs; alternatively preregister a separate isolated
wall-time experiment. Do not substitute an attention-only ratio or a generic
three-times-forward estimate, especially for a frozen-backbone gate experiment.
`matched_update_budget` floors to whole updates, caps candidates at 128 updates,
reports unused budget and refuses a matched label above 2% unused budget. It never
authorizes training. Fixed-token and fixed-compute results answer different
questions: extra examples under equal compute are not a same-data causal effect.
Do not average those ledgers or pad runs with fake work to claim equality.

A future trained-local recovery effect must be contrasted with trained-global
recovery, e.g. `(A6_trained-A3_frozen) - (A4_trained-A0_frozen)`, in a common
endpoint and budget regime. A6 beating its damaged frozen counterpart alone is
not evidence that local attention beats global attention.

## 4. Evidence gates and stopping rules

| Gate | Required evidence | What it licenses |
|---|---|---|
| P0 infrastructure | All applicable fixtures; native HF random-model integration separately required; negative-path tests | Implementation qualification only, never R2 performance |
| P1 identity | Real local weights and tokenizer, hashes, native config and primary shape; no LFS pointer substitutions | Inventory only; config assets alone do not qualify a model |
| P2 real checkpoint | Current code/environment/backend/block binding; native/global logits and argmax parity; full/stream equivalence; compact storage; identity gate; both new topology controls; unchanged assets | New open DEV diagnostics on that exact checkpoint/runtime only |
| P3 open DEV | Matched case IDs/hashes/answers; all planned arms; actual per-case work/storage checks; all gate identities | Descriptive boundary/readout evidence; **not preservation, intelligence, speed, or deployment approval** |
| P4 training | Reviewed provenance/license/derivation, dedup/split audit, frozen optimizer and loss masks, seeds/budgets, gate allowlist and selection rule | CLOSED; needs a separately reviewed implementation/registration |
| P5 confirmation | Independent approved data, adequate power, locked candidate/endpoints/margins, multiplicity and isolation plan | CLOSED; no historical or external sealed suite may be opened by this code |

P2 uses 17 synthetic token IDs with tiny windows and unequal chunks (0/7/8/17),
FP32 `atol=rtol=2e-5`, finite logits and identical argmax. It is a mechanics test.
The two new topology checks and study-version/unchanged-checkpoint guards make
old receipts insufficient. Numerical failures remain serializable failed receipts;
NaN/Inf values must not destroy the evidence file. Any mismatch stops the run.

P3 retains exactly the existing 72 deterministic prompts: six boundary lengths
(511/512/513, 1023/1024/1025), three depths, and retrieval/two-hop/absent/conflict.
A `--max-cases` prefix is an incomplete diagnostic, not a representative sample.
Use one complete run per bound configuration; do not tune windows or prompts until
it looks good. Primary A3 remains primary; W1024 and placement results are secondary,
not an undisclosed search for whichever arm wins. Any revised recipe is a new
registration and cannot reuse a held-out result as validation.

If all-global itself fails to produce useful exact answers, or every arm repeats
UNKNOWN to the token cap, classify the endpoint as **readout-limited/inconclusive**.
Do not call equality at zero a compatibility success or a negative architecture
result. Report strict/trimmed accuracy, false/proper abstention, emptiness,
repetition and budget exhaustion separately. A valid whole UNKNOWN abstention is
not a loop of UNKNOWN. These correlated synthetic cases and unadjusted McNemar
numbers cannot establish noninferiority or a broad reasoning gain.

For a later independent confirmation registration, proposed margins are a 2
percentage-point noninferiority bound on strict answerable accuracy and proper
abstention, a 2-point ceiling on excess false abstention, and a 5-point worst-slice
regression bound. These are **planning defaults, not executed or automatically
approved tests**. Freeze sample size/power, source identities, evaluation units,
paired confidence-interval method and multiplicity handling before data access.
Do not treat a nonsignificant p-value as equivalence. A quality claim requires an
appropriate confidence bound; an efficiency claim additionally requires a
measured quality/resource Pareto improvement against a qualified native baseline.

An isolated speed study must fix prompt/decode lengths independently of EOS,
use warm-ups and synchronized repeated measurements, record actual kernel,
dtype/device/software/driver and power/clock/thermal state, and establish lack of
competing GPU work. Randomize or counterbalance paired arm order. Separate TTFT,
prefill, decode token latency, peak allocation and resident KV. Shared-GPU or fixed-
order DEV timing remains non-causal. The current reference implementation has no
such benchmark, no trained model, and no performance release gate.

## 5. Run and record without promoting

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

Use a new run ID every time. For CUDA, supply the same `--device cuda --backend
sdpa --block-size 128 --cap-mib 6144` in qualification and diagnosis, lowering the
cap if needed; set deterministic CUDA prerequisites before launch. No silent CPU
fallback is allowed. Do not substitute random weights for a missing R2 checkpoint.

Commit infrastructure as infrastructure. Put actual reviewed receipts in a
**separate evidence-only commit**, identifying tested source hashes, environment,
commands, tests/skips/failures and limitations. A local subset receipt is not the
full-suite CI receipt, and neither is a real-checkpoint result. Preserve historical
reports and negative results rather than rewriting their evidence level. Changes
here stay under this experiment; do not modify shared trainers, legacy audits,
model assets, existing datasets, sealed suites, main, or other branches/tags.
