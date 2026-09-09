# Query Refinement v1: read, revise the query, read again

**Status: implementation and mechanical validation only. No Rineya efficacy,
long-context improvement, speedup or production-readiness claim.**

Branch: `research/query-refinement-attention-v1`.
Base: `main@3532c7bc8dbdd48ef8801ba6080cb80b0e90379a` (canonical audit).
The existing composition, cornerstone2 and hybrid-attention branches are not merged
or modified. The main-branch `configs/heal_b0.yaml` identifies R2 as
`artifacts/rineya-core-en32k-28L`. Its checked config has 28 layers, width 1024,
16 Q heads, 8 KV heads, head dimension 128, and vocabulary size 32768.
Thus Q's total width is 2048, NOT the residual width 1024.

## Hypothesis and exact intervention

At zero-indexed layer 14 only, let q0 be Qwen3's **Q-normalized, unrotated**
query and let K be its already rotated keys. Preserve native QK-Norm, RoPE,
GQA grouping, scaling, residual paths, MLPs and output head.

```
o0 = Attention(RoPE(q0), K, V)
U(s) = tanh(gain_per_head) * tanh(up(SiLU(down(RMSNorm_without_weights(s)))))
q1 = q0 + U(o0)
o1 = Attention(RoPE(q1), K, V)
out = (o0 + o1) / 2
```

K and V are the SAME tensors for both reads at this layer. RoPE is applied once
to each query, at its original absolute position. Updating earlier-layer outputs
would still change subsequently computed upper-layer representations; we do not
claim all K/V in the entire network remain invariant between architectures.

`down` and `up` share weights across heads; each head has its own bounded gain.
Rank 8 adds exactly `2*128*8 + 16 = 2064` trainable parameters. Only `up` starts
zero. The nonzero down/gain initialization avoids a double-zero gradient trap.
At initialization all arms recover the original attention result. The update is
bounded coordinate-wise; this is NOT a convergence theorem.

## Registered arms and controls

| Arm | Selected-layer output | Reads | Added trainable parameters |
|---|---|---:|---:|
| baseline | o0 | 1 | 0 |
| adapter | o0 + U(o0) | 1 | 2064 |
| independent | (o0 + Attn(RoPE(q0 + U(q0)), K,V))/2 | 2 | 2064 |
| feedback | (o0 + Attn(RoPE(q0 + U(o0)), K,V))/2 | 2 | 2064 |

Primary: feedback minus independent. Capacity check: feedback minus adapter.
Both reads affect the two-read control's output: no discarded-computation control.
The two-read arms have the same projection dimensions, attention MAC counts,
optimizer, examples, answer tokens, steps and seeds. Baseline/adapter are NOT
compute-matched to two-read arms. Report measured latency and CUDA memory, not
just nominal attention arithmetic. MAC counts omit non-attention operations and
are not total FLOPs. Eager reference attention and explicit GQA expansion are
correctness implementations, not optimized production kernels.

A stop-gradient-first-read arm was removed after review: with one intervention
and a frozen backbone, the first read has no trainable ancestors, making that
control redundant. It becomes meaningful only with different trainable pathways.

## What is intentionally not implemented

This is the bounded current-token, within-layer variant, not the user's entire
historical-query workspace. Updating old queries alone has no path into the next
prediction. A future all-history design needs a defined readout into the current
state, explicit workspace-time causality and an O(n^2) per-step attention budget.
Previous-token latent feedback also requires training that reproduces its recurrent
schedule. These remain separate future arms, not hidden changes to this experiment.

K/V updates, mixtures with local/global attention, deeper iterative reads, learned
halting, precision/quantization changes, padding/packing and production integration
are deferred. Broadcast key additions can be a softmax no-op; broadcast value
additions are direct output shifts. Unit tests preserve these mathematical controls.

## Executable pipeline

From the repository root, install the pinned requirements in an isolated environment:

```bash
python -m pip install -r experiments/query_refinement_v1/requirements.txt
python -m pytest experiments/query_refinement_v1/tests -ra
```

The native HF test uses a newly initialized tiny Qwen3 with no downloaded weights.
Local tests can skip that one test when Transformers is absent; CI explicitly
requires it. Neither a synthetic fixture nor a random native model qualifies R2.

Provide the actual local R2 weights and tokenizer; network downloads, remote code,
LFS pointer substitution, quantized models and mismatched target shapes are refused.
The source directory is read-only to the pipeline. Use NEW external run directories:

```bash
python -m experiments.query_refinement_v1.pipeline qualify \
  --model artifacts/rineya-core-en32k-28L --out artifacts/qref_runs/p2
python -m experiments.query_refinement_v1.pipeline smoke \
  --model artifacts/rineya-core-en32k-28L --receipt artifacts/qref_runs/p2/receipt.json \
  --out artifacts/qref_runs/p3
python -m experiments.query_refinement_v1.pipeline dev \
  --model artifacts/rineya-core-en32k-28L --receipt artifacts/qref_runs/p3/receipt.json \
  --out artifacts/qref_runs/p4
```

`--device cuda` is optional; default is CPU. Float32 only, original context limit,
batch size one without padding, and eager training. CUDA requires 4096 MiB free
before model load; this is a conservative gate, NOT an OOM guarantee. Each run
uses deterministic algorithms, disabled TF32 and one CPU thread. No resumable
trainer is claimed: interrupted runs are failed/incomplete, never silently resumed.

P2 compares all zero-update arms with the untouched native HF forward, then tests
nonzero logit reachability, token/chunk cache equivalence, future-token causality,
eager/SDPA agreement, finite active adapter gradients and backbone freezing.
P3 repeats qualification and executes 25 optimizer steps per trained arm on one
seed. This is only an optimization smoke, not an efficacy measurement.
P4 requires a successful, matching P3 receipt, repeats qualification, and executes
100 steps per trained arm for seeds 17/29/43, then open DEV scoring. No implicit
promotion from an infrastructure pass to a quality claim exists.

Receipts bind every source model/tokenizer file, code/tests/requirements, protocol,
environment and execution device. Hashes detect accidental drift, not malicious
forgery or independent attestation. Mechanics are rerun for each invocation. Output
directories are exclusive; failures receive failure receipts where initialization
has progressed far enough. Adapter checkpoints are separate from source weights.
No existing sealed suite is read; no stage grants sealed evaluation or deployment.

## Data and analysis

`data.py` creates fresh, public-to-this-experiment graph-following tasks, never
imports the repository's historical evaluation datasets. Train/dev entities have
disjoint namespaces, edges have unique source keys, facts are shuffled, unrelated
distractors are included and an exact graph oracle validates answers in tests.
The registered small budget is 64 train and 48 DEV rows. Train has 1/2-hop cases;
DEV has balanced 1/2/3/4-hop cases, including held-out chain lengths. This is a
small mechanism diagnostic, NOT representative long-context or language evaluation.
Split-specific entity prefixes introduce a deliberate distribution shift; report
that limitation and do not generalize a negative to all query feedback mechanisms.

Plain-text prompt tokenization and separately encoded answer continuations are
identical across arms. Loss is answer/EOS-only, with the first answer predicted
from the final prompt position. Oversized rows fail; no truncation, dropping,
retokenization ambiguity or packed cross-example attention is permitted. Exact
match is strict after decoding and trimming outer whitespace; additional prose
is wrong, and substring matching is never used. Raw generated IDs are retained.

Reports preserve per-example predictions and per-seed/per-hop accuracy. The
replication unit is the training seed, not thousands of correlated tokens.
Three-seed bootstrap intervals are descriptive and cannot support a robust
confirmatory significance claim. The present test bank has no power justification.
Positive DEV results require fresh preregistration with sufficient sample sizes,
independent confirmations, longer contexts/distractors, general-language regression
checks and measured hardware costs before any architecture-performance claim.

## Implementation and test coverage

`core.py`: Q-only mechanism, explicit absolute-position masks and parameter controls.
`runtime.py`: isolated Qwen3 traversal; no global patches; transactional cache/reset.
`pipeline.py`: native qualification, frozen-backbone training, stage receipts.
`data.py`: open graph tasks and paired seed-level summaries.
`tests/`: mathematical identities, backend/gradient agreement, zero initialization,
nonzero cache/chunk equivalence, no-future Jacobians, active parameter checks,
freeze/checkpoint roundtrip, invalid-input resets, split oracles and stale receipts.

The structural fixture is independently implemented but still synthetic. The native
HF comparison is the stronger implementation oracle; actual R2 qualification is
stronger again. Test results at one level do not transfer automatically to another.

## Provenance and related work

- Repository integration contract inspected read-only at
  `research/hybrid-local-global-attention-v1@a196318ec75cacccc8ac9ec446e4c4b6a29c2802`,
  `experiments/hybrid_attention_v1/runtime.py`. No experiment results inherited.
- Native Qwen3 source inspected at Transformers `v4.57.1`:
  https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py
- PyTorch SDPA mask semantics (True means allowed; non-square causal alignment must
  not be guessed): https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Sukhbaatar et al., *End-To-End Memory Networks* (2015), multi-hop query updates:
  https://papers.nips.cc/paper/5846-end-to-end-memory-networks

These motivate a hypothesis and integration choices, not a novelty or efficacy claim.
