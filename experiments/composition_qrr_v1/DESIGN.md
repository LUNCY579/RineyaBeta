# QRR-1 design and implementation review

## Computation

Given a causally computed query hidden vector h and accessible evidence bank H,
initialize z0 = Win RMSNorm(h), K = Wk RMSNorm(H), V = Wv RMSNorm(H). At each read:

```text
q_t     = Wq RMSNorm(z_t)
a_t     = softmax(q_t K^T / sqrt(rank) + causal_visibility_mask)
r_t     = a_t V
c_t     = concat(RMSNorm(z_t), r_t)
z_{t+1} = z_t + 0.25 sigmoid(Wgate c_t) tanh(Wupdate c_t)
delta   = Wout RMSNorm(z_T)
```

The query changes after the preceding read. This is not a repeated identical
forward pass. The evidence keys/values are reused within one invocation; no hidden
state is retained across examples. Read count is 1, 2 or 4, never an unbounded
convergence loop. Stabilizing the workspace is not evidence of a correct answer.
The RMS normalization has no learned parameters. Projections are bias-free.

For tied reads the stored parameter count is 4*H*R + 5*R^2. H=1024 and R=64 yield
282,624 parameters. The fully active two-matrix MLP control has width 138 and
2*1024*138 = 282,624 parameters, without unused padding scalars. Its evidence access
is different; a future trained study must also compare a one-read evidence-aware
control and match actual inference/training work. The untied-two arm deliberately
stores more parameters and cannot be called parameter-matched to the tied arm.

`initialized_controls` copies common tensors exactly between 1/2/4 reads and gives
untied cells initially equal values but separate storage. It preserves the caller's
CPU RNG state. The fixed 0.25 update multiplier does not change with read count.
`arm_table` is an exact parameter audit, not a FLOP, memory or latency benchmark.

## Identity initialization and gradients

Wout starts at zero. The residual and bias-free head projection are exactly zero
on finite tensor fixtures, preserving native logits in the explicit integration
seam. There is no additional closed scalar gate. At the first backward pass only
the output projection has task gradient; controller gradients are expected to be
zero. Tests then perturb Wout explicitly and verify finite nonzero gradients in
every controller matrix. This is a manipulation check, not model training.

Do not install a zero scalar gate on top of a zero output projection: that would
kill both first-step paths. Do not infer effective training participation from
nominal parameter count; measure gradients during any future approved microfit.

## Causality and integration

Visibility is the intersection of an explicit boolean mask, nonnegative key
positions and key_position <= query_position. Empty visibility is rejected.
Masked future/padding tensors are zeroed before projection; even masked NaNs do
not affect the output. Visible NaNs fail. Tests change future evidence by large
values and NaNs and verify unchanged outputs and exactly zero future attention.

This mask cannot remove future information already embedded by a bidirectional
encoder or teacher-forced noncausal hidden-state construction. The caller must
supply prefix-causal states and a query derived without gold. No gold token or
operator label is an input to the module. A future native bridge must prove
prefill/full/cached equivalence, batch isolation, padding behavior, actual retained
bank accounting, zero-initialization identity, save/load and uninstall rollback.

`add_logit_residual` is an explicit, bias-free linear-head integration seam. It is
not a Transformers hook or a qualified replacement for model.generate(). This
revision does not retain a pretrained hidden-state bank or modify attention/KV
storage. Tensor checks cannot authorize any of those runtime paths.

## Instrument and bias safeguards

The instrument generates worlds first, then prompts. Its list-index solver is
cross-checked with a separately implemented map/successor traversal. Both still
share the declared task semantics, so this does not substitute for human semantic
review or a truly independent benchmark. Lookup/next/next2 and red/blue count pairs
have different answers by construction. Absent code IDs come from the same random
identifier process rather than a fixed visible sentinel.

The current two renderers are both open DEV. There is no train/test split and no
claim of linguistic or structural generalization. Counting rows is not the same
as counting distinct vaults with duplicates; that task must be added explicitly,
not relabeled after seeing outcomes. A registry problem does not establish broad
long-context reasoning. Padding notes never count as registry rows.

The parser accepts a complete vault ID, decimal integer or UNKNOWN, with only the
declared optional Answer:/vault prefix. It never searches for a gold substring.
Oracle evidence preserves original row indices. Oracle binding supplies privileged
target-location information and is labeled diagnostic in every artifact.

The bias regression demonstrates a second-token change despite unchanged first
choice. `bias.inspect_trace` stops immediately at any counterfactual divergence;
later saved logits belong to the old prefix and cannot be reused. The helper uses
explicit float64 reference arithmetic, not native FP16/BF16 and processor
qualification. The native R2 runner performs actual zero-bias decoding and does
not use this helper to manufacture a bias curve.

## Review status

Focused CPU tests cover code identity, incomplete checkpoints, LFS pointers,
symlinks, changing snapshots, duplicate JSON keys, output overwrite refusal,
scoring artifacts, cluster statistics, causality, cache fixtures and gradient
liveness. The native CUDA loader is implemented but not exercised in this build
environment. Full pretrained QRR integration, training objectives, compute-matched
arms, transfer banks and production runtimes are intentionally unfinished stages.
These are explicit remaining work items, not inferred negative results.
