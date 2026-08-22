# The Stabilizer Matrix — pre-registration

**Status: designed, not started. Branch `exp/stabilizer-matrix`.**
Written before any measurement, deliberately including the ways it can fail.

---

## 1. The idea

A **stabilizer matrix**: a single persistent matrix, global to the computation
rather than per-layer, which projects an incoming hidden state into a similarity
space and from that decides *which neurons to fire* in the feed-forward blocks.
Neurons outside the selected set are never computed — their rows and columns are
not read, not multiplied, not written. The claim is that this reduces memory
traffic and wall-clock, while preserving most of the output quality.

This is the same intellectual move as the significance matrix, applied to a
different axis. There, a persistent accumulator decides *which cached positions
survive*; here, a persistent projector decides *which neurons participate*. Both
put a small piece of persistent state in charge of a large, mostly-redundant
computation.

## 2. Why this checkpoint is a reasonable place to try it

| | value |
|---|---|
| MLP parameters | 9.4M per layer, **264.2M total** |
| Model total | ~600M |
| MLP share | **~44% of all parameters** |
| Intermediate width | 3,072 per layer, 28 layers = 86,016 neurons |

Skipping FFN neurons skips the single largest parameter block in the model. If
half of them could be skipped per token, roughly a fifth of the model's weights
go unread on that token.

## 3. Prior art, stated honestly

This idea is **not new in kind.** It is a form of *contextual sparsity*, and the
literature is substantial. The design below is worth running anyway — but as a
replication-plus-variation on a known line, not as an invention. Getting this
wrong in the write-up would be the most damaging thing we could do.

- **Deja Vu** [1] is the closest prior work: small per-layer predictors choose
  which MLP neurons and attention heads to compute per token, reported at ~2×
  lower latency with preserved quality. Our proposal differs in that the selector
  is **one persistent global matrix** rather than a learned predictor per layer.
  Whether that difference buys anything is an open question and should be framed
  as the contribution, if any survives.
- **ReLU Strikes Back** [2] is the most important paper for us, because it says
  the substrate may not be there: models with SiLU/GELU activations have **much
  lower activation sparsity** than ReLU models, and recovering it needs
  "relufication" plus finetuning. **This checkpoint uses SiLU** (`hidden_act:
  silu`, SwiGLU MLP), so we should expect weak natural sparsity.
- **PowerInfer** [3] and **LLM in a flash** [4] exploit hot/cold neuron
  distributions to decide what to keep resident versus stream. Their gains come
  largely from avoiding *transfers*, which matters when weights do not fit —
  a condition that does not hold here.
- **MoEfication** [5] converts a dense FFN into experts post hoc, which is the
  structured version of the same observation.

*References are recalled, not resolved against a database; verify before use.*

## 4. The honest problem: our own measurements argue against the speed claim

Three results from the KV program bear directly on this, and two of them are
discouraging.

**Decode on this host is overhead-bound, not bandwidth-bound.**
`probe_decode_floor.py` measured a floor of **34.70 ms/token** that is flat across
a 512× range of cache size, implying **27 GB/s** for a 904 MiB weight read on a
card rated near 256 GB/s — about 11% of peak. If the machine is not waiting on
bytes, *reading fewer bytes cannot make it faster*. Sparse gathers typically make
this worse, not better: irregular access and extra kernels are precisely what an
overhead-bound regime punishes.

**The weights were never the memory problem here.** The model is 904 MiB resident
on an 8,188 MiB card. The pressure came from the KV cache — 14,336 MiB at 128K —
which the significance matrix already addresses. Skipping FFN neurons does not
reduce resident weights unless they are being streamed, and they are not.

**A prior compression result did not convert into speed for exactly this reason.**
A 3.8× smaller KV cache changed decode by 0.29 ms — nothing — because it was
already below the latency knee. The same trap is available here.

So the speed claim should be treated as **the most likely thing to fail**, and the
experiment must be able to kill it in minutes rather than weeks.

Where the idea could still matter, and what would have to be true:

| regime | why sparsity could pay | do we have it? |
|---|---|---|
| larger model, weights dominate | weight traffic becomes the bottleneck | no — 904 MiB |
| offloaded / streamed weights | skipping avoids a transfer, not a multiply | not currently |
| batched serving | compute-bound rather than launch-bound | no — batch 1 |
| **memory ceiling for a bigger model on this card** | never materialise unused rows | **plausible** |

The last row is the honest motivation: this is a route to running a *larger* model
on 8 GB, not to making this one faster.

---

## 5. Phase 0 — three kill-tests, each cheap, each decisive

Following the pattern that worked for the KV program: the cheapest measurement
that can end the idea runs first.

### P0.1 — Is there any sparsity to exploit? (no training, minutes)

Measure the activation concentration of the SwiGLU intermediate, exactly as
`concentration.py` did for attention. For each layer, sort |activation| and
compute the fraction of the output's L2 norm recovered by the top-*k* neurons:

```
rho(k) = || down_proj( topk_k(act) ) ||  /  || down_proj(act) ||
```

**Pre-declared thresholds.** Measured over ≥8 prompts and all 28 layers:

- rho(25% of neurons) ≥ 0.95 → strong substrate; proceed to P0.2.
- 0.85 ≤ rho(25%) < 0.95 → weak; proceed only if P0.2 shows a speed win.
- rho(25%) < 0.85 → **kill.** SiLU does not concentrate enough, [2] is right about
  this family, and no selector however good can fix a substrate that is not there.

This single probe can end the project in one afternoon, which is the point.

### P0.2 — Even with perfect selection, is sparse faster here? (no model, minutes)

Synthetic, no checkpoint required: time a dense `[1,1024]×[1024,3072]` matmul
against a gather-then-matmul at 75%, 50% and 25% density, on this GPU, at batch 1
and at batch 8.

**Pre-declared threshold.** If gather+matmul at **25% density is not at least 1.3×
faster** than dense at batch 1, the speed claim is dead on this host regardless of
how good the selector is, and the project continues — if at all — as a *memory*
result only. Given the floor measured in the KV program, this is the outcome I
expect.

Report batch 8 as well, because a negative at batch 1 and a positive at batch 8
localises the finding to the serving regime rather than to the idea.

### P0.3 — Can a single global matrix predict the firing set at all?

The proposal's distinguishing feature is that the selector is *one persistent
matrix*, not a per-layer learned predictor. Before building anything, check that
this is even possible:

1. Collect the true top-25% firing set per layer over a corpus.
2. Fit one shared projection `S` (hidden → k) by SVD on the concatenated
   activations, and rank neurons by `|S x|` similarity.
3. Compare against three controls: **random** selection, **per-layer** oracle
   top-*k*, and a **static frequency** baseline (the globally most-often-fired
   neurons, no input dependence at all).

**Pre-declared threshold.** The global matrix must beat the *static frequency*
baseline in recall of the true firing set. If it does not, then the input
dependence is contributing nothing and the honest result is "a fixed hot-neuron
set is as good", which is a real finding and worth reporting — but it is not the
proposed mechanism.

That third control matters. Without it, a selector that merely rediscovers "these
neurons are usually active" would look like it works.

---

## 6. If Phase 0 survives: what Phase 1 measures

- **Quality against sparsity**, on rag_dev and chat_dev with the frozen v4.5
  scorer, at 75/50/25% density, reported with byte-identical reply counts — the
  metric that catches damage aggregates hide.
- **Attention-output-style calibration**: relative error of the FFN output at each
  density, against perturbations whose end-to-end cost is already known, so the
  error has a scale rather than floating free.
- **Peak memory** with de-duplicated storage pointers, because a saving that is a
  view is not a saving.
- **Latency** with rotated arms and minimum-of-rounds, because this host's thermal
  band is 2.17×.

## 7. Success criteria, declared now

**Worth publishing** if: rho(25%) ≥ 0.95, the global selector beats static
frequency on recall, and quality at 50% density costs less than 0.02 on rag_dev.

**Worth keeping as a negative result** if it fails in an explainable way — for
example if SiLU concentration is genuinely too weak, which would be a useful
confirmation of [2] on a small model and a reason nobody else need repeat it.

**Abandoned** if it fails for reasons we cannot attribute.

## 8. What this does not do

It does not touch the KV cache, and the KV cache is what limited this program.
The significance matrix and the stabilizer matrix are orthogonal: one bounds the
memory that grows with *context*, the other the compute that scales with *model*.
Nothing here supersedes or competes with the work on `main`.

---

## References

*Recalled from the literature, not resolved against a bibliographic database.
Verify before citing.*

[1] Z. Liu et al. *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference
Time.* ICML 2023. arXiv:2310.17157.

[2] I. Mirzadeh et al. *ReLU Strikes Back: Exploiting Activation Sparsity in Large
Language Models.* ICLR 2024. arXiv:2310.04564.

[3] Y. Song et al. *PowerInfer: Fast Large Language Model Serving with a
Consumer-grade GPU.* arXiv:2312.12456, 2023.

[4] K. Alizadeh et al. *LLM in a flash: Efficient Large Language Model Inference
with Limited Memory.* arXiv:2312.11514, 2023.

[5] Z. Zhang et al. *MoEfication: Transformer Feed-forward Layers are Mixtures of
Experts.* Findings of ACL 2022. arXiv:2110.01786.
