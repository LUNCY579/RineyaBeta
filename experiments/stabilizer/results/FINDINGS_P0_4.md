# P0.4 — stabilizer as a compression anchor (FAIL)

An addition to the Phase 0 record in [`FINDINGS.md`](FINDINGS.md), testing a
different use of the stabilizer matrix than the one P0.1–P0.3 examined.
`src/residual_rank.py`, `results/residual_rank*.json`. CPU only; the forward pass
is bf16 to match P0.1, all analysis in float32.

## The proposal, and why it needed reframing first

The stabilizer matrix was proposed as a **selector**: predict the top-k firing
neurons per token. P0.2 killed that on speed. The compression proposal was to reuse
it as "the anchor point of a compressed model".

**A selector cannot compress.** Choosing the top-k of 3,072 neurons per token
requires all 3,072 weight columns to stay resident — which is why P0.3 measured
firing-set stability and `predict_firing.py` says of its static arm, *"the weights
can be pruned permanently. If this works, nothing else is needed."* It did not: rho
0.7468 against the predictor's 0.9027, a gap of 0.156, so the verdict was predict
rather than prune.

So for the idea to compress, the stabilizer has to stop selecting and start
**reconstructing**: delete the low-firing neurons permanently, then approximate what
they contributed with a cheap term.

```
F = a @ W_down.T                  full FFN output          [T, 1024]
K = a[:,S] @ W_down[:,S].T        kept, static set S
Y = F - K                         what the prune discarded
Y_r                               best rank-r approximation of Y
```

Per layer, at keep fraction k and correction rank r:
`3*1024*3072k + 2048r` against `9,437,184`. At k=0.25, r=64 that is 2,490,368 —
saving 194.5M over 28 layers, taking R2 from 474.0M to ~279.5M. **Larger than the
vocabulary trim's 122M.** Break-even is r < 3,457, so rank was never the parameter
constraint; the only question was whether a *small* r suffices.

**Threshold declared before measuring:** rank at 90% of Y's energy must be ≤ 128.

**This measures a ceiling, not a method.** Y_r is the oracle rank-r approximation,
computed by SVD of Y on the very tokens being scored. A real correction is a fixed
matrix that has never seen them, so it can only do worse. A pass would have meant
"not yet refuted"; a fail means dead.

## Result: FAIL, and it hardens with more data

| prompts | tokens | rank @ 90% of discarded energy |
|---:|---:|---:|
| 8 | 1,770 | 133 |
| 16 | 3,571 | 145 |
| 24 | 5,417 | **149** |

The rank **rises with sample size and has not plateaued**, so the first reading was
an undersampling artifact moving *toward* the threshold. `rho_static` converged over
the same sweep (0.8019 → 0.8005 → 0.8002), which is what a settled quantity looks
like — the contrast is why the sweep was worth running rather than accepting a
4%-margin verdict. Per P1, a borderline number is a reason to sample more, not a
reason to move the line.

Relative reconstruction error against the full FFN output, 28-layer mean, 8 prompts:

| correction | rel. error | params/layer | saves ×28 | R2 → |
|---|---:|---:|---:|---:|
| none (static prune only) | **0.5300** | 2,359,296 | 198.2M | 275.8M |
| rank 16 | 0.4200 | 2,392,064 | 197.3M | 276.7M |
| rank 64 | 0.2817 | 2,490,368 | 194.5M | 279.5M |
| rank 128 | 0.1770 | 2,621,440 | 190.8M | 283.2M |
| rank 256 | 0.0751 | 2,883,584 | 183.5M | 290.5M |
| rank 512 | 0.0150 | 3,407,872 | 168.8M | 305.2M |

A static 25% prune with no correction discards **53% of the output in relative
error** — far worse than `rho_static` = 0.80 suggests, because rho is a per-token
norm ratio and misses directional error.

## The layer-structure claim: retracted

An earlier draft of this file reported "strong layer structure, and it is the useful
part", citing layer 27 at rank 51 / 3.3% error against ~145 / ~34% mid-stack. Those
were single layers from the **8-prompt** run. At 24 prompts the structure largely
disappears:

| layers | mean rank @90% | mean err @ r=64 |
|---|---:|---:|
| 0–6 | 144.6 | 0.3017 |
| 7–13 | 160.1 | 0.3603 |
| 14–20 | 153.3 | 0.3075 |
| 21–27 | 138.7 | 0.2012 |

A range of 138.7–160.1 is nearly flat. **The same undersampling artifact that made
the rank look like 133 also manufactured the depth trend** — twice in one probe,
which is worth recording as the more general lesson.

Tested anyway, since a schedule was the one surviving hypothesis. Greedy per-layer
rank allocation at **equal total parameter cost**, marginal-error-per-parameter:

| budget | uniform err | scheduled err | gain |
|---|---:|---:|---:|
| rank 64 equivalent | 0.2927 | 0.2847 | −2.7% |
| rank 128 equivalent | 0.1935 | 0.1863 | −3.7% |
| rank 256 equivalent | 0.0994 | 0.0947 | −4.7% |

A schedule buys **2.7–4.7% relative** error reduction. Errors stay at 9.5–29%
regardless. That is not the order of magnitude the idea needs, and it follows from
the flat depth profile rather than being a separate surprise.

## Verdict

**The cheap, uniform low-rank correction is refuted** — at its own oracle ceiling,
against a threshold set in advance, with the margin widening as data is added.

Three things this does **not** refute, each needing its own pre-declared threshold
rather than a rescue of this one:

1. ~~A per-layer rank schedule~~ — **tested and closed above**: 2.7–4.7% gain at
   equal budget, because the depth profile is flat once properly sampled.
2. **Rank 256 plus healing.** 7.5% per-layer reconstruction error still saves
   183.5M (R2 → 290.5M, −38.7%). Whether healing absorbs that is an empirical
   question needing a training run, and B0 recovering all 202 abstentions shows
   healing is not weak. But 7.5% compounded across 28 layers is a much larger
   perturbation than the 2% token shift that cost 9 abstentions.
3. **Attention.** Unmeasured. The FFN answer does not transfer, and
   the weight-spectra measurement (that work lives on the model-surgery branch,
   not here) found k_proj/v_proj the *most* compressible matrices — which is
   where `artifacts/attn_shape/` already went.

On the "per-state versus global" question: this probe measured the **per-state
oracle**, since the SVD is fitted per layer to the observed tokens. A single global
fixed matrix is strictly weaker, so the global variant fails harder. That ordering
is worth keeping — it means the cheap version never needed testing separately.

## Why all four Phase 0 probes failed for one reason

Worth stating together, because the shared cause is more informative than the
verdicts:

| probe | what failed | why |
|---|---|---|
| P0.2 speed | gather is 0.84× | per-token selection needs a gather; it costs more than the narrower matmul saves |
| P0.3 static prune | ρ 0.9586 → 0.7468 | the union of per-token top-k sets over a corpus is far larger than any single top-k |
| P0.4 low-rank | rank ≥149, rising | different tokens discard different directions, so the discarded subspace grows with the corpus |
| P0.4b schedule | 2.7–4.7% | the variation is token-driven, not layer-driven, so per-layer budgeting has little to allocate |

**The concentration is real and it is per-token.** Every attempt to freeze it into a
persistent artifact — a fixed set, a fixed low-rank term, a fixed per-layer budget —
fails because the structure being exploited is input-dependent, and compression
requires input-independence. Those two requirements are in direct tension, and no
variant of "make the stabilizer persistent" escapes it.

That is a conclusive result rather than four inconclusive ones, and it is the
argument for stopping rather than trying a fifth variant.

## What the probe did buy

It calibrated the ordinary width-pruning axis, which needs no stabilizer at all.
A static 25% FFN keep costs **53% relative output error** — so 25% is far too
aggressive, and the sane version of width pruning lives at much higher keep ratios
with healing. That number did not exist before this probe and it constrains C2 in
`docs/RESEARCH_PROGRAM.md` directly.
