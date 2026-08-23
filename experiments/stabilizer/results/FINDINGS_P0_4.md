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

## What survives

**Strong layer structure, and it is the useful part.** Uniform rank is the wrong
shape for this problem:

| layer | rho_static | rank @90% | err at r=64 |
|---:|---:|---:|---:|
| 0 | 0.7368 | 106 | 0.2874 |
| 7 | 0.7314 | 148 | 0.3369 |
| 14 | 0.8024 | 143 | 0.3361 |
| 21 | 0.8736 | 147 | 0.2417 |
| **27** | **0.9330** | **51** | **0.0327** |

The last layer needs rank 51 and reconstructs to 3.3% error; the middle band needs
~145 and sits near 34%. That matches the independent finding in
`manifests/weight_spectra_*.json` that effective rank falls with depth, and it means
a **rank schedule** is a different hypothesis from the one tested here.

Also worth keeping: ranking the static set by *mean |activation|* gives rho 0.80
against P0.3's 0.7468 from top-k frequency. A small methodological improvement to
the prune baseline, independent of whether the correction works.

## Verdict

**The cheap, uniform low-rank correction is refuted** — at its own oracle ceiling,
against a threshold set in advance, with the margin widening as data is added.

Three things this does **not** refute, each needing its own pre-declared threshold
rather than a rescue of this one:

1. **A per-layer rank schedule.** Late layers are genuinely cheap (rank 51, 3.3%).
   Whether a schedule beats uniform rank at equal total parameters is untested.
2. **Rank 256 plus healing.** 7.5% per-layer reconstruction error still saves
   183.5M (R2 → 290.5M, −38.7%). Whether healing absorbs that is an empirical
   question needing a training run, and B0 recovering all 202 abstentions shows
   healing is not weak. But 7.5% compounded across 28 layers is a much larger
   perturbation than the 2% token shift that cost 9 abstentions.
3. **Attention.** Unmeasured. The FFN answer does not transfer, and
   `weight_spectra` found k_proj/v_proj the *most* compressible matrices — which is
   where `artifacts/attn_shape/` already went.

On the "per-state versus global" question: this probe measured the **per-state
oracle**, since the SVD is fitted per layer to the observed tokens. A single global
fixed matrix is strictly weaker, so the global variant fails harder. That ordering
is worth keeping — it means the cheap version never needed testing separately.
