# Stabilizer matrix — Phase 0 results

Three kill-tests with thresholds declared in
[`docs/EXPERIMENT_DESIGN_STABILIZER.md`](../../../docs/EXPERIMENT_DESIGN_STABILIZER.md)
before any measurement. Two passed, one failed, and the failure is decisive.

## P0.1 — the substrate exists (PASS, contrary to expectation)

`src/concentration_mlp.py`, 8 rag_dev prompts, 2,016 tokens, 28 layers, 3,072
neurons. ρ(k) is the fraction of the layer output's L2 norm preserved by keeping
only the top-k neurons.

| keep | by activation | by contribution | random | margin over random |
|---:|---:|---:|---:|---:|
| 6.25% | 0.8569 | 0.8581 | 0.2229 | +0.6351 |
| 12.5% | 0.9103 | 0.9117 | 0.3235 | +0.5882 |
| **25%** | 0.9581 | **0.9586** | 0.4682 | **+0.4905** |
| 37.5% | 0.9804 | 0.9805 | 0.5828 | +0.3977 |
| 50% | 0.9916 | 0.9916 | 0.6804 | +0.3112 |

**ρ(25%) = 0.9586 ≥ 0.95 → STRONG.** Per-layer minimum 0.9275 (layer 10), max
0.9874.

The design predicted this would fail. The relufication literature reports that
SiLU/GELU families have far lower activation sparsity than ReLU, and this
checkpoint is SwiGLU. That literature is about *exact zeros*; concentration is a
different property, and SwiGLU activations turn out to be heavy-tailed without
being sparse. **Weighting by the down-projection column norm changes nothing**
(0.9581 vs 0.9586), so a selector may rank on activation magnitude alone.

## P0.2 — the speed claim is dead (FAIL)

`src/probe_sparse_matmul.py`, synthetic, no checkpoint required. One SwiGLU block
at this model's shape (hidden 1024, intermediate 3072), minimum of 300 iterations.

| batch | density | gather | pregathered |
|---:|---:|---:|---:|
| 1 | 25% | **0.84×** | 1.45× |
| 1 | 50% | 0.82× | 1.38× |
| 1 | 12.5% | 0.72× | 1.38× |
| 8 | 25% | 0.62× | 1.14× |

**Threshold was 1.30× at batch 1, 25% density. Measured 0.84× — a slowdown.**

Gathering the live weight columns each token costs more than the narrower matmul
saves. The pregathered upper bound *does* clear the bar (1.45×), which localises
the cost to the gather rather than to the matmul.

**But the gather is not the real problem.** The dense FFN is 0.1106 ms/layer,
3.10 ms across 28 layers, against a measured decode floor of 34.70 ms/token — so
FFN matmuls are **8.9% of decode latency**. Even the unreachable pregathered case
saves 0.97 ms, which is **2.8% of a decode step**. A perfect, free selector is
worth under three per cent on this host, and no data layout changes that.

## P0.3 — the predictor works; the premise does not

`src/predict_firing.py`, 10 prompts, 28 layers, 25% keep, 70/30 train/test split
per layer.

| selector | recall of true top-k | **output preserved (ρ)** |
|---|---:|---:|
| random | 0.2500 | — |
| static fixed set | 0.4638 | 0.7468 |
| **global matrix** | **0.7034** | **0.9027** |
| global + significance | 0.7037 | 0.9030 |
| oracle top-k | 1.0000 | 0.9567 |

**A single persistent global matrix works as a predictor.** 0.70 recall against
0.25 for random and 0.46 for a static frequency-chosen set, preserving 0.9027 of
the layer output against the oracle's 0.9567. The input dependence is real, and
the static control — included precisely to catch a selector that merely
rediscovers "these neurons are usually on" — does not explain it.

**The KV significance matrix contributes nothing: +0.0002 recall, +0.0003 ρ.**
The hypothesis was that the shape of the attention distribution constrains which
neurons can fire — entropy, sink mass, recent-window mass, effective attended
count. Measured against a control identical except for those four features, the
shape carries no information the hidden state does not already contain. This is a
clean negative: the features are cheap and already computed, so there was no cost
to trying and there is no ambiguity in the answer.

**The firing set is not stable.** Overlap between adjacent tokens is 0.4341, and
0.3753 at eight tokens apart. The memory saving requires that unused weights need
not be resident; a set that turns over by half every token needs all of them
available anyway, so the memory motivation does not follow either.

## Verdict

Phase 0 ends the project **on this host**, for two independent reasons:

- **speed** — the FFN is 8.9% of decode; a perfect selector saves 2.8%
- **memory** — the firing set turns over too fast for unused weights to stay
  unloaded

What is established and worth keeping:

1. **SwiGLU concentrates.** ρ(25%) = 0.9586, margin +0.49 over random. The
   relufication result about exact-zero sparsity does not imply an absence of
   concentration, at least at this scale.
2. **One global matrix predicts the firing set** at 0.70 recall / 0.9027 ρ,
   clearly beating a static set. The core of the proposal works as a mechanism.
3. **Attention-distribution shape does not predict FFN firing.** Positions and
   feature directions are different spaces, and summarising the first says
   nothing extra about the second.

Where it could still pay, none of which is this machine: a model large enough that
weight traffic dominates decode; weights streamed rather than resident; or batched
serving where the FFN is compute-bound. Findings 1 and 2 would transfer. Finding 3
would not need retesting.

## Reproduce

```bash
python experiments/stabilizer/src/concentration_mlp.py --prompts 8
python experiments/stabilizer/src/probe_sparse_matmul.py
python experiments/stabilizer/src/predict_firing.py --prompts 10
```
