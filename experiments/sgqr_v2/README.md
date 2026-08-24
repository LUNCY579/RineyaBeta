# SGQR-v2 — Remnant Recovery vs Generic Capacity

**Branch:** `research/sgqr-v2-remnant-recovery`, based on
`research/csgqr-quantized-remnant` at `dc977dd`.

## The question

> When a transformer layer is structurally retired, is retaining a low-bit copy
> of **that exact deleted computation** a uniquely efficient recovery mechanism,
> or is its benefit merely equivalent to adding **generic parameter capacity**
> back to the damaged model?

CSGQR-v1 established that a frozen INT4 copy of a retired MLP recovers 96.3% of
the divergence deleting it causes, and that filling the same bytes with a
*different layer's* weights is 2.8× worse than installing nothing. That rules out
one alternative — a wrong frozen donor — and leaves the important one standing:
**maybe any few megabytes would have done.** This campaign tests that.

Read [`PREREGISTRATION.md`](PREREGISTRATION.md) first. It fixes the layer set,
the byte budget, the hypotheses and the verdict rules **before** any result
existed, and it is its own commit so the timestamp is verifiable.

---

## What is already established

### Four-layer deletion causes real capability damage — unlike one layer

`results/exp0_damage_smoke.json` (40 held-out prompts, 343 reply positions):

| arm | KL vs dense | gold top-1 | perplexity |
| --- | ---: | ---: | ---: |
| A dense | 0.00000 | 0.3149 | 144.17 |
| B four-layer deletion | **0.84821** | 0.2828 | **327.08** |
| C four fp16 remnants | 0.00000 | 0.3149 | 144.17 |
| C four INT8 remnants | 0.00133 | 0.3120 | 144.83 |
| C four INT4 remnants | 0.02603 | 0.3236 | 148.14 |

This is the result that makes the campaign possible. At **one** layer, CSGQR-v1
measured gold-token accuracy moving by exactly `+0.00000` (CI [−0.0075, +0.0076])
while KL moved 0.28 nats — so recovery there was *fidelity*, not capability, and a
capacity comparison would have had nothing to compare on. At **four** layers,
perplexity more than doubles and gold-token accuracy drops 0.032. There is now
real capability damage for a recovery mechanism to repair.

Two further observations:

* **Damage is slightly superlinear.** Joint deletion costs 0.84821 against 0.7301
  for the sum of the four individual single-layer damages — a ratio of 1.16.
  Layers 13 and 14 are adjacent, which the preregistration flagged in advance as
  a confound for exactly this reading.
* **INT4 remnants still recover ~97%** at four-layer scale (96.9% on KL, 97.8% on
  perplexity), so H5 survives rather than collapsing.

### Both capacity controls are built and byte-verified

| arm | what | params | stored bytes | vs remnant budget |
| --- | --- | ---: | ---: | ---: |
| — | 4 × INT4 remnants (the budget) | — | 19,759,488 | 100.0% |
| **H** | LoRA r=18, 7 projections × 24 surviving layers | 9,732,096 | 19,464,192 | **98.5%** |
| **I** | learned gated MLP at the retired site, bottleneck 804 | 9,879,552 | 19,759,104 | **100.0%** |

Arm I is the sharper control. LoRA differs from the remnant in **placement** as
well as origin, so a win for the remnant could be about the insertion point. Arm I
removes that: same architecture family, same insertion point, same byte budget,
weights **learned from scratch** instead of preserved. The only remaining
difference between arm C and arm I is where the weights came from.

Arm I initialises as an exact no-op (`down_proj` zeroed), so a freshly installed
adapter is numerically identical to deletion and any measured difference is
attributable to training rather than to initialisation.

**The asymmetry is priced, not hidden.** The remnant is frozen and untrained; both
controls are trained on 128,365 rows of the B0 heal mix that the remnant never
sees. That favours the controls. It is the honest form of the question: does free,
untrained, exactly-preserved computation beat trained generic capacity at equal
storage cost?

---

## Layout

```
PREREGISTRATION.md          hypotheses, verdict rules, budgets -- fixed in advance
src/multilayer.py           retire N layers; global vs uniform vs random allocation
src/run_arms.py             arms A-G on the paired teacher-forced endpoint
src/adapters.py             arm I: byte-matched learned MLP at the retired site
src/train_capacity_control.py   trains arms H and I with one shared protocol
src/eval_recovery.py        scores H and I on the identical endpoint as A-G
results/                    machine-readable outputs
```

Built as a thin layer over the CSGQR-v1 primitives, which already carry 144
verification checks and 40 independent re-derivations (packed INT4 decoded by
hand, additivity from raw tensors, executed sparsity proven by instrumenting
`dequantize` and counting rows touched). Reimplementing the remnant would have
discarded that evidence.

## What was dropped from CSGQR-v1, and why

| dropped | evidence |
| --- | --- |
| vocabulary-head confidence gate | costs `\|V\|/(3·d_ff)` = 3.56× the layer it gates in MACs, 13.96× measured — cannot pay for itself at any threshold |
| per-token dynamic significance router | genuinely state-conditioned, and beat by neither margin nor interval over a fixed global ranking at ρ=0.25 or ρ=0.50 |
| three-level skip/partial/full ramp | ties a binary gate at matched compute |

All three remain runnable in `experiments/csgqr/` as historical baselines. They
are dropped from the default, not deleted, and the negative results that motivated
dropping them are preserved in `experiments/csgqr/NEGATIVE_RESULTS.md`.

## Reproducing

```bash
.venv/Scripts/python.exe experiments/sgqr_v2/src/run_arms.py --arms A-dense,B-delete4,C-int4-full --prompts 200 --skip-prompts 96
```

```bash
.venv/Scripts/python.exe experiments/sgqr_v2/src/train_capacity_control.py --kind lora --rank 18 --steps 300
```

```bash
.venv/Scripts/python.exe experiments/sgqr_v2/src/eval_recovery.py --kind lora --artifact artifacts/sgqr_v2_adapters/lora_r18_lr1e-04_s300
```

## Operating constraint, recorded

Three concurrent `experiments/kv_significance` jobs have been saturating the same
8 GiB card throughout (100% utilisation, at times under 250 MiB free), giving
roughly a **15× throughput penalty** against CSGQR-v1 baselines — a dense
reference that took 0.09 s/prompt there takes ~1.4 s/prompt here, and LoRA
training runs at ~30 s/step. Every timing number from this campaign carries
`SHARED_GPU_TIMING — NOT SUITABLE FOR ABSOLUTE PERFORMANCE CLAIMS`. Quality and
memory numbers are unaffected: they are deterministic, and
`memory_allocated` tracks this process only.
