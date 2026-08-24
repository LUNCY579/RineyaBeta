# CSGQR — GPU Benchmarks

Measured, not inferred. Source: `results/gpu_bench.json`, produced by
`src/bench_gpu.py` on layer 20 of `artifacts/rineya-core-en32k-28L`.

---

## 0. Read this first: the card was shared

**Every latency number below was taken with a different experiment
(`experiments/kv_significance/src/passkey.py --lengths 131072`) running
concurrently on the same 8 GiB RTX 4070 Laptop.** That job was not mine to stop.

The cost is visible and large. Across three runs of the *same* benchmark the
dense MLP microbenchmark measured **0.4649, 0.4782 and 1.7218 ms** — a 3.7×
spread — and the exact-lens monitor's share of it moved between **391% and
1396%**. One of those runs also died with `CUBLAS_STATUS_EXECUTION_FAILED`.

So:

* **Absolute latencies here are not usable.** Do not quote them.
* **Orderings and same-run ratios are usable**, because each run measures every
  arm back to back in one contended window, and every ordering below reproduced
  across all three runs.
* **Memory numbers are usable.** `torch.cuda.memory_allocated` tracks this
  process's own live allocations and is unaffected by another process's
  occupancy. Peak *reserved* is this process's reservation, also unaffected.

`IMPLEMENTATION_READINESS.md` marks the systems block `FAIL` for this reason and
gates a systems verdict on an exclusive card. Nothing below is offered as one.

Method: `torch.cuda.synchronize()` around every timed region, 3 warmup
iterations, 12 timed iterations, P50 reported. `nvidia-smi` sampled alongside as
a cross-check.

---

## 1. The finding that decides Gate F: the quantized remnant is slower than the layer it replaces

Remnant sub-block alone, 512 tokens, bf16 activations. Ratios are within-run.

| path | P50 | vs dense MLP |
| --- | ---: | ---: |
| dense `nn.Linear` MLP | 0.4649 ms | 1.00× |
| FP passthrough remnant, all groups | 0.6592 ms | **1.42×** |
| INT8 remnant, all groups | 2.2656 ms | **4.87×** |
| INT4 remnant, all groups | 2.3221 ms | **5.00×** |

**Dequantization costs about 3.5× what the matmul saves.** The INT4 remnant is
five times slower than simply running the dense bf16 layer. That is not a
routing question and no routing policy fixes it — it is the price of
reconstructing weights on every forward.

The FP passthrough isolates the two contributions: the *code path* costs 1.42×
(bucketing, gather, concatenation), and dequantization adds the remaining 3.5×.

### 1.1 Routing does save time — but only when there is nothing to dequantize

From the same run:

| ρ | FP passthrough | vs dense | INT4 | vs dense |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 0.2279 ms | **0.49×** | 1.1105 ms | 2.39× |
| 0.25 | 0.2570 ms | **0.55×** | 1.1241 ms | 2.42× |
| 0.50 | 0.3687 ms | 0.79× | 1.3795 ms | 2.97× |
| 0.75 | 0.5364 ms | 1.15× | 1.3357 ms | 2.87× |
| 1.00 | 0.6592 ms | 1.42× | 2.3221 ms | 5.00× |

The mechanism works: at ρ=0.25 the FP remnant runs in **55%** of the dense MLP's
time, and the trend is monotone in ρ. So "execute only the selected groups" is a
real compute saving and the hard-skip implementation delivers it.

But for the **quantized** remnant — the only version that saves storage — every
operating point is 2.4–5.0× the dense layer. Routing down to an eighth of the
groups still leaves it 2.4× slower than not compressing at all.

The INT4 column is also visibly **non-monotone** (ρ=0.75 is faster than ρ=0.50),
which is what a launch-bound, shape-sensitive workload looks like at this size.

---

## 2. Monitor cost: measured, and nothing like the MAC ratio

Per 512 tokens, against the dense MLP in the same run.

| monitor | P50 | measured share | **analytic MAC share** |
| --- | ---: | ---: | ---: |
| need probe (`d_model → 1`) | 0.0581 ms | **12.50%** | 0.0109% |
| group router (`d_model → 24`) | 0.0571 ms | **12.28%** | 0.2604% |
| need + router together | 0.0950 ms | **20.44%** | 0.2713% |
| exact lens, projection only | 1.4120 ms | 303.7% | 355.6% |
| exact lens + feature reduction | 6.4890 ms | **1395.8%** | — |

**This is a correction to a claim this campaign had been making.** The
"probe costs 0.0109% of the MLP it gates" figure is a FLOP and byte ratio, and
it is right as such. As a *latency* it is wrong by nearly three orders of
magnitude: a `[512, 1024] × [1024, 1]` GEMM is small enough to be dominated by
launch overhead, so its MAC share badly understates its wall-clock share.

The honest statement of the deployment argument:

> The probe monitor costs **~12%** of the gated MLP in wall-clock and the exact
> lens costs **~1400%**. The probe is roughly **110×** cheaper, not 32,000×.

110× is still decisive, and the argument survives intact. But "four orders of
magnitude cheaper" is a FLOPs claim masquerading as a latency claim, and it is
withdrawn.

Note also that the closed form `|V|/(3·d_ff) = 3.556×` predicts the
*projection-only* cost well (measured 3.04×, within contention noise) and
badly understates the **full** monitor, because the feature reduction —
log-softmax, entropy and top-2 over 32,768 classes in float32 — is memory-bound
and costs 3.6× the projection itself. A design that budgeted for the projection
alone would have under-budgeted by 4.6×.

---

## 3. Resident bytes

Measured as `memory_allocated` deltas around construction, compared against the
analytic model rather than reconciled with it.

| format | measured | analytic | ratio |
| --- | ---: | ---: | ---: |
| FP passthrough | 18.000 MiB | 18.000 MiB | **1.0000** |
| INT8 | 10.211 MiB | 9.211 MiB | 1.1086 |
| INT4 | 6.211 MiB | 4.711 MiB | 1.3185 |

**The FP row at exactly 1.0000 is a bug fix, not luck.** It previously measured
6.000 MiB against 18.000 accounted (ratio 0.3334), because
`weight.detach().to(dtype).contiguous()` is three no-ops when the source is
already that dtype on that device — so the buffer *aliased* the donor's storage.
That broke the byte accounting and, worse, the freezing guarantee: an aliased
buffer is not a frozen copy, and `keep_donor_resident=False` freed nothing. See
`BUG_LEDGER.md` **S19**. The comparison in this table is what caught it.

**The INT8/INT4 excess is unexplained and is reported as such.** Investigated:

* On a **clean allocator**, `PackedMatrix` accounting is **exact** — the sum of
  buffer storage `nbytes` equals the measured delta to the byte (diff +0) for
  both INT4 and INT8, and no buffer is a view into a larger storage.
* The excess appears only when the 904 MiB model is already resident, and is
  exactly 524,288 B per matrix at INT4 and 1,048,576 B per matrix at INT8 —
  i.e. one third of each `qweight`.
* It is **not** allocator power-of-two rounding: a direct 1,572,864-byte
  allocation reports exactly 1,572,864.

I could not attribute it to a specific retained tensor, so no explanation is
offered. **The conservative figure to quote is the measured one**, which makes
INT4's effective compression against bf16 **2.90×** rather than the accounted
3.82×, and INT8's **1.76×** rather than 1.95×.

Per-arm `memory_delta_mib` in the JSON is trustworthy **only for the first
installed arm** (`b1-delete`, −12.414 MiB: 18 MiB donor freed, 4.711 MiB packed
added). For later arms the baseline is contaminated by the previous iteration's
model and the delta reads as roughly the whole model. Documented in the source.

---

## 4. End-to-end: every CSGQR arm is slower and uses more VRAM than dense

Whole model, prefill at 32 tokens, 32 decode tokens.

| arm | prefill@32 | decode tok/s | peak reserved | Δ peak vs dense |
| --- | ---: | ---: | ---: | ---: |
| b0-dense | 45.93 ms | **22.9** | **956 MiB** | — |
| b1-delete | 47.78 ms | 22.6 | 956 MiB | +0 |
| b2-fp-all | 54.32 ms | 22.1 | 1056 MiB | +100 |
| b3-int8-all | 50.43 ms | 21.8 | 1068 MiB | +112 |
| b4-int4-all | 55.94 ms | 21.9 | 1128 MiB | +172 |
| b6-static-25 | 48.12 ms | 21.8 | 1140 MiB | +184 |
| b7-conf-whole | 54.36 ms | 21.0 | 1160 MiB | +204 |
| b12-probe-whole | 51.00 ms | 21.7 | 1200 MiB | +244 |
| **b16-csgqr-dyn** | **87.20 ms** | 21.7 | **1226 MiB** | **+270** |

Reproduced in all three runs. The full method is the **worst arm on every
systems axis**: 1.90× the dense prefill latency and **+28% peak reserved VRAM**.

**Peak VRAM goes up, not down, and the reason is instructive.** `b1-delete`
matches dense at 956 MiB — so the packed storage itself is not the problem. The
entire increase is *execution transient*: dequantized weight blocks, per-bucket
gathers and concatenations, and the monitor's workspace. A method whose stated
purpose is to reduce resident weight bytes ends up costing more peak VRAM than
it saves, because what it saves is persistent and what it adds is transient —
and peak is what the card actually gives up.

The 13.3 MiB of persistent saving is real (`b1-delete`'s −12.414 MiB delta
confirms the donor is freed) and is swamped by ~200 MiB of transient.

---

## 5. Gate F verdict

**`FAILED`, on measured evidence, and for reasons that are not fixable by
tuning at this scope.**

1. Dequantization costs ~3.5× what the saved matmul buys, so the quantized
   remnant is 5× slower than the dense layer at ρ=1 and still 2.4× slower at
   ρ=0.125.
2. The monitor adds ~12% (probe) or ~1400% (lens) of the gated MLP.
3. Peak reserved VRAM rises 100–270 MiB because execution transients exceed the
   13.3 MiB of persistent saving.
4. The analytic ceiling already bounded the best case at 1.47% of model VRAM and
   1.99% of model MACs — below allocator and clock noise — so there was no
   headroom to absorb any of the above.

This was **predicted before measurement** (`results/budget.json`, computed before
any GPU run) and the measurement is consistent with the prediction, plus a
dequantization cost the analytic ceiling did not model.

### What would have to change

* **Many retired layers, not one.** Every number here is capped by one MLP being
  1.99% of the model.
* **A fused dequantize-GEMM.** The 3.5× dequantization penalty is a
  reference-backend property. A kernel that dequantizes into registers during
  the GEMM removes it in principle. Nothing here demonstrates that it does, and
  H12 is `NOT_ESTABLISHED` rather than supported — a hypothetical optimised
  backend is not demonstrated performance.
* **Bucketing that is not a Python loop.** Per-token routing costs 622 s against
  39 s for a fixed group set over 414 prompts in the quality harness — 16× — and
  shows up here as `b16-csgqr-dyn`'s 1.90× prefill.
* **Transient reuse.** A preallocated dequantization workspace would remove most
  of the +200 MiB peak, which is the difference between a method that costs VRAM
  and one that saves it.

### And one positive systems result worth keeping

The FP passthrough at ρ=0.25 runs in **55%** of the dense MLP's time, monotone in
ρ. **Hard group skipping is real and it works** — the executed-sparsity
implementation delivers genuine compute reduction. The problem is entirely that
dequantization currently costs more than the sparsity saves. That separates a
*mechanism* result (positive) from a *systems* result (negative), which is the
distinction the brief insists on and the one that tells you where to spend
effort next.
