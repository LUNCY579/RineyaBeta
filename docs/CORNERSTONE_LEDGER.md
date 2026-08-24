# Cornerstone 1 → Cornerstone 2: difference ledger and idea map

All paths relative to `C:/Users/clclc/RineyaExperiment`. C1 = branch `main`, C2 = branch `exp/stabilizer-matrix`. Every number carries its artefact. Where a verification pass corrected a figure, only the corrected figure appears here.

---

## 0. What changed after this document was generated

This ledger was assembled before P2.3–P2.6 and stage 1 existed. Rather than regenerate
it — a generated document that silently goes stale is how the "directional pooling is
untested" error got in here — the corrections are listed once, here, and the sections
below are left as the record of what was known at the time.

| # | then | now |
|---|---|---|
| 1 | floor law `B_half = 0.141·L^0.535` | **retracted.** Fitted from the centred window; with a backward-reaching window every length is lossless at or below B=16, so no exponent can be fitted |
| 2 | lossless 128K at B=256 (492×) | **B=16 (7,864×)**, peak 1,501 MiB — with a backward-reaching pool window |
| 3 | directional pooling listed as an untested open direction | **it is cornerstone 2.** 21 committed files set it; its pre-declared kill test was run and failed to kill it |
| 4 | pool window a tuned constant | **backward reach alone decides** (forward reach inert, spread 0.000), and the requirement **tracks span length**: back=4 gives 1.000 at d=5 and 0.000 at d=7 |
| 5 | — | **derived reach** (`grow_spans`) matches or beats the best *tuned* fixed reach at every span length without being told it: 1.000 / 1.000 / 0.889 at d=4/5/7 against 0.972 / 1.000 / 0.778 |
| 6 | no result outside passkey | **stage 1 run.** 16 query-shaped decoys take B=64 from 1.000 to **0.611**; 16 neutral ones cost nothing (0.972); exact McNemar **p = 0.0010** |
| 7 | — | that collapse is **capacity, not scoring**: budget alone restores it (B=512 → 1.000). Competition costs **32×** of compression — 7,864× down to **246×** |

| 8 | no baseline existed | **the score is what works.** At 128K/B=16, significance 36/36 against random 0/36 and streaming 0/36 — both p = 2.9 × 10⁻¹¹. Neither baseline degrades; both retrieve nothing |
| 9 | — | **the mechanism is purely inference-time**: 0 added parameters, no training, 1.5625% cache overhead, and it runs on the published Halo core with no code change |
| 10 | — | **it preserves accuracy, it does not improve it.** Two hints (+0.056 at W=12, 1 needle at 32K) are both inside noise |
| 11 | stage 2 ungated | **the model does not degrade with span length** (uncompressed 0.889 / 0.806 / 0.944 at 3 / 12 / 20 tokens), so "declines monotonically with span" is retracted at its middle point; the W=20 cost survives |
| 12 | — | **stage 3 is viable**: 2-hop 0.833 and 3-hop 0.722 uncompressed, so chains measure the mechanism rather than the model's incapacity |

**The single most important line for planning:** 7,864× is a property of a haystack
with **no competitors at all**, and passkey's filler contains no digits whatsoever, so
a regex scores 1.000 on it with no model. 246× is what the same mechanism delivers
against sixteen plausible competitors. Quote whichever fits the claim, never the first
without its condition.

---


## 1. The one-paragraph difference

Cornerstone 1 asks a **ranking** question — "given B slots, which B cached positions are the most important?" — and its whole apparatus is downstream of that: a per-position, per-KV-head accumulated attention score, count-normalised as `S/C^e`, spent on a `sinks + recent + top-B` keep rule, evaluated by end-to-end accuracy at a budget. Importance is a property of a single position, and top-k is a per-position order statistic with no notion of adjacency. Cornerstone 2 opened on a different axis entirely (a persistent projector selecting FFN neurons), killed that on this host in three cheap probes, and then **reversed the direction of its own question** — from "does the significance matrix help a new predictor?" to "does anything help the significance matrix?" — which turned it into a **sufficiency-and-audit** question: what must survive *together* for the answer to be reconstructible, and what is the score actually selecting? The methodological invariant behind the shift is that C1 measures **endpoints** (accuracy at a cell) while C2 measures **identity through the transformation** (original position IDs carried alongside the cache through every gather; a prompt split so attention received *before* can be separated from attention received *after*). That single change of instrument produced every C2 finding, including the discovery that C1's own span-protection switch (`Policy.pool`) had been off in every long-context result it ever published.

---

## 2. Side-by-side ledger

| | **Cornerstone 1** (`main`) | **Cornerstone 2** (`exp/stabilizer-matrix`) |
|---|---|---|
| **Thesis** | A persistent per-position, per-KV-head accumulator of received attention mass governs eviction at 0 added parameters and 1.56% memory overhead; its ratchet failure mode is corrected by ranking on `S/C^e`, e=1. | Strand A: a persistent projector selecting which FFN neurons fire — **qualified negative on this host**. Strand B (unregistered): accuracy tracks the *span-coherence* of the retained set, not its cardinality; the absolute budget floor grows sub-linearly in length. |
| **Mechanism** | `_SCORES[l]`, `_COUNTS[l]` as `[b, n_kv, n]` fp32; SDPA monkey-patch scores a trailing window w=16 pre-`repeat_kv`; keep = 4 sinks ∪ top-heavy(`S/C^e`) ∪ recent; mask-before-pool; raw `S` (not the ranking key) gathered through eviction. `scripts/kv/significance.py`, 571 lines. | Strand A: per-layer ridge from hidden state → \|activation\|, top-25% selection (`experiments/stabilizer/src/predict_firing.py`). Strand B: turn on SnapKV max-pool (pool=7/15) in the C1 mechanism; carry original position IDs through gathers (`trace_needle_eviction.py`, `trace_span_positions.py`); split a prompt at 60% to separate past from future attention (`predict_future_attention.py`). |
| **Unit of analysis** | The **position**, independently rankable. | The **span** — and, after P2.2, more precisely "position plus a contagion radius", because pooling propagates score to *neighbours of high scorers*, not to spans. |
| **Governing law** | Accuracy is a function of the **keep fraction**; length drops out. 1.56% → 0.61 (64K) / 0.53 (128K); 6.25% → 0.97 / 0.92; 12.5% → 1.00 / 1.00. Implies a **linear** budget law and a constant compression ratio. [`PAPER_DRAFT_SIGNIFICANCE.md` §5.3] | **Accuracy is set by the pool's BACKWARD REACH alone** — not by keep fraction, budget, or window width: (4,0) width 5 scores 1.000 where (3,3) width 7 scores 0.278 at 128K/B=64, and forward reach is inert (matched-pair spread 0.000 at back=4, p=1.0). The required reach tracks the **span**, not the context: d=5 needs back≥4, and d=7 still scores 0.000 at back=4. *(RETRACTED: `B_half = 0.141·L^0.535`, the √L floor law — fitted from the centred window only; with the causal window every length is lossless at or below B=16, so no exponent can be fitted.)* Second, independent law: at fixed budget and fixed memory, span-coherence moves 128K/B=2048 from 0.53 to 1.00. [`FINDINGS_P2.md`] |
| **Lossless 128K operating point** | B=16,384 = 12.5% nominal keep (13.02% of the 125,826 real tokens), peak 3,202.3 MiB, **pool=1**, chunk=2048. [`pk_ladder_131072_B16384.json`] | B=256 = 0.195% nominal keep, **492× on real tokens**, peak 1,536.8 MiB, pool=7, e=1, recent=64. [`pk_128k_pool7_B256_rf.json`] — a **64× budget reduction for the same 12/12**. |
| **What it explains** | The ratchet (survivors are scored more often than newcomers, so a raw sum closes the retained set); why sinks must be pinned and masked before pooling; why chunked prefill makes peak memory independent of length; why decode is a floor with a knee rather than a line. | *Why* query-aware scoring is the biggest win (future attention is predictable neither from the past nor from a position's key, so the only information about the future is to look at it); *why* pooling works and where it will not; *what failure looks like* (a clean one-token truncation, not degradation to noise); *why* the demotion tier was never worth building. |
| **What it cannot answer** | Whether the score predicts *future* attention; what the score prefers *within* a piece of evidence; what a failure at 0.53 actually consists of; anything about reasoning at length. | Whether the stabilizer matrix works as an efficiency mechanism anywhere (never carried into a sparse forward pass); whether one *genuinely global* matrix beats per-layer predictors (never built); whether any of Strand B survives non-contiguous evidence, a second question, or a properly powered sample. |
| **Pre-registered?** | Yes — `docs/EXPERIMENT_DESIGN_KV_LONGCTX.md`, plus 93 pinned numbers in `scripts/design/verify_claims.py`. | Strand A yes (`docs/EXPERIMENT_DESIGN_STABILIZER.md`). **Strand B — the valuable strand — no.** No design document declares thresholds for the pooling result, the √L law, the 4-of-5 prefix finding, or the label-vs-payload trace. |
| **Primary artefacts** | `docs/PAPER_DRAFT_SIGNIFICANCE.md`, `docs/FINDINGS_KV_PROGRAM.md`, `docs/NEGATIVE_RESULTS.md` | `experiments/stabilizer/results/FINDINGS.md` (P0), `FINDINGS_P1.md`, `FINDINGS_P2.md`, plus `docs/FINDINGS_KV_PROGRAM.md` §4f |

---

## 3. Cornerstone 1 in detail

### 3.1 Mechanism (what the code actually does)

- **State.** Two module-global dicts keyed by `layer_idx`: `_SCORES[l]`, `_COUNTS[l]`, each fp32 `[b, n_kv, n]`. Never per query head — eviction decides about stored tensors, and query-head granularity would need an arbitrary reduction.
- **Scoring.** `patch_attention()` monkey-patches `sdpa_attention_forward` (process-local). Per call, with q post-RoPE and k post-RoPE but **pre-`repeat_kv`** (G = 16/8 = 2): take the last w=16 query rows, `logits = einsum("bgcwd,bgnd->bgcwn") * scaling`, causal-mask, softmax, sum over G and w → `s [b, n_kv, n]`. The n×n matrix is never materialised (eager at 16K would be 16·16384²·2 = 8.59 GB/layer); cost is `[heads, w, n]` at prefill, `[heads, 1, n]` at decode. The main path keeps its `is_causal` fast path (≈1.9× prefill).
- **Accumulation.** Equal length → add. Cache grew → pad and carry history onto the prefix it covers. Cache shrank → `RuntimeError` (overwriting would silently degrade the policy to "prefers recent tokens").
- **Ranking key.** `e == 0` → raw `S`; else `S / C.clamp_min(1).pow(e)`. e=1 is the mean.
- **Ordering contract.** `update()` runs *before* the attention call that produces the scores, so full prefill attention → scores recorded → the first decode step compresses. The first version evicted during prefill and looked 2.6× "faster" at 16K, which is how the bug was caught.
- **Decision rule.** `keep = arange(n_sink) ∪ topk(middle, n_heavy) ∪ arange(n − n_recent, n)`, with `middle` **masked before pooling**; `max_pool1d(kernel=pool, stride=1, padding=pool//2)` then re-masked, because pooling propagates finite values back into masked regions. Load-bearing: R2 puts 46.3% of attention mass on the first 4 positions, so with pool=7 slots 4–6 would inherit that mass and win top-k unconditionally. *(Note: `select_keep` caps `n_recent`/`n_heavy` against the cache length `n`; the budget-relative caps live in `Policy.n_recent()`/`Policy.heavy()`. Behaviourally identical whenever `n > budget`, which is the only condition under which `_evict` fires.)*
- **Gather.** Each KV head keeps its own top-B set, so the cache stays rectangular `[b, n_kv, B, d]`. Cached keys already carry RoPE, so reordering is safe. The **raw** `S` is gathered, not the ranking key (writing the mean back would compound the division each round); `_COUNTS` pads with zeros over lag.
- **Position reporting.** `get_seq_length()` returns `true_length`, not `keys.shape[-2]` — otherwise RoPE silently re-indexes after compression.
- **Degenerate-policy guard.** `heavy() == 0` means the policy is exactly StreamingLLM; it warns and counts rather than raising (a raise mid-sweep truncates the run). Fired on 20/752 rag_dev prompts at every keep fraction.
- **Query-aware inter-chunk scoring** (`chunked_prefill.py:score_with_query`). Between chunks the *full question* is run as queries against the cache with recording on, then the cache is restored by snapshot with byte-equality verification, so the question is never itself cached.

### 3.2 Findings that stand

| Claim | Number | Source |
|---|---|---|
| Model / cache arithmetic | 28 layers, 1024 hidden, 16 Q / 8 KV heads, head_dim 128, **474,021,888 params**, weights 904 MiB bf16; KV = 114,688 B/token = 112 KiB/token | `config.json`, safetensors header |
| Host | RTX 4070 Laptop, 8,187.5 MiB; 128K uncompressed cache = 14,336 MiB, 64K = 7,168 MiB | manifests `device_total_mib` |
| Overhead — **exact, not measured-approximate** | scores + counts = 2·8·4 B/pos/layer against K+V 2·8·128·2 B = **2/head_dim = 1.5625%**, independent of length and budget; 0.78% for scores alone; **0 added parameters** | `measure_sig.json`; `verify_claims.py` |
| Count normalisation, 128K B=2048, pool=1, 12 paired needles | e=0.00: 0.58/0.08/0.00 (mean 0.22) → e=1.00: 0.42/0.33/0.83 (mean **0.53**). At depth 0.9 the raw sum retrieves **0 of 12**, the mean **10 of 12** | `pk_128k_pe{0.0…1.0}.json` |
| Its statistics | depth 0.9 McNemar +10/−0, p=0.0020; pooled over 36 needle-positions +13/−2, p=0.0074 | `PAPER_DRAFT_SIGNIFICANCE.md` §5.4 |
| Ratchet severity scales with rounds | rounds = n/C: 8 at 32K, 16 at 64K, 31–32 at 128K; 6-trial sum-vs-mean 0.89/0.83, 0.55/0.89, 0.22/0.39 | `FINDINGS_KV_PROGRAM.md` §4c |
| **The only matched eviction cost in the program** — 32K, uncompressed vs 6.25% keep | 0.9722 (peak 4,698.7 MiB) vs 0.7778 (peak 1,742.1 MiB); cost **0.19**; 7 needles lost, 0 gained, p=0.0156. *Chunk-mismatched: 1024 vs 4096.* | `pk_32k_reference_yarn.json`, `pk_32k_paired.json` |
| Query-aware scoring, depth-matched pair, 32K B=2048 | blind 0.00/0.00/1.00 = 0.33 → aware 1.00/0.33/1.00 = **0.78**. Largest single quality effect in the program | `pk_blind.json` / `pk_aware.json` |
| Query-blind failure is budget-independent | 64K depth 0.99 = 1.00 and depth 0.90 = 0.00 at **both** B=2048 and B=8192 | `FINDINGS_KV_PROGRAM.md` §4b |
| Short-context ladder (rag_dev, 752 paired items) | off 0.66939; keep 0.75 0.65677 (−0.01262, 711/752 byte-identical); 0.50 0.61401; 0.50+pool7 0.63371; 0.25 0.33863 (−0.33076) | `FINDINGS_KV_PROGRAM.md` §2 |
| **Operating range** — significance loses to StreamingLLM below keep 0.5 | keep 0.75 +0.18132 (p=1.3e-34); 0.50 +0.13719 (p=3.9e-08); **0.25 −0.08345 (p=1.5e-02)** | `FINDINGS_KV_PROGRAM.md` §2.1 |
| Attention concentration (precondition) | ρ(k) = 0.7911 / 0.8576 / 0.9100 / 0.9506 / 0.9796 / 0.9961 at k=4…128; sinks removed ρ(32)=0.898 | `concentration.json` |
| Attention sinks | first 4 tokens receive **46.3%** of mass; 14/28 layers exceed 50%; layer 27 = 80.5%; layers 0–2 under 3% | `sink_profile.json` |
| Importance persistence | top-32 overlap between consecutive query positions **0.584** vs chance 0.138 — but 42% churns each step and eviction is irreversible | `FINDINGS_KV_PROGRAM.md` §3 |
| Chunked prefill correctness (fp32, compression off) | max \|Δlogit\| 1.12e-04 / 1.19e-04 / 9.47e-05 at chunk 32/64/256; argmax agreement 1.0000 | §4a |
| Chunked prefill memory identity | 32K/64K/128K all store 6,144 KV, resident cache exactly 672.0 MiB, peaks 1744.1 / 1744.4 / **1744.9** MiB — 0.0% spread over 4× length; prefill 7.1/13.3/27.8 s | §4a |
| **Decode is a floor with a knee** | 34.532 ms at 8 tok; 34.696 at 1,024; 34.779 at 4,096; 45.252 at 8,192; 76.365 at 16,384. Fit: floor 34.696 ms, knee 4,096, slope 3.384 µs/token, flat spread 1.043 ms over a 512× range | `probe_decode_floor.json` |
| Neither term is bandwidth-saturated | floor ⇒ 27.3 GB/s, slope ⇒ 33.9 GB/s = **11% and 13%** of ~256 GB/s peak → overhead-bound at batch 1 | same |
| The one measured compressed-vs-uncompressed decode | 16K: full 82.573 ms (1,798.1 MiB resident, 79→86 °C) vs evict 46.494 ms (118.1 MiB, 81→81 °C) = **1.78×**; evict_noscore 38.664 ms = 2.14× | `measure_decode_full_16384.json` |
| Decode with eviction is flat in length | 47.72 / 47.12 / 46.663 / 47.204 / 47.987 ms at 4K/16K/32K/64K/128K, constant 118.1 MiB cache | `measure_decode.json` |
| Operating point moves because of the floor | 128K: B=2048 → 0.53 @ 34.7 ms; B=4096 → 0.78 @ 34.7 ms; B=8192 → 0.92 @ **45.25 ms (+30%)**. *(`FINDINGS_KV_PROGRAM.md` §4e says 48.6 ms / +40% — stale; the probe measures 45.252.)* | `PAPER_DRAFT_SIGNIFICANCE.md` §5.6 |
| Score reuse for bit allocation — strict Pareto win, unexploited | pure int8 relerr 0.00762 @ 1.94×; **mixed (sinks bf16 / top-25% int8 / rest int4) 0.00693 @ 3.04×**; pure int4 0.11179 @ 3.75× | `mixed_precision.json` |
| Calibration ladder that makes relerr interpretable | keep 0.75 → 0.00937 (quality −0.01262); 0.50 → 0.03729 (−0.05538); 0.25 → 0.11381 (−0.33076) | `store_probe.json` |
| Grouping, not bit width, decides int8 quality | K per-channel g=128 + V per-token: 0.00774. Both per-token: **0.04134** at identical bytes (5.3×) | `store_probe.json` |
| Power floor that set cell size | exact McNemar all-one-direction gives p = 2/2^d, so p<0.05 needs d ≥ 6 | `PAPER_DRAFT_SIGNIFICANCE.md` §4 |

### 3.3 C1's limitations

1. **Everything at length is retrieval** — a 5-digit passkey in repetitive filler, chosen to isolate retrieval from comprehension. Named in both documents as the single largest gap.
2. **No uncompressed control above 32K.** 7,168 MiB (64K) and 14,336 MiB (128K) of cache against an 8,187.5 MiB card holding 904 MiB of weights and 430–1,340 MiB of compositor. Saturation supplies the bound only because passkey ceilings out.
3. **The 32K cell is anomalous and unexplained** — at 6.25% keep, 32K scores 0.78 while 64K scores 0.97 and 128K 0.92.
4. **Single checkpoint, single host, batch 1, eager PyTorch, no CUDA graphs.** The regime where a smaller cache should matter most is the one not measured.
5. **The short- and long-context arms are not the same model** — rag_dev ran without YaRN, every long-context run with YaRN ×4.
6. **Two unstated design confounds in the headline ladders:** (a) the two 12.5%-keep cells ran at chunk=2048 while every other rung ran at chunk=4096, doubling the compression-round count on exactly the axis the paper says drives ratchet severity; (b) `recent` is pinned at 512 across every rung, so the share of the budget chosen by significance sweeps from 49.6% at B=1,024 to 93.7% at B=8,192 — the keep-fraction ladder is partly a sweep of *how much of the policy is significance versus StreamingLLM*.
7. **The checker constrains numbers it knows about, not claims it does not.** The 0.25% overhead figure survived in the abstract while 74/74 checks passed, because it was unpinned. Now 93/93.

---

## 4. Cornerstone 2 in detail — two strands

### 4.1 Strand A: the pre-registered stabilizer study (qualified negative)

**Proposal** (`docs/EXPERIMENT_DESIGN_STABILIZER.md`): one persistent projector `S`, *global to the whole computation rather than one predictor per layer*, ranks SwiGLU neurons by `|Sx|` so only the top-k fire. Target 3,072 × 28 = 86,016 neurons; **9,437,184 MLP params/layer, 264,241,152 total = 55.7% of the 474.0M checkpoint**. *(The design doc's own "~9.4M/layer, 264.2M, ~44% of a ~600M model" has the right numerator and the wrong denominator — the checkpoint is 474.0M, and 904.2 MiB bf16, the same 904 MiB the doc uses elsewhere.)*

| Probe | Result | Verdict against its own threshold |
|---|---|---|
| **P0.1** concentration | ρ(25%) = **0.9586** by contribution (0.9581 by activation) vs random control 0.4682, margin +0.4905; per-layer min 0.9275 / max 0.9874. Curve: ρ(6.25%)=0.857, ρ(12.5%)=0.910, ρ(37.5%)=0.980, ρ(50%)=0.992. 8 rag_dev prompts, 2,016 tokens. | **PASSED** (≥0.95 STRONG) — and the pre-registration predicted it would fail |
| **P0.2** sparse matmul speed | gather+matmul at batch1/25% density = **0.84×** (a slowdown); batch 8 = 0.62×. Pregathered upper bound 1.45× at batch 1. Dense 0.1106 ms/layer at batch 1. | **FAILED** (threshold 1.30×) |
| **P0.3** predictability | recall of true top-25% set: random 0.2500, static-frequency control 0.4638, **learned 0.7034**, oracle 1.0000. Output preserved ρ: static 0.7468, learned **0.9027**, oracle 0.9567. 10 prompts, 70/30 split per layer. | **PASSED** its declared bar |
| **P0.3** memory sub-measurement | top-25% set **overlap** 0.4341 between adjacent tokens, 0.3753 at eight apart ⇒ **turnover ≈ 0.566/token**. | No pre-registered threshold — this is where the memory motivation died |
| **P0.3** cross-axis ablation | adding the KV matrix's four attention-shape features (entropy, sink mass, recent-window mass, effective attended count): **+0.0002 recall, +0.0003 ρ** | Null |

**Two corrections that matter for brainstorming from this strand:**

- **The distinguishing feature was never tested.** `predict_firing.py` line 174 opens `for li in range(L):` and line 217 fits `W = ridge(F_[:tr], Y[:tr])` *inside* that loop — **28 independent `[1025 × 3072]` predictors, not one shared matrix.** What the 0.7034 / 0.9027 numbers measure is exactly the per-layer predictor (Deja Vu) that the design doc says the proposal differs from. The sole claimed novelty of Strand A is **unmeasured**, and the file's own docstring contradicts its code.
- **The fit is ridge by normal equations, not the pre-registered SVD.** No SVD anywhere in the file. The deviation is not recorded in `FINDINGS.md`.

**Why it is dead on this host, and only on this host.** Two independent routes: (a) FFN = 28 × 0.1106 ms = **3.10 ms of a 34.70 ms decode floor = 8.9%**, and the unreachable pregathered case saves 0.966 ms = **2.8% of a step**; (b) turnover ≈ 0.57/token means all 264.2M MLP parameters must stay resident. Caveat on (a): 3.10 ms is a **synthetic** standalone SwiGLU benchmark with no checkpoint loaded, and 34.70 ms is a real end-to-end floor from a different probe — the ratio is an Amdahl estimate stitched from two measurements, not a profile. The 2.8% is measured independently, not derived from the 8.9%.

**What Strand A never did:** carry either surviving positive (SwiGLU concentrates; a learned selector reaches 0.70 recall / 0.9027 ρ) into an actual sparse forward pass. Phase 1 — quality vs sparsity on rag_dev/chat_dev, byte-identical reply counts, FFN error calibration, de-duplicated peak memory, rotated-arm latency — **was never run**. There is no end-to-end quality, latency or memory number for the mechanism, and `1 − ρ = 0.0973` has no calibration ladder attached (the KV side has one; the FFN side does not).

### 4.2 The reversal

P1 opens by inverting its own question in one line: P0.3 asked whether the significance matrix helps a global predictor; **P1 asks whether a global predictor helps the significance matrix.** Everything valuable on the branch came after that sentence. Read as a heuristic: *stop bolting a second mechanism onto a second axis and go instrument the one you already have.*

**P1.1 — the key does not predict future attention.** One 3,923-token prompt split at position 2,353 (60%, not half); 527,072 (layer, head, position) triples; one ridge from key → future attention, held out **by (layer, head) group** so it must generalise across streams. Recall of the true top-147 by future attention: random 0.0647, **past-only (the current mechanism) 0.3944**, key-only 0.1564, past+predicted **0.3811 — worse than past alone by 0.0133**, oracle 1.0000. Headroom +0.6056. The key is not weakly informative; it is noise.

**The consequence is a mechanism for C1's biggest empirical win.** Future attention is predictable neither from the past (0.39 against an oracle 1.00) nor from the position's content. So the only available information about the future is *the question* — which is exactly what `score_with_query` supplies, and why 0.33 → 0.78 at 32K. It also bounds a family: no cheap content-feature predictor closes that headroom.

### 4.3 Strand B: KV span preservation — the real yield, **never pre-registered**

This must be stated plainly. `docs/EXPERIMENT_DESIGN_STABILIZER.md` §8 says *"It does not touch the KV cache"* and *"Nothing here supersedes or competes with the work on main."* Strand B violates both. There is **no design document declaring a threshold** for the pooling result, the √L law, the 4-of-5 prefix finding, or the label-vs-payload trace. All of it is post-hoc. It is also where every finding worth brainstorming from lives.

**B.1 — the premise of the planned build was false, and testing it first is what redirected the branch.** The next thing scheduled was a reversible demotion tier, premised on "the needle gets evicted and eviction is irreversible." `trace_needle_eviction.py` at 128K/B=2048/pool=1, 30 rounds: streams retaining **at least one** needle token = 178/224, 181/224, 201/224 (79%/81%/90%) at depths 0.1/0.5/0.9, while accuracy over the same cells is 0.42/0.33/0.83. Retention moves ~10 points while accuracy moves 0.50. **The tier would have fixed nothing and went unbuilt.**

The same trace reports `final_token_frac` = 0.5716 / 0.5864 / 0.6530. Two cautions: that is the fraction of the **26-token needle sentence** (`needle_tokens [12606, 12632]`, framing included), not of the five digits — the digit-level figure comes later and is 35%. And it does not establish "completeness is the variable that matters": its range (8.1 points) is *smaller* than the stream-retention range it is meant to displace (10.3 points), and it is non-monotonic against accuracy (depth 0.1 has lower completeness than depth 0.5 but higher accuracy). What the trace legitimately did was point at fragmentation as a candidate, which pointed at pooling.

**B.2 — pooling had been off in every long-context result in the entire program.** `Policy.pool` defaults to 1 and `main`'s `passkey.py` never set it. Turning it on, at the **same** budget (2048), the **same** keep fraction, and the **same** peak allocation (1,742.9 MiB):

| 128K, B=2048, e=1, YaRN ×4, query-aware, 12 paired needles | d0.1 | d0.5 | d0.9 | mean |
|---|---|---|---|---|
| pool=1 [`pk_128k_grid12.json`] | 0.42 | 0.33 | 0.83 | 0.53 |
| pool=7 [`pk_128k_pool7.json`] | 1.00 | 1.00 | 1.00 | **1.00** |
| pool=15 [`pk_128k_pool15.json`] | 1.00 | 1.00 | 1.00 | **1.00** |

Same seed 20260820, byte-identical 12-key list, manifests differ only in `pool` — so this is genuinely paired. On the honest unit (12 needles, not 36): 2 of 12 needles were 3/3 correct at pool=1 vs 12 of 12 at pool=7 → 10 discordant pairs, all one direction, **one-sided exact binomial p = 0.00098**. The result is significant; only "36/36" is the wrong way to say it.

Not a knife-edge (both widths give 1.00), zero memory cost, zero latency cost.

**B.3 — the floor curve, in absolute budgets down to B=16.** Territory C1 never entered (its smallest 128K cell was B=2048). pool=7, e=1, recent=B/4, query-aware, YaRN ×4, 12 needles × 3 depths:

| | B=16 | B=32 | B=64 | B=128 | B=256 | B=512 |
|---|---|---|---|---|---|---|
| **32K** | 0.028 | 0.389 | 0.833 | 1.000 | 1.000 | 1.000 |
| **64K** | 0.000 | 0.000 | 0.806 | 0.972 | 1.000 | 1.000 |
| **128K** | 0.000 | 0.000 | 0.278 | 0.972 | 1.000 | 1.000 |

Peak allocation across the entire table: 1,500–1,563 MiB. Headline cell: **128K lossless at B=256 = 492× on 125,826 real tokens, 1,536.8 MiB** [`pk_128k_pool7_B256_rf.json`]. `FINDINGS_P2.md` headlines the same cell as "512×" by dividing by the nominal 131,072; the two committed documents disagree by 4% and 492× is the conservative one.

**B.4 — the scaling law, RETRACTED by P2.3.** It was fitted from the *centred* window only. With `pool_back=6, pool_fwd=0` every length is lossless at or below B=16, so B_half lies below the sampled range everywhere and **no exponent can be fitted**. What survives is the negative half: the linear keep-fraction law is excluded. The withdrawn fit, for the record — B_half interpolated at the 0.5 crossing in log2(B): **38.05 / 49.20 / 79.89** at 32K/64K/128K. Least squares in log-log: coefficient **0.1407**, exponent **0.5350**. A fixed keep fraction anchored at 32K predicts 76.2 and 152.4; observed 49.2 and 79.9 — the linear law is off by ~1.9× at 128K, and B=128 (below even its own half-accuracy point) already scores 0.972. **Honesty attached:** three points, each interpolated between budgets spaced 2× apart, 12 independent needles each; 0.5 sits well inside the uncertainty; the fit uses nominal L, and refitting on real_tokens moves the coefficient. **What the data excludes is the linear law, not much more.**

**B.5 — count normalisation survives pooling and gets larger.** 128K, pool=7, B=256, recent=64:

| e | d0.1 | d0.5 | d0.9 | mean |
|---|---|---|---|---|
| 0.00 (raw sum) | 1.00 | 0.33 | 0.17 | 0.500 |
| 0.50 | 1.00 | 0.92 | 1.00 | 0.972 |
| 1.00 (mean) | 1.00 | 1.00 | 1.00 | **1.000** |

e=0 → e=1 is +18/−0 pooled over 36 needle-positions, exact McNemar **p = 7.6e-06**; gap 0.50 at pool=7 against 0.31 at pool=1 (p=0.0074). The two mechanisms repair **orthogonal** defects — pooling is spatial (which positions inside a span survive together), count normalisation is temporal (arrival time inflating long-survivors' scores) — and the ratchet's signature became legible for the first time: at pool=7/e=0 the depth pattern is **1.00 / 0.33 / 0.17**, perfect early and failing late, exactly what a score accumulating with survival time predicts. At pool=1 that was masked by fragmentation (0.58/0.08/0.00 — damaged everywhere).

**Where the "neither alone reaches 1.00" claim is incomplete.** The pooling-alone arm at B=256 is measured (pool=7, e=0 → 0.500). The **count-normalisation-alone arm (pool=1, e=1) at 128K/B=256 does not exist in the repository** — on `main` the smallest 128K cell is B=2048 at 6 trials/1 depth, and on the branch every pool=1 128K point is B≥2048. That leg is a downward extrapolation from an 8× larger budget scoring 0.53. Also, "0.31 → 0.50" changes two variables (pool 1→7 *and* budget 2048→256); B=256 is well-justified because at B=2048/pool=7 every arm is at ceiling, but it is not a clean 2×2. **One cell closes it: 128K, B=256, pool=1, e=1.**

**B.6 — what failure actually is.** Among failures at B=32–128 across all three lengths: mean digits recovered **4.00 of 5**, and **zero failures recover zero digits**. At 128K/B=64, all **26 of 26** failures are correct prefixes (6644 for 66445, 4264 for 42649, 9987 for 99873). The model localises the needle and reproduces the span verbatim including the trailing ". Remember it", missing precisely the **final token of the key**. Only at B=16, far below the floor, does it shred (1.9–2.7 digits).

**B.7 — query-aware scoring keeps the label and discards the payload.** `trace_span_positions.py`, 128K, pool=7, e=1, key 75526, survival across all 224 (layer, KV head) streams:

| cell | framing | digits | gap |
|---|---|---|---|
| d0.1 / B=64 | 41% | 25% | +16 |
| d0.1 / B=128 | 55% | 41% | +14 |
| d0.5 / B=64 | 45% | 23% | +22 |
| d0.5 / B=128 | 60% | 42% | +18 |
| d0.9 / B=64 | 57% | 31% | +26 |
| d0.9 / B=128 | 66% | 50% | +16 |
| **overall** | **54%** | **35%** | **+19** |

Six of six cells, same direction. The reason is lexical: `score_with_query` runs "What is the pass key? The pass key is" against the cache; the framing tokens *are* the query, and the digits are what the query asks for and look nothing like it.

**B.8 — pooling does not protect spans; it propagates score to neighbours of high scorers.** Within-span survival is U-shaped in 6/6 cells (d0.5/B=64: 34%, 22%, 31%, 30%, 28% across d1..d5; d0.5/B=128: 52/44/46/48/49; d0.9/B=128: 59/49/56/55/54). The last digit of the *second* key copy rebounds to 41–56% solely because it sits beside " is the pass key", while interior neighbours sit at 15–29%. **Corollary that inverts the obvious remedy: a wider pool cannot help a token whose entire neighbourhood is low.**

**B.9 — tooling.** `verify_claims.py` now raises on quarantined `_invalidated_*` paths, on ambiguous multi-row matches, and on manifest fields that disagree with a claim's `require` dict. Pins went 85 → 88 → 93; all **93/93** reproduce on the checked-out tree.

**One thing described as a code change that was not.** `git diff main..HEAD -- experiments/kv_significance/src/passkey.py` adds only `--pool`, `pool=args.pool`, and `"pool"` in the manifest. There is **no recent-scaling logic** — `--recent` still defaults to 512 and is passed through unmodified. The `recent = B/4` rule was applied by hand per invocation (manifests confirm B=16/32/64/128/256/512 ran with recent=4/8/16/32/64/128). This is exactly the mechanism behind the file-collision hazard below.

### 4.4 What Strand B costs epistemically

- The pooling discovery **superseded the operating point of the whole prior program** — the 32K/64K/128K keep-fraction ladders, the exponent sweep, the budget grids, the 12.5%-keep saturation result, Figure 1, and the paper's central table were all measured at pool=1. They are correct measurements *of pool=1*. `docs/FINDINGS_KV_PROGRAM.md:525` records the supersession and `:1007` lists what must be re-run; a banner was added rather than an edit, on the stated grounds that rewriting them to a configuration nobody measured would be worse. **The ladders have not been re-run.** The paper's central table cannot yet be rebuilt.
- Passkey retrieves a **contiguous literal span**, and pooling is precisely what keeps a contiguous span intact. 492× is the friendliest possible task for this mechanism. `§4f` states this itself.
- The word "lossless" is retracted downward: a perfect cell means "no failure in 12 needles", a one-sided 95% lower bound of **0.779**.

---

## 5. The inversions — findings that reversed an expectation

These are the richest material because each one carries a mechanism, not just a number.

| # | Expectation | What happened | Number | Why it inverted |
|---|---|---|---|---|
| 1 | The mechanism's long-context results reflect the mechanism as designed | **The mechanism had been measured with its own span protection switched off.** `Policy.pool` defaults to 1; `main`'s `passkey.py` never set it. C1's §3.2 *describes* SnapKV pooling and §5.2 rationalises its keep-0.25 loss with a span argument — it just never ran it at length. | 0.53 → 1.00 at identical budget and identical 1,742.9 MiB; lossless 128K went 12.5% keep / 3,202 MiB → 1.56% / 1,743 MiB → 0.195% / 1,537 MiB, a **64× budget reduction from a flag** | A default in a Policy dataclass, never asserted in any manifest check |
| 2 | Accuracy tracks the keep fraction; length drops out | **The law fails outside the two pool=1 points it was fitted on, in the opposite direction.** At matched nominal keep 0.0977%: 0.389 (32K/B32), 0.806 (64K/B64), **0.972 (128K/B128)** — a 0.583 spread, longer context scoring *higher*. At matched keep 0.0488%: 0.028 / 0.000 / 0.278. At matched **budget** B=128: 1.000 / 0.972 / 0.972, spread 0.028. Mean cross-length spread: 0.257 by keep fraction, 0.167 by budget. | B_half = 0.141·L^0.535; linear law off by ~1.9× at 128K | The "law" was a 1-D projection of a 2-D surface with the second variable (span coherence) silently pinned at its worst setting — which is also why the two ladders coincided so cleanly: both were fully fragmented |
| 3 | The needle is being evicted, so build a reversible demotion tier | **The needle was not being evicted.** 178/181/201 of 224 streams retain it while accuracy is 0.42/0.33/0.83 | retention moves ~10 pts, accuracy moves 0.50 | The planned build was premised on a property nobody had measured. Testing the premise cost one trace and cancelled the build |
| 4 | Removing fragmentation might soften the program's central contribution | **It hardened.** The risk was pre-registered in writing when §4f landed. Re-measured at pool=7/B=256: e=0 → e=1 goes 0.500 → 1.000 | gap 0.31 → **0.50**; McNemar p 0.0074 → **7.6e-06** | Orthogonal defects: pooling is spatial, count normalisation is temporal. Removing the spatial loss made the temporal signature (1.00/0.33/0.17) legible for the first time |
| 5 | Adding a content feature to the score should help a little | **Adding it made the predictor worse.** past-only 0.3944, +key 0.3811 | −0.0133, against a random floor of 0.0647 and key-only 0.1564 | The key is noise, not weak signal. Held out by (layer, head) group over 527,072 triples so it could not memorise a stream |
| 6 | Query-aware scoring is the program's best intervention | **It is systematically biased against the answer.** Framing 54% survival, digits 35% | +19 points, 6/6 cells | The framing tokens *are* the query lexically; the payload is what the query asks *for* and looks nothing like it. Invisible to any end-to-end accuracy measurement |
| 7 | Pooling protects spans | **It propagates score to neighbours of high scorers.** Within-span survival is U-shaped in 6/6 cells (22% at the interior digit) | d0.5/B=64: 34/22/31/30/28 | Corollary that inverts the obvious fix: **a wider pool cannot help**. Width does nothing for a token whose whole neighbourhood is low |
| 8 | Near-floor failure looks like degradation | **It is a clean one-token truncation.** 4.00 of 5 digits, 0% zero-digit failures; 26/26 correct prefixes at 128K/B=64 | 6644 for 66445; 4264 for 42649 | The scoring pass simulates exactly one decode step, and exactly one token past that step is what survives — an open causal slot with a mechanism attached |
| 9 | The trailing token is lost because a centred pool under-protects the trailing edge | **Refuted by reading the code**, before any GPU time: `kernel=pool, stride=1, padding=pool//2` is symmetric. Then both pre-registered explanations fell: "tie-break at the boundary" predicts flat survival (observed 19-pt gap); "scoring gradient along the span" predicts a falling ramp (observed offsets 20–25 *rising* to 62–70%) | shape is a **trough**, and the split is semantic (label vs payload), not positional | Two hypotheses that were both positional, one variable that is lexical |
| 10 | SwiGLU will not concentrate (relufication literature, SiLU/GELU) | **ρ(25%) = 0.9586** vs random 0.4682, per-layer min 0.9275 | margin +0.4905, "STRONG" | The literature is about **exact zeros**; concentration is a different property. SwiGLU is heavy-tailed without being sparse. Nested inversion: weighting by the down_proj column norm changes nothing (0.9581 vs 0.9586), so the refinement anyone would build first is unnecessary |
| 11 | The gather is what kills sparse FFN | **The prediction held, the reason inverted.** gather+matmul 0.84× vs 1.30× threshold — but the pregathered upper bound *clears* it at 1.45× | FFN is 3.10 ms of a 34.70 ms step = 8.9%; best case saves 2.8% | The optimisation was aimed at a term that is not the bottleneck. No data layout fixes that |
| 12 | The two cornerstones inform each other | **Cross-axis is null in the tested direction.** Attention-shape features → FFN firing: +0.0002 recall, +0.0003 ρ. Reverse direction (key → future attention): 0.1564, and −0.0133 when combined | positions and feature directions are different spaces | Cheap, unambiguous, and would not need retesting on other hardware |
| 13 | The proposal's novelty (one global matrix vs per-layer predictors) was tested | **It was not.** `predict_firing.py:174,217` fits a separate ridge per layer. The measured "global" arm *is* the per-layer baseline the design doc says it differs from. Also: ridge by normal equations, not the pre-registered SVD | 0.7034 / 0.9027 are correct as printed and attached to the wrong claim | The file's docstring contradicts its code |
| 14 | n = 36 per cell | **n = 12.** All three depths draw an identical key list ([75526, 85667, 99873, 42649, …], verified byte-identical) | one-sided 95% CP lower bound **0.779** at n=12 vs 0.920 at n=36 | 12 independent needles observed three times |

---

## 6. Dead ends, each with the number that closes it

| Direction | Closing number | Source |
|---|---|---|
| Predicting future attention from a position's post-RoPE key | recall 0.1564 vs random 0.0647 and past-only 0.3944; **combining is −0.0133** | `predict_future_attention.json` |
| Attention-shape features → FFN firing prediction | +0.0002 recall, +0.0003 ρ | `predict_firing.json` |
| Contextual FFN sparsity as **speed** on this host | 0.84× at batch 1 / 25% vs a 1.30× threshold; 0.62× at batch 8; even the unreachable pregathered bound saves 2.8% of a step | `sparse_matmul.json` |
| Contextual FFN sparsity as **memory** on this host | firing-set turnover 0.566/token (overlap 0.4341 adjacent, 0.3753 at eight) | `predict_firing.json` |
| Diversity / NMS eviction | top-k already keeps 60.4 of 64 slots distinct vs random's 61.9; 91% of apparent redundancy is anisotropy (mean pairwise cosine 0.661); in attention-profile space suppression is +33.7% worse at τ=0.90 and worse than matched-random at every τ | `NEGATIVE_RESULTS.md` |
| Sliding-window attention as a memory strategy | peak goes **up** (w=256 and w=1024 both 3,832.5 MiB — the mask scales with context); resident cache 1,792.0 MiB, identical to baseline, because the layer holds a view; every SWA variant OOMs at 32K where gqa-1KV fits in 2,111 MiB; w=512 is provably the identity (510/510 byte-identical) | `NEGATIVE_RESULTS.md` §1 |
| Positional-only policy at long context (the `heavy()==0` degenerate) | 128K/B=256 and B=512 with recent=512: **0.00 at all three depths** where significance scores 1.00 | `pk_128k_pool7_B256.json` vs `_rf.json` |
| Ranking on the raw sum; tuning the exponent's interior | pool=7/B=256: e=0 → 0.500, e=1 → 1.000, +18/−0, p=7.6e-06. Exponent column is monotone (0.22/0.28/0.33/0.47/0.53) with no interior optimum | `pk_128k_pool7_e*.json`, `pk_128k_pe*.json` |
| Query-blind prefill eviction; "restate the question at the top" | blind 64K: depth 0.99 = 1.00, depth 0.90 = 0.00 at **both** B=2048 and B=8192. `--query-first` scores 0.00/0.00 at 32K/B=2048, identical to query-last | `FINDINGS_KV_PROGRAM.md` §4b; `pk_qfirst.json` |
| Uniform int4 KV | relerr 0.11179 = the keep-0.25 calibration point that cost −0.33076 end-to-end (0.12804 without bf16 sinks) | `mixed_precision.json` |
| Per-token grouping for int8 K | 0.04134 vs 0.00774 at identical 1.94× store | `store_probe.json` |
| Cross-layer KV sharing (CLA) | held-out relerr 0.1720 mean over 27 adjacent pairs (best 0.0550, worst 0.2796, 6/27 below 0.10) — with a 4.19M-param adapter removing 2.10M params | `NEGATIVE_RESULTS.md` |
| Symmetric K/V low-rank | at equal rank, V costs 2.3× K (rank-128: V-only 0.19383 vs K-only 0.08326). Rank-512 at 0.00147 is a **per-prompt SVD oracle** | `NEGATIVE_RESULTS.md` |
| Unhealed GQA head merging | attention relerr **1.0206** (uncorrelated, not degraded); chat_dev 0.83498 → 0.11384 | `NEGATIVE_RESULTS.md` |
| Compressing below the decode knee to buy latency | decode flat to 1.043 ms across a 512× cache range; B=2048 and B=4096 both cost 34.7 ms, so B=2048 gives up 0.25 mean accuracy for zero latency; at 4K context eviction is a net 0.68× slowdown | `probe_decode_floor.json` |
| Per-layer budget allocation as a large lever (short context) | 4.2% proxy-objective error reduction over uniform (down from a retracted 7.4%); **0.00%** at mean 128/layer; never validated against a task | `allocation.json` |
| A demotion tier motivated by "the needle gets evicted" | 79–90% stream retention at 128K/B=2048 while accuracy spans 0.50. *(Does not rule out demotion at B=64, where retention has not been re-traced.)* | `needle_eviction_trace.json` |
| chat_dev aggregates as a damage detector | score **rose** +0.00247 at keep 0.75 while only 361/510 replies were byte-identical; `prompt_function_alignment` (weight 0.28, fires 502/510) rewards length and brevity (0.01) + completeness (0.02) cannot outvote it | `FINDINGS_KV_PROGRAM.md` |
| Cells below ~6–12 needles; unpaired analysis of paired arms | p = 2/2^d ⇒ d ≥ 6 for p<0.05. A 3-trial run gave "normalization hurts, 0.87→0.60" and reversed at 6. A 6-trial depth curve read as monotone (0.17/0.50/0.83) was 0.42/0.33/0.83 at 12. Cochran-Armitage and Fisher on needle-paired arms produced **four withdrawn p-values** | `NEGATIVE_RESULTS.md` |
| Reading a depth law out of keep-fraction data | the one significant contrast (128K/B=2048, d0.9 > d0.5, +6/−0) does not replicate at 64K (trends p = 0.40/0.91/1.00) and vanishes at B=8192 (permutation p = 0.8879) | `PAPER_DRAFT_SIGNIFICANCE.md` |

---

## 7. Structural gaps — what **neither** cornerstone measured

1. **Task diversity at length.** Every long-context number in both cornerstones is one task: a 5-digit key in repetitive filler, scored by substring match. No multi-hop, aggregation, summarisation, real-document QA, or code. The frozen demo suite (0.2031 overall, `count_with_exclusions` 0.031) says reasoning is where this checkpoint is weakest.
2. **Multi-needle and distractors.** Every prompt has exactly one key, stated twice. Note the existing trace shows the **second copy is far worse protected** (d0.5/B=64 digits: 76/50/69/68/63 for copy 1 vs 42/34/19/26/66 for copy 2) — source redundancy is not being converted into robustness, and nobody has asked why.
3. **Payload and span length.** The needle is 26 tokens with a 5-token payload; pool width 7 was chosen from a *short-context* sweep. The pool-width × span-length interaction is entirely unmeasured, so 492× has no known dependence on how long the answer is.
4. **Non-contiguous evidence.** "A task whose answer is not contiguous would not compress this way" is a prediction, not a measurement, and it bounds the entire headline.
5. **Cache reuse and multi-turn.** One cache, one question, everywhere. No second question against a compressed cache, no conversation, no streaming setting where the question does not exist during prefill.
6. **Generation quality at length.** `max_new_tokens=12` in every long-context run, scored by substring containment. **Perplexity has never been measured at any length under any policy.** No free-form generation, no v4.5 scorer above 330 tokens.
7. **Eviction *during* decode.** With max_new=12 the cache never exceeds budget after prefill, so the self-compressing `update()` path is exercised only at short context. Long generations under an active budget — where the score keeps accumulating and the ratchet keeps running — are unmeasured.
8. **Batch > 1.** Everything is batch 1, and both central negatives are explicitly localised to that regime (34.70 ms overhead-bound floor; 0.84× at batch 1 vs 0.62× at batch 8). Whether per-head keep sets remain practical when batch elements disagree is unknown.
9. **Model scale and checkpoint generality.** One 474.0M Qwen3 derivative, one host. The 46.3% sink share, the ρ curves, the 4-position pin and the 8.9% FFN share are single-checkpoint facts, several measured on 8 prompts.
10. **Short-context regression from YaRN ×4.** The quality sweeps ran *without* YaRN; every long-context result *with* it. Two different models. One rag_dev arm would close it; nobody has run it.
11. **Eviction composed with quantisation, end to end.** Mixed precision (0.00693 at 3.04×) exists only as an offline probe on a 4K haystack — never in the live cache path, never composed with eviction, never scored on a task, and the √rounds requantisation constraint never tested end to end.
12. **Per-query-head resolution.** Scores are **summed** over the 2 query heads sharing each KV head. Max-over-group vs sum-over-group has never been checked; it is a one-line change in `_score_rows`.
13. **The observation window w=16 and chunk size as *quality* variables.** w=16 defines what "past attention" means and has never been swept. Chunk was swept only for memory (274 MiB, 5.5%, verdict unchanged) — yet at 128K, chunk 4096 vs 1024 changes query-scoring passes from 31 to 123 and eviction rounds by 4×, both of which the theory says are decisive.
14. **The sink count.** `sinks=4` in every run ever made here, despite 46.3% sink mass being the most-cited structural fact and 4 sinks being 6.25% of the budget at B=64. Never swept.
15. **FFN density → end-to-end quality.** The KV side has a calibration ladder (0.00937 → −0.013; 0.03729 → −0.055; 0.11381 → −0.331). The FFN side has none, so `1 − ρ = 0.0973` is uninterpretable. One rag_dev arm at 25% density converts Strand A's surviving positive into a transferable number.
16. **The untested cross-axis direction.** P0.3 tested attention-shape → FFN firing (null). P1 tested key → future attention (0.156). **Nobody has tested residual-stream / FFN-side state → KV eviction value**, even though the hidden state is what the P0.3 predictor showed *is* predictive of something, and it is free at prefill time.

---

## 8. Open directions

| Idea | Grounded in | Cost | Kill test |
|---|---|---|---|
| **Roll the query forward.** `score_with_query` runs one fixed string at each chunk boundary; instead greedily decode k=4–8 tokens against the current cache, score all k query rows, restore as it already does. | 26/26 failures at 128K/B=64 are correct 4-digit prefixes; mean 4.00/5 at every length and budget in the transition band. The scoring pass simulates exactly one decode step, and exactly one token past it survives. P1: key-only 0.156 vs past-only 0.394 vs oracle 1.00. | ~30 lines in `chunked_prefill.py`; 2 cells (128K B=64, B=32) ≈0.5 GPU-h | **Zero-code oracle first:** run `--query-aware` with the QUESTION extended by the true key's first four digits at 128K/B=64. If a query literally containing four of five digits still yields 4.00/5, no rollout reaches the fifth token and the family dies for 0.25 GPU-h. |
| **Directional pooling.** ~~open~~ **DONE — this became cornerstone 2 (P2.3/P2.4).** `pool_back=6, pool_fwd=0` takes 128K/B=64 from 0.278 to **1.000**, and holds at B=32 and B=16 (all 36/36; `pk_128k_B{64,32,16}_pool_b6f0.json`). P2.4 then separated the variables: forward reach is **inert** and backward reach alone decides — (4,0) width 5 scores 1.000 while (3,3) width 7 scores 0.278. | P2.2, 6/6 cells: framing 54% vs payload 35%. | Run: ~2 GPU-h across 7 cells. | The pre-declared kill test — "accuracy stays within ±1 needle of 0.42/0.00/0.42" — was run and **failed to kill it**, by the maximum possible margin. |
| **All-or-nothing spans.** After top-k, close gaps: if two selected positions lie within w, keep the positions between them, displacing the lowest-scoring isolated singletons. Budget-neutral. | U-shaped within-span survival in 6/6 cells (interior digit worst: 22/44/49%) — pooling cannot reach a token whose neighbourhood is all payload. Recovered digits track budget as a *prefix length* (B=16→1.9–2.7, B=32→3.89, B=64→4.00). | ~40 lines in `select_keep`; 2 cells ≈0.5 GPU-h | **Offline, no policy change:** extend `trace_span_positions.py` (already carries orig_ids) to report the longest contiguous surviving run starting at the payload's first digit, per stream, over the 12 needles at 128K/B=64. Correlate per-trial mean run length with digits recovered. Spearman \|ρ\| < 0.2 ⇒ contiguity is not the currency. ≈0.2 GPU-h |
| **Weight the score by value norm** — rank on `a_i·‖v_i‖`. | The program's only theoretical bound contains the value norm and the mechanism ignores it: ‖o − ô‖ ≤ 2(1−ρ)·max‖v_i‖. The FFN-side analogue was null (0.9581 vs 0.9586) but that is a different space. P1 rules out the *key*, not the class. | Offline probe reusing `store_probe.py` ≈0.1 GPU-h, then 2 cells ≈0.5 | On the 4K haystack compute attention relerr for top-B by `a` vs by `a·‖v‖` at B=32/64 against the calibration ladder. Improvement under ~0.003 relerr is below the smallest end-to-end effect ever measured here. **Mask sinks before weighting** or this rediscovers the pool-before-mask bug in a new place. |
| **Test the remaining cheap content features** — value vector, pre-RoPE key, residual-stream hidden state — in the harness P1 already built. | P1 tested only the *post-RoPE* key, which mixes content with position. Its own limitation section says the finding "concerns THIS feature set". Largest stated headroom in the program: +0.6056 to oracle. | Feature switch in `predict_future_attention.py`; ≈0.05–0.1 GPU-h for all three | **It is its own kill test, pre-declarable:** if no feature exceeds 0.25 recall (halfway between 0.156 and past-only's 0.394), content-based prediction is dead *as a class* and P1 strengthens from "the key" to "content". Add a second prompt and split before believing a **positive**; a negative is safe at n=1 given the margin. |
| **Make the 224 (layer, KV head) keep sets agree.** Within a layer, vote across its 8 KV heads; keep if ≥ m heads want it, same total bytes. | Retention/accuracy decoupling (178–201/224 vs 0.42–0.83) says per-stream retention is not the binding variable. Per-token survival ranges 19/224 to 160/224 within one 26-token span. Nothing in `select_keep` couples the 224 decisions. A shared per-layer index also makes the gather cheaper — relevant since top-k+gather ate most of the 15× bandwidth saving (1.78× at 16K). | ~20 lines; 2 floor cells ≈0.5 GPU-h + one rag_dev arm ≈0.5 | **Offline, before implementing:** dump the 224 keep sets from one 128K/B=64 run and compute union span-coverage of the payload at matched bytes under m = 2/4/6. If forced agreement reduces coverage at every m, coordination costs more than it buys. ≈0.15 GPU-h. Mask sinks out of the vote or it degenerates to keeping only sinks. |
| **Demote instead of evict:** B bf16 positions + a T-position int4 shadow tier gated by the existing score. At 28 × 1088 B/token, T=8,192 ≈ 238 MiB. | Mixed precision is measured and unexploited (0.00693 @ 3.04×, *beating* pure int8). Memory headroom is measured: the whole floor table peaks at 1,500–1,563 MiB. At B=64 the failure is "the payload's trailing positions are gone", and 238 MiB buys 32× more positions in int4 than in bf16. | Implementation ~1 day; measurement 2 cells ≈0.5 GPU-h | **Do not write cache code first.** Run 128K/B=8192 where all non-top-64 positions are round-tripped through int4 (K per-channel g=128, V per-token) once per round, 31 rounds. If accuracy falls below the B=64 bf16 baseline (0.42/0.00/0.42), the tier is worse than deletion. ≈0.3 GPU-h. **Design constraint:** requantisation error grows as √rounds (int8 0.00276→0.00637 over 5; int4 0.04583→0.10396), so the tier must quantise **once** and gather, never round-trip. |
| **Budget schedule and partition.** (a) *Probation*: a position is ineligible for eviction until observed ≥ 2 times — `_COUNTS` already tracks and carries this. (b) Ablate `recent` and `sinks` **at the floor**, where recent = B/4 is convention, not measurement. | The single largest loss is the **first** compression after arrival (26 span tokens → 18.7 kept; 224 → 205 streams), while the following 26 rounds cost 18.7 → 14.9. At B=64, 4 sinks + recent 16 leave only 44 heavy slots; recent=4 buys +27% heavy slots for nothing, and the last 16 tokens are the regenerated question suffix. `sinks=4` has never been swept despite 46.3% sink mass. | ~15 lines for probation; arms recent ∈ {B/4, B/8, 4} × sinks ∈ {4, 1} at 128K/B=64 ≈1.0 GPU-h | **Cheapest first:** re-run `trace_needle_eviction.py` at pool=7/B=64 (the existing trace is pool=1/B=2048). If the arrival round no longer dominates (<10% of the span), the front-loaded story was a pool=1 artefact. ≈0.15 GPU-h. For partition: recent=4 must move 128K/B=64 by ≥3 needles pooled over 36 needle-positions. Run probation at e=1 **and** e=0.5 — both advantage newcomers and may be redundant. |
| **Break the lexical coupling between query and evidence.** Three matched arms at one budget: (a) current; (b) paraphrase with zero token overlap between question and needle framing; (c) semantic-only. | P2.2 states the mechanism: the score rewards the label because the framing *is* the query (54% vs 35%, 6/6). Everything downstream — including 492× — rests on query-aware scoring. | ~20 lines in `passkey.py`; 3 arms × 1 cell at 128K/B=256 ≈0.75 GPU-h, +B=64 ≈0.25 | **Pre-declare:** if arm (b) at 128K/B=256 falls below 0.5 (against 1.00), "lossless at 492×" is a lexical-overlap result and must be reported as one. Run arm (b) alone first, 0.25 GPU-h. **Mandatory control:** an uncompressed or high-budget arm at the same wording — the program has already retracted one result for subtracting two differently-worded prompts. |
| **Can a compressed cache answer a *second* question?** Prefill a document with two needles, score and compress against question A, then query for B. | `score_with_query` compresses using exactly one question, and P2.2 shows the surviving set is lexically shaped by it. Every long-context number is one cache, one question. The cost has never been measured. | ~30 lines; 2×2 grid at 128K/B=256 ≈0.6 GPU-h | Run the grid where the same-question diagonal is known to be 1.00. **Pre-declare:** off-diagonal ≥0.9 ⇒ cache reuse is free, direction closes as a positive in 0.6 GPU-h; ≤0.2 ⇒ the mechanism is *query-conditional retrieval*, not cache compression, and every compression ratio in the paper needs that qualifier. Run the two-needle single-question control first. |

---

## 9. How to read the numbers

**The n=12 caveat.** Every depth of a cell draws the **same key sequence** — verified byte-identical across depths 0.1/0.5/0.9 in `pk_ladder_*.json`, `pk_128k_pool7*.json`, `pk_128k_grid12.json`, and all three absolute-budget files (`[75526, 85667, 99873, 42649, 66365, 90190, 96659, 66445, 70176, 92722, 60745, 84699]`). A "12-trial, 3-depth" cell is **12 independent needles observed three times**, not 36 observations. Consequences:

- A perfect cell supports a **one-sided 95% Clopper-Pearson lower bound of 0.7791** (= 0.05^(1/12)), against 0.9202 at n=36. "Lossless" means "no failure in 12 needles" — compatible with a true rate as low as ~0.78. *(The 0.903 sometimes quoted is the two-sided 2.5%-tail bound at n=36, which pairs with 0.7354 at n=12; either pairing leaves the point intact.)*
- Reaching 0.97 needs **99 needles per depth**, 0.99 needs **299** — roughly 100 minutes per cell at 128K. Not run.
- **Paired arms rescue this.** Exact McNemar / binomial on the honest unit is valid and often strong: pool=1 → pool=7 at 128K/B=2048 is 10 discordant pairs all one direction, one-sided p = 0.00098; e=0 → e=1 at pool=7/B=256 is +18/−0, p = 7.6e-06. `p = 2/2^d` means **d ≥ 6 discordant pairs** for p < 0.05, which is what set the cell size at 12.
- Never compare cells that share no needles. `scripts/design/paired_passkey.py` refuses to.

**Allocated vs reserved memory.** Three distinct quantities are routinely conflated:

| Quantity | What it is | Example |
|---|---|---|
| `peak_mib` in a manifest | `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` per trial (`passkey.py:240-241`) — allocator high-water mark | 1,536.8 MiB at 128K/B=256 |
| Resident KV cache | `(B + C) × 112 KiB` | 224 MiB at 128K/B=2048 — **~7.8× smaller than the 1,742.9 MiB peak** for the same run |
| Process VRAM occupancy | reserved bytes + CUDA context + compositor (430–1,340 MiB here) | never recorded |

`set_per_process_memory_fraction` bounds **reserved** bytes; `max_memory_allocated` reports **allocated** ones; the gap is allocator fragmentation (`torch.cat` on a growing cache keeps two buffers live). This is what produced the retracted claim "above 16K the full cache does not fit": at cap 6144 a 32K uncompressed run dies at a *reported* peak of 4,571 MiB (chunk 1024) or 4,897 MiB (chunk 4096); at cap 7800 both succeed at 1.00 with peaks 4,699 and 4,973 MiB. The binding constraint was a harness default. It does **not** extend to 64K — a cap-8100 run OOMs at peak 6,426 MiB. So: never read a `peak_mib` as "what the mechanism costs", never read a cap failure as a hardware verdict, and never quote a peak as a cache size.

**The file-collision hazard.** `pk_128k_pool7_B256.json` and `pk_128k_pool7_B256_rf.json` share length, budget and pool, differ only in `recent` (512 vs 64), and report **0.00 vs 1.00**. The cause is real: `Policy.n_recent()` clamps to `budget − sinks`, so recent=512 at B=512 or B=256 drives `heavy()` to zero and the policy silently becomes plain StreamingLLM. And `passkey.py` never gained recent-scaling code — the B/4 rule was applied by hand per invocation, so nothing in the harness enforces it.

Rules that follow:

1. **Always select cells by asserting `recent == budget // 4` from the manifest**, plus `pool`, `count_exponent` and `query_aware`. Never by filename. An earlier analysis picked the wrong twin exactly this way.
2. **Never cite anything under `_invalidated_*` or `_superseded_unmatched`.** A quarantined file holds `score_pct_of_cache = 0.781` at budget=1024 in two rows — the *same* value the published 0.78% overhead claim pins — so a path slip would have returned the right answer for the wrong reason and nothing would have looked wrong. `verify_claims.py` now raises on quarantined paths, on ambiguous multi-row matches, and on manifest fields failing a `require` assertion. 93/93 pins reproduce.
3. **Check `pool` before comparing anything long-context.** Everything on `main` above §4f is pool=1. The two lossless 12.5% cells are additionally chunk=2048 while the ladder rungs they are read against are chunk=4096 — not schedule-matched, on precisely the axis (compression-round count) that count normalisation is about.
4. **Nominal vs real length.** Keep fractions in the paper divide by the *requested* length (2048/131072 = 1.56%); manifests record `real_tokens` = 62,898 (64K) and 125,826 (128K), giving 1.63%. Compression ratios in the same documents use real tokens (492× = 125,826/256). `FINDINGS_P2.md` headlines the same cell as 512× (nominal). Pick one convention per comparison and say which; 492× is the conservative one.
5. **Check whether a row is measured or fitted.** In the decode speedup table only the 16,384 row is measured (82.573 vs 46.494 ms = 1.78×); 32K/64K/128K are read off a hinge fit whose slope was measured over 4,096→16,384 tokens and extended 8× at 128K. The predicted column also mixes denominators (measured row uses the scoring-on 46.494 ms arm; predicted rows use the 34.7 ms no-score floor, which is the configuration behind every accuracy number). The program's own rule — "no throughput claim below 2× is defensible here" given a 2.17× thermal band — makes the one measured 1.78× marginal, and its arms ran at 79→86 °C vs 81→81 °C.
6. **Known stale-in-place figures on `main`:** `FINDINGS_KV_PROGRAM.md` §4e says 48.6 ms / "+40%" for the 128K B=8192 operating point; `probe_decode_floor.json` measures **45.252 ms / +30%**. `PAPER_DRAFT_SIGNIFICANCE.md` §6 and §8 say "74 pinned numbers" while Appendix A says 85 and the checker now holds **93**.