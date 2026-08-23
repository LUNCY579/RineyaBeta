# Model compression: what has been measured

One scoreboard for every compression axis touched so far, with verdicts and the
evidence behind them. This is the `I6` deliverable from
[`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md) — without it the Pareto claim cannot be
checked, because the findings otherwise live scattered across
[`MINI_MODEL_ROADMAP.md`](MINI_MODEL_ROADMAP.md) §1–18, three manifests, and two
experiment directories.

Reference model: **Qwen3-0.6B, 596,049,920 resident params**.

---

## 1. The frontier

| Artifact | Params | vs stock | chat DEV | RAG DEV | Status |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B stock | 596,049,920 | — | 0.8334 | 0.6809 | reference |
| **R2** (vocab 32,768) | **474,021,888** | **−20.5%** | 0.8336 | 0.6682 | measured |
| **R2 + B0** (LoRA heal) | 474,021,888 | −20.5% | **0.8607** | **0.8306** | measured |
| r2-gqa4kv | 444,661,760 | −25.4% | — | **0.0047** | **collapsed, un-healed** |
| r2-gqa2kv | 429,981,696 | −27.9% | — | — | not run (see §2.2) |
| r2-gqa1kv | 422,641,664 | −29.1% | — | — | not run (see §2.2) |
| retired 19L soup | ~461M | −22.6% | — | — | degenerate, rejected |

**R2+B0 is the current Pareto point: 474M beating stock's 596M on both gates**
(chat +0.027, RAG +0.150). Stated precisely, because it is easy to overclaim: stock
carries no adapter, so the adapter does that work, not the compression. It is a real
deployment result and **not** evidence that trimming helped. The controlled version
needs stock+B0, which is unrun.

All scores come from the vendored v4.5 evaluator, which reproduces the upstream
numbers at max |Δ| = 0.000e+00 on 510 stored replies — so these are comparable with
every prior Rineya artifact, not just with each other.

---

## 2. Axis by axis

### 2.1 Vocabulary — **works, −20.5%, essentially free**

Qwen3's embedding is 26.1% of the model and the corpora use **7,609 of 151,936 ids
(5.01%)**. Trimming to 32,768 by *merge rank* (BPE merge order is a frequency
ranking learned from Qwen's own corpus) saves 122,028,032 params.

Selection rule mattered more than size. At a matched vocabulary, ranking by token
*length* inflated held-out English prose by **+25.5%**, past its own compute
break-even; ranking by merge rank inflated it **+7.2%** and in-domain text **+1.1%**.

| | stock 596M | R2 474M | paired delta |
|---|---:|---:|---:|
| chat DEV | 0.8334 | 0.8336 | **+0.0002**, 95% CI [−0.008, +0.008] |
| RAG DEV | 0.6809 | 0.6682 | **−0.0127**, 95% CI [−0.020, −0.006] |

Chat is indistinguishable from zero (45 wins / 48 losses). RAG is a **real** effect
(0 wins / 14 losses) and localises to one thing: **9 lost abstentions in
`conflict`**, with 11 of 16 categories bit-identical. Mechanism tested — the entity
names and the abstention phrase tokenize *identically*; the trim shifts positions
~2% and the most borderline category flips. **B0 healing recovered all of it** and
then some (179/202 → 202/202 correct abstentions).

Also: −14.7% compute, and it is what made 4k-context training feasible at all on
8 GB (logits tensor 4.98 GB → 1.07 GB).

### 2.2 KV heads — **measured: catastrophic without healing**

`artifacts/attn_shape/`, produced outside this line of work. Verified by
least-squares as **contiguous-group mean-pooling** (pooling matrix recovered exactly,
rows `[0.5, 0.5, 0, …]`, residual 4.9e-04 = bf16 rounding) — the principled method,
not truncation.

| kv heads | params | vs stock | KV cache/token | RAG DEV |
|---:|---:|---:|---:|---:|
| 8 (R2) | 474,021,888 | −20.5% | 114,688 B | 0.6682 |
| **4** | 444,661,760 | −25.4% | 57,344 B (50%) | **0.0047** |
| 2 | 429,981,696 | −27.9% | 28,672 B (25%) | not run |
| 1 | 422,641,664 | −29.1% | 14,336 B (12.5%) | not run |

**8 → 4 KV heads, un-healed, destroys the model.** 752 cases:
`mean_quality_score` **0.0047** against R2's 0.6682, `intelligibility_pass_rate`
**0.0652**, `contract_violation_case_rate` 0.9734. Not degradation — collapse:

| gold | R2 | gqa4kv |
|---|---|---|
| `June 9, 2032.` | `June 9, 2032.` | `19761.299_ACTEMPLATE
ABO
EPICES…` |
| `103 GB.` | `103 GB.` | `19241.199992*19	type…` |
| `Wi-Fi 7.` | `Wi-Fi 7.` | `1996EJ
EPONJESPECT
19679661E…` |

Token salad. Only 1 empty reply of 752 and 30.3 mean visible tokens, so it generates
fluently and produces nonsense — a real model failure, not a harness artifact
(tokenizer sha verified identical to R2's beforehand).

**The 2kv and 1kv runs were stopped rather than completed.** They are strictly more
aggressive and quality is already at 0.0047 with no headroom to distinguish, so
~1.5 h of shared CPU would have bought nothing. Recorded as a deliberate stop, not
an omission.

**Why this axis behaves so differently from the vocabulary trim.** Removing
never-addressed embedding rows changes nothing the forward pass reads — logits came
out bitwise identical. Averaging K and V breaks the learned correspondence between
all 16 query heads and their key/value groups, simultaneously, in every layer. The
GQA literature does mean-pooling **followed by uptraining** at roughly 5% of original
pretraining compute; the uptraining is not polish, it is half the method.

So these three are **not compressed models — they are initialisations for
uptraining**, and the axis is unresolved rather than dead. The 12.5% cache figure at
1 KV head remains the largest deployment prize on the board, and it now has a
measured price: a real training run, not a surgical pass.

**A prediction of mine that was wrong, instructively.** I expected "in-context
grounding degrades first", reasoning from retrieval heads. Nothing degrades
selectively — everything collapses at once. The per-category gate that localised the
vocabulary trim's damage to nine `conflict` abstentions has nothing to localise here.

### 2.3 Low-rank factorization of weights — **rejected**

`scripts/measure/weight_spectra.py`. Factorizing `W (d×k)` into `U(d×r)V(r×k)` only
saves below a break-even rank. Measured effective rank exceeds it **everywhere**:

| projection | break-even | rank @99% | saving |
|---|---:|---:|---:|
| q_proj | 683 | 852 | −0.52M |
| k_proj | 512 | 694 | −0.37M |
| v_proj | 512 | 737 | −0.46M |
| o_proj | 683 | 895 | −0.65M |
| gate/up/down | 768 | 935–949 | −0.70M |

At 99% energy the whole model would **cost 116M more** (+24.5%). Weights are near
full rank (657–960 of 1024). A ~4-minute CPU measurement removed the need to build
`surgery.factorize`.

Two structural facts survive: `k_proj`/`v_proj` are the most compressible matrices,
and effective rank falls with depth — both pointing where `attn_shape` already went.

### 2.4 Stabilizer matrix as a compression anchor — **rejected**

Full record in
[`experiments/stabilizer/results/FINDINGS_P0_4.md`](../experiments/stabilizer/results/FINDINGS_P0_4.md).

The parameter case was the best on the board — 25% FFN keep plus a rank-64
correction would save **194.5M**, taking R2 to ~279.5M, larger than the vocabulary
trim. It failed on quality at its own oracle ceiling: the discarded contribution
needs rank **≥149 and rising with corpus size**, against a threshold of 128 declared
in advance.

All four Phase 0 probes fail for **one** reason: the FFN concentration is real
(ρ(25%) = 0.9586) but **per-token**, and compression requires input-independence.
A gather costs more than the narrower matmul (0.84×); the union of per-token top-k
sets dwarfs any single one (ρ 0.9586 → 0.7468); different tokens discard different
directions; and the variation is token-driven rather than layer-driven, so a
per-layer rank schedule buys only 2.7–4.7%.

**What it bought:** a static 25% FFN keep costs **53% relative output error**. That
number did not exist before and it calibrates §2.5 directly.

### 2.5 Width (MLP) — **not yet attempted, now calibrated**

MLP is 44.3% of params; `intermediate_size` 3072 → 2048 saves 3,145,728/layer
(88.1M over 28). §2.4's 53%-error-at-25%-keep says **25% is far too aggressive**; the
sane version lives at much higher keep ratios with healing, using activation-aware
importance rather than magnitude.

### 2.6 Depth — **measured, not yet cut**

`layer_profile.py` on R2: drop `25..26` (angular distance 0.1203) → 442,560,000
(−25.8%). The retired 19L soup is the cautionary case: 9 of 28 layers, unhealed,
produced `imony` repetition loops and scored 0/4. Its layer *selection* was also
wrong — the deleted run {6,7,8,9,10} ranked 15–20 of 28 for prunability, i.e. the
*less* prunable half.

### 2.7 Quantization — **deliberately out of scope**

4-bit reduces bytes, not parameters, so it is tracked as a separate deployment
artifact to keep the parameter claim clean. R2 at 4-bit is roughly 170 MB.

---

## 3. What the measurements say in general

**Free compression comes from what the model never uses, not from what it uses
lightly.** The vocabulary trim worked because 95% of embedding rows are literally
never addressed. Every axis that instead tried to exploit *lightly used* structure —
low-rank weights, per-token FFN concentration — failed, because "lightly used" still
means used, and the usage is input-dependent.

**Aggregates hide the only thing that matters.** The trim's RAG cost read as diffuse
noise at −0.0127. Per category it was 9 abstentions in `conflict`, with 11 of 16
categories bit-identical. A gate on the mean would have passed it silently, and a
gate on the mean would also have missed that healing fixed exactly that.

**Healing is stronger than expected, which changes what is worth trying.** B0 took
`conflict` from 0.3948 to 0.8800 and abstention from 88.6% to **100%** — above
stock's 93.1%. Contract violations went 19.7% → 0. That raises the acceptable
surgical damage, and it is why "rank 256 plus healing" (7.5% per-layer error, saves
183.5M) is the one stabilizer variant still worth a run rather than a refutation.

**Cheap measurements keep killing expensive builds.** A 4-minute SVD sweep removed
`surgery.factorize`. A 10-minute tokenization check killed the leading explanation
for the abstention loss and saved a 3-hour run from answering the wrong question. A
CPU probe closed a compression axis whose parameter case looked better than
anything else available.

---

## 4. Honest gaps

1. **Seed variance is unmeasured (M1).** Every comparison here is single-run against
   single-run. Discrete counts (abstentions recovered: 9, 14) and effects with CIs
   excluding zero are safe. Any aggregate delta under ~0.02 is **not yet
   interpretable**, including B0's +0.027 on chat.
2. **The frontier claim rests on an un-adaptered baseline.** stock+B0 is unrun.
3. **The KV axis needs a training run, not a gate.** gqa4kv measured at 0.0047
   un-healed (§2.2); 2kv and 1kv were stopped as foregone. Whether uptraining
   recovers them is the open question, and it needs the GPU.
4. **Only 2 of 5 available eval sets are wired.** Puzzle validation (4,080), LCR-SFT
   dev (800) and LongContextDemo (256, frozen) are imported but unused, so every
   verdict above rests on two benchmarks.
5. **The environment is not reproducible from the repo** (I1) — no lockfile.
