# Where to start

A routing map. There are ~50 markdown files in this tree and most of them belong
to datasets; this lists the ones that describe the *work*, says what each is for,
and flags which have been superseded.

**If you are checking program history or experiment status:**
[`EXPERIMENT_PROGRAM_AUDIT.md`](EXPERIMENT_PROGRAM_AUDIT.md) is the canonical
single source of truth for track chronology, lineage, and whether work is established,
negative, withdrawn, synthetic-only, or built-but-unrun. Historical branch names are
provenance labels; at the 2026-08-30 audit snapshot the remote exposes only `main`.
Exact measurements remain authoritative in their committed result files and in the
specialized findings documents checked by `scripts/design/verify_claims.py`.

**If you read one thing:** [`PAPER_DRAFT_CORNERSTONE2.md`](PAPER_DRAFT_CORNERSTONE2.md) —
the submission draft. It opens with a plain-language summary, states every
limitation, and reproduces from committed data via Appendix C.

**If you want the raw record instead:**
[`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) — the consolidated record of
the KV-compression and long-context program, with every statistic, the procedure
that reproduces it, and every retraction stated rather than overwritten.

---

## The program

| Document | What it is | Status |
|---|---|---|
| [`EXPERIMENT_PROGRAM_AUDIT.md`](EXPERIMENT_PROGRAM_AUDIT.md) | **Canonical experiment-history/status index.** Timeline, track lineage, positives, negatives, withdrawals, built-but-unrun work, archival branch handling, and the update contract future history checkers must follow. | **current — history SSOT** |
| [`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md) | The objective and its workstreams — move the capability-per-parameter frontier. Governing principles P1–P8 are here and are worth reading first; each one is a scar. | current |
| [`MINI_MODEL_ROADMAP.md`](MINI_MODEL_ROADMAP.md) | Execution log for the vocabulary trim, the size ladder and the healing harness. R2's provenance. | current |
| [`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) | **Consolidated findings** for KV compression and long context. Mechanism spec, all measurements, all nine bugs, all retractions, and an explicit "not established" list. | current |
| [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) | Things that were tried and do not work, with the measurement that killed each. | current |
| [`PAPER_DRAFT_CORNERSTONE2.md`](PAPER_DRAFT_CORNERSTONE2.md) | **The submission draft, and the thing to read if you read one paper.** Span-aware KV eviction: why the unit is wrong, why the pooling window must reach backward only, why its reach must be derived rather than tuned, what it costs, and what can be deployed. Opens with a plain-language summary; Appendix C reproduces every figure. | **current** |
| [`PAPER_DRAFT_SIGNIFICANCE.md`](PAPER_DRAFT_SIGNIFICANCE.md) | Paper-shaped draft scoped to the significance matrix as the single contribution: abstract, related work, method, experiments, limitations, retraction table. | current |
| [`EXPERIMENT_DESIGN_STABILIZER.md`](EXPERIMENT_DESIGN_STABILIZER.md) | Pre-registration for the **stabilizer matrix**: a persistent global projector that selects which FFN neurons fire. **Phase 0 run and closed** — the predictor works (0.70 recall) but the significance matrix adds +0.0002 and the speed claim fails on this host, so the hypothesis was abandoned. What the branch produced instead is **cornerstone 2**: span-aware selection over the significance scores, in [`experiments/stabilizer/results/FINDINGS_P2.md`](../experiments/stabilizer/results/FINDINGS_P2.md). | **closed — negative** |
| [`EXPERIMENT_DESIGN_KV_LONGCTX.md`](EXPERIMENT_DESIGN_KV_LONGCTX.md) | Pre-registration for the KV-head × long-context study. Written before any result, deliberately not revised into agreement with them. | **§2a superseded** — the asset gap it describes has since been partly filled |

## The experiments

| Directory | Question | Verdict |
|---|---|---|
| [`../experiments/attention_shape/`](../experiments/attention_shape/README.md) | Window schedule or KV-head count — which is the better axis at matched cache? | **Closed.** KV-head count. The window axis raises peak memory, shortens reachable context, and collapses below the window. |
| [`../experiments/kv_significance/`](../experiments/kv_significance/README.md) | Does evicting KV by accumulated attention beat evicting by position? | **Yes**, at keep ≥ 0.5. Its README/THEORY stop before chunked prefill and long context — both point forward to `FINDINGS_KV_PROGRAM.md`. |
| [`../experiments/stabilizer/`](../experiments/stabilizer/README.md) | Can a global FFN selector exploit contextual sparsity, and what does its failure teach the KV program? | **Selector line closed negative; C2 positive.** The original speed/memory system fails; the same directory contains the span-aware Cornerstone-2 follow-on. |
| [`../experiments/stabilizer/peer_optimizer/`](../experiments/stabilizer/peer_optimizer/README.md) | Can peer layers identify useful sparse native-weight plasticity? | **Synthetic feasibility only.** Phase C real-gradient causal harness is executable on `main`, but no Rineya PCSM quality result is claimed yet. |
| `../experiments/geometric_significance/` | Does diversity-aware selection beat top-k? *(historical branch `exp/geometric-significance`; ref no longer live)* | **No.** Killed by two probes in four minutes of GPU; result preserved in `NEGATIVE_RESULTS.md`. |

## The code

| Path | What it does |
|---|---|
| `scripts/kv/significance.py` | The working mechanism. Significance-gated KV eviction: per-position, per-KV-head accumulated attention mass; keep = sinks + recent + top-B. Zero added parameters. |
| `scripts/kv/chunked_prefill.py` | Chunked prefill with inter-chunk compression, so peak cache is `B + C` with prompt length eliminated. Also `score_with_query`, which makes eviction query-aware. |
| `experiments/stabilizer/src/trace_needle_eviction.py` | Carries original position identity through every gather, so you can ask whether a specific span survived compression and how much of it. This is what showed the needle was fragmented rather than evicted. |
| `experiments/kv_significance/src/measure_query_prefill.py` | **What the query-aware prefill pass costs.** CPU-only by design — it asserts `torch.cuda.is_initialized()` is False before writing output, so a run that accidentally reaches the card fails rather than reporting contended numbers. Fits t(T) = a + bT to show the pass is dominated by a fixed weight-streaming floor, then self-instruments a real prefill so the tax needs no matched-arm assumption. |
| `experiments/kv_significance/src/measure_select.py` | Per-compression selection cost across pooling variants, with no model at all — it times the one step where the variants differ, on tensors of the real shape, in a few hundred MB. |
| `scripts/design/figure_query_cost.py` | Figure 2, generated from the same JSON `verify_claims.py` reads, so it cannot drift from the data behind it. |
| `experiments/kv_significance/src/measure_decode.py` | Decode throughput against an uncompressed baseline, three arms, arms rotated per round and minimum-of-rounds reported because this host's thermal band is 2.17×. |
| `experiments/kv_significance/src/probe_decode_floor.py` | What actually sets decode latency. Sweeps cache size with nothing else varying and fits a hinge; answers whether the runtime or memory traffic is the limit. |
| `experiments/kv_significance/src/store_probe.py` | What to STORE each kept position as — quantisation, low-rank, cross-layer sharing — measured against three eviction settings whose end-to-end cost is known, so the relative errors have a scale. |
| `experiments/kv_significance/src/mixed_precision.py` | Significance-gated mixed precision: bf16 sinks, int8 for the top-p by score, int4 for the rest. Lower error than pure int8 at 1.6× more compression. |
| `scripts/design/kv_budget.py` | Feasibility and the iso-memory frontier, from a checkpoint's own config. |
| `scripts/design/power_analysis.py` | Minimum detectable effect for the two units of replication (item and training run). |
| `scripts/design/verify_claims.py` | **Run this after editing any doc that states pinned C1/C2 numbers.** Cross-checks headline numbers against committed data *and* against the prose. Non-zero exit on drift. |
| `scripts/design/identity_counts.py` | Derives byte-identical reply counts into a small **committed** artifact, because the per-case `results.jsonl` files are gitignored (132 MB) and every `N/752` figure otherwise reproduces only on the machine that made it. `--check` re-derives and reports drift. |
| `scripts/design/paired_passkey.py` | **Run this instead of computing a p-value by hand.** Exact McNemar on discordant needles, within-needle permutation trend test, and a refusal to compare cells that share no needles. Added because the hand-rolled version was wrong. |

Nothing in `scripts/train/` is touched by this work: `code_fingerprint()` hashes
every `.py` there and an edit invalidates every checkpoint's resume path.

---

## Reproducing the headline results

```bash
python scripts/kv/chunked_prefill.py --gate          # correctness, CPU/GPU, ~1 min
python scripts/design/verify_claims.py               # docs vs data, CPU, instant
python scripts/design/identity_counts.py --check     # identity artifact vs raw records
python experiments/kv_significance/src/verify_significance.py   # 6 checks
python experiments/kv_significance/src/measure_chunked.py --lengths 32768 131072
python experiments/kv_significance/src/store_probe.py           # ~4 min GPU
python experiments/kv_significance/src/mixed_precision.py       # ~3 min GPU
python experiments/kv_significance/src/passkey.py --lengths 131072 \
       --budgets 2048 8192 --depths 0.1 0.5 0.9 --trials 12 \
       --yarn 4.0 --query-aware --count-exponent 1.0 --cap-mib 7400 \
       --out experiments/kv_significance/results/pk_128k_grid12.json
python scripts/design/paired_passkey.py \
       experiments/kv_significance/results/pk_128k_grid12.json --by budget
```

Two flags in that command are not decoration. `--cap-mib 7400` overrides a 6144
default that bounds *reserved* memory and silently blocked the **32K** uncompressed
reference for most of this program. It is not what blocks 64K: there the cache
alone is 7,168 MiB against a card of 8,188, so no cap setting reaches it. `--trials 12` is a power choice, not
a minimum: exact McNemar gives p = 2/2^d on d discordant needles, so reaching 0.05
needs d ≥ 6, and a 6-needle cell only gets there if every needle flips — see
[`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) §4c.

Use the project venv (`.venv/Scripts/python.exe`); the default `python` on PATH
is a CPU-only torch build and every CUDA measurement taken with it is void.

---

## Three things a newcomer should know before trusting any number here

**The mature extreme-length headline is retrieval, not general reasoning.** The C1/C2
128K headline is a five-digit key in repetitive filler, deliberately isolating span
retrieval/preservation. A later controlled program **did** run distractor and 2-/3-hop
chains: the score advantage is detectable at two hops in the valid matched analysis,
not detectable at three hops, and moving to much longer context breaks the chain largely
independent of the eviction policy. See
[`EXPERIMENT_PROGRAM_AUDIT.md`](EXPERIMENT_PROGRAM_AUDIT.md) and
[`PLAN_LONG_REASONING.md`](PLAN_LONG_REASONING.md). The repository therefore still
supports **no broad long-context reasoning claim** from the passkey compression ratio.

> **The table below is the `pool=1` record and is superseded.** Span pooling was
> off by default in every early long-context run; turning it on (`--pool 7`) made
> 128K lossless at 1.56% keep rather than 12.5%. Cornerstone 2 then went much
> further by making the pooling window **backward-only** — 128K is lossless at a
> **16-position budget**, of which eight are score-chosen, certified at n = 100
> with a 0.9705 lower bound. See
> [`PAPER_DRAFT_CORNERSTONE2.md`](PAPER_DRAFT_CORNERSTONE2.md) §4.7 and §4.11.
> The table is kept because it is the matched-length evidence that keep *fraction*,
> not length, is the governing variable for that earlier `pool=1` configuration.

**"Works at 128K" needs a budget attached.** Uncompressed, this checkpoint is
1.00 at every depth to **32K** — the largest reference this card can hold, since
the 64K cache alone is 7,168 MiB, putting the run at 8,072 MiB required against
~7,758 MiB available on an 8,188 MiB card. Compressed, what matters in the earlier
C1 ladder is the keep *fraction*:

| n | budget | keep | accuracy | peak |
|---:|---:|---:|---:|---:|
| 32,768 | 0 (no eviction) | 100% | 0.97 | 4699 MiB |
| 32,768 | 2048 | 6.25% | 0.78 | 1742 MiB |
| 65,536 | 2048 | 3.12% | 0.81 | 1742 MiB |
| 65,536 | 8192 | 12.5% | **1.00** | 2196 MiB |
| 131,072 | 2048 | 1.56% | 0.53 | 1743 MiB |
| 131,072 | 8192 | 6.25% | 0.92 | 2521 MiB |
| 131,072 | 16384 | 12.5% | **1.00** | 3202 MiB |

At a constant 6.25% keep there is no systematic degradation with length: 0.78 at
32K, 0.97 at 64K, 0.92 at 128K — the shortest context is the *lowest* of the
three, so the spread is sampling at 12 needles rather than a length effect. The
32K rows also give the only matched eviction cost in that C1 ladder: **0.19**
against an uncompressed control run with the same RoPE scaling. The "128K is weak"
reading that ran through this program for a while was about B=2048, which is 1.56%
keep at that length — a different experiment wearing the same label. C2 later
supersedes the operating point by changing the retention unit, so use the audit for
current status rather than treating this historical table as the final configuration.

**This host fails silently in four specific ways**, all of which have produced
wrong published numbers at least once:

- `enable_gqa=True` falls back to the SDPA math kernel — 16 GiB against 224 MiB
  at 16K. It surfaces as an OOM at a plausible length, reading as "the card is
  too small". Workaround: `experiments/attention_shape/src/compat.py`.
- Windows/WDDM does not OOM when VRAM runs out; it spills to host RAM and the run
  goes quiet at 0% GPU for minutes. Always set an allocator cap.
- Processes that hit CUDA OOM may not exit. One pair held ~2.3 GiB of GPU and a
  full core for ~27 minutes after its run had finished (an operator observation
  from `nvidia-smi` during the session; no artefact records it). Verify a kill
  against system state, not against its return code (P3).
- `set_per_process_memory_fraction` caps **reserved** memory; `max_memory_allocated`
  reports **allocated**. A run OOMs with a peak far below its own cap — 4571 MiB
  against a 6144 MiB cap — because the caching allocator holds fragmentation the
  peak figure never shows. Growing a KV cache with `torch.cat` makes that gap
  large. This kept the 32K uncompressed reference out of reach for the whole
  program until it was tested directly.
