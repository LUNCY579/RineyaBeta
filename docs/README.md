# Where to start

A routing map. There are ~50 markdown files in this tree and most of them belong
to datasets; this lists the ones that describe the *work*, says what each is for,
and flags which have been superseded.

**If you read one thing:** [`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) —
the consolidated record of the KV-compression and long-context program, with
every statistic, the procedure that reproduces it, and every retraction stated
rather than overwritten.

---

## The program

| Document | What it is | Status |
|---|---|---|
| [`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md) | The objective and its workstreams — move the capability-per-parameter frontier. Governing principles P1–P8 are here and are worth reading first; each one is a scar. | current |
| [`MINI_MODEL_ROADMAP.md`](MINI_MODEL_ROADMAP.md) | Execution log for the vocabulary trim, the size ladder and the healing harness. R2's provenance. | current |
| [`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) | **Consolidated findings** for KV compression and long context. Mechanism spec, all measurements, all nine bugs, all retractions, and an explicit "not established" list. | current |
| [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) | Things that were tried and do not work, with the measurement that killed each. | current |
| [`EXPERIMENT_DESIGN_KV_LONGCTX.md`](EXPERIMENT_DESIGN_KV_LONGCTX.md) | Pre-registration for the KV-head × long-context study. Written before any result, deliberately not revised into agreement with them. | **§2a superseded** — the asset gap it describes has since been partly filled |

## The experiments

| Directory | Question | Verdict |
|---|---|---|
| [`../experiments/attention_shape/`](../experiments/attention_shape/README.md) | Window schedule or KV-head count — which is the better axis at matched cache? | **Closed.** KV-head count. The window axis raises peak memory, shortens reachable context, and collapses below the window. |
| [`../experiments/kv_significance/`](../experiments/kv_significance/README.md) | Does evicting KV by accumulated attention beat evicting by position? | **Yes**, at keep ≥ 0.5. Its README/THEORY stop before chunked prefill and long context — both point forward to `FINDINGS_KV_PROGRAM.md`. |
| `../experiments/geometric_significance/` | Does diversity-aware selection beat top-k? *(branch `exp/geometric-significance`, unmerged)* | **No.** Killed by two probes in four minutes of GPU. |

## The code

| Path | What it does |
|---|---|
| `scripts/kv/significance.py` | The working mechanism. Significance-gated KV eviction: per-position, per-KV-head accumulated attention mass; keep = sinks + recent + top-B. Zero added parameters. |
| `scripts/kv/chunked_prefill.py` | Chunked prefill with inter-chunk compression, so peak cache is `B + C` with prompt length eliminated. Also `score_with_query`, which makes eviction query-aware. |
| `experiments/kv_significance/src/measure_decode.py` | Decode throughput against an uncompressed baseline, three arms, arms rotated per round and minimum-of-rounds reported because this host's thermal band is 2.17×. |
| `experiments/kv_significance/src/probe_decode_floor.py` | What actually sets decode latency. Sweeps cache size with nothing else varying and fits a hinge; answers whether the runtime or memory traffic is the limit. |
| `experiments/kv_significance/src/store_probe.py` | What to STORE each kept position as — quantisation, low-rank, cross-layer sharing — measured against three eviction settings whose end-to-end cost is known, so the relative errors have a scale. |
| `experiments/kv_significance/src/mixed_precision.py` | Significance-gated mixed precision: bf16 sinks, int8 for the top-p by score, int4 for the rest. Lower error than pure int8 at 1.6× more compression. |
| `scripts/design/kv_budget.py` | Feasibility and the iso-memory frontier, from a checkpoint's own config. |
| `scripts/design/power_analysis.py` | Minimum detectable effect for the two units of replication (item and training run). |
| `scripts/design/verify_claims.py` | **Run this after editing any doc.** Cross-checks headline numbers against committed data *and* against the prose. Non-zero exit on drift. |
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
default that bounds *reserved* memory and silently blocked every uncompressed
reference above 16K for most of this program. `--trials 12` is a power choice, not
a minimum: exact McNemar gives p = 2/2^d on d discordant needles, so reaching 0.05
needs d ≥ 6, and a 6-needle cell only gets there if every needle flips — see
[`FINDINGS_KV_PROGRAM.md`](FINDINGS_KV_PROGRAM.md) §4c.

Use the project venv (`.venv/Scripts/python.exe`); the default `python` on PATH
is a CPU-only torch build and every CUDA measurement taken with it is void.

---

## Three things a newcomer should know before trusting any number here

**Everything at length is retrieval.** A 5-digit key in repetitive filler, which
deliberately isolates retrieval from comprehension. Multi-hop reasoning at length
has never been measured, and the frozen demo suite indicates that is where this
checkpoint is genuinely weak.

**"Works at 128K" needs a budget attached.** Uncompressed, this checkpoint is
1.00 at every depth to **32K** — the largest reference this card can hold, since
the 64K cache alone is 7,168 MiB, putting the run at 8,072 MiB required against
~7,758 MiB available on an 8,188 MiB card. Compressed, what matters is
the keep *fraction*, not the length:

| n | budget | keep | accuracy | peak |
|---:|---:|---:|---:|---:|
| 32,768 | 2048 | 6.25% | 0.89 | 1744 MiB |
| 65,536 | 2048 | 3.12% | 0.81 | 1742 MiB |
| 131,072 | 2048 | 1.56% | 0.53 | 1743 MiB |
| 131,072 | 8192 | 6.25% | **0.92** | 2521 MiB |

At a constant 6.25% keep the accuracy is flat from 32K to 128K. The "128K is weak"
reading that ran through this program for a while was about B=2048, which is 1.6%
keep at that length — a different experiment wearing the same label.

**This host fails silently in four specific ways**, all of which have produced
wrong published numbers at least once:

- `enable_gqa=True` falls back to the SDPA math kernel — 16 GiB against 224 MiB
  at 16K. It surfaces as an OOM at a plausible length, reading as "the card is
  too small". Workaround: `experiments/attention_shape/src/compat.py`.
- Windows/WDDM does not OOM when VRAM runs out; it spills to host RAM and the run
  goes quiet at 0% GPU for minutes. Always set an allocator cap.
- Processes that hit CUDA OOM may not exit. One pair held 2.3 GiB of GPU and a
  full core for 27 minutes after its run had finished. Verify a kill against
  system state, not against its return code (P3).
- `set_per_process_memory_fraction` caps **reserved** memory; `max_memory_allocated`
  reports **allocated**. A run OOMs with a peak far below its own cap — 4571 MiB
  against a 6144 MiB cap — because the caching allocator holds fragmentation the
  peak figure never shows. Growing a KV cache with `torch.cat` makes that gap
  large. This kept the 32K uncompressed reference out of reach for the whole
  program until it was tested directly.
