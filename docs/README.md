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
| `scripts/design/kv_budget.py` | Feasibility and the iso-memory frontier, from a checkpoint's own config. |
| `scripts/design/power_analysis.py` | Minimum detectable effect for the two units of replication (item and training run). |
| `scripts/design/verify_claims.py` | **Run this after editing any doc.** Cross-checks headline numbers against committed data *and* against the prose. Non-zero exit on drift. |

Nothing in `scripts/train/` is touched by this work: `code_fingerprint()` hashes
every `.py` there and an edit invalidates every checkpoint's resume path.

---

## Reproducing the headline results

```bash
python scripts/kv/chunked_prefill.py --gate          # correctness, CPU/GPU, ~1 min
python scripts/design/verify_claims.py               # docs vs data, CPU, instant
python experiments/kv_significance/src/verify_significance.py   # 6 checks
python experiments/kv_significance/src/measure_chunked.py --lengths 32768 131072
python experiments/kv_significance/src/passkey.py --lengths 32768 --budgets 2048 \
       --yarn 4.0 --query-aware
```

Use the project venv (`.venv/Scripts/python.exe`); the default `python` on PATH
is a CPU-only torch build and every CUDA measurement taken with it is void.

---

## Two things a newcomer should know before trusting any number here

**Everything at length is retrieval.** Passkey works to 128K. Multi-hop reasoning
at length has never been measured, and the frozen demo suite indicates that is
where this checkpoint is genuinely weak.

**This host fails silently in three specific ways**, all of which have produced
wrong published numbers at least once:

- `enable_gqa=True` falls back to the SDPA math kernel — 16 GiB against 224 MiB
  at 16K. It surfaces as an OOM at a plausible length, reading as "the card is
  too small". Workaround: `experiments/attention_shape/src/compat.py`.
- Windows/WDDM does not OOM when VRAM runs out; it spills to host RAM and the run
  goes quiet at 0% GPU for minutes. Always set an allocator cap.
- Processes that hit CUDA OOM may not exit. One pair held 2.3 GiB of GPU and a
  full core for 27 minutes after its run had finished. Verify a kill against
  system state, not against its return code (P3).
