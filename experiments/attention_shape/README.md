# Attention shape: is the window schedule or the KV-head count the better axis?

Self-contained experiment. Nothing here is imported by the rest of the repo, and
nothing here writes outside `experiments/attention_shape/` and
`artifacts/attn_shape/`, so it cannot disturb the healing runs or the gate
baselines.

**In particular it does not touch `scripts/train/`.** `code_fingerprint()` hashes
every `.py` in that directory, and an edit there invalidates the resume path of
every existing checkpoint. The one behaviour change this experiment needs
(`src/compat.py`) is a process-local monkey patch, not an edit to shared code.

---

## The question

R2 spends **112 KiB per token** on KV cache — 14.0 GiB at 128K, on an 8 GiB card.
There is more than one way to spend less, and they are not equivalent:

- **fewer KV heads** (GQA 8 → 4 → 2 → 1) — irreversible weight surgery, needs a
  healing run per rung, and capacity falls with the head count until at MQA all
  16 query heads read one shared key subspace;
- **a sliding window on most layers** — a *config change*, no weight edit at all,
  and its saving grows with context length instead of being a constant factor.

So: at matched cache, which buys more quality — and at what cost to get there?
The answer decides whether the multi-axis framing pays before anyone writes the
MLA module (`THEORY.md` §3).

---

## What is here

```
README.md                     this file
THEORY.md                     the maths: MHA redundancy, RoPE, MLA, predictions
run_quality_sweep.sh          the eval plan, in dependency order
figures/
  01_attention_variants.svg   MHA / GQA / MQA / MLA — what each stores
  02_mla_vs_qwen_dataflow.svg Qwen3's layer vs an MLA layer + absorption identity
  03_cache_growth.svg         cache vs length — which curves bend    [generated]
  04_iso_cache.svg            context reachable at one budget        [generated]
  05_measured_vs_predicted.svg  cache vs PEAK — the punchline        [generated]
  06_attention_sink_cliff.svg   why the window collapses             [generated]
src/
  shapes.py                   analytic cache model for GQA/SWA/CLA/MLA
  verify_cache.py             proves the cache really shrinks (P4)
  compat.py                   the enable_gqa workaround — read this first
  variants.py                 the 20-variant catalogue
  gqa_surgery.py              8→4/2/1 KV merge, provenance manifest + sha256
  measure.py                  cache bytes, prefill latency, decode throughput
  mask_audit.py               why sliding window raises peak memory
  run_eval.py                 quality sweep, protocol-identical to run_gates.py
  make_figures.py             emits figures 03–05
  sink_audit.py               attention mass on the first tokens
  make_figure_06.py           the mechanism figure
  summarize.py                collects results/ into FINDINGS.md
results/
  measure_main.json/.log      efficiency sweep
  FINDINGS.md                 generated summary — read this for results
```

Run everything with the repo venv: `../../.venv/Scripts/python.exe`.

---

## Order to run things

Each step can invalidate the next, so the cheap checks come first (P7).

```bash
python src/shapes.py                 # analytic cache costs, no GPU
python src/verify_cache.py           # PROVE the cache shrinks — CPU, seconds
python src/variants.py --check       # all 20 configs build
python src/gqa_surgery.py --kv-heads 4 2 1                    # ~2.6 GB of checkpoints
python src/measure.py --variants base-8KV swa-w1024-g4 gqa-2KV        --lengths 4096 16384 32768 --cap-mib 6144 --out results/measure_main.json
python src/mask_audit.py             # why the peaks look like that
bash run_quality_sweep.sh            # the whole eval plan, ~45 min
```

Two things that are not optional politeness:

**`--cap-mib`.** Windows/WDDM does not OOM when VRAM runs out — it spills into
system RAM and the run goes quiet at 0% GPU utilisation for minutes before dying
with its stdout buffer unflushed. An uncapped 32K sweep was lost exactly this
way. The cap turns a silent stall into a catchable error, the same trade
`scripts/train/memory.py` makes for the trainer.

**`--self-check`.** It reruns the unmodified base in `--no-compat` mode and
asserts it reproduces `artifacts/gate_runs/r2_base_chat_dev` exactly. It runs in
a *different kernel mode* from the rest of the sweep, deliberately — see the
comment block in `run_eval.py`. Do not "fix" that mismatch by patching the
baseline run too; that reintroduces a kernel confound into every comparison.

---

## What is established so far

### The analytic model is correct, and that is now proven not assumed
`verify_cache.py` passes 3/3 on a tiny model; the GPU sweep agreed with
`shapes.py` on **14/14 cells within 1%** (worst relative error 0.0011). Sliding
layers genuinely hold `window − 1` keys — the config flag truncates the cache, it
does not merely mask attention.

### Measured, R2, RTX 4070 Laptop, 6 GiB allocator cap

| variant | cache 16K | peak 16K | prefill 16K | decode 16K | 32K |
|---|---:|---:|---:|---:|---|
| `base-8KV` | 1792 MiB | 3128 MiB | 2368 ms | 11.4 tok/s | **OOM** |
| `swa-w1024-g4` | 352 MiB | 3832 MiB | 4510 ms | 18.1 tok/s | **OOM** |
| `swa-w256-g4` | 280 MiB | 3832 MiB | 4670 ms | 18.1 tok/s | **OOM** |
| `swa-w1024-g2` | 232 MiB | 3832 MiB | 4851 ms | 19.2 tok/s | **OOM** |
| `gqa-2KV` | 448 MiB | 1700 MiB | 2553 ms | 15.7 tok/s | OK (2573 MiB) |
| `gqa-1KV` | 224 MiB | 1462 MiB | 2510 ms | 21.5 tok/s | OK (2111 MiB) |

### Correction: the SWA cache saving is logical, not resident

The cache figures above were counted with `numel() * element_size()`.
`DynamicSlidingWindowLayer.update` keeps `full[:, :, -w+1:, :]` — a **view** onto
the full prefill allocation — so that count credits a saving the allocator has
not made. Measured with de-duplicated storage pointers
(`measure.py:cache_bytes_pair`, `results/measure_fixed.json`), immediately after
prefill at 16K:

| variant | logical | **resident** | ratio |
|---|---:|---:|---:|
| `base-8KV` | 1792.0 MiB | 1792.0 MiB | 1.00 |
| `swa-w1024-g4` | 351.9 MiB | **1792.0 MiB** | **5.09** |
| `swa-w256-g4` | 279.9 MiB | **1792.0 MiB** | **6.40** |
| `gqa-2KV` | 448.0 MiB | 448.0 MiB | 1.00 |
| `gqa-1KV` | 224.0 MiB | 224.0 MiB | 1.00 |

**At the end of prefill a sliding-window model holds exactly as much KV as the
full-attention baseline.** The window saving only materialises once the first
decode step replaces the view with a fresh tensor. GQA's saving, by contrast, is
resident from the moment it exists.

This completes the §4a mechanism rather than replacing it. The byte-identical
peaks for `w=256` and `w=1024` (3832.5 MiB both) now have two causes, not one:
the `[1,1,n,n]` mask *and* a resident cache that is identical between them
because both are views onto the same full allocation. It also explains directly
why SWA never improved peak memory — peak is set at end of prefill, which is
exactly the moment the saving does not exist.

The reduction figures quoted elsewhere in this document as "cache" are
**logical** occupancy. They are the right quantity for checking the analytic
model in `shapes.py`, which predicts logical occupancy, and the wrong quantity
for any memory claim.

### One audit prediction refuted

The audit also held that `base-8KV @ 32768 -> OOM` was purely a harness artifact
— `measure.py` held two caches during timing — and that a single prefill would
fit (reconstructed at 5,345 MiB against a 6,144 MiB cap). The harness defect was
real and is fixed (`peak_alloc_mib` and `peak_timed_mib` now agree in all 12
cells), **but the verdict stands: base-8KV and every SWA variant still OOM at
32K, while both GQA variants measure.** The reconstruction was arithmetic, not a
measurement, and the measurement disagrees.


---

### The headline: the cache-only analysis predicts the wrong winner

Sliding window has the **smallest caches and the largest peaks**. At 32K under a
6 GiB cap, every SWA variant OOMs and both GQA variants fit comfortably.

The cause is visible in `src/mask_audit.py`, which instruments a real forward:

```
base-8KV      create_causal_mask                 -> None (is_causal fast path)
swa-w1024-g4  create_sliding_window_causal_mask  -> [1,1,16384,16384] bool, 256 MiB
```

A sliding window is expressed as a dense `[1,1,n,n]` mask that scales with
**context, not window** — which is why `w=256` and `w=1024` have different caches
and *byte-identical* peaks (3832.5 MiB). The explicit mask also blocks SDPA's
`is_causal` fast path, so prefill is ~1.9× slower rather than the ~1.75× faster
the FLOP model predicts.

**So on this stack the honest answer to "which axis extends context" is the KV-head
count, not the window schedule.** Sliding window still wins decode throughput
(18.1 vs 11.4 tok/s at 16K) and persistent cache — it trades prefill and reach for
steady-state speed. The remedy for the mask cost is a kernel with native windowing
(FlashAttention takes a `window_size`); `flash_attn` is **not installed** here, so
that is an unexplored branch, not a fix in hand.

### The cost asymmetry still holds
Of the 20 variants, **16 are reachable by editing a config** and 3 need surgery
plus a healing run each. Parameter counts after surgery match the analytic model
exactly (474,021,888 → 444,661,760 / 429,981,696 / 422,641,664), the merge is
reproducible, and 254 of 310 tensors are bit-identical to the source. These
checkpoints are **unhealed**; their quality numbers are a *damage floor*.

### Two findings that reach beyond this experiment

**1. `enable_gqa=True` silently falls back to the SDPA math kernel** (sm_89,
torch 2.13.0+cu130). At 16 Q / 8 KV / `head_dim=128` / seq 16384, bf16:

```
enable_gqa=True, is_causal=True   ->  allocates 16.00 GiB  ->  OOM
repeat_kv to 16 heads, is_causal  ->  peak 224 MiB
```

16 GiB is exactly `16 × 16384² × 4 bytes`: the score matrix in fp32.
`transformers` opts in whenever `attention_mask is None`. It surfaces as an OOM
at a plausible length, so it reads as "the card is too small" rather than "the
wrong kernel ran". **Any long-context work on this host is affected**, and the
8KV ceiling in `docs/EXPERIMENT_DESIGN_KV_LONGCTX.md` is optimistic until
`src/compat.py` is applied.

**2. Switching SDPA kernels changes 23% of greedy replies while moving aggregate
quality by +0.00138.** Measured on chat_dev: without the compat patch this
harness is **60/60 byte-identical** to the stored gate baseline; with it, 51/60.
The two kernels are mathematically equivalent and numerically are not, and one
flipped token cascades through greedy decoding. This is a textbook case for the
program's P8 — the aggregate says "no effect" while a quarter of the outputs
changed — and it is why the sweep validates protocol fidelity in `--no-compat`
mode and then runs every comparison against a base that used the *same* kernel.

### Quality: the window axis has no usable middle

Harness validated first: in `--no-compat` mode this harness reproduces
`artifacts/gate_runs/r2_base_chat_dev` **exactly, |delta| = 0.00e+00 over all 510
cases**. Comparability proved before comparing (P4).

| variant | dataset | mean quality | vs base | identical replies |
|---|---|---:|---:|---|
| `swa-w512-g4` | chat_dev | 0.83498 | **±0.00000** | **510/510** |
| `gqa-2KV` **unhealed** | chat_dev | 0.11384 | −0.72113 | 0/510 |
| `gqa-1KV` **unhealed** | chat_dev | 0.02360 | −0.81138 | 0/510 |
| `swa-w128-g4` | rag_dev | 0.03552 | −0.63387 | 20/752 |
| `gqa-2KV` **unhealed** | rag_dev | 0.04189 | −0.62750 | 0/752 |

**A sliding window is either exactly free or catastrophic, with nothing in
between.** At `w=512`, wider than every chat_dev prompt, it is provably the
identity — 510/510 byte-identical, delta exactly zero. At `w=128`, narrower than
99% of rag_dev prompts, the model emits empty strings.

And the damage does not scale with how much context is lost:

| prompt tokens past `w=128` | n | base | swa-w128 | delta |
|---|---:|---:|---:|---:|
| inside the window | 20 | 0.8886 | 0.8886 | **+0.0000** |
| 1–100 over | 307 | 0.6863 | 0.0197 | −0.6666 |
| 101–200 over | 422 | 0.6487 | 0.0068 | −0.6419 |

One token over the edge costs as much as two hundred. The only rag_dev category
untouched is `social`, whose 20 cases are 41–45 tokens — *exactly* the 20 prompts
in the set that fit inside `w=128`.

**Mechanism, measured not cited** (`src/sink_audit.py`): R2 sends **46.3%** of its
attention mass to the first 4 tokens on average, **14 of 28 layers above 50%**,
layer 27 at **80.5%**. A naive sliding window evicts precisely those positions.
This is the attention sink, and it explains both the cliff and its independence
from how much context was dropped. The fix — pinning the first few tokens —
is not reachable through `transformers`' `sliding_window`, which removes the
window axis's main selling point: that it was free.

### One result retracted
The first pilot showed a 3.2× prefill speedup for sliding window. It was an
artefact: SWA variants took the fast `repeat_kv` path (explicit mask) while the
baseline took the broken `enable_gqa` one. The comparison was between two
kernels, not two attention shapes. With the patch applied the sign reverses and
SWA is *slower*.

## Verdict on the question this experiment was built to answer

> At matched cache, is the window schedule or the KV-head count the better axis?

**On this stack: the KV-head count, and it is not close.** The window axis fails
on three independent counts — it raises peak memory, it *reduces* reachable
context (every SWA variant OOMs at 32K where both GQA variants fit), and its
quality profile is a cliff with no usable operating point between "free" and
"broken". The one thing it wins is decode throughput, 18.1 vs 11.4 tok/s at 16K.

The multi-axis framing therefore did **not** pay in the direction expected. That
was worth finding out for ~a day of compute rather than assuming, and the
cache-only analysis that motivated it would have pointed the other way.

## What this implies for MLA

The case is stronger than before, not weaker. The two axes tested both work by
*removing* something — the window removes positions (and the sink with them), head
merging removes capacity. MLA removes neither: every position stays present and
every head stays distinct; only what each position *stores* is compressed. It is
the only option examined that avoids both failure modes, and at `d_c=256` it costs
17.5 KiB/token against MQA's 14.0 — near-MQA memory with near-MHA structure.

It remains the expensive option: it needs a custom attention module (none exists
for Qwen3 in this stack), SVD initialisation from the current K/V projections, and
healing. `THEORY.md` §3 has the maths and the absorption identity;
`figures/02_mla_vs_qwen_dataflow.svg` has the dataflow.

## Open

- **Healing for the GQA arms.** Unhealed numbers are a damage floor only
  (0.11384 at 2KV), and until they are repaired the two axes are still not
  comparable on quality: one is free and intact, the other damaged and unrepaired.
- **Attention sinks.** The cheapest test left: pin the first 4 tokens in the
  window and re-run `swa-w128-g4`. If the cliff disappears, the window axis is
  back in contention and needs a small custom mask rather than a config flag.
- **A window sweep** between 128 and 512 to locate the cliff edge precisely —
  though the sink measurement predicts it sits wherever the prompt start falls
  out, not at a particular width.
- **FlashAttention** (`window_size`) would remove the O(n²) mask cost and change
  the memory verdict. `flash_attn` is not installed here.
- Long-context work (≥64K) still needs the corpus that
  `docs/EXPERIMENT_DESIGN_KV_LONGCTX.md` §2a shows does not exist: nothing in
  this repository exceeds ~2.7K tokens.
