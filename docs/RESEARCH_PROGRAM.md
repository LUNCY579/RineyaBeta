# Rineya small-model research program

An ambitious objective stated precisely, so it can be falsified:

> **Move the capability-per-parameter frontier.** Not "make it smaller" and not
> "make it better" — either alone is easy and uninteresting. The claim to earn is
> a checkpoint that is *simultaneously* materially smaller than Qwen3-0.6B and
> materially more capable than it, on gates frozen in advance.

Everything below is organised around that frontier. A change is worth keeping only
if it moves a point up-and-left on (parameters, capability); anything that trades
one for the other at par is a deployment decision, not a research result.

---

## 0. Governing principles

These are not aspirations. Each one is here because violating it cost real time in
the work that produced `MINI_MODEL_ROADMAP.md`, and the citation is the scar.

| # | Principle | Why |
|---|---|---|
| **P1** | **Measure to a plateau, never to a sample.** | Four consecutive memory predictions were wrong (14–67%) because 12–14-step samples were treated as steady state. The 8x16 run that read a flat **3,882 MiB at ~step 10** kept climbing to **5,090 MiB** — 30 MiB from the cap — and hung at **step 72** with no checkpoint. |
| **P2** | **Gate per category. Never conclude from an aggregate.** | The trim's RAG cost read as diffuse noise at −0.0127 aggregate. Per category it was one thing: `conflict` −0.1898, nine lost abstentions, 11 of 16 categories bit-identical. |
| **P3** | **Verify supervisory signals against system state.** | Three orphaned trainers survived "successful" stops. `Stop-Process -Force` and `taskkill /F` both failed *silently*; one orphan burned 8,029 CPU-seconds and probably caused a `CUBLAS_STATUS_INTERNAL_ERROR` I nearly attributed to a kernel bug. |
| **P4** | **Prove comparability before comparing.** | The vendored v4.5 scorer only became trustworthy once it reproduced upstream at max |Δ| = 0.000e+00 on 510 stored replies. Two subtle invocation details were worth 3e-3 on a fifth of rows. |
| **P5** | **Pre-declare thresholds.** | A floor chosen after seeing the number is a rationalisation, not a gate. Inherited from `READINESS_ASSESSMENT.md`. |
| **P6** | **One variable per run.** | The retired 19L soup confounded four changes at once and its null result was uninterpretable. |
| **P7** | **A cheap test that can invalidate an expensive one runs first.** | A 10-minute tokenization check killed the leading explanation for the abstention loss and saved a 3-hour run from answering the wrong question. |
| **P8** | **Distinguish "not measurable" from "not there".** | Chat DEV: +0.0002, CI spanning zero → no effect. RAG DEV: −0.0127, CI excluding zero, 0 wins / 14 losses → a real effect. Same magnitude scale, opposite conclusions. |

---

## 1. Where the frontier currently sits

| Artifact | Params | Chat DEV | RAG DEV | Status |
|---|---:|---:|---:|---|
| Qwen3-0.6B stock | 596,049,920 | 0.8334 | 0.6809 | reference |
| **R2** (vocab 32k) | **474,021,888** | **0.8336** | **0.6682** | measured, §16 |
| retired 19L soup | ~461M | — | — | degenerate, rejected |

R2 is **−20.47% parameters, −14.72% compute**, free on conversation, and −0.0127
on grounding localised entirely to `conflict` abstention. That is one Pareto point
and one known defect. Everything in this program builds from there.

**The honest hardware ceiling.** One RTX 4070 Laptop, 8,187 MiB. That admits
*surgery plus healing* on an existing checkpoint. It does **not** admit pretraining
a new architecture from scratch. Every workstream below is therefore tagged:

- **[S] surgical** — reachable by transforming Qwen3 weights + a healing pass. Runs here.
- **[R] retrain-light** — needs a few GPU-days; feasible here with patience, or rented.
- **[P] pretrain** — needs orders more compute. Design and simulate here; execute elsewhere or not at all.

Tagging this honestly is what keeps the program executable rather than aspirational.

---

## 2. Workstream I — infrastructure (unblocks everything)

From the audit of the current tree: 4,092 LOC of project code, 26 files, and eight
gaps (I1–I8 below). Ordered by how much each unblocks.

### I1. Reproducible environment **[blocking]**
`.venv` cannot be rebuilt from the repo — no lockfile, no requirements. Every
number in the roadmap is therefore reproducible in *content* but not in
*environment*, and `determinism.py` already refuses to resume across a torch
change. Fix: `requirements.lock` pinned to exact versions (torch 2.13.0+cu130 from
the cu130 index, transformers 5.15.0, peft 0.20.0, tokenizers 0.22.2, numpy 2.4.6,
rapidfuzz 3.14.5, pyyaml), plus a `make env` that builds it via `uv`.
**Acceptance:** a fresh clone reproduces `verify_scorer.py` at max |Δ| = 0.

### I2. Surgery library **[blocking for §3 and §4]**
Only 1 of 8 compression capabilities exists. The missing seven are not
independent scripts — they are one library over a common contract:

```
surgery.load(checkpoint) -> Model
surgery.drop_layers(model, indices)        # depth
surgery.prune_mlp(model, keep_per_layer)   # width
surgery.prune_heads(model, keep_heads)     # attention
surgery.factorize(model, rank_map)         # low-rank
surgery.trim_vocab(model, keep_ids)        # done; refactor into this
surgery.save(model, out_dir, provenance)   # manifest + sha256, always
```

Every operation must: (a) emit a provenance manifest recording exactly what was
removed, (b) be byte-deterministic, (c) be verifiable by a logit-equivalence check
on the untouched sub-space. The vocab trim already meets all three; it becomes the
reference implementation.
**Acceptance:** each operation has a `verify_*` proving the intended invariant.

### I3. Test suite + CI
No tests, no CI. The `verify_*` scripts are integration tests taking minutes and
needing a GPU. Add fast unit tests for the pure logic — `plan.py` determinism and
budget arithmetic, `dataset.py` label boundaries and the off-by-one in
`supervised_lengths`, `checkpoint.py` atomicity and every guard rail,
`determinism.py` fingerprint comparison, the keep-set builder's merge closure.
**Acceptance:** `pytest` green in under 60s on CPU; CI runs it per commit.

### I4. Shared utilities
`sha256_file` ×3, `norm_hash` ×2, `classify` ×2, `count` ×2. Collapse into
`scripts/lib/`. Low value alone, but it is where I2's contract will live.

### I5. Eval registry
Only 2 of 5 available eval sets are wired (chat DEV, RAG DEV). Add Puzzle
validation (4,080), LCR-SFT dev (800), LongContextDemo (256, frozen). One registry
declaring for each: path, sha256, protocol, scorer, category key, anchors, and
whether it is gate or sealed. `run_gates.py` becomes registry-driven.
**Acceptance:** adding a benchmark is one registry entry, no code.

### I6. Run registry and frontier tracker
Every artifact should land in one table: params, active params, bytes, each gate's
score with CI, wall clock, and provenance chain back to stock. The frontier plot is
the program's scoreboard, and without it the Pareto claim cannot be checked.

### I7. Remove hardcoded upstream paths
Two in `scripts/eval/`. Make them config, so the repo runs on another machine.

### I8. Process hygiene, generalised
The heartbeat mutex in `heal.py` fixed trainers. `run_gates.py` needs the same —
one orphan there also survived a "successful" stop.

---

## 3. Workstream C — compression, remaining axes

Each with a pre-declared hypothesis. Parameter arithmetic from §1 of the roadmap:
per layer = 15,730,944; embedding row = 1,024/token.

### C1. Depth **[S]** — the measured next rung
`layer_profile.py` on R2 says drop `25..26` (d = 0.1203). → 442,560,000 (−25.8%).
**H1:** removing the two least-consequential layers costs < 0.01 chat quality after
healing, and depth removal damages `reasoning` categories before `retrieval` ones.
The retired soup is the counter-example that motivates the caution: 9 of 28 layers,
unhealed, produced `imony` loops.

### C2. Width **[S]** — the largest remaining pool
MLP is 44.3% of parameters; `intermediate_size` 3072 → 2048 saves 3,145,728/layer.
Use activation-aware importance (Wanda-style `|W|·‖X‖`) calibrated on the Rineya +
RAG mix, not magnitude.
**H2:** at equal parameter reduction, width pruning beats depth pruning on
reasoning gates — the Minitron result, tested on *this* model. This is a clean
head-to-head: two checkpoints at ~442M, one by depth, one by width, same healing.

### C3. Attention and retrieval heads **[S]**
**H3 (the one I most want to test):** a sparse set of heads does the
copy-from-context work, and protecting them makes grounding robust to compression.
Identify them by per-head attention mass on known supporting spans in RAG DEV,
then prune with and without protection at matched parameter count.
**Prediction:** unprotected pruning damages `supported_with_distractor`,
`conflict` and `prompt_injection` far more than `social` or `supported`.
This is also the direct follow-up to §16: `conflict` abstention was the one thing
the vocab trim cost, and it is an in-context discrimination task.

### C4. Low-rank factorization **[S]** — not yet attempted
Replace `W (d×k)` with `U(d×r) V(r×k)` where the spectrum permits. Unlike
pruning, this removes parameters without removing *units*, so it degrades
differently — and it composes with C2.
**H4:** per-matrix effective rank varies systematically by layer depth and by
projection type (q/k/v/o vs gate/up/down), and a rank schedule derived from
measured spectra beats a uniform rank at equal size.
First step is pure measurement and needs no training: SVD every weight matrix,
plot the spectra, compute the parameter saving at 95/99% energy retention.

### C5. Vocabulary frontier revisited **[S]**
32,768 was chosen on a break-even argument at one point. The full curve is unmeasured.
**H5:** there is an interior optimum. Below it, token inflation costs more compute
than the embedding saves; above it, embedding rows are wasted. §3c has four points
(24,576 / 32,768 / 40,960 / 49,152) and 32,768 won on net compute — but capability
was not measured at any of them. Now it can be, cheaply: gate runs are 5 minutes.
**Also test:** does a larger vocabulary recover the 9 `conflict` abstentions? §16.3
says the entity names tokenize identically, so the prediction is **no** — and a
confirmed no is valuable, because it rules out vocabulary as the remedy.

### C6. Quantization **[S]** — separate artifact, not a frontier point
4-bit AWQ/GPTQ reduces bytes, not parameters. Tracked separately so the parameter
claim stays clean. R5 at 4-bit is roughly 170 MB.

---

## 4. Workstream A — architecture, where the ambition lives

The frontier cannot be moved far by removal alone. §0 of the roadmap states the
limit plainly: *pruning can only remove capability, and SFT healing can only
recover what the base already had.* Exceeding stock Qwen3-0.6B at 60% of its size
requires either a better teacher (§5) or a better architecture.

Ordered by expected value per unit of compute, most promising first.

### A1. Looped / recurrent depth **[S→R]** — the strongest idea for this goal
Reuse a block of layers *k* times instead of having *k* distinct blocks. Effective
depth rises with **no parameter cost**. This is the single most direct attack on the
frontier, and it is motivated by a measurement already in hand: the layer profile
shows depth carries reasoning, and the retired soup showed that removing depth
destroys it. If depth is what matters and parameters are what cost, then decoupling
the two is exactly the right lever.

**H6:** a looped-depth model with *L/2* distinct layers applied twice matches an
*L*-layer model on reasoning gates at ~55% of the parameters, after healing.

Surgical entry point: initialise the loop from alternating layers of the trained
model and heal. This is cheap to try and the failure mode is obvious and fast.
Deeper versions (training with the loop from the start, adaptive loop counts,
per-token early exit) are **[R]** or **[P]**.

Connects directly to the state-sufficiency lab: looping is *implicit* recurrence,
the SPLIT representation is *explicit* recurrence through the context. Testing both
against the same gates is a genuinely interesting comparison and it is already half
built.

### A2. Cross-layer parameter sharing **[S]**
ALBERT-style tying, but selective rather than global. The layer profile gives a
principled grouping: layers whose residual-stream deltas are most similar are the
best sharing candidates.
**H7:** tying the 22–26 band (measured most redundant) costs less than tying any
other five-layer band at equal parameter saving. Note this is a *different*
prediction from C1 — sharing keeps the computation and drops the storage.

### A3. Matryoshka / elastic width **[R]**
Train so that any prefix of hidden dimensions is a usable model, giving one
checkpoint that serves several size budgets. High deployment value, and it turns the
frontier from a set of points into a curve. Needs nested-dimension training, so
retrain-light rather than surgical.

### A4. KV-cache and inference-shape work **[S]**
Qwen3-0.6B is already GQA (16 heads / 8 KV). Reducing KV groups, cross-layer KV
sharing, and sliding-window attention on a subset of layers all cut *deployment*
cost more than parameter count. Worth doing, but scored on latency and memory, not
on the parameter frontier — and kept out of the bit-exactness path, since
`determinism.py` must keep `eager` attention (§15.2 measured flash and
mem-efficient backward as nondeterministic: 4.7e-10 and 1.5e-11).

**Now specified in full:** `EXPERIMENT_DESIGN_KV_LONGCTX.md` pre-registers this
as a study — KV-head compression against RoPE scaling and chunked attention, on
an iso-memory frontier. Three findings from its Phase 0 audit change the shape of
A4 as written above: the 8KV control **cannot reach 64K or 128K on this card at
all**, eager attention is impossible past 4K so the evaluation path needs its own
weaker determinism contract, and the repository contains **no long-context
evaluation asset of any kind** — the longest context ever measured here is 1,504
tokens.

### A5. Small-scale MoE **[P]** — define the metric before attempting
Sparse experts raise *total* parameters while lowering *active* ones. That is only
a frontier gain if "size" means active parameters. **Decide that definition before
running anything**, or the result will be unfalsifiable. My recommendation: track
both columns and never conflate them.

### A6. Depth-width reallocation **[R]** — the cleanest scientific question
Given a fixed parameter budget, what aspect ratio maximises reasoning? Qwen3-0.6B
is 28 × 1024. Is 20 × 1280 better? 40 × 768? This is a real, publishable question,
the eval harness to answer it now exists, and it directly informs every choice in
Workstream C.
**H8:** for reasoning gates at fixed parameters, capability is monotone in depth
over the range reachable here — which if true says width pruning should always be
preferred to depth pruning, generalising H2 into a design rule.

---

## 5. Workstream T — training and algorithms

### T1. Prune-then-distill **[S]**, the Minitron recipe
Logit KL from an unpruned teacher, optionally plus hidden-state matching. The
ordering constraint is already established: **trim the vocabulary before
distilling**, so teacher logits restrict to kept ids by a fixed gather instead of
becoming a retraining problem.

### T2. Larger-teacher distillation **[S]**
Qwen3-1.7B → R-series. The only route to *exceeding* stock at smaller size without
an architecture change. **H9:** distillation from 1.7B recovers more than SFT at
equal steps, and most on the categories where R2 is weakest (`conflict`,
`qualified_answer`, `procedure_with_distractor` — all under 0.40).

### T3. State sufficiency **[S]** — the user's own hypothesis, now testable
`P(r_{t+1} | q, H_t) ≈ P(r_{t+1} | q, s_t)`. The lab is built, the dataset exists,
and Puzzle-51k ships `compact_state` + `reasoning_steps`, which is exactly the
`(problem, state) → transition` format the SPLIT arm consumes. This is the most
mature unexecuted experiment in the repository.

### T4. DPO **[S]**, strictly after an accepted SFT checkpoint
16,000 v3 preference pairs plus 12,000 v2. Their own cards say post-SFT only.
**H10:** DPO improves `abstain` precision more than SFT does, because abstention is
a preference between two fluent continuations rather than a knowledge gap — which
is precisely the shape of the nine lost `conflict` cases.

### T5. Iterative prune-heal cycles **[R]**
Many small prunes with healing between beats one large prune, at equal final size.
**H11:** testable directly — R2 → 442M → 411M with healing at each rung, versus
R2 → 411M in one step. The rung ladder is already designed for this.

---

## 6. Workstream M — measurement science

The program's credibility rests here, and today proved the cost of getting it wrong.

### M1. Seed-variance floor **[do this first, it is cheap and blocking]**
**Nothing else in this document is interpretable without it.** Every comparison so
far is single-run versus single-run. Train the same config at 3–5 seeds and measure
the gate spread. If seed alone moves chat DEV by ±0.01, then a +0.01 result is
noise and several planned experiments are underpowered as designed.
This is the honest prerequisite for every hypothesis above, it costs a few GPU-hours,
and it should run before any conclusion is drawn from B0.

### M2. Category-level gate suite (P2, institutionalised)
Every gate reports per-category with counts, not just a mean. `conflict`,
`stale_evidence`, `prompt_injection`, `missing_answer` and `qualified_answer` are
flagged as calibration-sensitive: they break first under compression, and the
mechanism is lost abstention rather than lost knowledge.

### M3. Paired statistics as the default
Same cases, same scorer, paired deltas with 95% CIs and win/loss counts. §16 shows
why: +0.0002 with a CI spanning zero and 45–48 wins/losses is *no effect*, while
−0.0127 with a CI excluding zero and 0–14 is a *real* one.

### M4. Power analysis before each experiment
`READINESS_ASSESSMENT.md` §7b already established that ~200 paired problems give
±7 pp — too coarse for a 2 pp decision. Compute the detectable effect size before
running, not after.

### M5. Sealed-set discipline
RAG TEST (738) is imported and `run_gates.py` refuses to evaluate it. It opens once,
for one frozen release candidate. LongContextDemo (256) is likewise frozen.

---

## 7. Sequencing

Dependencies are real; this order respects them.

**Phase α — make the ground solid (days)**
`I1` env lock → `I3` tests + CI → `I2` surgery library → `I5` eval registry →
**`M1` seed-variance floor**. Nothing after this phase is trustworthy without it.

**Phase β — finish the compression map (weeks)**
`C4` spectra measurement (no training, pure measurement, do it early) →
`C1` depth rung → `C2` width rung → **`H2` head-to-head at matched size** →
`C3` retrieval-head protection → `C5` vocabulary capability curve.
Ends with a measured frontier over four axes instead of one.

**Phase γ — cross the frontier (weeks)**
`T2` larger-teacher distillation → `T1` prune-then-distill on the best Phase β
point → `T5` iterative cycles. This is where "smaller *and* better" becomes
attemptable rather than aspirational.

**Phase δ — architecture (open-ended)**
`A1` looped depth first — highest expected value, surgical entry, fast failure.
Then `A2` selective sharing, then `A6` depth-width reallocation if compute allows.
`A3`/`A5` only with rented compute and a definition of "size" fixed in advance.

**Continuous:** `T3` state sufficiency runs in parallel — it is a capability
question rather than a compression one, and it is already built.

---

## 8. What would make this program fail

Named in advance, per P5.

1. **Seed variance swamps the effects.** If M1 shows ±0.02 gate noise, most
   single-run comparisons here are underpowered and the plan needs multi-seed
   budgets throughout. This is the most likely failure and the cheapest to check.
2. **The 8 GB ceiling binds harder than expected.** §15.6 already showed
   bit-exactness and 4k context are mutually exclusive on this card. More
   ambitious architectures may simply not fit, and the honest response is renting
   compute, not quietly dropping the guarantee.
3. **Healing plateaus below stock.** If §0's limit holds strictly, only Workstream A
   or T2 can cross the frontier, and Workstream C becomes loss-minimisation only.
4. **Gate saturation.** Chat DEV is already at 0.833 with contract fulfilment at
   1.0000 and zero violations. There may be little headroom left to measure there,
   in which case RAG DEV and the unwired benchmarks carry the whole signal — an
   argument for doing I5 early.
5. **The frontier claim gets made on a single eval.** Guarded by I6: no artifact
   enters the frontier table on one benchmark.
