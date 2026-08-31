# DAST Phase-0 readiness report and investigation plan

**Branch:** `research/depth-aligned-semantic-transitions-v1`
**Base commit audited:** `a343e1f`
**Report date:** 2026-08-31
**Program status (unchanged by this report):** **BUILT — UNRUN on a real checkpoint**
**Sealed evaluation:** **CLOSED**

This document is the Phase-0 output required by
`../../../docs/PLAN_DEPTH_ALIGNED_SEMANTIC_TRANSITIONS.md` §7 ("Output: readiness
report and frozen preregistration"). That plan remains the single source of
truth; nothing here redefines the thesis, the arms, or the gates.

## 0. Claim boundary

Everything below is either (a) a reproducible engineering defect, or (b) a
Phase-0 diagnostic computed from the real checkpoint and the real dataset with
**no training and no fine-tuned model**. Per `EVIDENCE_STANDARD.md`:

- no arm was trained;
- no capability, quality, reasoning, memory, or DiffusionBlocks-equivalence claim
  is made or implied;
- no DEV comparative result exists;
- TEST was never opened, loaded, or hashed.

Diagnostics about the *target geometry* are properties of the dataset and the
frozen encoder. They are legitimate Phase-0 observations and are explicitly
**not** evidence for or against the depth-alignment hypothesis.

## 1. Environment actually used

| Item | Value |
|---|---|
| Host | RTX 4070 Laptop GPU, 8188 MiB, driver 610.62 |
| Python | 3.11.15 (uv-managed venv) |
| torch | 2.13.0+cu130, CUDA 13.0 available |
| transformers | **5.15.0** |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| bitsandbytes | **absent** (QLoRA path unavailable) |
| Base checkpoint | `Qwen/Qwen3-0.6B` from HF cache, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Resident weights, inference | ~2.5 GiB (measured) |

The shipped `pyproject.toml` requires `transformers>=4.51` with no upper bound.
The installed major version is 5.x, and that is the direct cause of blocker B1.

## 2. Verified blockers

All five are reproducible. Each is an engineering defect, not a scientific
result. G0 in `PREREGISTRATION.md` §8 does **not** currently pass on this host.

### B1 — `apply_chat_template` returns a dict on transformers 5.x (blocks every real-model path)

`dast.tokenization.apply_chat_template_ids` calls
`tokenizer.apply_chat_template(..., tokenize=True)` and passes the result to
`_as_int_list`. On transformers 5.x that call returns a `BatchEncoding`
mapping, so iteration yields **keys**:

```text
ValueError: invalid literal for int() with base 10: 'input_ids'
  dast/tokenization.py:41 in _as_int_list
```

This function is on the critical path of both training tokenization and
generation evaluation, so no real run of any arm can start. Fix: unwrap
`input_ids` when the return value is a mapping (or pass `return_dict=False`),
and pin a supported `transformers` range.

### B2 — Offline smoke crashes on any CUDA host (blocks G0 where the science must run)

`dast/smoke.py:189` snapshots the projector parameters **before** `train()` moves
the model to the accelerator, then compares with `torch.equal`:

```text
RuntimeError: Expected all tensors to be on the same device, but got other is
on cuda:0, different from other tensors on cpu
```

Measured on this host, same commit, same interpreter, varying only device
visibility:

```text
                                      GPU visible        CUDA_VISIBLE_DEVICES=-1
python -m compileall -q src tests      PASS               PASS
python -m ruff check src tests         FAIL (1 error)     FAIL (1 error)
pytest -q                              36 passed 1 FAIL   37 passed
python -m dast.cli smoke               FAIL (above)       PASS
```

This is a direct experiment, not an inference: `PROGRESS.md`'s "pytest -q PASS
(37 tests)" and "smoke PASS" are **reproducible only on a CPU-only host**, and
fail on the GPU host that must run the science. Fix: compare on a common device.
Then state the host and versions alongside the recorded PASS, so an
environment-specific result is never again recorded as unconditional.

### B3 — Ruff error fails CI independently of the runner blocker

`ruff check src tests` reports one unsorted-import error in
`tests/test_tokenization.py`, on both GPU and CPU-only runs (see the B2 table),
so it is host-independent. The `dast-ci.yml` Lint step would therefore fail even
with a healthy runner.

`PROGRESS.md` classifies the two Actions attempts as
`EXTERNAL BLOCKER — GITHUB-HOSTED RUNNER NOT ASSIGNED`, and that classification
is sound (no executed steps, `runner_id: 0`, zero billable milliseconds). But
the recorded "next permitted action" — re-run the **unchanged** workflow — will
now fail at Lint. B3 must land before that retry is meaningful.

### B4 — The shipped default `data.path` cannot be loaded

`configs/base.yaml` points `data.path` at
`.../data/manifests/problem_manifest.jsonl`. That file is a **metadata**
manifest (`problem_id`, `split`, `step_count`, hashes) and carries none of the
fields the loader requires. RUNBOOK §3's first command fails on every record:

```text
record 1: record is missing required fields: problem, answer, full_response,
          reasoning_steps, compact_state_trace
... (48000 records, reported to max_errors)
```

The reasoning text lives in `data/full_context/{train,dev,test}.jsonl`, which
`.gitignore` excludes (lines 30–31) — so the tracked tree contains only the
metadata manifest, which is why this default was never exercised.

There is a second, structural half to this: `build_dataloaders` derives **both**
train and eval from a single `data.path` by filtering the `split` field, but
`full_context` is one file per split. No existing file satisfies both
requirements.

Two clean options, either acceptable:

- **B4-a (no code change, hardens sealing):** build a derived
  `dast_train_dev.jsonl` = `train.jsonl` + `dev.jsonl` and record its SHA-256 in
  the preregistration. `test.jsonl` is then *physically absent* from the training
  input, which is strictly stronger than the config guard alone.
- **B4-b:** add `train_path` / `eval_path` to `DataConfig`, keeping
  `assert_split_allowed` as the guard.

Recommendation: **B4-a**, because it converts a policy guarantee into a physical
one. Verified, non-blocking: both files are pre-shuffled — the first 4,000 train
records and first 800 dev records each cover all 30 categories — so the
loader's first-N truncation does not bias category coverage.

### B5 — Right-padded batched generation corrupts the primary metric

`dast/hf.py` sets `tokenizer.padding_side = "right"`, and
`dast/evaluation.py` right-pads each generation batch and then decodes every row
from the batch-max prompt width. For any row shorter than the batch maximum, the
model is asked to continue *after* pad tokens, and the decode window is
misaligned. With the shipped `generation_batch_size: 4`, this affects most DEV
problems.

transformers corroborates the defect independently — every batched call emits:

```text
A decoder-only architecture is being used, but right-padding was detected!
For correct generation results, please set `padding_side='left'`.
```

**Correction to an earlier draft of this report.** The measured accuracy impact is
**nil at n=48**: 0.125 right-padded versus 0.125 left-padded, on the same problems
with the same greedy decode. transformers 5.x derives position ids from the
attention mask, which absorbs most of the damage. So this is a real defect worth
fixing — the decode window is misaligned and the library warns on every call — but
it is **not** the reason the primary endpoint is floored, and it should not be
described as corrupting the metric. The actual cause is B6.

Fix anyway: left-pad for generation and decode from each row's own prompt length.

### B6 — The primary endpoint measures the harness, not the model

This, not B5, is what floors the primary outcome. Two independent and additive
causes. Measured on 96 DEV problems, real checkpoint, greedy, left-padded, varying
only the generation cap:

| | cap 128 (shipped) | cap 384 |
|---|---:|---:|
| hit the cap without emitting EOS | **0.510** | **0.010** |
| strict accuracy | **0.115** | **0.229** |
| gold answer present anywhere in the response | 0.333 | **0.438** |
| extraction gap (present − scored) | 0.219 | **0.208** |
| responses on the weakest `last_nonempty_line` route | 47 / 96 | 13 / 96 |
| seconds per problem | 1.41 | 2.78 |

**Cause 1 — the cap truncates correct answers.** At the shipped `max_new_tokens:
128`, **51%** of generations are cut off before emitting EOS. Raising the cap to
384 removes truncation (1%) and **exactly doubles** strict accuracy. The failure is
visible in the raw output — this response reasoned correctly to the gold answer and
was still scored wrong:

```text
... - **Damage -9**: Subtract 9 → 255 - 9 = **246 bolts**

**Final stock: 24        <- cap reached here; extracted as "**Final stock: 24"
```

**Cause 2 — the scorer is narrower than the corpus.** The extraction gap is a
stable **~0.21 at both caps**, so it is independent of truncation.
`answers_equal` applies numeric tolerance only for
`{integer, float, quantity, percentage, fraction, number}`. Over the full 2,400
-problem DEV split:

| | share of DEV |
|---|---:|
| answer types receiving numeric tolerance | **30.0%** |
| answer types falling through to exact string equality | **70.0%** |
| of those, plainly numeric but uncovered (`measurement`, `time`, `probability`, `money`, `decimal`, `rate`, `statistic`) | **36.7%** |

That is why gold `129 N` versus extracted `129` scores **incorrect**: `measurement`
is 13.3% of DEV and gets no numeric tolerance. `set`, `ordering`, `boolean`,
`inequality`, `equation`, and `table_result` are each 3.3% and need their own
canonicalisation.

**Net effect.** True capability is roughly **0.44**; the shipped harness reports
**0.115**. It discards about **three quarters** of the model's correct answers.

Fix: raise the cap until the hit-cap rate is negligible, and make `answers_equal`
answer-type aware. This is not a nicety — it moves the endpoint from 0.115 toward
0.44, which is close to the maximum-power operating point for a paired binary test.
Both changes must be frozen before any comparative run and applied identically to
every arm.

### What is *not* broken

Worth recording explicitly, because it bounds the repair surface. On the real
checkpoint with the B1 shim applied, over 48 DEV problems:

| Check | Result |
|---|---|
| `apply_chat_template` prefix-stability contract | **0 failures** |
| Prompt truncation at `max_length: 512` | **0 examples** (seq len mean 151, max 308) |
| `hidden_states` returned | 29, sufficient for checkpoint 28 |
| Resolved checkpoints | 7, 14, 21, 28 |
| Precision resolved | bf16 |
| Base params / with LoRA | 596,049,920 / 606,142,464 (**+10,092,544**, 1.69%) |
| Model load | 20.5 s, ~2.5 GiB resident |

The tokenization contract, the prompt-mask seam, the depth resolution, and the LoRA
attachment all behave exactly as documented.

Quantification of B5 and of base-model headroom is in §3.4.

## 3. Phase-0 diagnostics

### 3.1 Depth schedule resolves as documented

| Item | Value |
|---|---|
| `num_hidden_layers` | 28 |
| `checkpoint_fractions` | 0.25, 0.50, 0.75, 1.00 |
| Resolved checkpoints | **7, 14, 21, 28** |
| `hidden_states` returned | 29 (index 0 = embeddings) |

Matches `PLAN…§4` and `PREREGISTRATION.md` §2 exactly. `pad_token`
(`<|endoftext|>`) and `eos_token` (`<|im_end|>`) are distinct, so the
`input_ids.ne(pad_id)` attention mask does not silently blank chat-template
delimiters.

### 3.2 Dataset shape

48,000 problems; splits train 43,200 / dev 2,400 / test 2,400; 30 categories at
1,600 problems each; `cross_split_problem_overlap: false`.

### 3.3 Target-geometry separability — the decisive Phase-0 finding

`FrozenTextTargetEncoder` mean-pools frozen input embeddings of the compact-state
text and applies a fixed QR-orthonormalised projection. Measured over 400 DEV
problems, K=4, `state_dim=128`:

| Quantity | Value |
|---|---|
| Within-problem, same stage (diagonal) | 1.000 |
| Within-problem, **different** stage (off-diagonal, mean) | **0.821** |
| Within-problem, adjacent stages | 0.867 |
| Within-problem, first vs last stage | 0.725 |
| **Order margin** (diagonal − off-diagonal) | **0.179** |
| Between-problem, **same** stage (by stage) | 0.676 / 0.719 / 0.724 / 0.775 |
| cos(ordered target, shuffled target) at the same stage | **0.858** |

Two consequences follow directly, and both are about *statistical power*, not
about whether the hypothesis is true:

1. **The target space encodes problem identity more strongly than reasoning
   stage.** Two *different stages of the same problem* (0.821) are more similar
   than the *same stage of two different problems* (~0.72). A projector can
   drive `L_state` down largely by encoding "which problem is this", nearly
   ignoring "how far along am I".
2. **`state_ordered` and `state_shuffled` share ~86% of their target signal.**
   A perfect ordered predictor still scores 0.858 against the shuffled
   assignment, so the largest possible cosine separation between treatment and
   its decisive control is about **0.142** — and the primary gate asks that
   residual to show up in *generated answer accuracy*, a much noisier
   downstream endpoint.

### 3.4 Degenerate-contrast census

`_progress_indices` pins both endpoints, so when a trace has fewer than K=4
targets some depth slots receive the **same** target. Any permutation that only
exchanges identical targets makes the shuffled control partially equal to the
treatment.

| Distinct targets | train | dev |
|---:|---:|---:|
| 2 | 1,438 (3.3%) | 97 (4.0%) |
| 3 | 16,721 (38.7%) | 924 (38.5%) |
| 4 | 14,135 (32.7%) | 775 (32.3%) |
| ≥5 | 10,906 (25.2%) | 604 (25.2%) |
| **< K = 4 (contrast diluted)** | **18,159 (42.0%)** | **1,021 (42.5%)** |

`SSOT §4` and `ALGORITHM.md §4` require repeats to be retained and recorded, and
the code does record them (`repeated_stage_mask`). What is not yet recorded is
that this dilutes the ordered-vs-shuffled contrast on **42% of the corpus**.

### 3.5 The anti-collapse term is inert as configured

`SSOT §6.3` and `ALGORITHM.md §6` rely on a stage-wise multi-positive InfoNCE
term to prevent collapse. `multi_positive_contrastive_loss` contrasts **across
the batch** and returns exactly zero when `batch < 2`. The shipped
`base.yaml` sets `batch_size: 1`:

```text
batch=1 contrastive loss : 0.0
batch=4 contrastive loss : 2.363
```

So with `contrastive_weight: 0.10`, the entire anti-collapse mechanism
contributes nothing to any Phase-1 arm as shipped. Combined with §3.3, the
collapse it was meant to prevent — one problem-conditioned vector reused at all
four depths — is both cheap to reach and nearly as good as the intended
solution, *and identical between the treatment and its control*.

Note also that this term separates *problems* within a stage, which is the axis
already dominant in §3.3. Nothing in the objective separates *stages within a
problem*, which is the axis the hypothesis actually rides on.

### 3.6 Frozen hidden-state diagnostics (SSOT §7, Phase 0.4)

300 DEV problems, frozen untrained Qwen3-0.6B, prompt-only mean pooling at
7/14/21/28. Descriptive only — decodability is neither causality nor capability,
and LoRA training can change these states, so this is a **floor, not a ceiling**.

**Checkpoint 4 reads a different class of representation from checkpoints 1–3.**

| `hidden_states[k]` | 7 | 14 | 21 | 27 | **28** |
|---|---:|---:|---:|---:|---:|
| elementwise RMS | 49.20 | 49.08 | 49.47 | 50.93 | **3.06** |

Verified directly: `lm_head(hidden_states[28])` reproduces `logits` to
**0.0** absolute error, while `lm_head(norm(hidden_states[28]))` differs by 18.7.
So index 28 is **post-final-RMSNorm** and indices 7/14/21 are raw residual-stream
states. `cos(h27, h28) = 0.104`.

Pooled cross-depth cosine, same prompt:

| pair | 7–14 | 7–21 | 14–21 | 7–28 | 14–28 | 21–28 |
|---|---:|---:|---:|---:|---:|---:|
| cosine | **0.995** | **0.954** | **0.964** | −0.018 | −0.018 | 0.037 |

Two structural consequences:

1. **Checkpoints 1–3 are nearly the same vector** (cosine 0.954–0.995), yet one
   *shared* projector must map them to three *different* targets. That is the
   tight bottleneck, and it is upstream of everything in §3.3.
2. **Checkpoint 4 is in an unrelated basis.** `SSOT §3.1` argues the shared
   projector is "essential" precisely because "separate probes would permit each
   checkpoint to encode the target in an unrelated basis". Reading post-norm at
   stage 4 and pre-norm at stages 1–3 reintroduces exactly that split *inside*
   the supposedly shared interface. The projector's input `LayerNorm` rescues the
   16× scale gap but not the direction.

   Two candidate repairs, either cheap: apply the model's final norm to **all**
   checkpoints before pooling (every depth then sits in the geometry the model
   itself reads from), or move the last checkpoint off the final index so all
   four are pre-norm. The first is the more principled and is recommended.

**The untrained floor for the G1 gate.** Best *shared linear* probe, fit on 150
problems, scored on 150 held out:

| ridge λ | held-out diagonal | held-out off-diagonal | **order margin** | checkpoint top-1 |
|---:|---:|---:|---:|---:|
| 1 | 0.909 | 0.781 | **0.127** | **0.785** |
| 100 | 0.919 | 0.836 | 0.083 | 0.653 |
| 10,000 | 0.870 | 0.822 | 0.048 | 0.517 |

Chance top-1 is 0.25. So an **untrained** backbone with a **linear** shared probe
already reaches 78.5% checkpoint top-1 and an order margin of 0.127 — which is
**71% of the entire 0.179 margin available in the target geometry** (§3.3).

This directly invalidates G1 as currently worded. `PREREGISTRATION.md` §8 asks
that "canonical order margin is positive on DEV", and a positive margin is very
nearly free. **G1's threshold must be set above this measured floor**, not above
zero, or the gate will pass while proving nothing.

### 3.7 Endpoint power

The endpoint loss decomposes cleanly, and the conclusion is **favourable**: the
instrument is repairable and lands near its maximum-power operating point.

| Layer | Accuracy | Owns |
|---|---:|---:|
| Model actually produces the gold answer (cap 384, loose) | **0.438** | — |
| After the scorer's extraction (cap 384, strict) | 0.229 | −0.208 to extraction |
| After the shipped 128-token cap (strict) | **0.115** | −0.114 to truncation |

**As shipped the endpoint is underpowered by roughly an order of magnitude.** At
p ≈ 0.115, with two arms sharing ~86% of their target signal (§3.3) on a common
base model with identical LoRA capacity, the discordant-pair count available to
exact McNemar is small, and most of the variance is harness noise rather than
model behaviour. This is the instrument failure the repository already documented
once, when a nine-arm sweep produced a table in which every ordering was noise.

**Repaired, it is a usable instrument.** Moving to ~0.44 puts a paired binary test
near p = 0.5, where its resolving power is greatest. This changes the Stage-C
calculus materially: the endpoint is no longer the reason to doubt Phase 1.

**Wall-clock is generation-bound, not VRAM-bound**, and is affordable at the
repaired cap:

| Configuration | s / problem | 800 DEV problems | × 18 runs |
|---|---:|---:|---:|
| batch 1, cap 128 | 13.8 | 3.1 h | 55 h |
| batch 4, cap 128 | 1.41–2.66 | 19–35 min | 5.6–10.6 h |
| batch 4, cap 384 (**measured**) | **2.78** | **37 min** | **11.1 h** |

Inference is ~2.5 GiB resident, so the 8 GiB budget is not binding — decode
throughput is. Fixing the cap costs only ~2× decode time, not the ~5× earlier
estimated, so the repaired matrix is an overnight job rather than a multi-day one.
A teacher-forced companion endpoint (§4, Stage B5) remains worthwhile for
resolving power, but it is no longer required to make Stage E feasible.

### 3.8 The decisive Phase-0 result: a factor mismatch, and a fix inside the existing mechanism

§3.3 and §3.6 each looked like a separate problem. They are one problem, and it is
measurable in a single table. Variance decomposition over 600 DEV problems, frozen
backbone, splitting each quantity into a stage (progress) component and a problem
(nuisance) component:

| quantity | η² stage | η² problem | ratio |
|---|---:|---:|---:|
| hidden states, raw | **0.908** | 0.060 | 15.2 |
| hidden deltas, raw | **0.972** | 0.006 | 154 |
| semantic targets `s*`, as shipped | **0.093** | **0.566** | **0.164** |

**The two sides of the objective encode orthogonal factors.** The hidden
representation is ~91% depth-determined and ~6% problem-determined. The target is
~57% problem-determined and ~9% stage-determined. The shared projector is asked to
map a signal carrying almost only depth onto a target carrying almost only problem
identity.

That single fact explains every symptom previously listed separately:

- checkpoint top-1 is easy (0.82) — depth is trivially identifiable from `H`;
- the order margin is tiny (0.129) — stage is only 9% of target variance, so even
  perfect stage identification buys almost no cosine;
- ordered ≈ shuffled — permuting stages rearranges only that 9%, while the 57%
  problem component is **bit-identical in both arms**.

#### The earlier "redundant checkpoints" reading was a measurement artefact

| cross-depth cosine | 7v14 | 7v21 | 14v21 | 21v28 |
|---|---:|---:|---:|---:|
| raw, as shipped | 0.995 | 0.954 | 0.964 | 0.036 |
| per-depth centered | 0.914 | 0.624 | 0.725 | 0.486 |
| **globally centered** | 0.821 | **0.059** | 0.419 | **−0.844** |

Removing one global mean vector separates the depths almost completely. The 0.995
was dominated by a large offset direction shared by every depth and every problem.
The projector's first linear layer has a bias and is therefore insensitive to that
offset — which is why a shared *linear* probe already reaches 0.82 checkpoint top-1,
something impossible if the depths were genuinely near-identical. §3.6 should be
read as "the raw cosine is uninformative", not "the depths are redundant".

#### The fix: correct both geometries, change no mechanism

Shared *linear* probe, frozen backbone, fit on 300 problems, scored on 300 held out:

| readout → target | stage margin | stage top-1 | problem top-1 by stage |
|---|---:|---:|---|
| raw → `s*` (**as shipped**) | **0.129** | 0.82 | .22 / .27 / .33 / .17 |
| destaged delta → target transition | 0.819 | 0.95 | .25 / .12 / .22 |
| raw → within-problem-centered target | 0.977 | 0.83 | .19 / .23 / .26 / .24 |
| **final-norm 7/14/21/27 → centered target** | **0.995** | 0.83 | .30 / .22 / .25 / .25 |

Problem top-1 chance is 0.0033, so every probe runs 36–100× above chance: none of
these are degenerate stage-matching.

Two independent corrections, and their effect sizes are very unequal:

1. **Within-problem target centering** — replace `s*_k` with
   `normalize(s*_k − mean_j s*_j)`. This is the dominant fix:

   | target variant | η² stage | η² problem | ratio | cos(ordered, shuffled) | cos(ordered, reverse) |
   |---|---:|---:|---:|---:|---:|
   | as shipped | 0.093 | 0.566 | 0.164 | **0.860** | 0.825 |
   | within-problem centered | 0.214 | 0.034 | **6.32** | **−0.019** | −0.189 |

   The treatment and its decisive control go from **86% identical to orthogonal**,
   and the target's factor ratio flips from 0.164 to 6.32 — now aligned with the
   hidden side's 15.2 instead of opposed to it. Order margin 0.129 → 0.977.

   This is not discarding signal. The removed component is the problem's identity,
   which is already fully present in the prompt; supervising it spends projector
   capacity on something the model can read directly, and it is the component that
   makes the arms identical. It also stays inside the shipped mechanism —
   cumulative states, one shared projector, one state loss — so it does **not**
   spend the SSOT §13 one-mechanism revision budget.

2. **Symmetric norm readout** — apply the model's final norm to all checkpoints and
   move the last one off the final index:

   | readout | 7v14 | 7v21 | 7v28 | 21v28 | pooled L2 | η² ratio |
   |---|---:|---:|---:|---:|---|---:|
   | raw 7/14/21/28 | 0.995 | 0.954 | **−0.018** | 0.036 | 131/148/222/57 | 15.2 |
   | norm(7/14/21) + raw 28 | 0.887 | 0.689 | **−0.111** | 0.093 | 77/76/70/57 | 8.30 |
   | **norm(7/14/21/27)** | 0.887 | 0.689 | **0.409** | 0.691 | 77/76/70/46 | **5.03** |

   Only the third has no near-orthogonal outlier and a uniform scale, and its
   factor ratio (5.03) is the closest match to the centered target's 6.32.
   Confirms the §3.6 caveat exactly: applying the norm alone does **not** remove
   the endpoint discontinuity — the last checkpoint must also leave the final
   index. Marginal gain over centering alone is small (0.977 → 0.995), but it is
   what makes "one shared interface" true rather than nominal.

#### What this means for the ΔH / semantic-motion proposal

The motion reading was diagnostically right — it is what exposed the factor
mismatch — and the predicted inequality holds strongly:

```text
cos(dH_k^p, dH_k^q)  = 0.94 - 0.99     (same stage, different problems)
cos(dH_i^p, dH_j^p)  = -0.32           (different stages, same problem)
```

an exact reversal of the absolute-state pattern (0.821 within > 0.72 between). And
after removing per-stage means, the destaged delta probe still identifies the
correct problem 40–63× above chance, so the motion carries real problem-specific
content, not just a constant per-depth displacement.

**But the remedy it suggests is not the best available one.** The corrected
*state* route (margin 0.995) beats the *motion* route (0.819) on the same held-out
split, with better and more uniform problem discrimination. So semantic motion
should be recorded as a supported and interesting Phase-2 direction — it is what
`operators.py` already gestures at, though in projected `z` space rather than
hidden-state space — and **not** adopted as the V1 research object. Fixing the
target geometry is cheaper, stays in scope, and measures better.

#### One proposed fix that does not work

Restricting Phase 1 to traces with ≥K distinct states moves cos(ordered, shuffled)
from **0.860 to 0.849** while discarding **43.8%** of the corpus. The degenerate
stratum is not the driver; target problem-dominance is. Recording this so the
effort is not spent.

#### A consequence for the G1 gate

Centering pushes the *untrained* order margin to 0.977–0.995, near saturation. So
mechanism metrics become **necessary but non-discriminative**: after this fix a
positive or even large order margin proves nothing, because the frozen model
already has one. G1 must be stated as a delta over the measured floor, and the
decisive gate must rest on generated capability — which §3.7 shows is repairable
to ≈0.44, near the maximum-power point for a paired binary test. The two
corrections therefore work together: one makes the arms differ, the other makes the
endpoint able to see it.

## 4. Plan

Ordered, with an explicit exit criterion per stage. Stages A and B require no
GPU training. No stage after C may begin until C is recorded.

### Stage A — Repair G0

| # | Work | Verifies |
|---|---|---|
| A1 | Unwrap mapping return in `_as_int_list`; pin a tested `transformers` range; migrate `torch_dtype` → `dtype` | B1 |
| A2 | Device-safe projector comparison in `smoke.py` | B2 |
| A3 | Fix the ruff import-order error | B3 |
| A4 | Real data source per B4-a, with recorded SHA-256 | B4 |
| A5 | Left-pad generation; decode from each row's own prompt length | B5 |
| A6 | Read all K checkpoints from one representation class — apply the model's final norm to every checkpoint before pooling (§3.6). Treat as a **correctness** repair, not a design change: the current code silently mixes post-norm and pre-norm states inside an interface the SSOT calls shared | §3.6 |

**Exit:** on this GPU host — `compileall` PASS, `ruff` clean, `pytest -q` 37/37,
`dast.cli smoke` PASS, `dast validate-data` PASS on the real source, and the
RUNBOOK §3 `inspect-alignment` checks manually confirmed for ordered, shuffled,
and reverse. Update `PROGRESS.md` to state the host and versions under which
qualification passed, replacing the unqualified "PASS".

### Stage B — Complete the readiness report

| # | Work |
|---|---|
| B1 | Tokenizer prefix-stability and prompt-mask seam over ≥2,000 real records; report truncation rate and its balance across arms |
| B2 | Extend §3.3 to ≥2,000 DEV problems, stratified by category and by distinct-target count |
| B3 | **Done at n=300 — see §3.6.** Extend to ≥2,000 problems and re-measure after the §3.6 pre/post-norm repair, since the current floor is contaminated by stage 4 being trivially identifiable from its different basis. Then set G1's threshold above the measured floor |
| B4 | Record the §3.4 census as a protocol fact and choose a frozen policy for it (§Stage C) |
| B5 | **Endpoint power.** Measure DEV accuracy for the base model and for an LM-only (`cot_baseline`) run, then compute the minimum detectable paired effect at n=800 and n=2,400 for 3 seeds. If the endpoint is floored or saturated, the primary metric cannot resolve the contrast at any effect size |
| B6 | Confirm the VRAM **and wall-clock** budget for one arm at the shipped settings (inference already measured at ~2.5 GiB, so VRAM is not the binding constraint; generation throughput may be) |

On B5, the repository has already paid for this lesson once: a nine-arm sweep was
run on a generation-and-score endpoint that could not resolve the effect, and
every ordering in the resulting table turned out to be noise. DAST's primary
outcome is the same *kind* of endpoint — strict generated-answer accuracy, greedy,
`max_new_tokens: 128`, n=800.

The recommended mitigation is to add a **teacher-forced paired secondary endpoint**
(per-position gold-token accuracy and KL against the matched control) with a
bootstrap **clustered by `generator`**, which yields thousands of correlated
observations from the same problems at a fraction of the generation cost. Two
caveats, both verified rather than assumed:

- this does **not** replace the preregistered primary metric — `EVIDENCE_STANDARD.md`
  is explicit that mechanism metrics cannot substitute for generated capability. It
  is a power-bearing companion that decides whether the primary gate is worth
  running at the shipped n;
- there is **no** in-repo implementation to reuse. `experiments/csgqr/src/eval_forced.py`
  is gone (only a stale `.pyc` remains), and the state-sufficiency lab's
  `evaluation/aggregate.py` provides stratified summaries but no clustered paired
  inference. It has to be written, or recovered from git history.

**Exit:** readiness report complete, with §3.6 filled.

### Stage C — Design decision gate

This gate exists because §3.3, §3.4 and §3.5 were found *before* the 18-run
budget was spent, and because `SSOT §4` explicitly permits Phase-0 diagnostics
to motivate a frozen alternative before confirmatory runs. Choose exactly one,
record it, and freeze it:

- **C-a — Proceed as specified.** Only if B5 shows the primary endpoint can
  resolve an effect of the size §3.3 permits.
- **C-b — One bounded revision**, per `SSOT §13`. Candidates, any subset, applied
  **symmetrically to every arm** and frozen before comparative runs:
  - `batch_size ≥ 2`, so the existing anti-collapse term is not inert. This is a
    repair of a dead regularizer, not a new mechanism, but it still changes the
    objective and must be frozen symmetrically.
  - add a **stage**-contrastive term (separate stages within a problem) — the
    axis §3.3 shows is currently unsupervised. This is a new mechanism and
    consumes the one-mechanism revision budget.
  - restrict Phase 1 to traces with ≥K distinct targets, or set K per example, to
    remove the 42% diluted-contrast stratum. Report the excluded stratum
    explicitly; do not let it disappear silently.
  - a target geometry less dominated by surface wording — `SSOT §3.3` already
    flags V1's as "not a proof of a language-independent semantic manifold" and
    `§13` lists "target geometry appears too tied to surface text" as a revision
    trigger.
  - **a depth schedule that is not three near-duplicates plus one outlier.**
    §3.6 shows checkpoints 7/14/21 sit at cosine 0.954–0.995 of one another. Even
    after the A6 norm repair, a shared projector asked to send three
    near-identical inputs to three different targets is the binding constraint.
    `SSOT §4` explicitly anticipates this: the 7/14/21/28 partition is "a first
    controlled schedule, not an assumption that natural reasoning roles are
    exactly equal in depth", and Phase-0 diagnostics "may later motivate a frozen
    alternative schedule". This is that evidence. Any alternative must be chosen
    **once**, frozen, and never searched against DEV — and never against TEST.

`SSOT §13` forbids searching many schedules until one passes. Stage C is one
decision, recorded once, before any comparative run.

**Exit:** `PREREGISTRATION.md` placeholders all resolved; file committed;
SHA-256 recorded; `status: FROZEN`.

### Stage D — G1 learnability rehearsal

SSOT §11 gate 3: 64–128 problems, one seed, `cot_baseline` + `state_ordered` +
`state_shuffled`. Confirm the state loss falls from initialisation, the order
margin is positive **above the Stage-B3 untrained floor**, the projector does not
collapse to a problem-identity vector, and LM loss is not damaged. Record VRAM,
wall time, and tokens/s to budget Stage E.

**Exit:** G1 pass/fail recorded. A collapse here closes the tested formulation
without spending Stage E.

### Stage E — Phase-1 confirmatory DEV matrix

Six arms × seeds 17/29/43 = 18 immutable runs, per RUNBOOK §6. Copy configs per
seed; never overwrite a run; retain failed and interrupted runs.

### Stage F — G2/G3 decision

Ordered vs shuffled **and** reverse (jointly required, per `PREREGISTRATION.md`
§6), plus ordered vs final-repeated. Record **CLOSED — NEGATIVE** for the tested
V1 formulation if the gates fail, and do not advance to operators or feedback.

### Stage G — Documentation duty

Two items already outstanding, both required by RUNBOOK §12:

- `../../../docs/EXPERIMENT_PROGRAM_AUDIT.md` does not mention DAST at all, and
  its line 11 still asserts the remote exposes "**one live branch: `main`**".
  Origin now also carries `research/depth-aligned-semantic-transitions-v1` and
  `research/rge-peft-v1`.
- `SSOT §10` promises "problem-cluster bootstrap intervals", but
  `compare.py` bootstraps individual problems i.i.d. Because problems from one
  generator/template are correlated, i.i.d. intervals are anti-conservative.
  `SSOT §10` also offers to "reuse the mature state-sufficiency evaluator after an
  explicit adapter bridge" — but that evaluator does not implement clustered paired
  inference either, so the bridge would not supply the promised interval. Either
  cluster the bootstrap by `generator` (already carried on every `ReasoningRecord`,
  so this is a small change) or amend the plan. Do not leave the document and the
  code disagreeing.

## 5. What would change the program's classification

Nothing in this report supports or weakens the depth-alignment thesis. It
changes only what a Phase-1 run could *detect*. The honest summary is:

> The DAST V1 harness is well specified and, apart from six repairable defects,
> faithfully implements its own algorithm document. The open risk is mostly
> **power**, and it compounds from both ends of the interface. On the target
> side, the treatment and its decisive control share ~86% of their signal, 42%
> of the corpus dilutes the contrast further, and the term meant to prevent the
> collapse that would erase the contrast entirely is inert at `batch_size: 1`.
> On the input side, three of the four checkpoints are 95–99% the same vector
> while the fourth sits in an unrelated basis — so one shared projector is asked
> to separate what is nearly unseparable at three depths and to learn a second
> regime at the fourth. An untrained linear probe already captures 71% of the
> available order margin, so the G1 gate as worded would pass on the pretrained
> prior alone.

Two of these are plain repairs (A6, and `batch_size ≥ 2`). The rest is a real
design question that the SSOT already anticipated in §4 and §13. Resolving Stage
B5 and recording Stage C is the shortest honest path to either a meaningful
Phase-1 result or a documented, non-wasteful revision — and it costs a small
fraction of the 18-run matrix it protects.
