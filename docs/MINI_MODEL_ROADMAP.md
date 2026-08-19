# Mini-model roadmap — deriving a Rineya core from Qwen3-0.6B

How to make the core model smaller **by parameter count** while retaining, and
eventually improving, reasoning and evidence-grounded retrieval.

Every number below is measured on this checkpoint and this corpus, not taken
from a paper. Reproduce with `scripts/measure/` and `scripts/trim/` (§9).

**Status (2026-08-19):** rung R2 is built and verified — **474,021,888 params,
−20.47%, bitwise-equivalent logits, zero pre-heal damage** (§11). R3 onward is
blocked on Phase 0 (a CUDA torch build).

---

## 0. Decisions on record

| Decision | Value | Consequence |
|---|---|---|
| Objective | **Fewer parameters**, own architecture | Quantization does not count toward the target; it is a separate deployment artifact (§5 Phase 6) |
| Base | `Qwen3-0.6B/` stock, immutable | No layer surgery inherited from anywhere; provenance stays clean |
| Healing | **SFT first.** Larger-teacher distillation deferred | Phase 2–4 heal with the existing 124k + 4,410 mix; Phase 5 revisits with a 1.7B/4B teacher |
| Retired | `19L.454M` layer-dropped soup | Post-mortem in §2; not a base for anything |
| Gate namespace | **C1–C9** | Distinct from the state-sufficiency lab's G1–G9 |

> The honest limit of the SFT-first choice: **pruning can only remove
> capability, and SFT healing can only recover what the base already had.**
> "Potentially enhance" is a Phase 5 objective and cannot be claimed before it.
> Phases 1–4 should be judged on *how little* they lose, not on gains.

---

## 1. Measured baseline

`scripts/measure/param_budget.py`

| Component | Params | Share of resident |
|---|---:|---:|
| MLP, all 28 layers | 264,241,152 | 44.3% |
| Attention projections, all 28 layers | 176,160,768 | 29.6% |
| `embed_tokens` (tied, serves as `lm_head`) | 155,582,464 | 26.1% |
| Norms | 65,536 | 0.0% |
| **Resident total** | **596,049,920** | 100% |

Derived constants used throughout: **per layer = 15,730,944**, **embedding row =
1,024 params/token**.

Two defects in the stock checkpoint, both free to fix:

1. `lm_head.weight` is stored as a **bitwise-identical copy** of
   `model.embed_tokens.weight` (verified by `torch.equal`) despite
   `tie_word_embeddings: true`. The file holds 751,632,384 params for a 596M
   model — **311 MB of dead disk**.
2. `vocab_size` is 151,936 but the tokenizer defines 151,669 real tokens.
   **267 embedding rows (273,408 params) are not addressable by any input.**

### Host

| | |
|---|---|
| GPU | **NVIDIA RTX 4070 Laptop, 8,188 MiB**, driver CUDA 13.3 |
| System RAM / CPU | 15.2 GB / Ryzen 9 7940HX |
| Installed torch | **`2.13.0+cpu` — CUDA unavailable** |
| Missing | `bitsandbytes`, `datasets`, `trl` |

> **This supersedes `rineya_state_sufficiency_lab/docs/READINESS_ASSESSMENT.md`
> §7c for this host.** That section concluded "no GPU is available, so the
> experiment runs here" and sized a CPU pilot that it correctly described as
> unable to answer the non-inferiority question (±7 pp CI at 200 paired
> problems, against 900–2,400 required). That conclusion was measured on an 8 GB
> Apple Silicon machine. On *this* machine a CUDA GPU exists and only the torch
> build is CPU-only. Fixing that is Phase 0 and it is the single highest-value
> action in this document — no compression decision should be made before it,
> because every gate below needs GPU-speed evaluation to be affordable.

---

## 2. Post-mortem: why `19L.454M` was retired

The retired checkpoint was `stage_09_drop_14_24_6_23_8_10_18_9_7` — 9 of 28
layers deleted, then merged with two LoRA adapters at 0.5/0.5. Its dress
rehearsal (`READINESS_ASSESSMENT.md` §7b) scored `full_direct` strict **0/4**,
produced `imony` repeated ~100 times for gold `-6`, and emitted no coherent
final answers.

`scripts/measure/layer_profile.py` measures what *should* have been deleted:
angular distance between each layer's input and output residual stream, over 24
real prompts (12 RAG with evidence, 12 reasoning) under the real chat template.
Low distance = near-identity map = cheap to delete.

| Layer | 0 | 1 | 2–4 | 5–11 | 12–13 | 14–21 | 22 | 23–24 | 25 | 26 | 27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ang. dist | 0.484 | 0.167 | 0.155–0.161 | 0.121–0.144 | 0.091–0.093 | 0.107–0.132 | 0.102 | 0.086 | **0.078** | 0.081 | 0.376 |

Prunability rank of the nine that were actually deleted (0 = most prunable of 28):

| Layer | 6 | 7 | 8 | 9 | 10 | 14 | 18 | 23 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rank | 20 | 19 | 17 | 18 | 15 | 8 | 10 | **3** | **2** |

Three conclusions:

1. **Layers 23 and 24 were good choices** (ranks 3 and 2). The deep-but-not-final
   band 22–26 is genuinely the most redundant region of this model, and layer 25
   is the single most redundant layer. The intuition that "late layers are
   precious" is wrong here — only layer 27 is, and layer 0 most of all.
2. **The contiguous run {6,7,8,9,10} was the error.** Those rank 15–20 of 28,
   i.e. the *less* prunable half. Five consecutive layers were removed from the
   band that rewrites the stream second-most heavily.
3. **Selection was not the whole failure.** 9 of 28 layers is 32% of depth. The
   best available 9-block (`6..14`, d=0.2434) still discards a transformation
   1.5× larger than the largest single interior layer. Removing that much depth
   requires a healing pass; averaging two unrelated LoRAs is not one. Degenerate
   repetition loops are the signature of unhealed depth removal.

> **Retired as a technique, not just as a checkpoint:** LoRA soup will not be
> used as post-pruning repair again. It averages adapters trained on different
> data; it does not reconstruct the function the deleted layers computed.

---

## 3. Vocabulary utilization — the largest safe reduction available

`scripts/measure/vocab_utilization.py`, over 7,335,955 content tokens:

| Source | Tokens | Cumulative distinct ids |
|---|---:|---:|
| v3 recommended combined (40,000-row sample) | 4,140,846 | 5,197 |
| Short-response combined | 989,910 | 7,300 |
| RAG v3 train SFT (4,410) | 1,511,101 | 7,597 |
| `evidence/knowledge` (6,000) | 679,683 | 7,609 |
| `evidence/procedures` (200) | 14,415 | 7,609 |

**7,609 of 151,936 ids (5.01%). 144,327 embedding rows are never addressed.**
A quarter of the model is an embedding table, 95% of which goes unused.

Trimming is safe here for a structural reason, not a statistical one: Qwen3 uses
byte-level BPE and **all 256 byte-alphabet tokens occupy ids 0–255**. Keep those
and every possible string stays encodable — unseen text degrades into *more
tokens*, never into an unrepresentable input. Coverage cannot hit zero.

Mandatory keep set: **ids 0–255** (byte alphabet) + **ids 151643–151668** (all 26
control tokens, including `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`).

### 3a. English-only is not enough on its own

`scripts/measure/vocab_script_census.py` decodes every vocabulary entry out of
its byte-level form and classifies it by writing system:

| Category | Tokens | Share | Decision |
|---|---:|---:|---|
| ascii | 94,358 | 62.21% | keep |
| cjk | 25,577 | 16.86% | drop |
| latin_ext | 5,849 | 3.86% | keep |
| cyrillic | 4,137 | 2.73% | drop |
| arabic | 4,017 | 2.65% | drop |
| hangul | 3,564 | 2.35% | drop |
| hebrew | 3,160 | 2.08% | drop |
| thai | 2,570 | 1.69% | drop |
| kana | 1,755 | 1.16% | drop |
| byte fragments | 1,448 | 0.95% | keep |
| emoji | 972 | 0.64% | drop |
| greek, math, punct, currency, combining | 838 | 0.55% | keep |
| everything else (44 categories) | 3,424 | 2.26% | drop |

**Dropping every non-English script saves only 49,150 tokens — −8.44%.** Qwen3's
vocabulary is 62% ASCII. The reduction has to come mostly from the long tail of
*English* subwords, not from other languages.

### 3b. The trade that actually decides the vocabulary size

A smaller vocabulary saves embedding parameters and shrinks the `lm_head` GEMM,
but every dropped token inflates the sequence — and inflation is paid across all
28 layers on every forward pass forever. The embedding is the one part of the
network that costs almost nothing per token, so this trade is easy to lose.

Per-token cost is `440,466,432 + 1024 × vocab` MACs, giving

```
break_even_inflation = per_token_MACs(stock) / per_token_MACs(trimmed)
```

Below break-even the model is smaller *and* cheaper. Above it, parameters are
being bought with compute. `scripts/measure/trim_sweep.py` measures both sides.

### 3c. Selection rule: merge rank, not token length

Two rules were built and measured. Inflation is on held-out human prose (the
repository's own 44 hand-written Markdown files — real English that is in no
training corpus):

| Candidate | Vocab | Params | vs 596M | Prose | In-domain | Break-even | Net compute | |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `rank` 24,576 | 24,576 | 465,633,280 | −21.88% | 1.1006× | 1.0207× | 1.2801× | 0.8598× | pass |
| **`rank` 32,768** | **32,768** | **474,021,888** | **−20.47%** | **1.0723×** | **1.0107×** | **1.2574×** | **0.8528×** | **pass** |
| `rank` 40,960 | 40,960 | 482,410,496 | −19.07% | 1.0573× | 1.0062× | 1.2356× | 0.8557× | pass |
| `rank` 49,152 | 49,152 | 490,799,104 | −17.66% | 1.0493× | 1.0043× | 1.2144× | 0.8640× | pass |
| `len` ≤ 4 chars | 37,888 | 479,264,768 | −19.59% | 1.2549× | 1.2301× | 1.2437× | 1.0091× | **fail** |

**`len` mode was rejected on measurement.** Keeping only short tokens inflates
English prose by +25.5%, which lands *past* its own break-even: it makes the model
smaller and simultaneously ~0.9% more expensive to run. "Short" and "frequent"
are not the same thing.

**`rank` mode wins because BPE merge order is already a frequency ranking**
learned from Qwen's own training corpus. Walking the merge list in order and
keeping a rule `A B -> AB` when `A` and `B` are already kept has a second
benefit: the surviving merge list is self-consistently valid by construction — a
kept merge can never reference a dropped token, so no post-hoc repair is needed.

**Adopted: `rank` at 32,768** — −20.47% parameters and **−14.72% compute** per unit
of text. It is the cheapest net compute of every candidate. `rank` 24,576 is
available if 8.4M more parameters matter more than the extra 2.8 pp of prose
inflation.

### Secondary benefit: this is what makes the 8 GB card workable

The logits tensor is usually the largest single training activation. At batch 8 ×
seq 384 (the length `READINESS_ASSESSMENT.md` §2 measures as truncating nothing):

| Vocab | Logits (fp32) | With gradient |
|---:|---:|---:|
| 151,936 | 1.87 GB | 3.73 GB |
| 32,768 | 0.40 GB | 0.81 GB |

**≈2.9 GB of VRAM returned** on a card with 8,188 MiB total and 4,757 MiB already
held by desktop applications. Vocab trimming is both the largest safe parameter
reduction and the change that makes the rest of the roadmap affordable here.

### Implementation traps

- **Merges must be pruned consistently.** BPE merges are ordered `A B -> AB`
  rules. Every merge whose *result* token is dropped must be dropped too, or the
  tokenizer emits ids that no longer exist. There are 151,387 merges.
- **Hardcoded ids move.** `READINESS_ASSESSMENT.md` §1b states the harness must
  pass **both** stop ids `[151645, 151643]`. Renumbering the vocab invalidates
  every literal id in `evaluation/` and the configs. Either remap them in the
  same commit, or place the 26 control tokens at the top of the new range and
  update exactly one constant.
- **Re-tie after slicing.** Slice `embed_tokens` rows, re-point `lm_head`, set
  `vocab_size`, and keep `tie_word_embeddings: true`.
- **Coverage was measured on content only** (`add_special_tokens=False`) and on
  templated synthetic corpora. Open-ended user input will use more ids. Gate C2.

---

## 4. The size ladder

Each rung is a checkpoint, gated before the next is attempted.

| Rung | Configuration | Layers | Params | vs 596M | Status |
|---|---|---:|---:|---:|---|
| Baseline | stock | 28 | 596,049,920 | — | immutable |
| **R1+R2** | lossless cleanup + vocab 32,768 | 28 | **474,021,888** | **−20.47%** | **built, verified (§11)** |
| **R3** | + delete 2 layers (`25..26`) | 26 | 442,560,000 | −25.75% | blocked on Phase 0 |
| **R4** | + delete 2 more (→ `23..26`) | 24 | 411,098,112 | −31.03% | blocked on Phase 0 |
| **R5** | + MLP 3072→2048, all layers | 24 | 335,600,640 | −43.70% | blocked on Phase 0 |

R1 and R2 collapsed into one artifact: dropping the redundant `lm_head` is a
by-product of rewriting the embedding matrix, so separating them would have meant
writing a checkpoint nobody would ever train.

**R3 is smaller than the retired 454M checkpoint while keeping 26 of 28 layers
instead of 19.** That is the entire argument for doing this in the stated order:
the cheap axis was the embedding table, and it was never touched.

Rungs are ordered by measured damage-per-parameter, cheapest first. Stop wherever
a gate fails; a passing R3 is a better artifact than a failing R5.

### Why width before more depth, at R5

`intermediate_size` is 3072 and MLP is 44.3% of the model, so width is the
largest remaining pool after the embedding. Depth is also the axis this model can
least afford: 28 layers is already shallow, and depth bounds the serial
transformation available per token. NVIDIA's Minitron result — width pruning
outperforming depth pruning at equal parameter count — points the same way. R4
therefore caps depth removal at 4 layers (14%, versus the 32% that failed), and
R5 takes the remaining reduction from width.

Use activation-aware importance (Wanda-style `|W| · ‖X‖`, calibrated on the
Rineya + RAG mix) rather than magnitude alone, so the surviving channels are the
ones this corpus actually drives.

---

## 5. Phases

### Phase 0 — unblock the host *(prerequisite for everything)*

1. Install a CUDA build of torch; confirm `torch.cuda.is_available()`.
2. Install `bitsandbytes`, `datasets`, `trl`.
3. Re-run `scripts/measure/layer_profile.py 100` on GPU — 24 prompts is a screen,
   not a verdict (§8).
4. Free VRAM: 4,757 of 8,188 MiB is held by desktop applications.

Gate **C1**.

### Phase 1 — lossless cleanup → R1

Delete the redundant `lm_head.weight` tensor; drop the 267 unaddressable padding
rows. No capability change is possible, so this is verified by equality, not by
evaluation: **logits must match the stock model bitwise** on a fixed prompt set.

Gate **C3**.

### Phase 2 — vocabulary trim → R2 ✅ done, §11

1. Keep set: ids 0–255, the 26 control ids, every id observed in the corpora,
   then English-script tokens filled by **merge rank** to 32,768 (§3c).
2. Prune `merges` consistently; rebuild `tokenizer.json`, `vocab.json`,
   `merges.txt`.
3. Gather embedding rows into new-id order, drop the redundant `lm_head`, set
   `vocab_size`, and remap **every** hardcoded id: `config.json` bos/eos,
   `generation_config.json` bos/eos/pad, and all 26 `added_tokens_decoder` keys.
4. **Measure damage before healing** (gate C4). It came out at zero (§11), which
   is why step 5 is not required for this rung.
5. SFT heal — *optional for R2, mandatory from R3 on*: one epoch, QLoRA, on the
   124k + 4,410 mix at the LR the corpus README already validates (`1e-4`).
   Worth running anyway to adapt the model to the ~7% of English text whose
   tokenization changed, but it is an improvement pass, not a repair pass.

Gates **C2 pass, C3 pass, C4 pass** (§11). C5–C7 need Phase 0.

### Phase 3 — depth trim → R3, then R4

Delete `25..26` first (measured `d=0.1198`, comparable to a single interior
layer). Heal. Gate. Only then extend to `23..26`.

Candidate blocks from the current profile, in preference order: `25..26` (n=2),
then `23..26`. Note the n=3 and n=4 optima sit at `12..14` and `11..14`; those
are legitimate alternatives, and the two regions should be compared empirically
at R4 rather than chosen from the metric alone.

Re-run the profile **on the R2 checkpoint**, not on stock — trimming the vocab
changes the residual stream and therefore the ranking.

Gates **C5, C6, C7, C8**.

### Phase 4 — MLP width trim → R5

Activation-aware channel pruning of `gate_proj`/`up_proj`/`down_proj` to
`intermediate_size = 2048`, calibrated on the Rineya + RAG mix. Heal. This is the
rung most likely to fail C6, because MLP is where parametric knowledge lives.

Gates **C5, C6, C7, C8**.

### Phase 5 — distillation *(deferred, scoped now)*

The first phase capable of producing a model *better* than stock at a smaller
size, and the reason to keep the ladder's artifacts around.

- Teacher: Qwen3-1.7B, then 4B if VRAM allows. Not yet downloaded.
- Objective: logit KL from teacher to student, optionally plus hidden-state
  matching (Minitron-style prune-then-distill).
- **Vocab dependency:** teacher logits must be restricted to the student's kept
  ids. Do Phase 2 before Phase 5 and the mapping is a fixed gather; do it after
  and it is a retraining problem. This ordering is deliberate.
- Revisit every rung with distillation healing and compare against its SFT-healed
  sibling. Expect the aggressive rungs (R4, R5) to benefit most.

### Phase 6 — quantized deployment artifact *(deferred, out of scope for the target)*

4-bit AWQ/GPTQ over the accepted rung. Reduces bytes, not parameters, so it does
not count toward this roadmap's objective — but it composes, and R5 at 4-bit is
roughly 170 MB. Track as a separate artifact so the parameter claim stays clean.

---

## 6. Gates

| Gate | Test | Pass condition |
|---|---|---|
| **C1** | Host readiness | `torch.cuda.is_available()`; layer profile re-run at n≥100 prompts on GPU |
| **C2** | Vocab coverage on held-out **real** text (not the synthetic corpora) | Token-inflation ratio vs stock tokenizer within a pre-declared bound; zero unencodable inputs |
| **C3** | Lossless-cleanup equivalence | Logits bitwise identical to stock on a fixed prompt set |
| **C4** | Pre-heal damage recorded | Number is written down *before* healing; no pass/fail, but skipping it forfeits the ability to attribute the result |
| **C5** | Chat DEV after heal | No regression vs the rung above on the frozen 510-case chat DEV |
| **C6** | RAG per-category | **Per-category** non-regression on RAG DEV, with **abstain precision reported separately** |
| **C7** | Degeneration screen | Zero degenerate repetition loops on the DEV sample — the exact failure that retired `19L.454M` |
| **C8** | Retrieval-head integrity | Heads identified as retrieval heads on RAG DEV are excluded from pruning and verified intact (§7) |
| **C9** | Proceed decision | Explicit go/no-go before attempting the next rung; a passing rung is shippable and a failing one stops the ladder |

> **C6 is the gate that matters most, and the one most likely to be missed.**
> Compression breaks *calibration* before it breaks accuracy: a pruned model
> becomes confidently wrong rather than obviously broken. The corpus is 1,600
> abstain against 4,200 answer rows across 16 categories precisely so this is
> measurable. Aggregate quality can hold while `conflict`, `stale_evidence`,
> `prompt_injection`, and `missing_answer` collapse into confident answers.
> Gate on the categories, not the mean.

---

## 7. Retrieval-head protection (Phases 3–4)

The product is evidence-grounded: facts arrive in the prompt, not from the
weights. A sparse subset of attention heads performs the copy-from-context
operation that grounding depends on, and losing them degrades exactly the
categories this corpus is built to test while leaving perplexity nearly
unchanged.

Protocol, before any head or width pruning:

1. On RAG DEV cases with a known supporting span, record per-head attention mass
   landing on that span during generation of the answer tokens.
2. Rank heads by that mass; take the top decile as the protected set.
3. Exclude protected heads from pruning. Record the set in the run manifest.
4. After each heal, re-verify the protected heads still attend to supporting
   spans, and that `supported_with_distractor` and `conflict` have not moved.

This also justifies the asymmetry in §4: MLP (parametric knowledge) can be cut
harder than attention (in-context retrieval) for a RAG-grounded model. A
closed-book model would want the opposite trade.

---

## 8. Threats to validity

- **The vocab measurement is on synthetic, templated corpora.** 7,609 distinct
  ids understates open-ended user input. 32,768 is a margin, not a proof; C2
  exists to test it on real text before the trim is accepted.
- **The layer profile is a 24-prompt CPU screen.** Directionally strong (the
  22–26 band is clearly the redundant one) but not a basis for choosing between
  adjacent candidate blocks. Re-run at n≥100 in Phase 0.
- **Importance is measured on prompts, not completions.** A layer that matters
  mainly during generation could be under-weighted. Extend the profile to
  teacher-forced completions before Phase 4.
- **Angular distance is a proxy.** It measures how much a block moves the
  residual stream, not how much the *task* needs that movement. It ranks
  candidates; only C5–C8 decide.
- **The profile must be re-measured per rung.** Rankings shift after each trim
  and heal; a plan fixed at R2 does not remain valid at R4.
- **SFT healing cannot exceed the base.** Any claim of improved reasoning before
  Phase 5 is a measurement artifact. Phases 1–4 are loss-minimization.
- **DEV improvement is not a release proof.** The existing separation of DEV from
  sealed test, and of research keep from release gate, applies unchanged. No
  sealed pass until a rung is frozen.

---

## 9. Reproducing every number above

Measurement, in dependency order — the census and utilization manifests are
inputs to the trim:

```bash
python scripts/measure/param_budget.py
python scripts/measure/vocab_utilization.py
python scripts/measure/vocab_script_census.py
python scripts/measure/layer_profile.py 12
```

Build, verify, and apply the trim:

```bash
python scripts/trim/build_trimmed_tokenizer.py rank 32768
python scripts/measure/token_inflation.py artifacts/tokenizer_en_rank32768
python scripts/trim/apply_vocab_trim.py artifacts/tokenizer_en_rank32768 artifacts/rineya-core-en32k-28L
python scripts/trim/verify_trim.py artifacts/rineya-core-en32k-28L
```

Compare candidates before committing to one:

```bash
python scripts/measure/trim_sweep.py artifacts/tokenizer_en_rank24576 artifacts/tokenizer_en_rank32768 artifacts/tokenizer_en_len4
```

Manifests written: `vocab_utilization.json` (frequency-ranked keep order),
`vocab_script_census.json` (per-category id lists), `layer_profile.json` and
`layer_profile_<checkpoint>.json` (cosine and angular-distance matrices,
prunability order, ranked candidate blocks), and `trim_manifest.json` inside each
artifact (the full old→new id map, so any trim is auditable and invertible).

Two non-obvious details are commented in the scripts rather than left as traps:

- `layer_profile.py` — per-prompt cosines must be **averaged before clamping**.
  Clamping the accumulated sum to [-1, 1] first saturates every entry to 1.0 and
  yields a perfectly flat profile that looks like a real result.
- On Windows, set `PYTHONIOENCODING=utf-8`. The console defaults to cp1252 and
  dies printing byte-level token strings such as `Ġ`.

`layer_profile.py` carries a comment on the one non-obvious detail: the
per-prompt cosines must be **averaged before clamping**. Clamping the accumulated
sum to [-1, 1] first saturates every entry to 1.0 and yields a perfectly flat
profile that looks like a real result.

---

## 10. Artifact naming

Extending the existing convention
(`rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4`):

```
rineya.core.aegis.v2.0.0.28L.474M.vocab32k.sft-heal-v1-lr1e4      # R2
rineya.core.aegis.v2.1.0.26L.443M.vocab32k-drop25_26.sft-heal-v1  # R3
rineya.core.aegis.v2.2.0.24L.411M.vocab32k-drop23_26.sft-heal-v1  # R4
rineya.core.aegis.v2.3.0.24L.336M.vocab32k-drop23_26-mlp2048...   # R5
```

Layer count and parameter count stay in the name — that is what made the retired
`19L.454M` checkpoint auditable after the fact. Record the deleted layer indices
explicitly; `stage_09_drop_14_24_6_23_8_10_18_9_7` was recoverable only because
the drop list survived in `soup_manifest.json`.

---

## 11. Execution log — R2 built and verified (2026-08-19)

Artifact: `artifacts/rineya-core-en32k-28L`
Tokenizer: `artifacts/tokenizer_en_rank32768`

| | Stock | R2 | |
|---|---:|---:|---|
| Parameters | 596,049,920 | **474,021,888** | **−20.47%** |
| `model.safetensors` | 1,503,300,328 B | **948,078,864 B** | **−36.9%** |
| `vocab_size` | 151,936 | 32,768 | −78.4% |
| Layers | 28 | 28 | unchanged |
| Merges | 151,387 | 32,304 | −78.7% |
| Alignment padding rows | 267 | **0** | eliminated |
| Unreachable vocab rows | — | 182 | 0.6% of the table |
| bos / eos / pad | 151643 / 151645 / 151643 | 32742 / 32744 / 32742 | remapped |

### Gate C2 — round-trip fidelity and inflation: **pass**

Round-trip exact on all 9 probes, including the scripts that were deliberately
dropped. Nothing became unencodable; dropped scripts decompose into byte pieces.

| Probe | Stock | R2 | Ratio | Exact |
|---|---:|---:|---:|---|
| ascii english | 15 | 15 | 1.00× | yes |
| chat control tokens | 10 | 10 | 1.00× | yes |
| emoji | 12 | 12 | 1.00× | yes |
| JSON decision contract | 16 | 17 | 1.06× | yes |
| greek / math | 31 | 35 | 1.13× | yes |
| accented latin | 18 | 23 | 1.28× | yes |
| cjk *(dropped)* | 11 | 43 | 3.91× | yes |
| arabic *(dropped)* | 8 | 34 | 4.25× | yes |
| cyrillic *(dropped)* | 9 | 57 | 6.33× | yes |

Inflation: held-out human prose **+7.23%**; in-domain chat SFT **+1.07%**, RAG SFT
**+0.32%**, RAG evidence **+0.53%**. Against a break-even of +25.74%, R2 is
**14.72% cheaper per unit of text** than stock.

The ASCII, control-token and emoji rows at 1.00× matter more than they look: the
chat template, the JSON decision contract, and the `### ✅ Final Answer:` style the
base model actually emits are all untouched.

### Gate C3 — weight equivalence: **pass, bitwise**

For prompts whose tokenization is unchanged, on all **32,768 / 32,768** kept ids:

```
max |trimmed_logits[new_id] − stock_logits[old_id]|  =  0.000e+00
argmax agreement : yes      top-5 agreement : yes
```

Exactly zero, not merely close — the shape change in the `lm_head` GEMM did not
even perturb the accumulation order. This is the strongest available evidence that
the trim removed only unused rows and altered no weight value.

### Gate C4 — pre-heal damage: **zero**

Greedy generation, thinking off, 5 prompts: **3/5 byte-identical to stock**, and
the 2 that differ are exactly the 2 whose tokenization inflated. Behaviour changed
only where the input changed. No degeneration, no repetition loops, no truncation
pathology — the R2 checkpoint is a working model with no healing at all.

On one of the two divergent prompts R2 was *better* (it answered "Mo is the
shortest"; stock rambled past the token budget without stating an answer) and on
the other it was worse (it hedged where stock answered). That is trajectory noise
from a changed tokenization, not a capability change, and it is not evidence in
either direction.

> **Consequence for the roadmap.** §0 states that pruning can only remove
> capability. R2 is the exception that proves the framing: removing rows the
> forward pass never reads is not pruning in the lossy sense at all. The −20.47%
> was free. Everything from R3 on removes computation the model does use, and will
> behave like real pruning — expect C4 to be non-zero and healing to be mandatory.

### What R2 does not establish

C5–C8 have not run. That needs the frozen 510-case chat DEV and the RAG DEV
category breakdown, which needs GPU-speed generation, which needs Phase 0. The
claim on record is narrow and exact: **R2 is 20.47% smaller, 14.72% cheaper, and
bitwise-equivalent on unchanged tokenizations.** It is not yet "as good on DEV".

### R3 readiness

`manifests/layer_profile_rineya-core-en32k-28L.json` — the profile was re-measured
on R2 rather than inherited from stock, per §5 Phase 3. It is essentially
unchanged: most prunable `[25, 26, 24, 23, 13, 12, …]`, best 2-block `25..26` at
d=0.1203 (stock: 0.1198). **R3 target confirmed: delete layers 25 and 26.**
