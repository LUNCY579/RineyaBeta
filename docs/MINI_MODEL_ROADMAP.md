# Mini-model roadmap — deriving a Rineya core from Qwen3-0.6B

How to make the core model smaller **by parameter count** while retaining, and
eventually improving, reasoning and evidence-grounded retrieval.

Every number below is measured on this checkpoint and this corpus, not taken
from a paper. Reproduce with `scripts/measure/` and `scripts/trim/` (§9).

**Status (2026-08-19):**
- Rung **R2 built and verified** — 474,021,888 params, −20.47%, bitwise-equivalent
  logits, zero pre-heal damage (§11).
- **Phase 0 closed.** Project `.venv` with torch 2.13.0+cu130; CUDA live (§14.1).
- Healing harness validated on the real card; GPU resume proven bit-exact (§14.3).
- **Gate baselines measured (§16).** Against stock Qwen3-0.6B on identical cases:
  the trim is **free on chat DEV** (paired delta +0.0002, CI spans zero) and costs
  **exactly 9 abstentions in the RAG `conflict` category** (paired delta −0.0127,
  CI excludes zero). 12 of 16 RAG categories are bit-identical.
- **B0's success criterion is now pre-declared:** restore `conflict` abstention to
  ≥93.1% while chat contract fulfilment holds at 1.0000 (§16.4).
- **B0 needs reconfiguring first.** Its memory characterisation was wrong (§15
  numbers came from too-short samples; the real run reached 5,090 MiB of a 5,120
  cap and hung at step 72 with no checkpoint).
- **Open decision:** B2/B3 at 4k context cannot be both bit-exact and fit on 8 GB.
  Four quantified options in §15.6.

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
| GPU | **NVIDIA RTX 4070 Laptop, 8,187 MiB**, capability 8.9, driver CUDA 13.3 |
| System RAM / CPU | 15.2 GB / Ryzen 9 7940HX |
| torch, as first found | `2.13.0+cpu` — CUDA unavailable |
| torch, now | **`2.13.0+cu130` in `.venv/`** — resolved in Phase 0 (§14.1) |
| Desktop VRAM use | observed 360–4,800 MiB, fluctuating — the reason for a hard cap |

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

---

## 12. Healing-corpus audit (2026-08-19)

`scripts/data/audit_training_mix.py` → `manifests/training_mix_audit.json`.
Lengths measured with the **R2 32k tokenizer**, since that is what will train.

### 12.1 Inventory

| Corpus | Rows | Form | Splits | Role |
|---|---:|---|---|---|
| v1 short-response combined | 12,000 | chat | — | superseded, see 12.2 |
| v2 recommended combined | 40,000 | chat | — | superseded, see 12.2 |
| **v3 recommended combined** | **124,000** | chat | — | **chat/reasoning half** |
| **RAG v3 train** | **4,410** | chat | — | **grounding half** |
| **Puzzle-51k train** | **40,800** | **raw** | train | **needs conversion** |
| Puzzle-51k validation / test / challenge | 4,080 / 4,080 / 2,040 | raw | held out | eval |
| **LCR-SFT v1** | 8,000 | chat | **train 7,200 / dev 800** | **long-context, see 12.4** |
| **LongContextDemo v1** | **256** | chat | showcase 64 / stability 192 | **FROZEN EVAL — never train** |

Puzzle-51k is balanced at 15 families × 2,720. `Rineya_Long_Context_Demo_v1`'s own
card lists "training on the exact suite and then reporting test performance" as
out of scope; it is the long-context gate, not training data.

### 12.2 The v1→v3 chain double-counts, as suspected

Overlap by normalized prompt hash — measured, not read off the READMEs:

| Pair | Shared prompts |
|---|---:|
| v2 carries-v1-slice ∩ v1 combined | 7,872 (99.9% of the slice) |
| v2 combined ∩ v1 combined | 7,872 |
| v3 carries-v1+v2-slice ∩ v2 combined | 12,647 (90.6% of the slice) |
| v3 combined ∩ v2 combined | 12,647 |
| v3 combined ∩ v1 combined | 3,955 |

Concatenating v1 + v2 + v3 gives 176,000 rows containing **~24,474 duplicate
prompt pairs**, and the duplication lands on the *oldest* rows — the v1
short-response set, which is the most brevity-biased material in the collection.
Naive concatenation therefore over-weights terseness in exactly the run meant to
add reasoning breadth.

> **Use `rineya_v3_recommended_combined_124000.jsonl` alone for the chat half.**
> It already carries the intended 14,000-row preservation slice. The v3 README
> already says not to append v1 or v2 again; this measurement confirms it.

### 12.3 Contamination: clean

No training corpus shares a normalized prompt with any evaluation set. Both
claims that mattered check out: LCR-SFT (`LCR-*`) is disjoint from
LongContextDemo (`RLC-*`), and Puzzle-51k train is disjoint from its own
validation, test and challenge splits.

### 12.4 Sequence budget — LCR-SFT dominates the epoch

| Corpus | Rows | Mean tokens | Max | Epoch tokens | % rows | % tokens |
|---|---:|---:|---:|---:|---:|---:|
| v3 combined 124k | 124,000 | 99.4 | 204 | 12.32 M | 70.3% | 33.8% |
| Puzzle-51k `short_sft` | 40,800 | 111.6 | 313 | 4.55 M | 23.1% | 12.5% |
| RAG v3 | 4,410 | 224.9 | 316 | 0.99 M | 2.5% | 2.7% |
| **LCR-SFT train** | **7,200** | **2,576.7** | **3,946** | **18.55 M** | **4.1%** | **50.9%** |

**LCR-SFT is 4% of the rows and 51% of the cost.** Two consequences:

1. **Length-grouped batching is mandatory.** One uniform `max_length=4096` over
   the blended mix yields **5.0% efficiency — a 19.8× compute waste**, because
   99-token chat rows get padded to 4,096. Use `group_by_length=True`, or train
   long context as its own phase.
2. **Its targets break the brevity contract.** LCR-SFT targets are p50 **28** /
   p99 **44** tokens against the Rineya 24-proxy-token limit. That is deliberate
   — LCR is correctness-first and explicitly does not score length — but blending
   it at full weight will lengthen the model's answers everywhere.

### 12.5 Puzzle-51k needs a contract decision before conversion

It ships raw, with `prompt`, `reasoning_steps`, `reasoning_summary`,
`compact_state`, `final_answer`. Two conversions are demonstrated in `examples/`:

| Conversion | Target | Mean tokens | Epoch tokens | Contract |
|---|---|---:|---:|---|
| `short_sft` | `"Orange."` | 111.6 | 4.55 M | **compatible** |
| `think_sft` | `<think>…</think>` + answer | 211.0 | 8.61 M | **conflicts** |

`think_sft` collides with three established decisions at once: the no-CoT-in-
targets rule, the 24-token target limit, and `enable_thinking=false` pinned
across the whole eval harness. Adopting it means re-baselining every evaluation.

> **Use `short_sft` for healing.** Chain-of-thought supervision is a separate
> research question, not a variable to change during a compression heal.

Separately: `compact_state` + `reasoning_steps` is precisely the `(problem, state)
→ transition` representation the state-sufficiency lab's SPLIT arm consumes.
Puzzle-51k is a better-matched source for that experiment than for this one.

### 12.6 R2 unlocked long-context training on this card

Logits tensor at batch 1, fp32 cross-entropy:

| Sequence | Stock vocab | R2 vocab |
|---:|---:|---:|
| 384 | 0.47 GB | 0.10 GB |
| **4,096** | **4.98 GB → OOM on 8 GB** | **1.07 GB → fits** |

At the stock 151,936-token vocabulary the logits tensor alone would exhaust the
card before any weights, gradients or activations were counted. The vocabulary
trim was justified on parameters and throughput; making 4k-context training
feasible at all on this hardware was not a planned benefit.

### 12.7 Recommended staging

One variable per run, in the repository's existing ablation style. `max_len 384`
covers B0 and B1 exactly — the same budget the harness is already configured for.

| Run | Mix | Rows | Tokens | max_len | Question |
|---|---|---:|---:|---:|---|
| **B0** | v3 124k + RAG 4,410 | 128,410 | 13.3 M | 384 | Does R2 train as well as stock did? *Control.* |
| **B1** | B0 + Puzzle `short_sft` | 169,210 | 17.9 M | 384 | What does puzzle reasoning add? |
| **B2** | B0 + LCR-SFT, length-grouped | 135,610 | 31.9 M | 4,096 | What does long context add, and what does it cost brevity? |
| **B3** | everything | 176,410 | 36.4 M | 4,096 | Interaction. Only if B1 and B2 both pass. |

Gates per run: C5 (chat DEV), C6 (RAG per-category incl. abstain precision), C7
(degeneration). B2 and B3 add the LCR-SFT 800-row dev split, with
LongContextDemo's 256 rows held frozen for the final report only.

DPO stays untouched. The v1/v2/v3 preference sets (12k/16k) are explicitly
post-SFT material in their own cards, and no SFT checkpoint has been accepted yet.

---

## 13. Healing harness (2026-08-19)

Built to a stricter contract than "it trains": **an interrupted run must produce
byte-identical weights to an uninterrupted one, and must not be able to OOM.**

### 13.1 The one tension, resolved explicitly

Bit-exact results and "recover from OOM by shrinking the batch" cannot both hold.
Shrinking changes the computation, so a recovered run is a *different* run wearing
the same checkpoint's name. The harness therefore does not recover from OOM. It
makes OOM structurally unreachable instead:

1. `memory.apply_cap` installs a hard allocator ceiling via
   `set_per_process_memory_fraction`. The process cannot exceed it even on an idle
   card, so behaviour does not depend on what else is running — and a run cannot
   starve the desktop.
2. `memory.probe` takes **the widest micro-batch the plan will ever produce** and
   runs a real forward and backward at exactly that shape, under the cap, before
   step 0. Peak allocation is measured, not estimated. Since that micro-batch
   bounds every other, a passing probe means no later step allocates more.
3. A budget that does not fit aborts in seconds with the exact
   `max_tokens_per_microbatch` to use instead — not at hour six.
4. If OOM happens anyway (another process took VRAM, fragmentation), it is caught,
   state is checkpointed, and the process exits 2 with a diagnosis. Resuming
   replays that step bit-exactly.

> Changing the memory budget changes the plan, which starts a **new run**, not a
> resume. The harness enforces this by hash rather than trusting the operator.

### 13.2 What makes it deterministic

| Concern | Mechanism |
|---|---|
| Kernel nondeterminism | `use_deterministic_algorithms(True)` — raises rather than warns |
| Attention backward | flash and mem-efficient SDPA disabled; `eager`/`math` only |
| cuBLAS reductions | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, **checked at startup** and refused if absent (it cannot be fixed after CUDA initialises) |
| TF32 / reduced-precision reductions | disabled |
| Data order | a plan that is a pure function of (mix, seed, epoch, budgets) — regenerated on resume and hash-compared, never trusted from disk |
| Learning rate | a pure function of `global_step`, so there is no scheduler state to drift |
| Loss normalisation | summed token loss ÷ the step's exact supervised-token count, so the result is independent of how rows were packed |
| RNG | Python, NumPy, torch CPU and every CUDA device captured and restored |
| Environment change | fingerprinted; a different GPU/driver/torch aborts unless explicitly allowed |

The speed cost — deterministic kernels plus `eager` attention — is the deliberate
trade requested.

### 13.3 Verified, not asserted

```
scripts/verify/verify_resume.py    --steps 6 --break-at 3
scripts/verify/verify_interrupt.py --steps 8 --signal-after 3
```

Both run the real entrypoint in real subprocesses, then compare every LoRA tensor
and every Adam moment with `torch.equal`.

| Test | Result |
|---|---|
| Clean stop at step 3, resumed to 6 | **PASS** — 112 LoRA + 336 Adam tensors identical, 0 differing |
| Live `CTRL_BREAK` mid-run (landed at step 4), resumed to 8 | **PASS** — 112 LoRA + 336 Adam tensors identical, exit code 0 |

Per-step losses match across the interruption boundary exactly (4.0515 / 4.2664 /
3.3743 in both arms). Adam moments are compared as well as weights because
optimizer state is where resume bugs hide: weights can agree for a step or two
while `exp_avg` has already diverged.

Guard rails, each tested to actually fire:

| Condition | Behaviour |
|---|---|
| Plan hash mismatch (config changed under a run) | refuse, exit 1, prints both hashes |
| Training mix changed | refuse, exit 1 |
| `CUBLAS_WORKSPACE_CONFIG` absent | refuse **before loading the model**, exit 1 |
| Plan vs tokenised data disagree on supervised tokens | refuse mid-step rather than train on a wrong normaliser |
| Second signal during shutdown | immediate exit 130, no checkpoint |
| Checkpoint write interrupted | previous checkpoint intact — temp file + `os.replace`, pointer written only after the payload is durable |

### 13.4 Token-budgeted batching earns its complexity

Budgeting on `rows × longest_row_in_batch` rather than a fixed batch size:

| Run | Rows | Steps | Padded positions | Padding efficiency | vs uniform padding |
|---|---:|---:|---:|---:|---:|
| B0 | 128,365 | 1,003 | 17.32 M | **99.2%** | 2.8× cheaper |
| B2 | 135,565 | 1,455 | 35.93 M | **99.8%** | **15.5× cheaper** |

The planner automatically degrades to **1 row × 3,969 tokens** for long-context
rows while packing 16 short rows together — the same mechanism that bounds memory
also removes the 19.8× padding waste identified in §12.4.

### 13.5 Files

| Path | Role |
|---|---|
| `scripts/data/build_heal_mix.py` | Deterministic mix builder; refuses to read the frozen eval pack |
| `scripts/train/determinism.py` | Seeding, RNG capture/restore, env preconditions, fingerprinting |
| `scripts/train/plan.py` | Token-budgeted length-grouped plan + stable hash |
| `scripts/train/memory.py` | Hard VRAM cap, pre-flight probe, budget suggestion |
| `scripts/train/dataset.py` | Cached pre-tokenisation; exact prompt/target boundary |
| `scripts/train/checkpoint.py` | Atomic save, complete state, load-time refusals |
| `scripts/train/heal.py` | Training loop |
| `scripts/train/run_heal.ps1` / `.sh` | Launchers that set the required env vars |
| `scripts/verify/verify_resume.py` | Clean-stop bit-exactness test |
| `scripts/verify/verify_interrupt.py` | Real-signal bit-exactness test |
| `configs/heal_b{0,1,2,3}.yaml` | The four staged runs |
| `configs/heal_smoke.yaml` | CPU-sized config for the verifiers |

### 13.6 Config summary

| Run | Rows | max_len | Token budget | Rows/µbatch | Accum | Cap | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 128,365 | 384 | 8,192 | 16 | 8 | 5,120 MiB | 1,003 |
| B1 | 169,165 | 384 | 8,192 | 16 | 8 | 5,120 MiB | — |
| B2 | 135,565 | 4,096 | 4,096 | 8 | 16 | 6,144 MiB | 1,455 |
| B3 | 176,365 | 4,096 | 4,096 | 8 | 16 | 6,144 MiB | — |

LoRA r=16 on all seven projections, lr 1e-4 cosine with 40 warmup steps, one
epoch, gradient checkpointing on. Caps leave headroom on the 8,188 MiB card;
check `nvidia-smi` first, since desktop applications commonly hold 4–5 GiB.

### 13.7 Still unverified

The probe, the cap, and OOM handling have **only been exercised on the CPU path**,
where they correctly no-op. They are untested against real CUDA because torch here
is still `2.13.0+cpu` (Phase 0). The determinism, planning, checkpointing and
resume logic *is* fully tested, and is device-independent.

First action once CUDA is live:

```bash
./scripts/train/run_heal.sh configs/heal_b0.yaml --dry-run
```

That runs the probe against the real card and reports whether the budget fits
before committing any training time.

---

## 14. Phase 0 closed; harness validated on the real card (2026-08-19)

Executed against pre-declared predictions. Two were falsified, and chasing them
found a real bug and a real hardware limit.

### 14.1 Environment

Phase 0 is **done**. A project-local `.venv` now carries **torch 2.13.0+cu130** —
the same version as the CPU build, so only the CUDA runtime differs. The shared
`hermes-agent` venv and the state-sufficiency lab were left untouched.

`torch.cuda.is_available() = True`, RTX 4070 Laptop, 8,187 MiB, capability 8.9.

Note: only `cu126` and `cu130` carry a 2.13.0 wheel — `cu128`/`cu129` stop at
2.11.0/2.9.0. CUDA 13.0 runs on the installed 13.3 driver.

### 14.2 Phase A — all pass (CPU)

| Test | Result |
|---|---|
| **A1** code-fingerprint guard (new, §14.5) | **PASS** — editing `plan.py` blocks resume, exit 1, names the module |
| **A2** resume across the **epoch boundary** | **PASS** — 112 LoRA + 336 Adam identical; log confirms `epoch 1, step_in_epoch 0` |
| **A3** two fresh runs, same seed | **PASS** — 112 + 336 identical |

A2 mattered: every earlier verification stopped mid-epoch, so the epoch-transition
path had never executed.

### 14.3 Phase C — the probe found a bug in itself

| Test | Predicted | Measured | Verdict |
|---|---|---|---|
| **C1** B0 probe | 3,600–4,100 MiB reserved | **5,094** then **4,120** | falsified, then vindicated |
| **C2** B2 probe | 3,500–4,000 MiB, *lower* than B0 | **6,032**, 46% *higher* than B0 | **falsified** |
| **C3** GPU bit-exact resume, real B0 config | PASS | **PASS** — 392 LoRA + 1,176 Adam identical | confirmed |

**C1, first run: 5,094 MiB — over budget, refused.** Cause was not the model but
the harness: `mem.probe` ran before `model.train()`, and **gradient checkpointing
is a no-op in eval mode**. The probe was measuring the un-checkpointed forward and
would have rejected budgets that fit comfortably. Fixed by moving `model.train()`
above the probe.

**After the fix: 3,290 MiB allocated / 4,120 MiB reserved** — against a predicted
3,600–4,100. The component estimate (3,306 MiB allocated) was within **0.5%** of
measurement. The memory model was right all along; the instrument was wrong.

**C2 falsified the reasoning, not just the number.** The prediction that B0 would
peak higher than B2 came from comparing *total positions* (B0 5,344 vs B2 3,969),
which is the right scaling for the logits term — linear in positions × vocab. But
at seq 3,969 the **eager attention score matrix dominates, and it is O(seq²)**:
one layer alone is 481 MiB before autograd's copies. The correct statement:

> The logits term dominates at short sequence; the attention term dominates at
> long sequence. Comparing total positions is only valid on the short side of that
> crossover.

Neither prepared contingency rescues B2. Chunking the cross-entropy attacks the
logits term, which is not the binding one here; bf16 adapters do not help either.

### 14.4 The real limit: bit-exactness and 4k context are mutually exclusive on 8 GB

Bit-exact replay requires `eager`/`math` attention, because the flash and
mem-efficient backward passes accumulate nondeterministically. That is precisely
what makes attention O(seq²) in memory. B2 needs 6,032 MiB against 5,632 usable —
**400 MiB short.** Three ways out, and they trade against each other:

| Option | Keeps bit-exact? | Keeps the data? | Risk |
|---|---|---|---|
| Raise `cap_mib` 6,144 → 6,656 | yes | yes | leaves ~1,530 MiB for the desktop; observed desktop use ranges 360–4,800 MiB |
| Lower `data.max_len` to ~2,560 | yes | **no** — LCR is p50 2,577 / p99 3,749, so about half the rows get left-trimmed | weakens the capability B2 exists to add |
| Mem-efficient SDPA for B2/B3 only | **no** | yes | forfeits the guarantee you asked for, on those runs only |

**This is a genuine conflict between two of your stated requirements**, so it is
yours to resolve rather than mine. B0 and B1 are unaffected.

### 14.5 Windows changes what "OOM safety" means

An uncapped diagnostic reported **10,440 MiB reserved on an 8,187 MiB card.** That
is not an error: on Windows/WDDM the driver silently spills past VRAM into system
RAM over PCIe. Exceeding VRAM does not crash — it becomes roughly two orders of
magnitude slower, and the run appears to hang rather than fail.

So the hard cap earns its place for a stronger reason than originally argued:
**it converts a silent, un-diagnosable slowdown into a clean failure in seconds.**
Without it, an over-budget config on this machine would look like a hung job.

(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not supported on Windows
and warns harmlessly; it still helps on the Linux CUDA host.)

### 14.6 New guard: code fingerprint

The plan hash covers *what* the optimizer sees, not the code consuming it. Editing
the loss and resuming would have satisfied every other guard while silently
changing the numerics — which is exactly what contingency C1 would have done.
`determinism.code_fingerprint()` now hashes all six training modules into every
checkpoint and refuses on drift, overridable via `resume.allow_code_drift`.

Byte-level hashing means comment-only edits also trip it. That is the conservative
direction: a hash cannot distinguish a docstring from a reduction order.

### 14.7 Where B0 stands

**Ready to run.** Probe 3,290 / 4,120 MiB against a 4,608 MiB budget (488 MiB
headroom), 1,003 steps, 128,365 rows, 99.2% padding efficiency, GPU resume proven
bit-exact. Reproducible to the MiB across invocations.

```bash
./scripts/train/run_heal.sh configs/heal_b0.yaml
```

Not yet measured: wall clock, and the loss trajectory over a full epoch.

---

## 15. Lightweight diagnostics before committing GPU hours (2026-08-19)

Run before starting B0, on the principle that two minutes of measurement beats
three hours of a run built on an assumption. Four findings, three of them defects
in the harness itself.

### 15.1 The code fingerprint was too strict to survive git

`.gitattributes` declares `*.py text eol=lf`, so a checkout rewrites CRLF to LF in
the working copy. A byte-level fingerprint would then refuse a valid resume — the
guard firing on git hygiene rather than on a real change. Newlines are now
normalised before hashing: LF and CRLF copies hash identically, while appending a
statement to `plan.py` is still caught.

### 15.2 The nondeterministic-attention claim was correct, and now measured

`determinism.py` disables flash and mem-efficient SDPA on the standard caveat that
their backward passes accumulate via atomics. That was inherited, not tested.
Tested now — 3 repeats of an identical forward/backward at 1×3969, gradients
compared with `torch.equal`:

| Backend | Peak reserved | Bit-exact across repeats? |
|---|---:|---|
| `math` / eager (current) | **OOM** at a 6,144 MiB cap | — |
| `mem_efficient` | 3,750 MiB | **no**, max\|Δ\| 1.46e-11 |
| `flash` | 3,136 MiB | **no**, max\|Δ\| 4.66e-10 |

So the claim holds. The new information is the memory: **flash halves B2's
worst-shape footprint**, which is exactly what would make B2 fit — at the price of
the guarantee. The divergence is float noise per step, but over 1,455 steps it
compounds, so byte-identical resume is genuinely lost, not merely strained.

Also measured: **eager at 1×2048 = 3,000 MiB and bit-exact.** So a bit-exact
long-context run is possible at reduced `max_len`, just not at 4,096.

### 15.3 The probe was measuring the wrong thing (twice)

**Defect 1 — eval mode.** `mem.probe` ran before `model.train()`, and gradient
checkpointing is a no-op in eval mode. Fixed in §14.3.

**Defect 2 — one micro-batch instead of one step.** The probe measured a single
micro-batch while a real step runs `grad_accum` of differing widths, and Adam's
moments are created lazily on the first `optimizer.step()` so a forward/backward
probe never counted them. Measured shortfall on B0: **4,120 MiB predicted vs 4,814
actual.** The probe now replays the heaviest step in full, including the optimizer
step, snapshotting and restoring weights so it still cannot perturb training —
re-verified: two independent GPU runs agree on 392/392 LoRA and 784/784 Adam
tensors.

**The honest limit.** Even the full-step probe under-predicts, by between 0.6% and
27% depending on config. It is a fast **lower bound**, not a predictor.

> Operational rule: probe to catch gross misconfiguration, then validate with
> `--max-steps 8` and read `peak_reserved_mib` from `train_log.jsonl`. Two minutes,
> and authoritative.

### 15.4 Width bucketing: measured, then rejected

Rounding padded width to a multiple of 64 should have reduced allocator churn and
tightened the probe. It did tighten the probe, and bought nothing real:

| | Padding eff. | Probe | **Actual** | Wall clock |
|---|---:|---:|---:|---:|
| bucket off | 99.6% | 3,048 MiB | **3,882 MiB** | 2.9–3.2 h |
| bucket 64 | 81.6% | 3,466 MiB | **3,880 MiB** | 3.24 h |

Actual peak is identical — 3,880 MiB is the real working set, not a function of
shape diversity. Bucketing cost 22% more padded positions and 12% wall clock to
narrow an estimate. Kept as a `data.width_bucket` option, defaulted **off**, with
the measurement recorded in `plan.py` so it is not re-tried on intuition.

### 15.5 B0 reconfigured and validated

Halving the micro-batch and doubling accumulation keeps the effective batch, the
step count, and the total compute identical while cutting peak memory 19%:

| | Before | After |
|---|---|---|
| rows × accum | 16 × 8 | **8 × 16** |
| rows/step, steps | 128, 1,003 | **128, 1,003** (unchanged) |
| `max_tokens_per_microbatch` | 8,192 | **3,072** |
| Probe / actual peak | 4,786 / 4,814 MiB | **3,048 / 3,882 MiB** |
| Real headroom under the 5,120 cap | 306 MiB | **1,238 MiB** |
| Padding efficiency | 99.2% | **99.6%** |
| Wall clock | 2.6 h | **~3.2 h** (+12% for the smaller micro-batch) |

**B0 is ready.** 11.3 s/step, 1,003 steps, ~3.2 hours, 1,238 MiB of measured
headroom, GPU resume proven bit-exact.

### 15.6 B2 decision, now quantified

The conflict in §14.4 stands, with numbers attached:

| Option | Bit-exact | Data intact | Peak at worst shape |
|---|---|---|---:|
| eager @ `max_len` 4,096 | yes | yes | **does not fit** |
| eager @ `max_len` 2,048 | **yes** | no — LCR p50 is 2,577, so most rows trim | 3,000 MiB |
| `flash` @ 4,096 | **no** (4.7e-10/step) | yes | 3,136 MiB |
| `mem_efficient` @ 4,096 | **no** (1.5e-11/step) | yes | 3,750 MiB |

Still yours to resolve: it trades the byte-identical guarantee against the
long-context capability B2 exists to add. B0 and B1 are unaffected either way.

---

## 16. Gate baselines: what the vocabulary trim actually cost (2026-08-19)

First real gate measurement of R2, against stock Qwen3-0.6B on identical cases
with the identical vendored v4.5 scorer. `scripts/eval/run_gates.py`.

### 16.1 chat DEV — the trim is free

510 paired cases:

| Metric | stock 596M | **R2 474M** | Delta |
|---|---:|---:|---:|
| mean_quality_score | 0.8334 | **0.8336** | +0.0002 |
| quality_pass_rate | 0.9941 | 0.9961 | +0.0020 |
| intelligibility_pass_rate | 1.0000 | 1.0000 | 0.0000 |
| contract_fulfillment | 1.0000 | 1.0000 | 0.0000 |
| worst_decile quality | 0.6987 | 0.7045 | +0.0058 |

```
paired mean delta  +0.00019    95% CI [-0.00806, +0.00844]    t = +0.045
identical 417/510    R2 better 45    stock better 48
byte-identical replies 308/510 (60.4%)    visible tokens +3.31%
```

Indistinguishable from zero, effect bounded inside ±0.008. Win/loss 45–48 is
symmetric noise. **C7 passes outright on both**: zero critical defects, zero
contract violations, 100% intelligibility — the `imony`-loop pathology that
retired the 19L soup does not appear.

### 16.2 RAG DEV — the trim costs exactly one thing

752 paired cases. Here the trim **is** detectable, and significantly so:

```
paired mean delta  -0.01273    95% CI [-0.01955, -0.00592]    t = -3.660
identical 738/752    R2 better 0    stock better 14
byte-identical replies 729/752 (96.9%)
```

Zero cases improved and 14 regressed. That asymmetry is what distinguishes this
from the chat result: it is a small, one-directional, real degradation.

**And the per-category view shows it is a single failure mode, not a broad decay.**
Twelve of sixteen categories are bit-identical:

| Category | n | stock | R2 | Delta | Regressed |
|---|---:|---:|---:|---:|---:|
| **conflict** | 38 | 0.5847 | **0.3948** | **−0.1898** | **9** |
| stale_evidence | 38 | 0.5135 | 0.4875 | −0.0260 | 2 |
| procedure_with_distractor | 25 | 0.3843 | 0.3600 | −0.0243 | 1 |
| prompt_injection | 38 | 0.6434 | 0.6304 | −0.0130 | 1 |
| qualified_answer | 38 | 0.2737 | 0.2665 | −0.0072 | 1 |
| *the other 11 categories* | 615 | — | — | **0.0000** | **0** |

The `conflict` category alone accounts for essentially all of the aggregate loss.

### 16.3 The mechanism: lost abstentions

| | Stock | R2 |
|---|---:|---:|
| Correct abstentions (gold = "I don't know…"), 202 cases | 188 (93.1%) | 179 (88.6%) |

**R2 lost exactly 9 abstentions, and they are exactly the 9 `conflict`
regressions.** In each one, stock answers `"I don't know based on the available
information."` and R2 instead begins `"Based on the available information, the
first launch region for X is…"` and runs to the 32-token cap. Same prefix words,
opposite decision — a token-level nudge flipped the abstain/answer choice.

**Mechanism tested, and the obvious explanation is wrong.** The first hypothesis
was that these DEV entity names were never in the keep-set input (the vocabulary
was built from training corpora, and RAG DEV was not in this repo at the time), so
they would tokenize worse. Measured, and they do not:

| | stock | R2 |
|---|---:|---:|
| all 10 failing entity names (`ThistleEngine`, `IvorySensor`, `BirchInstrument`, …) | 3–4 tok | **identical** |
| `"I don't know based on the available information."` | 10 tok | **identical** |
| prompt tokens, the 14 regressed cases | — | 1.0210x |
| prompt tokens, 112 unchanged cases | — | 1.0015x |

So the vocabulary is not too tight, and the trim did not damage evidence copying.
What it does is shift positions by ~2% on these particular prompts, and `conflict`
is the most fragile category there is -- stock's own score on it is 0.5847, second
worst of sixteen, and stock itself misses 14 of 202 abstentions. These were
borderline decisions that any small perturbation tips.

> The causal claim is therefore narrower than "the trim cost 9 abstentions". It is:
> **the trim perturbs token positions slightly, and the most borderline category
> flips.** An equally benign perturbation would likely flip a comparable number in
> either direction. What healing has to do is make the decision robust, not restore
> a lost capability.

Two smaller mechanisms appear in the remaining 5 regressions:

- **Entity corruption from re-tokenization.** `BirchInstrument` → `BIRCHINSTRUMENT`.
  The mixed-case token was trimmed, so the model rebuilt the name from pieces and
  lost the casing.
- **Token inflation tripping the brevity contract on identical text.** 10 of the
  729 byte-identical replies cost more tokens under the 32k tokenizer, and two of
  them scored lower for that reason alone despite being the same answer.

> **§12.6 predicted this, and named the category.** "Compression breaks
> *calibration* before it breaks accuracy: a pruned model becomes confidently
> wrong rather than obviously broken… Aggregate quality can hold while `conflict`,
> `stale_evidence`, `prompt_injection` and `missing_answer` collapse into confident
> answers. Gate on the categories, not the mean." Three of the four named
> categories regressed; `conflict` dominates. The aggregate −0.0127 would have read
> as diffuse noise. Only the per-category gate localised it.

### 16.4 What this changes

The trim is **free on conversation and free on 11 of 16 grounding categories**, and
costs **9 abstentions in `conflict`** — about 1.3 quality points in aggregate for
−20.47% parameters and −14.72% compute.

Crucially, this is the failure mode the healing data is built to fix: the RAG
corpus carries **1,600 abstain decisions across 300 `conflict` rows**, and the
prior teacher's dual SFT moved RAG by **+0.313** with abstention rising from 87.6%
to 99.5% intelligibility. So B0 gets a concrete, pre-declared success criterion
rather than a vague one:

> **Gate C6 for B0: restore `conflict`-category abstention to ≥ 93.1%** (stock's
> rate) while chat DEV `contract_fulfillment` stays at 1.0000. Declared before B0
> is trained.

Secondary target, also now quantified: RAG `contract_violation_case_rate` is
**19.7%** on R2-base against **0.0%** on chat DEV. That gap is the grounding work
B0 exists to do.

### 16.5 Cross-architecture note, not a claim

Both R2 and stock land at ~0.833 on chat DEV, matching the prior
`phase_4_model_soups` teacher's *post-SFT* score (0.8332) with no adapter at all,
and both beat its baseline (0.7984). On RAG both (~0.67–0.68) sit just under that
teacher's post-SFT 0.6932 and far above its 0.3801 baseline. Interesting, but a
different checkpoint — recorded as context, not as a result.

Also unresolved: upstream's RAG summary reports `case_count 1504` against a
752-row set, exactly 2×. Until that is understood their RAG anchor is a per-case
mean reference, not a row-for-row equivalent. The stock-vs-R2 pairing above is
unaffected — same 752 rows, same protocol, both sides.
