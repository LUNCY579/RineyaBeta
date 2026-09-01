# DAST preregistration — DRAFT, PENDING APPROVAL. TEST CLOSED.

**Status:** `DRAFT`. Every placeholder is now resolved from measurement, but this
file is **not frozen** and its hash cannot open the sealed TEST split until the
freeze block in §11 is completed and approved.

Everything below is derived from work committed on
`research/depth-aligned-semantic-transitions-v1` up to `239b894`. Where a value came
from a measurement rather than a convention, the measurement is cited.

## 1. Primary question

Does ordered alignment of cumulative compact reasoning states to increasing
Transformer depth improve held-out reasoning relative to matched **shuffled** and
**reversed** depth assignments?

## 2. Base checkpoint

| | |
|---|---|
| model | `Qwen/Qwen3-0.6B` |
| revision | `c1899de289a04d12100db370d81485cdf75e47ca` (pinned from the HF cache) |
| architecture | 28 decoder layers, hidden 1024, 16 heads / 8 KV heads |
| parameters | 596,049,920 base; 606,142,464 with LoRA (**+10,092,544**, 1.69%) |
| precision | bf16 resolved from `precision: auto` |
| tokenizer | same revision; `pad` `<|endoftext|>` distinct from `eos` `<|im_end|>` |

**Resolved checkpoints must be 7 / 14 / 21 / 27, or the run is invalid.**

This supersedes the original "must be 7/14/21/28". `hidden_states[28]` is
post-final-RMSNorm — `lm_head(hidden_states[28])` reproduces `logits` to 0.0
absolute error — while 7/14/21 are raw residual states (elementwise RMS 49/49/49.5
versus 3.06). A schedule ending at layer 28 therefore feeds two representation
classes into the single shared projector, which is exactly what `SSOT §3.1` says the
sharing exists to prevent. Measured pooled cosine 7-vs-last: **−0.018** raw,
**−0.111** after norming only 7/14/21, **+0.409** for 7/14/21/27 — the only variant
with no near-orthogonal outlier. `DASTModel` refuses `final_norm` whenever the last
checkpoint resolves to the top layer.

## 3. Data

| | |
|---|---|
| source | `Rineya_state_experiment_bundle/rineya_sequence_independence_dataset_v1/data/full_context/` |
| train file | `train.jsonl`, SHA-256 `705bfd968c14e2b5eea77ad6889e0c01f5f5c9069d2ef7cdf6857649f0f31e9f` |
| DEV file | `dev.jsonl`, SHA-256 `74a9cac6c7e0597ad4c57a31e7c73860a92dbdef753a0281140eb226af5338b3` |
| TEST file | `test.jsonl`, SHA-256 `bab8c05e354baff1a15b891787f9e57a6d011338a88c8cfdda693c02d2f1dab1` |
| train records used | **4,000** (first-N; the file is pre-shuffled, and the first 800 already cover all 30 categories) |
| DEV records used | **800 per seed** |
| TEST | **SEALED.** Its hash is recorded so tampering is detectable. It has never been read. No DEV config names the file. |

Strata for analysis: 30 categories, 30 generators, `answer_type` (21 values), and
distinct-target count. **42.0% of train / 42.5% of DEV traces have fewer than K=4
distinct compact states**, so repeated depth slots are reported, not hidden.
Restricting to full traces was measured and **rejected**: it moves the
ordered/shuffled target contrast only 0.860 → 0.849 while discarding 43.8% of the
corpus.

## 4. Arms and seeds

Six Phase-1 arms: `cot_baseline`, `state_ordered`, `state_shuffled`,
`state_reverse`, `state_final_repeated`, `literal_ordered`.

Seeds **17, 29, 43** — three, not one. One seed at 800 DEV gives only **56.7%**
power (§6), so a single-seed result cannot be confirmatory.

No arm-specific tuning after comparative runs begin. The only per-arm differences
are those in the arm contract.

## 5. Training budget, matched across arms

| | |
|---|---|
| LoRA | r=16, alpha=32, dropout 0.05, on q/k/v/o/gate/up/down |
| batch size | **2**, gradient accumulation **8** |
| optimizer steps | **250** per arm (identical across arms and dataset sizes) |
| learning rate | 2e-4, auxiliary 2e-4, warmup ratio 0.03, cosine |
| epochs | 1 |
| readout | `checkpoint_readout: final_norm` |
| target geometry | `target_centering: within_problem` |
| state / contrastive weight | 1.0 / 0.10 |
| hardware | RTX 4070 Laptop, 8188 MiB; ~4.1 GiB peak observed |
| cost | 21.8 min train + ~37 min generation ≈ **60 min per arm** |

`batch_size: 2` is required, not cosmetic: `multi_positive_contrastive_loss` returns
exactly 0.0 when `batch < 2`, so the documented anti-collapse term was inert at the
previous `batch_size: 1`. Accumulation was halved so the effective batch and the
optimizer-step count are unchanged (verified identical at n = 4000/16/799/4001).

`target_centering: within_problem` is a **protocol decision requiring approval**. As
shipped, targets carried 56.6% of their variance in the problem factor and 9.3% in
the stage factor while hidden states carried 90.8% stage / 6.0% problem — the two
sides of the objective encoded orthogonal factors, leaving `state_ordered` at
**0.860** cosine from `state_shuffled`. Centering moves that to **−0.019**.

## 6. Primary outcome — resolved from measurement

| | |
|---|---|
| primary metric | strict generated-answer accuracy, greedy, `max_new_tokens: 384` |
| pairing | by `problem_id`, identical problem sets enforced by `compare.py` |
| decisive contrast | ordered minus the **better** of shuffled and reverse |
| **practical minimum effect** | **2.0 pp** |
| uncertainty | **clustered** bootstrap, unit = `generator` (30 clusters), 95% |
| significance | exact McNemar on discordant pairs, α = 0.05, two-sided |
| multiplicity | ordered-vs-shuffled and ordered-vs-reverse **jointly required** |

The margin is derived, not guessed. Measured arm-to-arm discordance is **6.0%**
(12 of 200 in the G1 pilot). At 3 seeds × 800 DEV = 2,400 paired problems that gives
n_d ≈ 144 and a **1.56 pp** minimum detectable effect at 80% power, with **97.8%**
power for the pilot's observed 2:1 split. **2.0 pp** is therefore achievable and
leaves headroom against a noisy discordance estimate.

`max_new_tokens: 384` is required: at 128, **51%** of generations hit the cap without
emitting EOS and strict accuracy read 0.115 against a true ~0.44. At 384 the hit-cap
rate is 1.0%.

Quote the **clustered** interval. On real data it is 22% wider than the i.i.d. one,
which is anti-conservative because problems from one generator share a template.

## 7. Secondary outcomes

Order margin; checkpoint top-1; final-state cosine; decoded progress monotonicity;
`L_state` per arm (the only mechanism number comparable **across** arms, since
margin and monotonicity are computed against canonical order and so penalise reverse
by construction); teacher-forced LM loss/perplexity; per-category, per-generator and
per-`answer_type` accuracy; extraction-route distribution; peak VRAM; wall time.

Secondary outcomes cannot rescue a failed primary gate.

## 8. Phase gates

### G0 — Engineering — **PASSED** at `f5ae82f`
Local qualification PASS across all six steps in both device configurations;
artifact `qualification/latest.json`. Hosted CI is retired and is not a gate.

### G1 — Learnability — threshold restated
Ordered must raise the canonical order margin **above the measured frozen floor**,
not above zero. A fitted shared *linear* probe on the **untrained** backbone already
reaches margin **0.977** under this exact geometry, so "margin is positive" is
satisfied by the pretrained model and proves nothing.

**Threshold:** Δmargin(ordered) > Δmargin(shuffled) **and** > Δmargin(reverse),
each measured against that frozen floor, with LM perplexity within **+2%** of
`cot_baseline`.

### G2 — Depth-order causality
Ordered passes **both** contrasts on the primary metric at the frozen 2.0 pp margin,
with the clustered interval excluding zero, jointly across three seeds.

### G3 — Progressive-state specificity
Ordered beats `state_final_repeated` on the primary metric or on a preregistered
secondary; generic deep supervision must not explain the effect.

### G4 — Literal vs semantic
Report `literal_ordered`. Interpretive; does not replace G2.

### G5 — Operators — only after G2 and G3.
### G6 — Causality — only after G2 and G3. Requires the no-cache generation path to be qualified first; feedback is contractually incompatible with cached generation, and the zero-initialised decoder makes every intervention a bitwise no-op until it has trained.

## 9. Exclusion and failure handling

No generated response is dropped for formatting failure. OOM, NaN, interrupted and
zero-output runs are retained and reported. A seed may be rerun only for a
documented infrastructure failure, never because its result is unfavourable.
Leakage or budget mismatch invalidates the affected comparison and requires
withdrawal, not reinterpretation.

**Scorer is frozen at `239b894`.** It may not be changed once comparative runs
begin. Its two prior corrections were both driven by absolute values being
impossible (0.0000 on visibly correct responses; wrong answers scored correct), not
by any arm difference, and both are pinned by tests against real model output.

## 10. Stop rules

Stop Phase 1 and close the tested formulation if ordered fails G2 or G3. Stop
operators on router collapse or absent transfer. Stop feedback if interventions are
non-directional. **Do not open TEST to resolve an ambiguous DEV result.**

## 11. Freeze block — REQUIRES APPROVAL

Two decisions in §2 and §5 change what the experiment measures and are recorded here
rather than applied silently:

1. **checkpoints 7/14/21/27 with `final_norm`**, rather than 7/14/21/28 raw. This
   abandons \(d_K = L\) as written in `SSOT §4`; the model's final layer is no longer
   read by any checkpoint. The alternative preserves \(d_K=L\) at the cost of stage 4
   reading a near-orthogonal vector through a shared projector. Old behaviour remains
   reachable as an ablation.
2. **`target_centering: within_problem`**, which removes the problem-identity
   component the arms share. It stays inside the existing mechanism, so it does not
   spend the `SSOT §13` one-mechanism revision budget.

```text
status: DRAFT
commit: 239b894
sha256_of_this_file: (compute after approval, before any TEST config references it)
frozen_at_utc: PENDING
approved_by: PENDING
```
