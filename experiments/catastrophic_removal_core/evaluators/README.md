# Historical 64-row long-context evaluator — recovered and registered

**Status: REGISTERED — ANCHORS REPRODUCED.**
**This is infrastructure, not a CRCC result.** No CRCC donor exists (discovery
stopped at G0), so nothing here is a comparison with, or a claim about, `30/64`.

`docs/PLAN_CATASTROPHIC_REMOVAL_CORE.md` §6.4 made this a *blocking* evidence
requirement: the branch may not reference the historical `14/64` / `4/64` / `30/64`
family comparison until the exact evaluator is recovered, pinned, and shown to
reproduce its known anchors. This directory closes that requirement.

## What the evaluator is

The **showcase split of the Rineya Long-Context Demonstration Suite v1.0.0** — 64
rows, 16 scenario packs, each contributing base / paraphrase / counterfactual /
unanswerable variants. Contexts are 853–1183 words of invented material, so a
pretrained model cannot answer from memory.

The suite lives **outside this repository** and is not vendored here. Two identical
copies exist locally:

```text
C:\Users\clclc\RineyaExperiment\Rineya_Long_Context_Demo_v1
C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Long_Context_Demo_v1
```

`longctx_showcase64_registry.json` pins content hashes instead, so drift is
detectable without copying 256 long-context prompts into the repo.

## Pinned protocol

Recovered from the stored anchor artifacts, not guessed:

| Item | Value |
|---|---|
| Cases | `data/showcase_prompts.jsonl`, sha256 `7fbf6a06…182c6d`, **64 rows** |
| Answer key | `data/showcase_answer_key.jsonl`, sha256 `dc1541b6…b6d53b` |
| Prompt | `apply_chat_template(add_generation_prompt=True)`, `system`+`user` only |
| Tokenization | `add_special_tokens=False` |
| Generation | greedy, `do_sample=False`, `max_new_tokens=24`, `use_cache=True`, seed 17 |
| Context | 4096-token cap, overflow is an error — **never truncate** |
| Scoring | `strict_exact`: normalized prediction ∈ normalized `accepted_answers` |

`enable_thinking` was determined by measurement rather than assumption. The stored
Qwen anchor records `input_tokens=1145` for `RLC-001-B`; re-rendering that row gives
**1145 with `enable_thinking=False`** and 1141 with the default. The stored Aegis
anchor records **1141**, so the two historical runs did *not* use the same setting.
Each anchor below was reproduced under its own historical condition.

## Anchor reproduction

| Model | Historical | Reproduced | Agreement with stored predictions |
|---|---:|---:|---|
| Qwen3-0.6B (official, pinned revision) | 14/64 | **13/64** | 61/64 identical text, 64/64 identical prompt tokens |
| Rineya-Aegis-1.1.0-devfreeze | 4/64 | **4/64** | **64/64 bit-identical predictions** |
| Rineya-Halo-1.2.0-devfreeze | 30/64 | **30/64** | score exact; no stored per-row artifact exists |

The one Qwen row that flips is `RLC-005-P` (historical `Answer: 5`, reproduced `2`).
A 61/64 own-model agreement is the *same* band the upstream family record reports
for its own standardized rerun, so this executable path sits inside the established
equivalence class rather than being a new protocol.

### Scorer fidelity

`register_longctx64.py` re-implements the suite's `strict_exact` rule and refuses to
report a number until it reproduces the suite's own stored reports. All three
reproduce exactly: `aegis_showcase` 4/64, `greedy_soup_v2_showcase` 0/64,
`qwen3_06b_showcase` 14/64.

## Two protocol epochs — do not mix them

`Rineya_Published_Cores\FAMILY_PERFORMANCE_AND_RESEARCH.md` §1.2 records that both
of these are valid and different:

| Model | Historical selection run | Standardized 2026-08-26 rerun |
|---|---:|---:|
| Halo 1.2 | 30/64 | 29/64 |
| Nimbus 1.6 | 29/64 | 30/64 |

The reproduction above targets the **historical** epoch, which is the epoch the CRCC
plan quotes. Halo's `30/64` has no stored per-row artifact; its only provenance is
`showcase_hits` in Halo's `CORE.json`. That it reproduces exactly is meaningful, but
it is a score-level match, not a row-level one.

## Two defects found while recovering this

Both would have silently corrupted any future comparison.

1. **RoPE base collapse.** Halo and Aegis configs were written by transformers
   5.14.1, which stores `rope_parameters.rope_theta`. Under the pinned transformers
   4.57.1 that key is ignored and `rope_theta` falls back to the class default
   `10000.0` instead of the declared `1000000` — a 100× error in the RoPE base, on a
   *long-context* benchmark. Both staged copies now carry an explicit top-level
   `rope_theta` translated from their own declared value.
2. **Tokenizer config incompatibility.** The same transformers-5 configs make 4.57.1
   attempt a slow-tokenizer path that demands `protobuf`. Substituting Qwen's
   tokenizer files is exactly equivalent and was verified, not assumed: identical
   token IDs on **64/64** prompts for both Halo and Aegis, byte-identical
   `merges.txt` and `vocab.json`, and byte-identical chat templates.

## Suite integrity note

`MANIFEST.sha256` verifies for **all 12 evidence files** — every data file, the
prompt template, the schema, and the scorer. It fails for exactly two *driver*
scripts (`run_local_hf.py`, `inspect_token_lengths.py`), which have drifted since the
manifest was written and also differ between the two local copies. The registry
therefore pins evidence files only, and the anchors were reproduced with the
2026-08-19 copy contemporaneous with the historical runs.

## Reproduce

```bash
python scripts/critical_core/register_longctx64.py --suite-root <suite> --prediction qwen3_0_6b_official=<preds.jsonl> --anchor qwen3_0_6b_official=14 --out experiments/catastrophic_removal_core/evaluators/longctx_showcase64_registry.json
```
