# Aegis RAG train split (4,410)

Copy of the **grounded RAG rows actually used to train Aegis** (teacher dual-SFT v1, LR `5e-4`). This is not the 12,000-row short-response chat half of that mix, and it is not the 16,410-row shuffled `mix.jsonl`.

| | |
|---|---|
| Core | `rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4` |
| Mix seed | `20260816` |
| RAG rows | **4,410** (all unique IDs; all have an assistant target) |
| Chat rows in Aegis mix (not in this bundle) | 12,000 from Rineya Short-Response Behavior Corpus v1.0 |

## What to train on

Use **`rag_v3_train.sft.jsonl`**. That is the converted SFT view that went into Aegis: assistant text copied from `approved_answer`. The official reviewed file was not modified.

## Layout

```
rag_v3_train.sft.jsonl          # Aegis training rows (converted)
source/train.reviewed.jsonl     # official RAG v3 large train split (same 4,410 IDs)
source/RAG_REASONING_V3_README.md
evidence/knowledge.jsonl        # 6,000 knowledge records those cases cite
evidence/procedures.jsonl       # 200 procedures
evidence/rag_v3.sqlite3         # retrieval DB used in the Aegis dual-SFT pipeline
manifests/aegis_mix_manifest.json
manifests/build_mix.py          # conversion + mix script (chat path still points at Chat-Bubble)
manifests/bundle_hashes.json
```

Dev (752) and sealed test (738) from RAG v3 large are **not** included. They were not in the Aegis training mix.

## Provenance (Chat-Bubble)

- Source: `short-response-model-phase-0-1-2/data/phase_9_student_distillation/rag_reasoning_v3_large/reviewed_splits/train.reviewed.jsonl`
- Converted: `short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/rag_v3_train.sft.jsonl`
- Mix (chat + RAG, not copied here): `…/artifacts/teacher_dual_sft_v1/mix.jsonl`

SHA-256 of `source/train.reviewed.jsonl` matches the upstream corpus hash `b606173cc6dff8358981b23e3cd59dd494dc31ff6cb7c6797519a91c9e3fddb1`. Full file hashes: `manifests/bundle_hashes.json`.
