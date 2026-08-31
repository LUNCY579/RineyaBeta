#!/usr/bin/env python3
"""Materialize M4 component-local curriculum with byte-identical membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIFFICULTY_ORDER = {"basic": 0, "intermediate": 1, "compositional": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode()).hexdigest()


def canonical_row_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", default="artifacts/heal_mixes/B0/heal_mix_b0.jsonl")
    parser.add_argument("--metadata-source", default=(
        "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/"
        "data/sft/rineya_v3_recommended_combined_124000.jsonl"))
    parser.add_argument("--out",
                        default="artifacts/heal_mixes/RGE_M4/curriculum_mix.jsonl")
    parser.add_argument("--manifest",
                        default="experiments/rge_peft/evidence/m4_curriculum_manifest.json")
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if args.buckets < 2:
        raise SystemExit("curriculum needs at least two aligned buckets")
    mix_path, metadata_path = ROOT / args.mix, ROOT / args.metadata_source
    output, manifest_path = ROOT / args.out, ROOT / args.manifest
    if output.exists() or manifest_path.exists():
        raise SystemExit("curriculum output/manifest already exists; refusing overwrite")

    metadata = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata[str(row["id"])] = row["metadata"]
    rows = [json.loads(line) for line in mix_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["source_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("B0 source_id values are not unique")

    components: dict[str, list[dict]] = defaultdict(list)
    difficulty_counts = Counter()
    provenance_counts = Counter()
    enriched = []
    for row in rows:
        source_id = str(row["source_id"])
        source = str(row["source"])
        category = str(row["category"])
        if source == "v3_chat":
            if source_id not in metadata:
                raise SystemExit(f"missing v3 metadata for {source_id}")
            raw_difficulty = metadata[source_id].get("difficulty")
            if raw_difficulty in DIFFICULTY_ORDER:
                difficulty = str(raw_difficulty)
                rank_key = (DIFFICULTY_ORDER[difficulty],
                            stable_hash(args.seed, "v3", source_id))
                provenance = "v3_metadata_difficulty"
            else:
                difficulty = "fixed_random"
                rank_key = (0, stable_hash(args.seed, "preservation_replay", source_id))
                provenance = "fixed_random_no_difficulty_metadata"
        else:
            difficulty = "fixed_random"
            rank_key = (0, stable_hash(args.seed, "rag_replay", source_id))
            provenance = "fixed_random_no_comparable_score"
        component = f"{source}/{category}"
        payload = {"row": row, "source_id": source_id, "component": component,
                   "difficulty": difficulty, "rank_key": rank_key,
                   "row_hash": canonical_row_hash(row), "provenance": provenance}
        components[component].append(payload)
        enriched.append(payload)
        difficulty_counts[difficulty] += 1
        provenance_counts[provenance] += 1

    bucket_rows: list[list[dict]] = [[] for _ in range(args.buckets)]
    component_bucket_counts = {}
    for component, items in sorted(components.items()):
        items.sort(key=lambda item: item["rank_key"])
        counts = [0] * args.buckets
        for index, item in enumerate(items):
            bucket = min(args.buckets - 1,
                         math.floor(index * args.buckets / len(items)))
            bucket_rows[bucket].append(item)
            counts[bucket] += 1
        component_bucket_counts[component] = counts
    ordered = []
    for bucket, items in enumerate(bucket_rows):
        items.sort(key=lambda item: stable_hash(
            args.seed, "bucket_mix", bucket, item["source_id"]))
        ordered.extend(items)
    if {item["source_id"] for item in ordered} != set(ids):
        raise SystemExit("curriculum membership changed during materialization")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ordered:
            handle.write(json.dumps(item["row"], ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    membership_payload = "".join(
        f"{item['source_id']}:{item['row_hash']}\n"
        for item in sorted(enriched, key=lambda value: value["source_id"]))
    order_payload = "".join(f"{item['source_id']}\n" for item in ordered)
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_M4_RUNS",
        "seed": args.seed,
        "buckets": args.buckets,
        "source_mix": args.mix,
        "source_mix_sha256": sha256_file(mix_path),
        "metadata_source": args.metadata_source,
        "metadata_source_sha256": sha256_file(metadata_path),
        "ordered_mix": args.out,
        "ordered_mix_sha256": sha256_file(output),
        "rows": len(rows),
        "membership_sha256": hashlib.sha256(membership_payload.encode()).hexdigest(),
        "order_sha256": hashlib.sha256(order_payload.encode()).hexdigest(),
        "source_order_sha256": hashlib.sha256(
            "".join(f"{source_id}\n" for source_id in ids).encode()).hexdigest(),
        "difficulty_order": DIFFICULTY_ORDER,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "score_provenance_counts": dict(sorted(provenance_counts.items())),
        "component_count": len(components),
        "bucket_counts": [len(items) for items in bucket_rows],
        "component_bucket_counts_sha256": hashlib.sha256(json.dumps(
            component_bucket_counts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "membership_identical": True,
        "sealed_data_accessed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
