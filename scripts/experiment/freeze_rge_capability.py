#!/usr/bin/env python3
"""Freeze M6 puzzle validation bank and audit B0 contamination."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def grams(text: str) -> set[str]:
    words = normalize(text).split()
    if len(words) < 2:
        return set(words)
    return {f"{a} {b}" for a, b in zip(words, words[1:])}


def hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")


def simhash(features: set[str]) -> int:
    vector = [0] * 64
    for feature in features:
        value = hash64(feature)
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def bands(signature: int) -> list[tuple[int, int]]:
    return [(band, (signature >> (16 * band)) & 0xFFFF) for band in range(4)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation",
                        default="Rineya-Puzzle-Reasoning-51k/data/validation.jsonl")
    parser.add_argument("--mix", default="artifacts/heal_mixes/B0/heal_mix_b0.jsonl")
    parser.add_argument("--manifest",
                        default="manifests/rge_peft/m6_capability_dev.json")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m6_capability_freeze.json")
    args = parser.parse_args()
    validation_path, mix_path = ROOT / args.validation, ROOT / args.mix
    validation = [json.loads(line) for line in validation_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    prompts = [str(row["prompt"]) for row in validation]
    normalized = [normalize(prompt) for prompt in prompts]
    if len(normalized) != len(set(normalized)):
        raise SystemExit("validation prompts are not unique")
    validation_grams = [grams(prompt) for prompt in prompts]
    signatures = [simhash(value) for value in validation_grams]
    band_to_validation = collections.defaultdict(list)
    for index, signature in enumerate(signatures):
        for key in bands(signature):
            band_to_validation[key].append(index)

    exact_training = set()
    candidate_training = collections.defaultdict(list)
    training_rows = 0
    for line in mix_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line); training_rows += 1
        users = [str(message.get("content", "")) for message in row["messages"]
                 if message.get("role") == "user"]
        prompt = "\n".join(users)
        norm = normalize(prompt)
        exact_training.add(norm)
        feature = grams(prompt)
        signature = simhash(feature)
        interested = set()
        for key in bands(signature):
            interested.update(band_to_validation.get(key, []))
        for index in interested:
            candidate_training[index].append((str(row["source_id"]), feature))
    exact_matches = [validation[index]["id"] for index, value in enumerate(normalized)
                     if value in exact_training]
    high_similarity = []
    maximum = {"jaccard": 0.0, "validation_id": None, "training_source_id": None}
    for index, candidates in candidate_training.items():
        target = validation_grams[index]
        for source_id, feature in candidates:
            union = len(target | feature)
            score = len(target & feature) / union if union else 1.0
            if score > maximum["jaccard"]:
                maximum = {"jaccard": score,
                           "validation_id": validation[index]["id"],
                           "training_source_id": source_id}
            if score >= 0.80:
                high_similarity.append({"validation_id": validation[index]["id"],
                                        "training_source_id": source_id,
                                        "bigram_jaccard": score})
    families = collections.Counter(str(row["family"]) for row in validation)
    difficulties = collections.Counter(str(row["difficulty"]) for row in validation)
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_M6_EVALUATION",
        "asset": args.validation,
        "sha256": sha256_file(validation_path),
        "rows": len(validation),
        "families": dict(sorted(families.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "role": "targeted_capability_dev",
        "selection_timing": "candidate frozen before bank evaluation",
        "generation": {
            "chat_template": "single user message; add_generation_prompt; enable_thinking=false",
            "max_new_tokens": 32,
            "do_sample": False,
            "extractor": "dataset evaluator final/think/plain extraction"
        },
        "evaluation": "normalized exact match against accepted_answers",
        "test_and_challenge_access": "forbidden during M6 selection/confirmation",
        "license": "Apache-2.0",
        "sealed_rag_accessed": False,
    }
    evidence = {
        **manifest,
        "contamination_audit": {
            "training_mix": args.mix,
            "training_mix_sha256": sha256_file(mix_path),
            "training_rows": training_rows,
            "normalized_exact_matches": len(exact_matches),
            "exact_match_ids": exact_matches[:50],
            "simhash_bands": 4,
            "approximate_metric": "token-bigram Jaccard on SimHash-LSH candidates",
            "threshold": 0.80,
            "high_similarity_matches": len(high_similarity),
            "high_similarity_examples": high_similarity[:50],
            "maximum_candidate": maximum,
        },
        "sealed_data_accessed": False,
    }
    manifest_path, evidence_path = ROOT / args.manifest, ROOT / args.evidence
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
