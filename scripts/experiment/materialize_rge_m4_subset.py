#!/usr/bin/env python3
"""Freeze an exact same-membership M4 subset for complete-epoch comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stable_mod(seed: int, source_id: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{seed}|m4_subset|{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="artifacts/heal_mixes/B0/heal_mix_b0.jsonl")
    parser.add_argument("--curriculum",
                        default="artifacts/heal_mixes/RGE_M4/curriculum_mix.jsonl")
    parser.add_argument("--uniform-out",
                        default="artifacts/heal_mixes/RGE_M4/uniform_subset10.jsonl")
    parser.add_argument("--curriculum-out",
                        default="artifacts/heal_mixes/RGE_M4/curriculum_subset10.jsonl")
    parser.add_argument("--manifest",
                        default="experiments/rge_peft/evidence/m4_subset_manifest.json")
    parser.add_argument("--modulus", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    source_path, curriculum_path = ROOT / args.source, ROOT / args.curriculum
    uniform_out, curriculum_out = ROOT / args.uniform_out, ROOT / args.curriculum_out
    manifest_path = ROOT / args.manifest
    if any(path.exists() for path in (uniform_out, curriculum_out, manifest_path)):
        raise SystemExit("M4 subset output already exists; refusing overwrite")
    source_rows, curriculum_rows = load(source_path), load(curriculum_path)
    source_ids = {str(row["source_id"]) for row in source_rows}
    curriculum_ids = {str(row["source_id"]) for row in curriculum_rows}
    if source_ids != curriculum_ids:
        raise SystemExit("full source/curriculum membership differs")
    selected = {source_id for source_id in source_ids
                if stable_mod(args.seed, source_id, args.modulus) == 0}
    uniform = [row for row in source_rows if str(row["source_id"]) in selected]
    curriculum = [row for row in curriculum_rows if str(row["source_id"]) in selected]
    if {str(row["source_id"]) for row in uniform} != \
            {str(row["source_id"]) for row in curriculum}:
        raise SystemExit("subset membership differs")
    write(uniform_out, uniform); write(curriculum_out, curriculum)
    components = Counter(f"{row['source']}/{row['category']}" for row in uniform)
    membership = "".join(f"{source_id}\n" for source_id in sorted(selected))
    uniform_order = "".join(f"{row['source_id']}\n" for row in uniform)
    curriculum_order = "".join(f"{row['source_id']}\n" for row in curriculum)
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_M4_RUNS",
        "seed": args.seed,
        "selection": f"sha256(seed|m4_subset|source_id) mod {args.modulus} == 0",
        "source_rows": len(source_rows),
        "selected_rows": len(selected),
        "selection_fraction": len(selected) / len(source_rows),
        "membership_sha256": hashlib.sha256(membership.encode()).hexdigest(),
        "uniform_mix": args.uniform_out,
        "uniform_mix_sha256": sha256_file(uniform_out),
        "uniform_order_sha256": hashlib.sha256(uniform_order.encode()).hexdigest(),
        "curriculum_mix": args.curriculum_out,
        "curriculum_mix_sha256": sha256_file(curriculum_out),
        "curriculum_order_sha256": hashlib.sha256(curriculum_order.encode()).hexdigest(),
        "component_count": len(components),
        "component_counts_sha256": hashlib.sha256(json.dumps(
            dict(sorted(components.items())), sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "membership_identical": True,
        "consume_complete_epoch": True,
        "sealed_data_accessed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
