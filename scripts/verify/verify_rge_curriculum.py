#!/usr/bin/env python3
"""Verify M4 curriculum membership, hashes, and deterministic planner behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from plan import build_plan  # noqa: E402


def canonical_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
                        default="experiments/rge_peft/evidence/m4_curriculum_manifest.json")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m4_curriculum_unit.json")
    parser.add_argument("--subset-manifest",
                        default="experiments/rge_peft/evidence/m4_subset_manifest.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    source = rows(ROOT / manifest["source_mix"])
    ordered = rows(ROOT / manifest["ordered_mix"])
    source_map = {str(row["source_id"]): canonical_hash(row) for row in source}
    ordered_map = {str(row["source_id"]): canonical_hash(row) for row in ordered}
    if len(source_map) != len(source) or len(ordered_map) != len(ordered):
        raise AssertionError("source IDs are not unique")
    if source_map != ordered_map:
        raise AssertionError("curriculum membership or row content changed")
    membership = "".join(f"{source_id}:{digest}\n"
                         for source_id, digest in sorted(source_map.items()))
    if hashlib.sha256(membership.encode()).hexdigest() != manifest[
            "membership_sha256"]:
        raise AssertionError("membership hash does not reproduce")
    order = "".join(f"{row['source_id']}\n" for row in ordered)
    if hashlib.sha256(order.encode()).hexdigest() != manifest["order_sha256"]:
        raise AssertionError("order hash does not reproduce")

    total = [8 + (index * 7) % 43 for index in range(64)]
    supervised = [max(1, value // 3) for value in total]
    kwargs = dict(seed=19, epoch=0, max_tokens_per_microbatch=128,
                  max_rows_per_microbatch=4, micro_batches_per_step=2,
                  megabatch_rows=16, max_len=64, width_bucket=1)
    reference_a = build_plan(total, supervised, **kwargs)
    reference_b = build_plan(total, supervised, **kwargs)
    curriculum_a = build_plan(total, supervised, order_mode="curriculum", **kwargs)
    curriculum_b = build_plan(total, supervised, order_mode="curriculum", **kwargs)
    if reference_a.hash() != reference_b.hash() or curriculum_a.hash() != \
            curriculum_b.hash():
        raise AssertionError("planner is not deterministic")
    if reference_a.hash() == curriculum_a.hash():
        raise AssertionError("curriculum order did not change the plan")
    reference_members = sorted(index for step in reference_a.steps
                               for micro in step.micro for index in micro.indices)
    curriculum_members = sorted(index for step in curriculum_a.steps
                                for micro in step.micro for index in micro.indices)
    if reference_members != list(range(64)) or curriculum_members != list(range(64)):
        raise AssertionError("planner changed membership")
    first_curriculum_indices = {index for step in curriculum_a.steps[:2]
                                for micro in step.micro for index in micro.indices}
    if not first_curriculum_indices.issubset(set(range(16))):
        raise AssertionError("curriculum planner destroyed coarse megabatch order")

    subset = json.loads((ROOT / args.subset_manifest).read_text(encoding="utf-8"))
    uniform_subset = rows(ROOT / subset["uniform_mix"])
    curriculum_subset = rows(ROOT / subset["curriculum_mix"])
    uniform_ids = {str(row["source_id"]) for row in uniform_subset}
    curriculum_ids = {str(row["source_id"]) for row in curriculum_subset}
    if uniform_ids != curriculum_ids or len(uniform_ids) != subset["selected_rows"]:
        raise AssertionError("M4 subset membership differs")
    subset_membership = "".join(f"{source_id}\n" for source_id in sorted(uniform_ids))
    if hashlib.sha256(subset_membership.encode()).hexdigest() != subset[
            "membership_sha256"]:
        raise AssertionError("M4 subset membership hash does not reproduce")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": len(source),
        "membership_sha256": manifest["membership_sha256"],
        "source_order_sha256": manifest["source_order_sha256"],
        "curriculum_order_sha256": manifest["order_sha256"],
        "component_count": manifest["component_count"],
        "subset_rows": subset["selected_rows"],
        "subset_membership_sha256": subset["membership_sha256"],
        "bucket_counts": manifest["bucket_counts"],
        "checks": 10,
        "sealed_data_accessed": False,
    }
    path = ROOT / args.evidence
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
