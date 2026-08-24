#!/usr/bin/env python3
"""Run the controlled, deterministic CPU PDRR reference study."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from pdrr.analysis import aggregate_arm_results, claim_support, paired_quality_family  # noqa: E402
from pdrr.arms import arm_registry  # noqa: E402
from pdrr.provenance import atomic_write_json, atomic_write_text, config_hash  # noqa: E402
from pdrr.toy import config_from_mapping, run_reference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "configs" / "cpu_reference.yaml"))
    parser.add_argument("--out", default=str(HERE / "results" / "cpu_reference"))
    parser.add_argument("--quick", action="store_true",
                        help="one seed, 8 steps, and reduced data for CI/smoke testing")
    parser.add_argument("--arms", default=None, help="optional comma-separated arm subset")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config, seeds = config_from_mapping(payload)
    if args.quick:
        config = replace(config, train_examples=96, eval_examples=64, steps=8, batch_size=16)
        seeds = [17]
    arms = None if args.arms is None else [value.strip() for value in args.arms.split(",") if value.strip()]
    unknown = set(arms or []) - arm_registry().keys()
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")

    results = run_reference(config, seeds, arms=arms)
    records = [result.to_json() for result in results]
    aggregate = aggregate_arm_results(records)
    analysis_config = payload.get("analysis", {})
    contrasts = paired_quality_family(
        records,
        bootstrap_samples=int(analysis_config.get("bootstrap_samples", 5000)),
        permutation_samples=int(analysis_config.get("permutation_samples", 10000)),
        practical_threshold=float(analysis_config.get("practical_threshold", 0.005)),
        seed=int(payload.get("seed", 20260823)),
    ) if "causal_state_position_recovery" in aggregate else []
    claims = claim_support(contrasts, aggregate)

    expected_cells = len(seeds) * len(arms or arm_registry())
    compact = [row for row in records if row["arm"] != "full"]
    full_quality = aggregate.get("full", {}).get("mean_quality", 1.0)
    hard_quality = aggregate.get("hard_prune", {}).get("mean_quality", full_quality)
    proposed = aggregate.get("causal_state_position_recovery", {})
    checks = {
        "expected_arm_seed_cells": len(records) == expected_cells,
        "all_registered_arms_present": arms is not None or set(aggregate) == set(arm_registry()),
        "hard_pruning_causes_quality_damage": float(full_quality) > float(hard_quality),
        "all_gate_zero_physical_checks_pass": all(
            row.get("gate_zero_physical_max_abs") is None
            or float(row["gate_zero_physical_max_abs"]) <= 1e-5 for row in records),
        "all_compact_parameter_frontiers_match_hard_pruning": all(
            int(row["final_parameter_count"]) == int(row["hard_parameter_count"])
            for row in compact),
        "full_method_beats_random_router_in_reference": (
            not proposed or float(proposed["mean_quality"])
            > float(aggregate["random_matrix_recovery"]["mean_quality"])),
        "full_method_beats_gradual_kd_in_reference": (
            not proposed or float(proposed["mean_quality"])
            > float(aggregate["gradual_kd"]["mean_quality"])),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_write_json(out / "summary.json", {
        "study_type": "controlled CPU residual-sequence mechanism study; not Rineya evidence",
        "quick": args.quick,
        "config_hash": config_hash({"config": config.__dict__, "seeds": seeds, "arms": arms}),
        "config": config.__dict__,
        "seeds": seeds,
        "aggregate": aggregate,
        "contrasts": [contrast.__dict__ for contrast in contrasts],
        "claim_gates": claims,
        "validation": checks,
        "all_validation_checks_pass": all(checks.values()),
    })
    atomic_write_json(out / "validation.json", {
        "all_validation_checks_pass": all(checks.values()),
        "checks": checks,
        "interpretation": (
            "Passing validates executable distinctions and accounting only. "
            "It does not establish an LLM quality or novelty claim."
        ),
    })
    atomic_write_text(out / "report.md", report_markdown(aggregate, checks, claims, args.quick))
    print(f"wrote {out}")
    print(f"validation: {'PASS' if all(checks.values()) else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit(2)


def report_markdown(aggregate: dict, checks: dict, claims: dict, quick: bool) -> str:
    lines = [
        "# PDRR controlled CPU reference study",
        "",
        "> This is a residual-sequence mechanism study, not a Rineya checkpoint benchmark. "
        "It cannot support a claim that PDRR improves an LLM.",
        "",
        f"Mode: **{'quick smoke' if quick else 'full preregistered reference'}**",
        "",
        "| Arm | Quality ↑ | Recovery Ratio ↑ | Handoff ↑ | Route/gain ρ ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, row in aggregate.items():
        lines.append(
            f"| `{arm}` | {float(row['mean_quality']):.6f} | "
            f"{float(row.get('mean_recovery_ratio', float('nan'))):.3f} | "
            f"{float(row.get('mean_handoff_completeness', float('nan'))):.3f} | "
            f"{float(row.get('mean_route_alignment_spearman', float('nan'))):.3f} |"
        )
    lines += ["", "## Validation", ""]
    lines += [f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in checks.items()]
    lines += ["", "## Confirmatory claim gates", ""]
    lines += [f"- {'SUPPORTED' if passed else 'NOT SUPPORTED'} — `{name}`"
              for name, passed in claims.items()]
    lines += [
        "",
        "Claim gates may remain unsupported in this small study because three paired seeds "
        "cannot provide strong confirmatory power. That is an intended safeguard, not an error.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
