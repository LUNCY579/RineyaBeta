#!/usr/bin/env python3
"""Non-invasive analysis of a completed CRCC discovery scan.

Answers only the questions a discovery scan can answer on its own: whether
catastrophic-removal criticality and residual angular rewrite are the same signal,
and how criticality is distributed over depth.  Everything about donor value stays
out of reach until a nonempty core exists, so this module deliberately computes no
donor statistic and loads no weights.

Usage:
  python scripts/critical_core/analyze_discovery.py \
      --scan experiments/catastrophic_removal_core/results/discovery_v1/scan_summary.json \
      --out experiments/catastrophic_removal_core/results/discovery_v1/discovery_analysis.json
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    SCHEMA_VERSION,
    atomic_write_json,
    object_sha256,
    read_json,
    repo_relative,
    resolve_repo_path,
)

ANALYSIS_SCHEMA = "crcc.discovery-analysis.v1"

# Depth bands are descriptive groupings for reporting, fixed here rather than
# chosen per result so the same partition applies to any future rerun.
DEPTH_BANDS = (
    ("early", 0, 3),
    ("early_mid", 3, 10),
    ("mid", 10, 20),
    ("deep", 20, 27),
    ("final", 27, 28),
)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        raise ContractError("pearson needs two equal-length series of at least 2 points")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if denominator == 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def rank(values: Sequence[float]) -> list[float]:
    """Average ranks, so ties do not bias the rank correlation."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rank(x), rank(y))


def analyze(scan: Mapping[str, Any]) -> dict[str, Any]:
    if scan.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("scan summary schema is not CRCC v1")
    rows = sorted(scan["layers"], key=lambda row: int(row["layer"]))
    depth = int(scan["num_hidden_layers"])
    if [int(row["layer"]) for row in rows] != list(range(depth)):
        raise ContractError("analysis requires a complete 0..N-1 layer scan")
    if any(row.get("single_layer_angular_distance") is None for row in rows):
        raise ContractError("scan summary lacks the joined angular-distance profile")

    criticality = [float(row["criticality_score"]) for row in rows]
    angular = [float(row["single_layer_angular_distance"]) for row in rows]
    ratios = [row["signals"]["reference_nll_ratio"] for row in rows]
    if any(value is None for value in ratios):
        raise ContractError("a layer produced non-finite reference NLL; ratios are undefined")
    log_ratio = [math.log2(max(float(value), 1e-12)) for value in ratios]

    ranked_criticality = sorted(range(depth), key=lambda i: -criticality[i])
    ranked_angular = sorted(range(depth), key=lambda i: -angular[i])
    top_k = 6
    overlap = sorted(set(ranked_criticality[:top_k]) & set(ranked_angular[:top_k]))

    bands = {}
    for name, low, high in DEPTH_BANDS:
        window = criticality[low:high]
        bands[name] = {
            "layers": list(range(low, high)),
            "n": len(window),
            "mean_criticality": sum(window) / len(window),
            "max_criticality": max(window),
        }

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "DISCOVERY_ANALYSIS_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_scan": {
            "run_identity_sha256": scan["run_identity_sha256"],
            "summary_sha256": scan.get("summary_sha256"),
            "num_hidden_layers": depth,
        },
        "automatic_critical_layers": list(scan["automatic_critical_layers"]),
        "criticality_vs_angular": {
            "pearson_r": pearson(angular, criticality),
            "spearman_rho": spearman(angular, criticality),
            "pearson_r_log2_nll_ratio": pearson(angular, log_ratio),
            "top_k": top_k,
            "top_k_by_criticality": ranked_criticality[:top_k],
            "top_k_by_angular_distance": ranked_angular[:top_k],
            "top_k_overlap": overlap,
            "top_k_overlap_count": len(overlap),
        },
        "depth_bands": bands,
        "per_layer": [
            {
                "layer": int(row["layer"]),
                "criticality_score": criticality[index],
                "reference_nll_ratio": float(ratios[index]),
                "single_layer_angular_distance": angular[index],
                "angular_rank": ranked_angular.index(index),
                "automatic_crcc": bool(row["catastrophic"]),
                "overall_catastrophic": bool(row.get("overall_catastrophic", False)),
                "supporting_sources": list(row.get("supporting_sources", [])),
            }
            for index, row in enumerate(rows)
        ],
        "claim_boundary": (
            "Discovery-only diagnostics. No donor delta, merge, development gate or "
            "sealed result is represented here."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        default="experiments/catastrophic_removal_core/results/discovery_v1/scan_summary.json",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    scan_path = resolve_repo_path(args.scan, must_exist=True)
    analysis = analyze(read_json(scan_path))
    analysis["source_scan"]["path"] = repo_relative(scan_path)
    analysis["analysis_sha256"] = object_sha256(
        {key: value for key, value in analysis.items() if key != "created_at"}
    )

    correlation = analysis["criticality_vs_angular"]
    print("CRCC criticality vs residual angular distance")
    print(f"  pearson  r   = {correlation['pearson_r']:+.4f}")
    print(f"  spearman rho = {correlation['spearman_rho']:+.4f}")
    print(f"  log2(NLL ratio) vs angular r = {correlation['pearson_r_log2_nll_ratio']:+.4f}")
    print(
        f"  top-{correlation['top_k']} overlap "
        f"{correlation['top_k_overlap_count']}/{correlation['top_k']}: "
        f"{correlation['top_k_overlap']}"
    )
    print("\ndepth bands (mean criticality)")
    for name, band in analysis["depth_bands"].items():
        print(f"  {name:10s} n={band['n']:2d} mean={band['mean_criticality']:.4f}")

    if args.out:
        output = resolve_repo_path(args.out)
        atomic_write_json(output, analysis)
        print(f"\nwrote {repo_relative(output)}")


if __name__ == "__main__":
    main()
