#!/usr/bin/env python3
"""Compile SGQR quality/storage/compute comparisons from evaluation summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.sgqr.utility import recovery_ratio


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--run must use NAME=PATH")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError("--run must use non-empty NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True)
    parser.add_argument("--deleted", required=True)
    parser.add_argument("--run", action="append", default=[], help="NAME=summary.json")
    parser.add_argument("--metric", default="mean_quality_score")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dense = load(args.dense)
    deleted = load(args.deleted)
    dense_quality = float(dense[args.metric])
    deleted_quality = float(deleted[args.metric])

    rows = []
    for spec in args.run:
        name, path = parse_run(spec)
        payload = load(str(path))
        quality = float(payload[args.metric])
        sgqr = payload.get("sgqr", {})
        routing = sgqr.get("routing", {})
        storage = sgqr.get("storage", {})
        performance = payload.get("performance", {})
        try:
            ratio = recovery_ratio(quality, deleted_quality, dense_quality)
        except ZeroDivisionError:
            ratio = None
        rows.append(
            {
                "name": name,
                "path": str(path),
                "quality": quality,
                "recovery_ratio": ratio,
                "active_group_fraction": routing.get("active_group_fraction"),
                "packed_storage_bytes": storage.get("packed_storage_bytes", 0),
                "peak_allocated_bytes": performance.get("peak_allocated_bytes"),
                "peak_reserved_bytes": performance.get("peak_reserved_bytes"),
                "tokens_per_second": performance.get("tokens_per_second"),
                "p90_case_latency_ms": performance.get("p90_case_latency_ms"),
            }
        )

    result = {
        "format": "sgqr-comparison-v1",
        "metric": args.metric,
        "dense_quality": dense_quality,
        "deleted_quality": deleted_quality,
        "deletion_damage": dense_quality - deleted_quality,
        "runs": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# SGQR Comparison",
        "",
        f"Metric: `{args.metric}`",
        "",
        f"Dense: {dense_quality:.6f}",
        "",
        f"Deleted: {deleted_quality:.6f}",
        "",
        "| Arm | Quality | Recovery Ratio | Active groups | Packed bytes | Peak VRAM | tok/s | P90 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = "n/a" if row["recovery_ratio"] is None else f"{row['recovery_ratio']:.4f}"
        active = (
            "n/a"
            if row["active_group_fraction"] is None
            else f"{float(row['active_group_fraction']):.4f}"
        )
        peak = row["peak_allocated_bytes"]
        tok_s = row["tokens_per_second"]
        p90 = row["p90_case_latency_ms"]
        lines.append(
            f"| {row['name']} | {row['quality']:.6f} | {ratio} | {active} | "
            f"{row['packed_storage_bytes']} | {peak if peak is not None else 'n/a'} | "
            f"{float(tok_s):.3f} | {float(p90):.3f} |"
            if tok_s is not None and p90 is not None
            else
            f"| {row['name']} | {row['quality']:.6f} | {ratio} | {active} | "
            f"{row['packed_storage_bytes']} | {peak if peak is not None else 'n/a'} | n/a | n/a |"
        )
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path} and {out_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
