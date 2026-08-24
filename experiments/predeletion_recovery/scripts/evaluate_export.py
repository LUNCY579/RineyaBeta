#!/usr/bin/env python3
"""Evaluate an exported PDRR checkpoint with Rineya's existing v4.5 gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))

from pdrr.provenance import atomic_write_json, sha256_file  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", default="chat_dev,rag_dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cap-mib", type=float, default=4096.0)
    parser.add_argument("--run-manifest", default=None)
    args = parser.parse_args()

    gate_script = ROOT / "scripts" / "eval" / "run_gates.py"
    if not gate_script.exists():
        raise FileNotFoundError(f"existing Rineya gate harness not found: {gate_script}")
    model = resolve(args.model)
    if not model.exists():
        raise FileNotFoundError(model)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    run_meta = {}
    if args.run_manifest:
        run_meta = json.loads(resolve(args.run_manifest).read_text(encoding="utf-8"))
    all_rows = []
    summaries = {}
    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        destination = out / dataset
        command = [
            sys.executable, str(gate_script),
            "--dataset", dataset,
            "--model", str(model),
            "--out", str(destination),
            "--cap-mib", str(args.cap_mib),
        ]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        print(" ".join(command))
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        summary_path = destination / "summary.json"
        results_path = destination / "results.jsonl"
        summaries[dataset] = json.loads(summary_path.read_text(encoding="utf-8"))
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            all_rows.append({
                "dataset": dataset,
                "id": str(row["id"]),
                "category": str(row.get("category", "uncategorized")),
                "quality_score": float(row.get("quality_score", 0.0)),
                "quality_pass": bool(row.get("quality_pass", False)),
                "reply": row.get("reply"),
                "reference": row.get("reference"),
                "arm": run_meta.get("arm"),
                "seed": run_meta.get("seed"),
                "target_layer": run_meta.get("target_layer"),
            })

    rows_path = out / "benchmark_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_write_json(out / "benchmark_manifest.json", {
        "model": str(model),
        "datasets": list(summaries),
        "summaries": summaries,
        "rows": len(all_rows),
        "rows_sha256": sha256_file(rows_path),
        "run": run_meta,
        "cpu_forced": args.cpu,
    })
    print(f"wrote {rows_path} ({len(all_rows)} paired benchmark rows)")


if __name__ == "__main__":
    main()
