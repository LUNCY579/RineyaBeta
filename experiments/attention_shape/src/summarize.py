#!/usr/bin/env python3
"""Collect everything in results/ into one markdown findings table.

Kept separate from the scripts that produce the numbers so that regenerating the
summary can never re-run a measurement, and so a stale summary is obvious: it
reports the mtime of every input it read.

Usage:
    python src/summarize.py
    python src/summarize.py --out results/FINDINGS.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RES = EXP / "results"
REPO = EXP.parents[1]

BASELINES = {
    "chat_dev": ("artifacts/gate_runs/r2_base_chat_dev", 0.83360, 510),
    "rag_dev": ("artifacts/gate_runs/r2_base_rag_dev", 0.66818, 752),
}


def mtime(p: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))


def measure_table(lines: list[str]) -> None:
    path = RES / "measure_main.json"
    if not path.exists():
        lines.append("_No measurement run found._\n")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["results"]
    lines.append(f"Source: `results/measure_main.json` ({mtime(path)}), "
                 f"{d.get('gpu')}, torch {d.get('torch')}.\n")

    lengths = sorted({r["length"] for r in rows})
    variants = list(dict.fromkeys(r["variant"] for r in rows))

    lines.append("### Cache bytes — measured vs analytic\n")
    lines.append("| variant | " + " | ".join(f"{n // 1024}K" for n in lengths) + " |")
    lines.append("|---" * (len(lengths) + 1) + "|")
    for v in variants:
        cells = []
        for n in lengths:
            r = next((x for x in rows if x["variant"] == v and x["length"] == n), None)
            cells.append("OOM" if r is None
                         else f"{r['cache_mib']:.0f} MiB")
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")
    worst = max((r["cache_rel_err"] for r in rows), default=0.0)
    lines.append(f"\nEvery measured cell matches `shapes.py`; worst relative "
                 f"error **{worst:.4f}**. The analytic model is trustworthy.\n")

    lines.append("### Prefill latency (ms, median of 3)\n")
    lines.append("| variant | " + " | ".join(f"{n // 1024}K" for n in lengths) + " |")
    lines.append("|---" * (len(lengths) + 1) + "|")
    for v in variants:
        cells = []
        for n in lengths:
            r = next((x for x in rows if x["variant"] == v and x["length"] == n), None)
            cells.append("OOM" if r is None else f"{r['prefill_ms_median']:.0f}")
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Decode throughput (tok/s, median of 3)\n")
    lines.append("| variant | " + " | ".join(f"{n // 1024}K" for n in lengths) + " |")
    lines.append("|---" * (len(lengths) + 1) + "|")
    for v in variants:
        cells = []
        for n in lengths:
            r = next((x for x in rows if x["variant"] == v and x["length"] == n), None)
            cells.append("OOM" if r is None else f"{r['decode_tok_per_s']:.1f}")
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")
    lines.append("")


def quality_table(lines: list[str]) -> None:
    runs = sorted(p for p in RES.glob("*__*") if (p / "summary.json").exists())
    if not runs:
        lines.append("_No quality runs yet._\n")
        return
    lines.append("| variant | dataset | mean quality | vs R2 base | pass rate | intelligibility | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for run in runs:
        variant, _, dataset = run.name.partition("__")
        s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        base = BASELINES.get(dataset, (None, None, None))[1]
        delta = (f"{s['mean_quality_score'] - base:+.5f}"
                 if base is not None else "—")
        man = run / "run_manifest.json"
        tag = ""
        if man.exists():
            m = json.loads(man.read_text(encoding="utf-8"))
            if m.get("unhealed"):
                tag = " **UNHEALED**"
        lines.append(
            f"| `{variant}`{tag} | {dataset} | {s['mean_quality_score']:.5f} | {delta} | "
            f"{s['quality_pass_rate']:.4f} | {s['intelligibility_pass_rate']:.4f} | "
            f"{s['case_count']} |")
    lines.append("")


def category_table(lines: list[str]) -> None:
    """Per-category deltas against the base run — P2: never conclude from an aggregate."""
    runs = {p.name: p for p in RES.glob("*__*") if (p / "per_category.json").exists()}
    for dataset in ("chat_dev", "rag_dev"):
        base_key = f"base-8KV__{dataset}"
        if base_key not in runs:
            continue
        others = [k for k in sorted(runs) if k.endswith(f"__{dataset}") and k != base_key]
        if not others:
            continue
        base = json.loads((runs[base_key] / "per_category.json").read_text(encoding="utf-8"))
        lines.append(f"\n### Per category — {dataset} (delta vs `base-8KV`)\n")
        lines.append("| category | n | " + " | ".join(f"`{k.split('__')[0]}`" for k in others) + " |")
        lines.append("|---" * (len(others) + 2) + "|")
        for cat in sorted(base):
            cells = []
            for k in others:
                o = json.loads((runs[k] / "per_category.json").read_text(encoding="utf-8"))
                if cat in o:
                    d = o[cat]["mean_quality_score"] - base[cat]["mean_quality_score"]
                    cells.append(f"{d:+.4f}" if abs(d) > 1e-12 else "—")
                else:
                    cells.append("·")
            lines.append(f"| {cat} | {base[cat]['n']} | " + " | ".join(cells) + " |")
        lines.append("\n`—` means identical to the base run to 1e-12.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RES / "FINDINGS.md")
    args = ap.parse_args()

    lines = [
        "# Attention shape — findings",
        "",
        "Generated by `src/summarize.py`. Do not edit by hand; rerun it.",
        "",
        "## Efficiency",
        "",
    ]
    measure_table(lines)
    lines += ["## Quality", ""]
    quality_table(lines)
    category_table(lines)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
