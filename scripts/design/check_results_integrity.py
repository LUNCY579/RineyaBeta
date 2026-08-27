#!/usr/bin/env python3
"""Structural integrity check over every committed result. No model, no GPU.

WHAT THIS CATCHES that verify_claims.py cannot. verify_claims asserts that a
DOCUMENTED number reproduces from its file. It says nothing about files nobody
quotes, and nothing about whether a file is internally coherent. A run killed
mid-write, two queues appending to one path, or a summary that disagrees with
its own rows would all pass a 215/215 pin report.

WHAT IT DELIBERATELY DOES NOT FLAG. The cornerstone-1 language arms have no
`results.jsonl`: 132 MB of per-case records are gitignored by policy, which is
precisely why identity_counts.json and lang_contrasts.json exist to preserve the
derived figures. A first version of this check reported all eleven as corrupt.
A checker that cries wolf on a deliberate policy is worse than none, because the
next real finding gets waved through with the false ones.

Checks per language arm: summary and manifest present; row count matches the
manifest; no duplicate case ids; the summary's mean equals the mean of its own
rows to 1e-9.
Checks per passkey file: every cell's `acc` equals the hit rate of its own
trial records.

    python scripts/design/check_results_integrity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/kv_significance/results"
DERIVED = {"identity_counts.json", "lang_contrasts.json"}


def main() -> None:
    problems: list[tuple[str, str]] = []
    full = summary_only = passkey = 0

    for d in sorted(RES.glob("*__rag*")):
        if not d.is_dir() or any(p.startswith("_invalidated") for p in d.parts):
            continue
        name = d.name
        s_p, m_p, r_p = d / "summary.json", d / "run_manifest.json", d / "results.jsonl"
        if not (s_p.exists() and m_p.exists()):
            problems.append((name, "missing summary.json or run_manifest.json"))
            continue
        if not r_p.exists():
            summary_only += 1          # per-case records gitignored -- by policy
            continue
        full += 1
        s = json.loads(s_p.read_text(encoding="utf-8"))
        m = json.loads(m_p.read_text(encoding="utf-8"))
        rows = [json.loads(l) for l in r_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        ids = [r["id"] for r in rows]
        if len(rows) != m.get("case_count"):
            problems.append((name, f"{len(rows)} rows against manifest {m.get('case_count')}"))
        elif len(set(ids)) != len(ids):
            problems.append((name, f"{len(ids) - len(set(ids))} duplicate case ids "
                                   f"-- the signature of two writers on one path"))
        else:
            mean = sum(r["quality_score"] for r in rows) / len(rows)
            if abs(mean - s["mean_quality_score"]) > 1e-9:
                problems.append((name, "summary mean disagrees with its own rows"))

    for f in sorted(RES.glob("*.json")):
        if any(p.startswith("_invalidated") for p in f.parts) or f.name in DERIVED:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                                    # noqa: BLE001
            problems.append((f.name, f"unreadable: {type(e).__name__}"))
            continue
        if not isinstance(d, dict) or "results" not in d:
            continue
        passkey += 1
        for r in d["results"]:
            if not isinstance(r, dict) or r.get("acc") is None:
                continue
            tr = r.get("trials") or []
            if tr and abs(sum(1 for t in tr if t["hit"]) / len(tr) - r["acc"]) > 1e-6:
                problems.append((f.name, f"acc disagrees with its trial records "
                                         f"at depth {r.get('depth')}"))
                break

    print(f"  language arms fully verified : {full}")
    print(f"  language arms summary-only   : {summary_only}  "
          f"(per-case records gitignored by policy -- not a defect)")
    print(f"  passkey files verified       : {passkey}")
    print(f"\n{len(problems)} integrity problems")
    for n, why in problems:
        print(f"  BAD  {n}: {why}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
