#!/usr/bin/env python3
"""Verify the vendored quality evaluator reproduces upstream's own numbers.

Vendoring is only worth doing if the copy scores identically to the original.
Otherwise every comparison against a prior Rineya artifact is quietly wrong, and
it looks like a measurement rather than an error.

So: take a prior run's results.jsonl, feed each row's recorded `reply` and token
counts back through the vendored scorer, and compare the recomputed
`quality_score` against the value upstream stored. They must match to within
floating-point noise.

Also checks the aggregate against the stored summary.json, since a per-row match
with an aggregate mismatch would mean the summarising path differs.

Usage: python scripts/eval/verify_scorer.py [--tol 1e-9]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import (  # noqa: E402
    QUALITY_EVALUATION_VERSION, aggregate_quality_rows, build_corpus_profile,
    evaluate_response, load_quality_config,
)

UPSTREAM = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/short-response-model-phase-0-1-2"
                r"/artifacts/teacher_dual_sft_v1/baseline_chat_dev")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--results", default=str(UPSTREAM / "results.jsonl"))
    ap.add_argument("--summary", default=str(UPSTREAM / "summary.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print(f"scorer version : {QUALITY_EVALUATION_VERSION}")
    print(f"upstream rows  : {len(rows):,}\n")

    config = load_quality_config()
    # The profile must come from the DATASET records, not the result rows, and the
    # token count must be the *visible* count for both arguments. Both details are
    # taken from the upstream reference implementation (native_v45.py); getting
    # either wrong shifts quality_score by ~3e-3 on about a fifth of the rows.
    dataset = [json.loads(l) for l in
               (ROOT / "artifacts/eval_sets/chat_dev.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    profile = build_corpus_profile(dataset)

    worst = 0.0
    mismatches = []
    recomputed = []
    for row in rows:
        visible = int(row.get("visible_generated_tokens", 0) or 0)
        got = evaluate_response(
            row, str(row.get("reply", "")), visible,
            visible_model_tokens=visible,
            config=config, corpus_profile=profile)
        recomputed.append({**got, "category": row.get("category")})
        a = float(row.get("quality_score", float("nan")))
        b = float(got.get("quality_score", float("nan")))
        delta = abs(a - b)
        worst = max(worst, delta)
        if delta > args.tol:
            mismatches.append((row.get("id"), a, b, delta))

    print(f"per-row quality_score: max |delta| = {worst:.3e}")
    print(f"  mismatches beyond tol {args.tol:g}: {len(mismatches)}")
    for rid, a, b, d in mismatches[:8]:
        print(f"    {rid}: upstream {a:.6f} vs vendored {b:.6f}  (d={d:.2e})")

    stored = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    agg = aggregate_quality_rows(recomputed)
    print("\naggregate comparison:")
    ok_agg = True
    for key in ("mean_quality_score", "quality_pass_rate"):
        if key in stored and key in agg:
            d = abs(float(stored[key]) - float(agg[key]))
            ok_agg &= d <= 1e-6
            print(f"  {key:22s} upstream {float(stored[key]):.6f}  "
                  f"vendored {float(agg[key]):.6f}  d={d:.2e}")
        else:
            print(f"  {key:22s} not comparable (upstream has "
                  f"{'yes' if key in stored else 'no'}, vendored "
                  f"{'yes' if key in agg else 'no'})")

    print("\n--- verdict ---")
    if mismatches or not ok_agg:
        print("FAIL: the vendored scorer does not reproduce upstream numbers. "
              "Comparisons against prior artifacts would be invalid.")
        raise SystemExit(1)
    print("PASS: vendored scorer reproduces upstream per-row and aggregate scores. "
          "R2 results are directly comparable with the prior Rineya artifacts.")


if __name__ == "__main__":
    main()
