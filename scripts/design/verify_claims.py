#!/usr/bin/env python3
"""Cross-check every headline number in the docs against the committed data.

WHY THIS EXISTS. Over this program five published figures turned out to be wrong,
and none of them were caught by reading the code:

  allocation gain          7.4%   ->  4.2%   (fabricated rho column, bad floor)
  SWA prefill speedup      3.2x   ->  1.9x SLOWER (kernel dispatch, not shape)
  SWA cache saving         5.1x   ->  1.00x resident (counted on a view)
  significance vs SLM      +0.204 ->  -0.083 at keep 0.25 (corrupted baseline)
  MDE at k=3               9.6pp  ->  13.9pp (normal-quantile multiplicity fix)

Each survived because a number in a markdown table and a number in a JSON file
could drift apart with nothing comparing them. This compares them.

Each claim states BOTH the documented value and how to recompute it from a
committed result file, and additionally asserts that the documented value still
literally appears in the prose. So editing the doc without the data fails, and
regenerating the data without the doc fails too.

Exit code is non-zero on any mismatch, so this is usable as a pre-commit check.

Usage:
    python scripts/design/verify_claims.py
    python scripts/design/verify_claims.py --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KVS = ROOT / "experiments/kv_significance/results"
ATS = ROOT / "experiments/attention_shape/results"
DOC = ROOT / "docs/FINDINGS_KV_PROGRAM.md"


def summary(base: Path, tag: str, ds: str = "rag_dev"):
    p = base / f"{tag}__{ds}" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["mean_quality_score"]


def measured(path: Path, key: str, **match):
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            return r.get(key)
    return None


def concentration(k: int):
    p = KVS / "concentration.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["rho_mean"][d["ks"].index(k)]


# (label, documented value, how to recompute, tolerance, text that must appear)
CLAIMS = [
    ("quality: no eviction", 0.66939, lambda: summary(KVS, "off"), 5e-6, "0.66939"),
    ("quality: keep 0.75", 0.65677, lambda: summary(KVS, "frac0.75"), 5e-6, "0.65677"),
    ("quality: keep 0.50 pooled", 0.63371,
     lambda: summary(KVS, "frac0.5-pool7-fixed"), 5e-6, "0.63371"),
    ("quality: keep 0.50", 0.61401, lambda: summary(KVS, "frac0.5"), 5e-6, "0.61401"),
    ("quality: keep 0.25 recent-frac", 0.41932,
     lambda: summary(KVS, "frac0.25-rf0.25"), 5e-6, "0.41932"),
    ("quality: keep 0.25", 0.33863, lambda: summary(KVS, "frac0.25"), 5e-6, "0.33863"),

    ("control: StreamingLLM keep 0.75", 0.47545,
     lambda: summary(KVS, "frac0.75-rf1.0"), 5e-6, "0.47545"),
    ("control: StreamingLLM keep 0.50", 0.47682,
     lambda: summary(KVS, "frac0.5-rf1.0"), 5e-6, "0.47682"),
    ("control: StreamingLLM keep 0.25", 0.42208,
     lambda: summary(KVS, "frac0.25-rf1.0"), 5e-6, "0.42208"),

    ("chunked: peak 32K", 1744.1,
     lambda: measured(KVS / "measure_chunked_long.json", "peak_mib", length=32768),
     0.05, "1744.1"),
    ("chunked: peak 128K", 1744.9,
     lambda: measured(KVS / "measure_chunked_long.json", "peak_mib", length=131072),
     0.05, "1744.9"),
    ("chunked: resident cache 128K", 672.0,
     lambda: measured(KVS / "measure_chunked_long.json", "cache_resident_mib",
                      length=131072), 0.05, "672"),

    ("resident: swa-w1024-g4 @16K", 1792.0,
     lambda: measured(ATS / "measure_fixed.json", "cache_resident_mib",
                      variant="swa-w1024-g4", length=16384), 0.05, "1792.0"),
    ("resident: gqa-1KV @16K", 224.0,
     lambda: measured(ATS / "measure_fixed.json", "cache_resident_mib",
                      variant="gqa-1KV", length=16384), 0.05, "224.0"),

    ("concentration: rho(32)", 0.951, lambda: concentration(32), 1e-3, "0.951"),
    ("concentration: rho(64)", 0.980, lambda: concentration(64), 1e-3, "0.980"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    text = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    if not text:
        raise SystemExit(f"missing {DOC}")

    print(f"{'claim':<34}{'documented':>12}{'committed':>12}  {'in doc':>7}  verdict")
    bad = 0
    for label, doc_val, fn, tol, needle in CLAIMS:
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001
            got = None
            if args.verbose:
                print(f"  {label}: {type(e).__name__}: {e}")
        in_doc = needle in text
        ok = got is not None and abs(doc_val - got) <= tol and in_doc
        bad += not ok
        shown = f"{got:>12.5f}" if isinstance(got, (int, float)) else f"{'MISSING':>12}"
        print(f"{label:<34}{doc_val:>12.5f}{shown}  {str(in_doc):>7}  "
              f"{'OK' if ok else 'MISMATCH'}")

    n = len(CLAIMS)
    print(f"\n{n - bad}/{n} documented numbers reproduce from committed data")
    if bad:
        print("A mismatch means the prose and the data disagree. Fix whichever is")
        print("wrong -- and if the data is right, state the retraction rather than")
        print("silently overwriting the old figure.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
