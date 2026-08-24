#!/usr/bin/env python3
"""The claim this mechanism actually makes: maintain accuracy while using less memory.

Both halves have to be measured at the SAME length or the comparison is empty. That is
harder than it sounds here, because an uncompressed run does not fit above ~32K on an
8 GB card -- 64K full cache OOM'd at 6,425.6 MiB, and the highest peak ever survived in
this programme is 4,972.9 MiB. So the reference arms exist only at 16K and 32K, and
above that the honest statement is not "compression is cheaper" but "there is no
uncompressed alternative to be cheaper than".

Reads whatever cells are present and says plainly which comparisons it cannot make yet,
rather than filling gaps with the nearest available number.

Usage:
    python scripts/design/accuracy_vs_memory.py
    python scripts/design/accuracy_vs_memory.py --prefix halo
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/kv_significance/results"


def cells(pattern: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(RESULTS / pattern))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        rows = d.get("results") or []
        if not rows:
            continue
        hits = sum(sum(1 for t in (c.get("trials") or []) if t.get("hit")) for c in rows)
        n = sum(len(c.get("trials") or []) for c in rows)
        if not n:
            continue
        r0 = rows[0]
        out.append({
            "file": os.path.basename(p),
            "model": d.get("model", "?"),
            "length": r0.get("length"),
            "real": r0.get("real_tokens") or r0.get("length"),
            "budget": r0.get("budget"),
            "peak": max(c.get("peak_mib") or 0 for c in rows),
            "acc": hits / n,
            "hits": hits, "n": n,
            "ooms": sum(c.get("ooms", 0) for c in rows),
            "by_depth": {c["depth"]: (sum(1 for t in c["trials"] if t["hit"]) / len(c["trials"]))
                         for c in rows if c.get("trials")},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="halo",
                    help="result-file prefix to report on (default: halo)")
    args = ap.parse_args()

    rows = cells(f"{args.prefix}_*.json")
    if not rows:
        print(f"no cells matching {args.prefix}_*.json yet")
        return

    model = rows[0]["model"]
    print(f"ACCURACY vs MEMORY  --  {model}\n")

    by_len: dict[int, list[dict]] = {}
    for r in rows:
        by_len.setdefault(r["length"], []).append(r)

    for length in sorted(by_len):
        group = sorted(by_len[length], key=lambda r: (r["budget"] or 10 ** 9))
        ref = next((r for r in group if not r["budget"]), None)
        real = group[0]["real"]
        print(f"--- {length:,} nominal ({real:,} real tokens) ---")
        print(f"{'budget':>8} {'compression':>12} {'accuracy':>9} {'n':>8} "
              f"{'peak MiB':>9} {'vs full':>9}  depths")
        for r in group:
            b = r["budget"]
            comp = f"{real / b:,.0f}x" if b else "1x (full)"
            vs = ""
            if ref and b and ref["peak"]:
                vs = f"{r['peak'] / ref['peak']:.2f}x"
            d = " ".join(f"{r['by_depth'][k]:.2f}" for k in sorted(r["by_depth"]))
            oom = f"  OOM x{r['ooms']}" if r["ooms"] else ""
            print(f"{(b or 0):>8} {comp:>12} {r['acc']:>9.3f} {r['hits']:>3}/{r['n']:<4}"
                  f" {r['peak']:>9.0f} {vs:>9}  {d}{oom}")
        if ref is None:
            print("    no full-cache reference at this length. Above ~32K an "
                  "uncompressed run does not fit\n    on this card, so the comparison "
                  "is not 'cheaper than full' but 'runs at all'.")
        else:
            best = max((r for r in group if r["budget"]), key=lambda r: r["acc"],
                       default=None)
            if best:
                da = best["acc"] - ref["acc"]
                dm = 1 - best["peak"] / ref["peak"] if ref["peak"] else 0
                print(f"    best compressed arm: accuracy {da:+.3f} against full cache, "
                      f"memory {dm:.0%} lower")
        print()

    print("READING THIS. Accuracy at or above the full-cache row with a lower peak is")
    print("the claim holding. Accuracy below it is the price, and the price is only")
    print("meaningful where a full-cache row exists to compare against.")


if __name__ == "__main__":
    main()
