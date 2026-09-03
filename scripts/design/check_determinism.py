#!/usr/bin/env python3
"""Is a language arm bitwise reproducible on this host? The generator for the
determinism table in docs/FINDINGS_TRIVIALITY_CONTROL.md section 6.

WHY IT EXISTS. The preregistration's reproduction gate (section 6.1) demands that a
re-run of arm A be byte-for-byte identical to the committed arm. At 16K that gate
failed at 99/100 and the analyser duly dropped the optional legacy floor arm -- and
the failure was originally written up as a code-state or environment difference.
It was not. Running the identical configuration four times shows that ONE case of a
hundred varies run to run, that three of the four runs agree, and that the outlier is
the run made while another project held ~3 GB of the card at 86-88 C. Byte-identity
is therefore the wrong instrument at 16K: it measures nondeterminism, not drift.

What matters for every published number is the weaker property, and it holds
absolutely: the ENDPOINT never flips. This script reports both, so the distinction
cannot be lost again.

Usage:
    python scripts/design/check_determinism.py
    python scripts/design/check_determinism.py --check   # non-zero if an endpoint flips
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/kv_significance/results"
OUT = RES / "determinism.json"
HARD = 0.999

# Every group is the SAME configuration: keep 0.25, sinks 4, recent-frac 0.15,
# pool 7, window (6,0), seed 20260826, base core, greedy decoding.
GROUPS = {
    "rag_long4k": ["long4k-sig-k0.25-rf0.15", "c4-sig-b6f0", "_determinism-sig4k"],
    "rag_long16k": ["long16k-sig-k0.25-rf0.15", "c4-sig-b6f0",
                    "_determinism-sig16k", "_determinism-sig16k-b"],
}


def load(tag: str, dataset: str):
    p = RES / f"{tag}__{dataset}" / "results.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["id"]: json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def hard(r: dict) -> int:
    return 1 if float(r.get("contract_fulfillment_score") or 0.0) >= HARD else 0


def derive() -> dict:
    out = {}
    for dataset, tags in GROUPS.items():
        runs = {t: load(t, dataset) for t in tags}
        have = [t for t, v in runs.items() if v]
        pairs, varying, flips_total = [], set(), 0
        for a, b in itertools.combinations(have, 2):
            A, B = runs[a], runs[b]
            ids = sorted(set(A) & set(B))
            diff = [i for i in ids if A[i]["reply"] != B[i]["reply"]]
            flip = [i for i in ids if hard(A[i]) != hard(B[i])]
            varying.update(diff)
            flips_total += len(flip)
            pairs.append({"a": a, "b": b, "n": len(ids),
                          "identical_replies": len(ids) - len(diff),
                          "endpoint_flips": len(flip)})
        rates = {t: round(sum(hard(r) for r in runs[t].values()) / len(runs[t]), 4)
                 for t in have}
        out[dataset] = {
            "runs_present": have, "n_runs": len(have), "pairs": pairs,
            "cases_that_ever_varied": sorted(varying),
            "n_cases_that_ever_varied": len(varying),
            "total_endpoint_flips": flips_total,
            "hard_endpoint_per_run": rates,
            "endpoint_identical_across_runs": len(set(rates.values())) == 1,
            "bitwise_deterministic": len(varying) == 0,
        }
    return {"note": ("Every run in a group is the same configuration. "
                     "'bitwise_deterministic' is the property prereg 6.1 assumes; "
                     "'endpoint_identical_across_runs' is the property every "
                     "published number actually depends on."),
            "endpoint": f"contract_fulfillment_score >= {HARD}",
            "datasets": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    d = derive()
    bad = False
    for ds, blk in d["datasets"].items():
        print(f"\n=== {ds}: {blk['n_runs']} runs of the identical config ===")
        for p in blk["pairs"]:
            print(f"  {p['a']:28s} vs {p['b']:26s} "
                  f"replies {p['identical_replies']}/{p['n']}   "
                  f"endpoint flips {p['endpoint_flips']}")
        print(f"  cases that ever varied : {blk['n_cases_that_ever_varied']}")
        for c in blk["cases_that_ever_varied"]:
            print(f"      {c}")
        print(f"  bitwise deterministic  : {blk['bitwise_deterministic']}")
        print(f"  hard endpoint per run  : {blk['hard_endpoint_per_run']}")
        print(f"  endpoint identical     : {blk['endpoint_identical_across_runs']}")
        if not blk["endpoint_identical_across_runs"] or blk["total_endpoint_flips"]:
            bad = True
    if args.check:
        if bad:
            raise SystemExit("FAIL: an endpoint flipped between runs of the same "
                             "configuration -- published numbers are not stable")
        print("\ncheck: no endpoint flips in any group; every published rate is stable")
        return
    OUT.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
