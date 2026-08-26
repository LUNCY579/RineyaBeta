#!/usr/bin/env python3
"""Derive the cornerstone-2 NATURAL-LANGUAGE contrasts into a committed artifact.

WHY THIS EXISTS, and why it is not identity_counts.py. That script compares every
arm against a single baseline (`off__rag_dev`), which is the right shape for
"how much did compression change the replies". The cornerstone-2 question is a
different one: at a FIXED keep fraction, does the pooling window's DIRECTION
matter? That is a contrast between two compressed arms, paired case by case, and
neither arm is a baseline.

It also cannot reuse identity_counts.py in practice: `off__rag_dev/results.jsonl`
is not present in this working tree (per-case records are gitignored, 132 MB), so
any baseline-relative figure is currently underivable here while the arm-vs-arm
figures are fully derivable. Recording what CAN be derived, and not silently
substituting a different comparison, is the point.

Every figure this writes is paired on case id and computed from the per-case
records. Once written, the artifact is a few kB and IS committed, so the numbers
survive the records being cleaned.

Usage:
    python scripts/design/lang_contrasts.py           # write
    python scripts/design/lang_contrasts.py --check   # verify, non-zero on drift
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/kv_significance/results"
OUT = RES / "lang_contrasts.json"

# (label, arm_a, arm_b) -- b MINUS a, so a positive difference favours b.
# Each pair differs in exactly one manifest field: the pooling window.
CONTRASTS = [
    ("keep0.25_centred_vs_causal", "frac0.25-pool7-fixed__rag_dev", "frac0.25-b6f0__rag_dev"),
    ("keep0.75_centred_vs_causal", "frac0.75-pool7-fixed__rag_dev", "frac0.75-b6f0__rag_dev"),
    ("keep0.50_causal_vs_derived", "frac0.5-b6f0__rag_dev", "frac0.5-grow2__rag_dev"),
    ("keep0.25_causal_vs_derived", "frac0.25-b6f0__rag_dev", "frac0.25-grow2__rag_dev"),
    # The budget-ALLOCATION contrast. Window held causal in both arms; what
    # changes is recent (fixed 32 -> 25% of budget), which moves the score's
    # share of the budget from ~39% to ~68%. This is the single largest effect
    # measured on language.
    ("keep0.25_allocation_fixed_vs_fair", "frac0.25-b6f0__rag_dev",
     "frac0.25-rf0.25-b6f0__rag_dev"),
]


def _load(d: Path) -> dict:
    f = d / "results.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _sign_test(win: int, loss: int) -> float:
    """Exact two-sided sign test on discordant pairs only.

    Ties carry no information about direction and are excluded -- the same
    convention the passkey McNemar tests in this program use. n here is the
    DISCORDANT count, not the case count.
    """
    n = win + loss
    if n == 0:
        return 1.0
    k = min(win, loss)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def derive() -> dict:
    rows = {}
    for label, a, b in CONTRASTS:
        A, B = _load(RES / a), _load(RES / b)
        ids = sorted(set(A) & set(B))
        if not ids:
            rows[label] = {"arm_a": a, "arm_b": b, "status": "records absent"}
            continue
        da = [B[i]["quality_score"] - A[i]["quality_score"] for i in ids]
        win = sum(1 for d in da if d > 1e-9)
        loss = sum(1 for d in da if d < -1e-9)
        mean = sum(da) / len(da)
        se = st.stdev(da) / len(da) ** 0.5 if len(da) > 1 else 0.0
        rows[label] = {
            "arm_a": a, "arm_b": b, "n_paired": len(ids),
            "mean_quality_a": round(sum(A[i]["quality_score"] for i in ids) / len(ids), 5),
            "mean_quality_b": round(sum(B[i]["quality_score"] for i in ids) / len(ids), 5),
            "paired_mean_diff": round(mean, 5),
            "ci95": [round(mean - 1.96 * se, 5), round(mean + 1.96 * se, 5)],
            "b_better": win, "b_worse": loss, "tied": len(ids) - win - loss,
            "sign_test_p": _sign_test(win, loss),
            "identical_replies": sum(1 for i in ids if A[i]["reply"] == B[i]["reply"]),
            "mean_stored_kv": round(sum(B[i]["stored_kv"] for i in ids) / len(ids), 1),
            "mean_prompt_tokens": round(sum(B[i]["prompt_tokens"] for i in ids) / len(ids), 1),
        }
    return {"contrasts": rows,
            "note": ("Paired on case id. b MINUS a. Sign test is exact and two-sided "
                     "on DISCORDANT pairs only. quality_score is the repo's heuristic "
                     "rubric, which flags itself quality_pass_is_diagnostic_only -- it "
                     "is not a model judge.")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fresh = derive()
    if args.check:
        if not OUT.exists():
            raise SystemExit(f"{OUT} missing")
        old = json.loads(OUT.read_text(encoding="utf-8"))
        bad = 0
        for k, v in fresh["contrasts"].items():
            if v.get("status") == "records absent":
                print(f"  {k}: records absent, cannot re-derive (artifact retained)")
                continue
            if old["contrasts"].get(k) != v:
                print(f"  DRIFT in {k}")
                bad += 1
        print("lang_contrasts: OK" if not bad else f"lang_contrasts: {bad} drifted")
        raise SystemExit(1 if bad else 0)
    OUT.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    for k, v in fresh["contrasts"].items():
        if v.get("status"):
            print(f"  {k}: {v['status']}")
        else:
            print(f"  {k}: {v['mean_quality_a']:.5f} -> {v['mean_quality_b']:.5f}  "
                  f"diff {v['paired_mean_diff']:+.5f}  p={v['sign_test_p']:.2e}  "
                  f"identical {v['identical_replies']}/{v['n_paired']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
