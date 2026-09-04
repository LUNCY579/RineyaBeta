#!/usr/bin/env python3
"""Find the capacity floor. Implements only what docs/PREREG_CAPACITY_FLOOR.md fixed.

The prediction under test is a FORMULA, not a number:

    4 sinks + 0.15*B recent + |instructions| + |one record|  <=  B

which on rag_long4k (instructions 104 tokens mean, median record 63) gives
B* >= 196 tokens, a 21.6x reduction on a 4226-token prompt.

The floor is reported as the largest swept budget whose POOLED quality is not
significantly below the uncompressed reference, with every smaller swept budget
significantly below it. If no such point exists the sweep is reported as having no
knee and F1 is refuted -- that outcome is available and is not treated as a failure
to measure.

Usage:
    python scripts/design/analyse_capacity_floor.py
    python scripts/design/analyse_capacity_floor.py --check
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/kv_significance/results"
OUT = RES / "capacity_floor.json"
DATASET = ROOT / "artifacts/eval_sets/rag_long4k_d0.5.jsonl"
LIMIT, HARD = 600, 0.999
REFERENCE = "c7-off-n600"
PREDICTED_FLOOR = 196
BUDGETS = [512, 256, 196, 160, 128, 96]
ARM = "c9-sys-B{}"
# prereg 3: F1 is confirmed if the knee lands within one sweep step of 196.
CONFIRM_BAND = {160, 196, 256}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(tag: str):
    p = RES / f"{tag}__rag_long4k" / "results.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["id"]: json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def manifest(tag: str):
    p = RES / f"{tag}__rag_long4k" / "run_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def hard(r) -> int:
    return 1 if float(r.get("contract_fulfillment_score") or 0.0) >= HARD else 0


def derive() -> dict:
    order, meta = [], {}
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            meta[r["id"]] = r
            order.append(r["id"])
            if len(order) >= LIMIT:
                break
    off = load(REFERENCE)
    if off is None:
        return {"error": f"reference {REFERENCE} absent -- nothing to compare against"}

    rows, void = {}, {}
    for B in BUDGETS:
        tag = ARM.format(B)
        X, man = load(tag), manifest(tag)
        if X is None or man is None:
            rows[B] = {"status": "NOT RUN"}
            continue
        # prereg 3: void conditions, checked before anything is read off the arm.
        problems = []
        if man.get("heavy_zero_count"):
            problems.append(f"heavy_zero_count={man['heavy_zero_count']} "
                            f"(policy silently became StreamingLLM)")
        want = B / 4226.0
        mk = man.get("mean_kept_ratio")
        if mk is None or abs(mk - want) > 0.10 * want:
            problems.append(f"mean_kept_ratio {mk} is not within 10% of B/4226={want:.4f}")
        if man.get("select") != "lexical-doc-bm25-sys":
            problems.append(f"select={man.get('select')!r}")
        if problems:
            void[B] = problems
            rows[B] = {"status": "VOID", "problems": problems}
            continue
        cell = {"status": "ok", "budget": B, "mean_kept_ratio": mk,
                "reduction_x": round(4226.0 / B, 1), "elapsed_s": man.get("elapsed_s")}
        for stratum in ("answer", "abstain", "pooled"):
            ids = [i for i in order if i in X and i in off
                   and (stratum == "pooled" or meta[i].get("decision") == stratum)]
            ro = sum(hard(off[i]) for i in ids) / len(ids)
            rx = sum(hard(X[i]) for i in ids) / len(ids)
            b = sum(1 for i in ids if hard(X[i]) and not hard(off[i]))
            c = sum(1 for i in ids if hard(off[i]) and not hard(X[i]))
            cell[stratum] = {"n": len(ids), "off": round(ro, 4), "arm": round(rx, 4),
                             "diff": round(rx - ro, 4), "win": b, "lose": c,
                             "p": mcnemar_exact(b, c),
                             "below_uncompressed": bool(rx < ro and mcnemar_exact(b, c) < 0.05)}
        rows[B] = cell

    ok = [B for B in BUDGETS if rows[B].get("status") == "ok"]
    floor, reason = None, None
    for B in sorted(ok, reverse=True):
        smaller = [b2 for b2 in ok if b2 < B]
        if rows[B]["pooled"]["below_uncompressed"]:
            continue
        if smaller and all(rows[b2]["pooled"]["below_uncompressed"] for b2 in smaller):
            floor = B
            break
    if floor is None:
        reason = ("no knee: there is no swept budget that is itself not-below "
                  "uncompressed with every smaller swept budget below it")
    verdict = ("NOT RUN" if len(ok) < 2 else
               "F1 CONFIRMED" if floor in CONFIRM_BAND else
               "F1 REFUTED" if floor is not None else "F1 REFUTED (no knee)")

    f2 = None
    if floor is not None:
        below = [b2 for b2 in ok if b2 < floor]
        if below:
            nb = max(below)
            da = rows[nb]["answer"]["diff"]
            db = rows[nb]["abstain"]["diff"]
            f2 = {"first_budget_below_floor": nb, "answer_diff": da,
                  "abstain_diff": db,
                  "F2_confirmed": bool(db < da),
                  "reading": ("abstention falls faster, as predicted"
                              if db < da else
                              "answering falls at least as fast -- F2 refuted")}
    return {"preregistration": "docs/PREREG_CAPACITY_FLOOR.md",
            "predicted_floor_tokens": PREDICTED_FLOOR,
            "confirmation_band": sorted(CONFIRM_BAND),
            "reference": REFERENCE, "budgets": rows, "void": void,
            "observed_floor_tokens": floor, "no_knee_reason": reason,
            "F1": verdict, "F2": f2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = derive()
    if "error" in d:
        raise SystemExit(d["error"])
    print(f"predicted floor {d['predicted_floor_tokens']} tokens; "
          f"reference {d['reference']}\n")
    print(f"{'B':>5} {'x':>6} {'kept':>7} | "
          f"{'answer':>21} | {'abstain':>21} | {'pooled':>21}")
    for B in BUDGETS:
        r = d["budgets"][B]
        if r.get("status") != "ok":
            print(f"{B:5d} {r.get('status')}"
                  + (f"  {r.get('problems')}" if r.get("problems") else ""))
            continue
        cells = []
        for s in ("answer", "abstain", "pooled"):
            c = r[s]
            cells.append(f"{c['arm']:.4f} {c['diff']:+.4f} {c['win']:3d}/{c['lose']:<3d}"
                         f"{'*' if c['below_uncompressed'] else ' '}")
        print(f"{B:5d} {r['reduction_x']:6.1f} {r['mean_kept_ratio']:7.4f} | "
              + " | ".join(cells))
    print("\n  (* = significantly BELOW uncompressed, paired exact McNemar p<0.05)")
    print(f"\nobserved floor : {d['observed_floor_tokens']} tokens"
          + (f"   [{d['no_knee_reason']}]" if d["no_knee_reason"] else ""))
    print(f"F1             : {d['F1']}")
    if d["F2"]:
        print(f"F2             : {d['F2']['reading']} "
              f"(answer {d['F2']['answer_diff']:+.4f} vs "
              f"abstain {d['F2']['abstain_diff']:+.4f} at B={d['F2']['first_budget_below_floor']})")
    if a.check:
        if not OUT.exists():
            raise SystemExit(f"no committed artifact at {OUT}")
        if json.loads(OUT.read_text(encoding="utf-8")) != d:
            raise SystemExit("DRIFT: recomputed sweep differs from the committed one")
        print("\ncheck: committed capacity-floor analysis reproduces")
        return
    OUT.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
