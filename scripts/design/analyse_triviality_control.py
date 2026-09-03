#!/usr/bin/env python3
"""Analyse the triviality control. Implements ONLY what
docs/PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md (v2) fixed in advance.

THE ONE RULE THIS FILE EXISTS TO ENFORCE: **no verdict from missing data.** In this
repository an analyser once printed "REFUTED -- the closure is complete" from three
empty files. Every gate in prereg 6 and 8 is checked before any verdict is formed, and
a failed gate produces a refusal, not a null.

No scipy here, so the exact binomial machinery is written out: exact McNemar by
enumeration, Clopper-Pearson by bisection on the exact binomial tail. Both are
validated against R's binom.test and against this programme's own published p-values
in scripts/design/test_triviality_stats.py. Nothing is a normal approximation.

Usage:
    python scripts/design/analyse_triviality_control.py            # write + print
    python scripts/design/analyse_triviality_control.py --check    # verify, non-zero on drift
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/kv_significance/results"
OUT = RES / "triviality_control.json"

HARD = 0.999                 # contract_fulfillment_score >= this  == the endpoint
BOOT_N, BOOT_SEED = 10_000, 20260902
MIN_RANGE = 0.40             # prereg 6.5 dynamic-range gate
KEEP_FRAC, KEEP_TOL = 0.25, 0.01

# arm id -> (tag, label, required). Must agree with run_triviality_campaign.ARMS.
ARMS = {
    "A": ("c4-sig-b6f0", "significance, causal (6,0)", True),
    "B": ("c4-sig-centred", "significance, centred (3,3)", True),
    "C": ("c4-sig-nopool", "significance, no pooling", True),
    "D": ("c4-lex-idf-b6f0", "trivial: per-position IDF", True),
    "E": ("c4-lex-naive-b6f0", "trivial: naive question-token match", True),
    "G": ("c4-lex-docbm25-b6f0", "trivial: document-level BM25", True),
    "N": ("c4-random-prepool", "floor: contiguity-matched random", True),
    "F": (None, "floor: random, post-pool (committed, legacy)", False),
}
# Arm F is reused from committed data and its tag differs per length (prereg 2.4).
F_TAG = {"rag_long4k": "long4k-random-k0.25", "rag_long16k": "long16k-random-k0.25"}
TRIVIAL_ARMS = ("D", "E", "G")
POSITIONAL_TRIVIAL = ("D", "E")

DATASETS = {
    "rag_long4k": (200, "artifacts/eval_sets/rag_long4k_d0.5.jsonl"),
    "rag_long16k": (100, "artifacts/eval_sets/rag_long16k_d0.5.jsonl"),
}

CONTRASTS = [("A", "L"), ("A", "G"), ("A", "D"), ("A", "E"), ("A", "B"), ("A", "C"),
             ("L", "N"), ("D", "N"), ("F", "N")]

# prereg 6.2: fields allowed to differ between an arm and arm A.
# mean_kept_ratio is NOT here -- it is asserted, because an arm that never evicted
# would otherwise pass this gate and be scored as a policy.
FREE_FIELDS = {"tag", "elapsed_s", "recent", "recent_effective_mean",
               "heavy_zero_count", "case_count", "select"}
POOL_FIELDS = {"pool", "pool_back", "pool_fwd", "pool_back_flag", "pool_fwd_flag",
               "pool_window_uniform", "directional"}


# --------------------------------------------------------------------------- #
# exact statistics
# --------------------------------------------------------------------------- #

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(f, a: float = 0.0, b: float = 1.0, it: int = 80) -> float:
    fa = f(a)
    for _ in range(it):
        m = (a + b) / 2
        if (f(m) >= 0) == (fa >= 0):
            a = m
        else:
            b = m
    return (a + b) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact CI for a binomial proportion, by bisection on the exact tail."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else _bisect(lambda p: 1 - _binom_cdf(k - 1, n, p) - alpha / 2)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) - alpha / 2)
    return lo, hi


def paired_diff_ci(b: int, c: int, n: int) -> tuple[float, float]:
    """Exact 95% CI on the paired difference (prereg 3.2).

    Conditional on the discordant count: b ~ Binom(n_disc, p), and the difference is
    (2p - 1) * n_disc / n. With no discordant pairs the conditional argument is empty
    and the rule of three applies.
    """
    nd = b + c
    if nd == 0:
        return -3.0 / n, 3.0 / n
    pl, pu = clopper_pearson(b, nd)
    return (2 * pl - 1) * nd / n, (2 * pu - 1) * nd / n


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def is_strict(r: dict) -> bool:
    rc = r.get("response_contract") or {}
    return bool(rc.get("requirements")) and not rc.get("open_ended", False)


def hard(r: dict) -> int:
    return 1 if float(r.get("contract_fulfillment_score") or 0.0) >= HARD else 0


def decisions(dataset: str, limit: int) -> dict[str, str]:
    """case id -> decision, from the eval set. The results files do not carry it."""
    path = ROOT / DATASETS[dataset][1]
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines()[: limit * 4]:
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = r.get("decision")
        if len(out) >= limit:
            break
    return out


def load_arm(tag: str, dataset: str):
    d = RES / f"{tag}__{dataset}"
    res, man = d / "results.jsonl", d / "run_manifest.json"
    if not (res.exists() and man.exists()):
        return None
    rows = {}
    for line in res.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return {"dir": d.name, "manifest": json.loads(man.read_text(encoding="utf-8")),
            "rows": rows}


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_report(arms: dict, expect: int) -> dict:
    problems, notes = [], []
    present = [k for k, v in arms.items() if v]
    for k, (_t, _l, required) in ARMS.items():
        if required and k not in present:
            problems.append(f"arm {k} absent -- reported NOT RUN, never a null (prereg 8)")
    if "F" not in present:
        notes.append("arm F absent: F-relative rows are not run (it is optional)")
    if "A" not in present:
        return {"problems": problems + ["arm A absent: no reference to diff against"],
                "notes": notes, "arms_present": present}

    base = arms["A"]["manifest"]
    for k in present:
        m = arms[k]["manifest"]
        if m.get("case_count") != expect:
            problems.append(f"arm {k}: case_count {m.get('case_count')} != {expect}")
        if len(arms[k]["rows"]) != expect:
            problems.append(f"arm {k}: {len(arms[k]['rows'])} rows != {expect}")
        common = set(arms[k]["rows"]) & set(arms["A"]["rows"])
        if len(common) != expect:
            problems.append(f"arm {k}: paired with A on {len(common)}, expected {expect}")
        if m.get("heavy_zero_count"):
            problems.append(f"arm {k}: heavy_zero_count={m['heavy_zero_count']} -- the "
                            f"policy silently became StreamingLLM (prereg 6.3)")
        # prereg 6.2: mean_kept_ratio is ASSERTED, not whitelisted.
        mk = m.get("mean_kept_ratio")
        if mk is None or abs(float(mk) - KEEP_FRAC) > KEEP_TOL:
            problems.append(f"arm {k}: mean_kept_ratio {mk} is not {KEEP_FRAC} "
                            f"+/-{KEEP_TOL} -- this arm did not compress as configured")
        if k == "F":
            continue                      # committed arm, different tag family
        allowed = set(FREE_FIELDS) | ({"pool"} if k == "C" else set())
        if k in ("B", "C"):
            allowed |= POOL_FIELDS
        diff = {f for f in set(base) | set(m) if base.get(f) != m.get(f)}
        unexpected = diff - allowed
        if unexpected:
            problems.append(f"arm {k}: unexpected manifest difference in "
                            f"{sorted(unexpected)} (prereg 6.2)")
        if k != "A":
            same = sum(1 for i in common
                       if arms[k]["rows"][i]["reply"] == arms["A"]["rows"][i]["reply"])
            if common and same == len(common):
                problems.append(f"arm {k}: every reply identical to A -- a plumbing "
                                f"failure, not a null (prereg 6.4)")
            elif common:
                notes.append(f"arm {k}: {same}/{len(common)} replies identical to A")
    return {"problems": problems, "notes": notes, "arms_present": present}


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def contrast(vals: dict, a: str, b: str, ids: list) -> dict:
    if a not in vals or b not in vals:
        return {"status": "arm absent -- not run, not a null"}
    A, B = vals[a], vals[b]
    wa = sum(1 for i in ids if A[i] and not B[i])
    wb = sum(1 for i in ids if B[i] and not A[i])
    n = len(ids)
    lo, hi = paired_diff_ci(wa, wb, n) if n else (0.0, 0.0)
    return {"arm_a": a, "arm_b": b, "n_paired": n,
            "rate_a": round(sum(A[i] for i in ids) / max(n, 1), 4),
            "rate_b": round(sum(B[i] for i in ids) / max(n, 1), 4),
            "a_better": wa, "b_better": wb, "tied": n - wa - wb,
            "diff_a_minus_b": round((wa - wb) / max(n, 1), 4),
            "ci95_diff": [round(lo, 4), round(hi, 4)],
            "mcnemar_exact_p": mcnemar_exact(wa, wb)}


def recovery(vals: dict, ids: list, pool: tuple) -> dict | None:
    """R = (L - N) / (A - N) with L the best of `pool`. prereg 7."""
    if "A" not in vals or "N" not in vals or not any(k in vals for k in pool):
        return None
    if not all(k in vals for k in pool):
        return {"status": f"undefined: L = max{pool} needs all of them present; "
                          f"a missing arm can only lower the maximum (prereg 8)"}
    mean = {k: sum(vals[k][i] for i in ids) / len(ids) for k in ("A", "N") + pool}
    best = max(pool, key=lambda k: mean[k])
    denom = mean["A"] - mean["N"]
    rng = random.Random(BOOT_SEED)
    boots, undef = [], 0
    for _ in range(BOOT_N):
        s = [rng.randrange(len(ids)) for _ in ids]
        ma = sum(vals["A"][ids[j]] for j in s) / len(s)
        mn = sum(vals["N"][ids[j]] for j in s) / len(s)
        ml = sum(vals[best][ids[j]] for j in s) / len(s)
        if abs(ma - mn) < 1e-12:
            undef += 1
            continue
        boots.append((ml - mn) / (ma - mn))
    boots.sort()
    frac = undef / BOOT_N
    return {"best_arm": best, "rate_A": round(mean["A"], 4),
            "rate_N": round(mean["N"], 4), "rate_L": round(mean[best], 4),
            "range_A_minus_N": round(denom, 4),
            "R_point": round((mean[best] - mean["N"]) / denom, 4) if abs(denom) > 1e-12
                       else None,
            "R_ci95": ([round(boots[int(0.025 * len(boots))], 4),
                        round(boots[int(0.975 * len(boots))], 4)] if boots else None),
            "bootstrap_undefined_fraction": round(frac, 4),
            # prereg 6.5 + 7: a ratio needs its denominator safely positive.
            "R_reportable": bool(boots) and frac <= 0.05 and denom >= MIN_RANGE}


def analyse_dataset(dataset: str, expect: int) -> dict:
    arms = {}
    for k, (tag, _label, _req) in ARMS.items():
        arms[k] = load_arm(tag or F_TAG[dataset], dataset)
    gates = gate_report(arms, expect)
    dec = decisions(dataset, expect)

    out = {"dataset": dataset, "n_expected": expect, "gates": gates,
           "strata": {}, "arms": {}}
    for k, v in arms.items():
        if v:
            out["arms"][k] = {
                "tag": v["dir"], "label": ARMS[k][1], "n": len(v["rows"]),
                "mean_kept_ratio": v["manifest"].get("mean_kept_ratio"),
                "elapsed_s": v["manifest"].get("elapsed_s"),
                "composite_diagnostic_only": round(
                    sum(float(r.get("quality_score") or 0.0)
                        for r in v["rows"].values()) / max(len(v["rows"]), 1), 4)}

    for stratum in ("answer", "abstain"):
        vals, ids = {}, None
        for k, v in arms.items():
            if not v:
                continue
            sel = {i: hard(r) for i, r in v["rows"].items()
                   if is_strict(r) and dec.get(i) == stratum}
            vals[k] = sel
            ids = set(sel) if ids is None else ids & set(sel)
        ids = sorted(ids or [])
        # L is a derived arm, not a run one.
        rec_s = recovery(vals, ids, TRIVIAL_ARMS) if ids else None
        rec_p = recovery(vals, ids, POSITIONAL_TRIVIAL) if ids else None
        if rec_s and "best_arm" in rec_s:
            vals["L"] = vals[rec_s["best_arm"]]
        block = {
            "n": len(ids),
            "rates": {k: round(sum(v[i] for i in ids) / max(len(ids), 1), 4)
                      for k, v in vals.items() if k != "L"},
            "contrasts": {f"{a}_vs_{b}": contrast(vals, a, b, ids)
                          for a, b in CONTRASTS},
            "R_structural": rec_s, "R_positional": rec_p,
        }
        out["strata"][stratum] = block
    return out


def verdict_for(d: dict) -> dict:
    """prereg 7, on the ANSWER stratum, and only from a cell that cleared every gate."""
    if d["gates"]["problems"]:
        return {"verdict": "REFUSED",
                "reason": "coverage/validity gates failed; a verdict here would be the "
                          "default output of missing data (prereg 8)",
                "problems": d["gates"]["problems"]}
    s = d["strata"]["answer"]
    rec = s.get("R_structural")
    if not rec or "R_point" not in rec:
        return {"verdict": "REFUSED",
                "reason": (rec or {}).get("status", "recovery fraction not computable")}
    if not rec["R_reportable"]:
        return {"verdict": "REFUSED",
                "reason": f"instrument failure (prereg 6.5): A - N = "
                          f"{rec['range_A_minus_N']} < {MIN_RANGE}, or "
                          f"{rec['bootstrap_undefined_fraction']:.1%} of bootstrap "
                          f"resamples had a vanishing denominator"}
    con = s["contrasts"]["A_vs_L"]
    R, p = rec["R_point"], con["mcnemar_exact_p"]
    lo, hi = con["ci95_diff"]
    d_al = con["diff_a_minus_b"]
    if d_al < 0 and p < 0.05:
        v, why = "BEATEN", "a selector with no forward pass outperforms the mechanism"
    elif R >= 0.90 and p >= 0.05 and hi <= 0.10 and lo >= -0.10:
        v, why = "TRIVIAL", "the score is this programme's L_state"
    elif R >= 0.90 and d_al > 0 and p < 0.05:
        v, why = "PARTIAL (upper)", "residual is real but bounded at <=10% of the advantage"
    elif 0.50 <= R < 0.90:
        v, why = "PARTIAL", "a trivial selector recovers most of the advantage"
    elif R < 0.50 and d_al > 0 and p < 0.05:
        v, why = "SUBSTANTIVE", "no trivial selector reproduces the score, on this benchmark"
    else:
        v, why = "INCONCLUSIVE", "no preregistered row matches; close nothing"
    return {"verdict": v, "reading": why, "R_structural": R,
            "R_positional": (s.get("R_positional") or {}).get("R_point"),
            "best_trivial_arm": rec["best_arm"], "mcnemar_p_A_vs_L": p,
            "ci95_diff_A_minus_L": [lo, hi]}


def derive() -> dict:
    per = {ds: analyse_dataset(ds, n) for ds, (n, _p) in DATASETS.items()}
    primary = per.get("rag_long4k")
    rep = per.get("rag_long16k")
    return {"preregistration": "docs/PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md (v2)",
            "endpoint": f"contract_fulfillment_score >= {HARD}, strict cases, "
                        f"ANSWER stratum, paired exact McNemar",
            "datasets": per,
            "primary_verdict": (verdict_for(primary) if primary else
                                {"verdict": "REFUSED", "reason": "4K cell absent"}),
            "replication_16k": ({"note": "direction only -- n=72 cannot certify "
                                         "equivalence, and it is the SAME 100 items "
                                         "as the first 100 of the 4K cell (prereg 9.2)",
                                 **verdict_for(rep)} if rep else None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data = derive()

    for ds, d in data["datasets"].items():
        print(f"\n===== {ds} (expected n={d['n_expected']}) =====")
        for pb in d["gates"]["problems"]:
            print(f"  GATE FAIL: {pb}")
        for nt in d["gates"]["notes"]:
            print(f"  note: {nt}")
        for stratum, blk in d["strata"].items():
            if not blk["n"]:
                continue
            print(f"  --- {stratum} stratum, n={blk['n']} ---")
            print("      rates: " + "  ".join(f"{k}={v:.4f}"
                                              for k, v in sorted(blk["rates"].items())))
            for name, c in blk["contrasts"].items():
                if "status" in c:
                    continue
                print(f"      {name:8s} {c['rate_a']:.4f} vs {c['rate_b']:.4f}  "
                      f"diff {c['diff_a_minus_b']:+.4f} "
                      f"CI[{c['ci95_diff'][0]:+.4f},{c['ci95_diff'][1]:+.4f}]  "
                      f"{c['a_better']}/{c['b_better']}  p={c['mcnemar_exact_p']:.3g}")
            for tag in ("R_structural", "R_positional"):
                r = blk.get(tag)
                if r and "R_point" in r:
                    print(f"      {tag}: {r['R_point']} CI {r['R_ci95']} "
                          f"(best {r['best_arm']}, reportable={r['R_reportable']})")

    print(f"\nPRIMARY (4K, answer stratum): "
          f"{json.dumps(data['primary_verdict'], indent=2)}")
    if data["replication_16k"]:
        print(f"16K: {json.dumps(data['replication_16k'], indent=2)}")

    if args.check:
        if not OUT.exists():
            raise SystemExit(f"no committed artifact at {OUT}")
        if json.loads(OUT.read_text(encoding="utf-8")) != data:
            raise SystemExit("DRIFT: recomputed analysis differs from the committed one")
        print("\ncheck: committed analysis reproduces")
        return
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
