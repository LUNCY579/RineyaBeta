#!/usr/bin/env python3
"""Analyse the triviality control. Implements ONLY what
docs/PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md (v3) fixed in advance.

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

# arm id -> (tag, label, required, EXPECTED select, EXPECTED pool).
# Must agree with run_triviality_campaign.ARMS.
# `select` and `pool` are ASSERTED, not whitelisted. They used to be free-running,
# which meant a directory holding one arm's run under another arm's name passed all
# eight gates and flipped the verdict -- the same defect shape as `--select
# streaming` once performing significance selection while the manifest said
# streaming. An arm's identity is now checked against what it claims to be.
ARMS = {
    "A": ("c4-sig-b6f0", "significance, causal (6,0)", True, "significance", 7),
    "B": ("c4-sig-centred", "significance, centred (3,3)", True, "significance", 7),
    "C": ("c4-sig-nopool", "significance, no pooling", True, "significance", 1),
    "D": ("c4-lex-idf-b6f0", "trivial: per-position IDF", True, "lexical-idf", 7),
    "E": ("c4-lex-naive-b6f0", "trivial: naive question-token match", True,
          "lexical-naive", 7),
    "G": ("c4-lex-docbm25-b6f0", "trivial: document-level BM25", True,
          "lexical-doc-bm25", 7),
    "N": ("c4-random-prepool", "floor: contiguity-matched random (7-token)", True,
          "random-prepool", 7),
    "Ndoc": ("c4-random-doc", "floor: granularity-matched random (record)", True,
             "random-doc", 7),
    "F": (None, "floor: random, post-pool (committed, legacy)", False, "random", 7),
}
# Arm F is reused from committed data and its tag differs per length (prereg 2.4).
F_TAG = {"rag_long4k": "long4k-random-k0.25", "rag_long16k": "long16k-random-k0.25"}
# Each trivial family is scored against the floor with ITS OWN contiguity, so a
# selector is credited only for what it achieves above noise of the same shape.
#   positional  D, E  keep pooled 7-token runs   -> floor N   (random-prepool)
#   structural  G     keeps whole ~64-token records -> floor Ndoc (random-doc)
FAMILIES = {"positional": (("D", "E"), "N"), "structural": (("G",), "Ndoc")}

DATASETS = {
    "rag_long4k": (200, "artifacts/eval_sets/rag_long4k_d0.5.jsonl"),
    "rag_long16k": (100, "artifacts/eval_sets/rag_long16k_d0.5.jsonl"),
}

CONTRASTS = [("A", "L"), ("A", "G"), ("A", "D"), ("A", "E"), ("A", "B"), ("A", "C"),
             ("L", "N"), ("G", "Ndoc"), ("D", "N"), ("F", "N"), ("Ndoc", "N")]

# prereg 6.2: fields allowed to differ between an arm and arm A.
# mean_kept_ratio is NOT here -- it is asserted, because an arm that never evicted
# would otherwise pass this gate and be scored as a policy.
# cap_mib is NOT free: every arm in a cell must share it, or arms run in different
# GPU-contention conditions are silently pooled. It is asserted equal to arm A's.
FREE_FIELDS = {"tag", "elapsed_s", "recent", "recent_effective_mean",
               "heavy_zero_count", "case_count", "select"}
# `select` stays in FREE_FIELDS only so the arm-vs-A DIFF does not double-report it;
# its VALUE is asserted against ARMS[k][3] in gate_report.
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

# prereg 6.1: the committed arm each cell's arm A must reproduce byte-for-byte
# before any committed data may be reused as arm F.
REPRO_REFERENCE = {"rag_long4k": "long4k-sig-k0.25-rf0.15",
                   "rag_long16k": "long16k-sig-k0.25-rf0.15"}


def reproduction_check(arms: dict, dataset: str) -> dict:
    """Does this campaign's arm A reproduce the committed reference exactly?

    prereg 6.1 makes arm F's reuse conditional on this, and prereg 2.4 repeats the
    condition -- but until now nothing enforced it, so the licence to reuse committed
    data rested on the operator remembering to check. That is the failure class this
    repository has already paid for once.
    """
    ref_dir = RES / f"{REPRO_REFERENCE[dataset]}__{dataset}"
    f = ref_dir / "results.jsonl"
    if not arms.get("A"):
        return {"status": "arm A absent", "identical": None, "licenses_F": False}
    if not f.exists():
        return {"status": f"no committed reference at {ref_dir.name}",
                "identical": None, "licenses_F": False}
    ref = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ref[r["id"]] = r
    rows = arms["A"]["rows"]
    ids = sorted(set(rows) & set(ref))
    same = sum(1 for i in ids if rows[i]["reply"] == ref[i]["reply"])
    ok = bool(ids) and same == len(ids)
    return {"reference": ref_dir.name, "n_paired": len(ids), "identical": same,
            "licenses_F": ok,
            "status": ("PASS -- the campaign inherits the committed arms' provenance"
                       if ok else
                       "FAIL -- arms stay internally comparable, but no result may be "
                       "compared to a committed number from the earlier code state, "
                       "and arm F is dropped (prereg 6.1)")}


def gate_report(arms: dict, expect: int) -> dict:
    problems, notes = [], []
    present = [k for k, v in arms.items() if v]
    for k, (_t, _l, required, _sel, _pool) in ARMS.items():
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
        want_sel, want_pool = ARMS[k][3], ARMS[k][4]
        if m.get("select") != want_sel:
            problems.append(f"arm {k}: manifest select={m.get('select')!r} but this "
                            f"arm is defined as {want_sel!r} -- the directory holds a "
                            f"different arm than its name claims (prereg 6.2)")
        if m.get("pool") != want_pool:
            problems.append(f"arm {k}: manifest pool={m.get('pool')} but this arm is "
                            f"defined at pool={want_pool} (prereg 6.2)")
        if k == "F":
            continue                      # committed arm, different tag family
        # prereg 6.2: cap_mib is ASSERTED equal across the cell. Arm A was run at
        # 4500 while another project held part of the card; an arm at the runner's
        # 7000 default would be a different memory regime pooled in silently, and
        # the generic "unexpected manifest difference" message would send the
        # operator to relax a frozen gate after the data rather than to re-run.
        if m.get("cap_mib") != base.get("cap_mib"):
            problems.append(f"arm {k}: cap_mib {m.get('cap_mib')} != arm A's "
                            f"{base.get('cap_mib')}; re-run this arm at A's cap "
                            f"rather than widening the gate")
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


def recovery(vals: dict, ids: list, pool: tuple, floor: str) -> dict | None:
    """R = (L - floor) / (A - floor), L the best of `pool`. prereg 7.

    Returns the bootstrap CI as well, and the verdict logic USES it: a threshold
    counts as crossed only when the interval lies wholly on one side of it. R's
    half-width at the preregistered n is about +/-0.09, which straddles the 0.50 and
    0.90 boundaries at the very values section 4 predicts -- so branching on the
    point estimate alone would resolve a threshold the data cannot resolve.
    """
    if "A" not in vals or floor not in vals:
        return {"status": f"undefined: needs arm A and floor {floor}"}
    have = [k for k in pool if k in vals]
    if not have:
        return {"status": f"undefined: no arm of {pool} present"}
    mean = {k: sum(vals[k][i] for i in ids) / len(ids) for k in ("A", floor) + tuple(have)}
    best = max(have, key=lambda k: mean[k])
    denom = mean["A"] - mean[floor]
    rng = random.Random(BOOT_SEED)
    boots, undef = [], 0
    for _ in range(BOOT_N):
        smp = [rng.randrange(len(ids)) for _ in ids]
        ma = sum(vals["A"][ids[j]] for j in smp) / len(smp)
        mf = sum(vals[floor][ids[j]] for j in smp) / len(smp)
        ml = sum(vals[best][ids[j]] for j in smp) / len(smp)
        if abs(ma - mf) < 1e-12:
            undef += 1
            continue
        boots.append((ml - mf) / (ma - mf))
    boots.sort()
    frac = undef / BOOT_N
    ci = ([round(boots[int(0.025 * len(boots))], 4),
           round(boots[int(0.975 * len(boots))], 4)] if boots else None)
    return {"family_arms": list(pool), "arms_present": have, "complete": len(have) == len(pool),
            "floor": floor, "best_arm": best,
            "rate_A": round(mean["A"], 4), "rate_floor": round(mean[floor], 4),
            "rate_L": round(mean[best], 4), "range_A_minus_floor": round(denom, 4),
            "R_point": round((mean[best] - mean[floor]) / denom, 4) if abs(denom) > 1e-12 else None,
            "R_ci95": ci, "bootstrap_undefined_fraction": round(frac, 4),
            "R_reportable": bool(boots) and frac <= 0.05 and denom >= MIN_RANGE}


def _primary_family(strat: dict) -> tuple[str, dict] | tuple[None, None]:
    """The family with the highest reportable R -- the strongest trivial selector."""
    best, blk = None, None
    for name, rec in strat.get("recovery", {}).items():
        if not rec or not rec.get("R_reportable") or rec.get("R_point") is None:
            continue
        if blk is None or rec["R_point"] > blk["R_point"]:
            best, blk = name, rec
    return best, blk


def verdict_for(d: dict) -> dict:
    """prereg 7, on the ANSWER stratum, from a cell that cleared every gate."""
    if d["gates"]["problems"]:
        return {"verdict": "REFUSED",
                "reason": "coverage/validity gates failed; a verdict here would be the "
                          "default output of missing data (prereg 8)",
                "problems": d["gates"]["problems"]}
    s = d["strata"]["answer"]
    fam, rec = _primary_family(s)
    if rec is None:
        return {"verdict": "REFUSED",
                "reason": "no family has a reportable R: either the floor or arm A is "
                          "missing, or the dynamic-range gate (prereg 6.5) failed"}
    con = s["contrasts"]["A_vs_L"]
    if "status" in con:
        return {"verdict": "REFUSED", "reason": "A_vs_L not computable"}
    R, p = rec["R_point"], con["mcnemar_exact_p"]
    lo_d, hi_d = con["ci95_diff"]
    diff = con["diff_a_minus_b"]
    rl, rh = (rec["R_ci95"] or [R, R])
    # A THRESHOLD COUNTS AS CROSSED ONLY IF THE WHOLE INTERVAL IS PAST IT.
    above90, above50, below50 = rl >= 0.90, rl >= 0.50, rh < 0.50
    equivalent = p >= 0.05 and hi_d <= 0.10 and lo_d >= -0.10

    if diff < 0 and p < 0.05:
        v, why = "BEATEN", "a selector with no forward pass outperforms the mechanism"
    elif above90 and equivalent:
        v, why = "TRIVIAL", "the score is this programme's L_state"
    elif above90 and diff > 0 and p < 0.05:
        v, why = "PARTIAL (upper)", "residual real but bounded at <=10% of the advantage"
    elif above50 and rh < 0.90:
        # BOUNDED ABOVE, because prereg 7 row 4 is literally 0.50 <= R < 0.90. Left
        # unbounded, a cell with the whole R interval above 1 and the trivial arm
        # 11.8 pp BETTER printed "a trivial selector recovers most of the advantage"
        # and prescribed re-scoping, where the frozen table says close nothing.
        v, why = "PARTIAL", "a trivial selector recovers most of the advantage"
    elif below50 and diff > 0 and p < 0.05:
        # MONOTONE SAFETY. A subset of the trivial arms can only LOWER the maximum,
        # so R from a subset is a lower bound on the true R -- which makes BEATEN,
        # TRIVIAL and both PARTIAL rows safe under partial coverage, and makes
        # SUBSTANTIVE the one verdict a missing arm can manufacture. It therefore
        # needs complete coverage of its family; the others do not.
        if not rec["complete"]:
            v, why = ("INCONCLUSIVE",
                      f"R < 0.50 but family {fam} is incomplete "
                      f"({rec['arms_present']} of {rec['family_arms']}); a missing "
                      f"trivial arm can only lower R, so SUBSTANTIVE is not safe")
        else:
            v, why = "SUBSTANTIVE", "no trivial selector reproduces the score, on this benchmark"
    else:
        v, why = "INCONCLUSIVE", ("no preregistered row matches once the bootstrap CI on "
                                  "R is required to clear the threshold; close nothing")
    out = {"verdict": v, "reading": why, "primary_family": fam,
           "R_point": R, "R_ci95": rec["R_ci95"], "floor": rec["floor"],
           "best_trivial_arm": rec["best_arm"], "mcnemar_p_A_vs_L": p,
           "ci95_diff_A_minus_L": [lo_d, hi_d],
           "R_by_family": {k: (r or {}).get("R_point")
                           for k, r in s.get("recovery", {}).items()}}
    # Secondary: the directionality ablation, SYMMETRIC in both directions.
    b = s["contrasts"].get("A_vs_B")
    if b and "status" not in b:
        if b["diff_a_minus_b"] > 0 and b["mcnemar_exact_p"] < 0.05:
            out["directionality"] = "TRANSFERS: causal beats centred at length"
        elif b["diff_a_minus_b"] < 0 and b["mcnemar_exact_p"] < 0.05:
            out["directionality"] = ("DOES NOT TRANSFER: centred beats causal at length, "
                                     "at p<0.05 -- abstract-level consequence")
        else:
            out["directionality"] = (
                f"UNDERPOWERED/INDETERMINATE: A-B = {b['diff_a_minus_b']:+.4f}, "
                f"p = {b['mcnemar_exact_p']:.3g}. Exact McNemar first reaches p<0.05 at "
                f"6/0, so nothing below ~6pp can resolve here. NOT evidence of absence.")
    c = s["contrasts"].get("A_vs_C")
    if c and "status" not in c:
        out["pooling_worth"] = {"diff_A_minus_C": c["diff_a_minus_b"],
                                "p": c["mcnemar_exact_p"]}
    return out


def analyse_dataset(dataset: str, expect: int) -> dict:
    arms = {}
    for k, spec in ARMS.items():
        arms[k] = load_arm(spec[0] or F_TAG[dataset], dataset)
    repro = reproduction_check(arms, dataset)
    if not repro["licenses_F"] and arms.get("F"):
        # prereg 2.4/6.1: committed data may only be reused if arm A reproduced.
        arms["F"] = None
    gates = gate_report(arms, expect)
    if repro["identical"] is not None and not repro["licenses_F"]:
        gates["notes"].append(
            f"reproduction: {repro['identical']}/{repro['n_paired']} against "
            f"{repro['reference']} -- arm F dropped (prereg 6.1)")
    dec = decisions(dataset, expect)

    out = {"dataset": dataset, "n_expected": expect, "reproduction": repro,
           "gates": gates, "strata": {}, "arms": {}}
    for k, v in arms.items():
        if v:
            out["arms"][k] = {
                "tag": v["dir"], "label": ARMS[k][1], "n": len(v["rows"]),
                "mean_kept_ratio": v["manifest"].get("mean_kept_ratio"),
                "cap_mib": v["manifest"].get("cap_mib"),
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
        rec = {}
        if ids:
            for fam, (pool, floor) in FAMILIES.items():
                rec[fam] = recovery(vals, ids, pool, floor)
        # L MUST BE THE ELECTED FAMILY'S OWN BEST ARM, not a global argmax over
        # rates. Two independent arm selections used to coexist here: this line
        # picked by rate across D/E/G while _primary_family elected a family by R and
        # recovery() picked a best arm inside it. When they disagreed, verdict_for
        # printed one arm's name beside another arm's p-value and CI -- demonstrated
        # to flip TRIVIAL into BEATEN on constructed data, and the disagreement
        # already occurs on the committed 4K abstain stratum (best_arm G, global L D).
        # It is a no-op on both answer strata, where G is both, so no published
        # number moves; it is fixed because the next cell would not have been caught.
        cand = [k for k in ("D", "E", "G") if k in vals]
        if cand and ids:
            fam, frec = _primary_family({"recovery": rec})
            if frec and frec.get("best_arm") in vals:
                vals["L"] = vals[frec["best_arm"]]
            else:
                # No reportable family: fall back to the global best so the contrast
                # is still computable, but verdict_for will REFUSE on the missing
                # reportable R rather than read a verdict off it.
                vals["L"] = vals[max(cand, key=lambda k: sum(vals[k][i] for i in ids))]
        out["strata"][stratum] = {
            "n": len(ids),
            "rates": {k: round(sum(v[i] for i in ids) / max(len(ids), 1), 4)
                      for k, v in vals.items() if k != "L"},
            "contrasts": {f"{a}_vs_{b}": contrast(vals, a, b, ids) for a, b in CONTRASTS},
            "recovery": rec,
        }
    return out


def derive() -> dict:
    per = {ds: analyse_dataset(ds, n) for ds, (n, _p) in DATASETS.items()}
    primary = per.get("rag_long4k")
    rep = per.get("rag_long16k")
    return {"preregistration": "docs/PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md (v3)",
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
        rp = d.get("reproduction") or {}
        if rp.get("identical") is not None:
            print(f"  reproduction vs {rp['reference']}: "
                  f"{rp['identical']}/{rp['n_paired']} byte-identical -- {rp['status']}")
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
            for tag, r in (blk.get("recovery") or {}).items():
                if r and r.get("R_point") is not None:
                    print(f"      R[{tag}] = {r['R_point']} CI {r['R_ci95']}  "
                          f"floor {r['floor']}  best {r['best_arm']}  "
                          f"reportable={r['R_reportable']}"
                          + ("" if r["complete"] else "  INCOMPLETE FAMILY"))
                elif r:
                    print(f"      R[{tag}]: {r.get('status', 'not computable')}")

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
