"""Does record selection do anything once the instructions are kept?

Analyser for docs/PREREG_INSTRUCTION_IDENTIFICATION.md. Enforces the coverage
gates first and refuses to emit a verdict if any of them fails, then runs the
preregistered contrasts on the preregistered primary stratum.

    python scripts/design/analyse_instruction_identification.py
    python scripts/design/analyse_instruction_identification.py --check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

RESULTS = os.path.join("experiments", "kv_significance", "results")
DATASET = os.path.join("artifacts", "eval_sets", "rag_long4k_d0.5.jsonl")
ENDPOINT = "contract_fulfillment_score"
HARD = 0.999
CANNED = "i don't know based on the available information"

ARM_A = "c19-randomdocsys-k0.05-n600__rag_long4k"      # random-doc-sys
ARM_B = "c19-oraclesys-k0.05-n600__rag_long4k"          # oracle-support-sys (ceiling)
ARM_C = "c8-docbm25sys-k0.05-n600__rag_long4k"          # lexical-doc-bm25-sys
ARM_R = "c7-off-n600__rag_long4k"                       # uncompressed

# Arms that are NOT protected, for the contrast that made the pooled endpoint
# look as though selection did nothing.
ARM_BM25_BARE = "c7-docbm25-k0.05-n600__rag_long4k"
ARM_ORACLE_BARE = "c7-oracle-k0.05-n600__rag_long4k"

ABSTAIN_CATEGORIES = frozenset({
    "ambiguous_evidence", "missing_answer", "conflict", "partial_support",
    "procedure_missing_input", "procedure_conflict",
})

# Section 2 as amended by section 6a: compare only keys present in BOTH
# manifests, by value rather than repr.
ALLOWED_MANIFEST_DIFFS = {"tag", "select", "elapsed_s"}
MIN_CASES = 594


def sig6(x):
    """Round to 6 significant figures. round(1.46e-98, 10) is 0.0, and a p-value
    printed as exactly zero is a claim no finite test can make."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + 5)


def exact_mcnemar(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(arm):
    path = os.path.join(RESULTS, arm, "results.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row
    return out


def manifest(arm):
    path = os.path.join(RESULTS, arm, "run_manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def hard(row):
    return 1 if float(row.get(ENDPOINT) or 0.0) >= HARD else 0


def same_value(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) < 1e-9
    return json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)


def manifest_diff(arm, ref):
    ma, mc = manifest(arm), manifest(ref)
    if ma is None or mc is None:
        return None
    shared = set(ma) & set(mc)
    return sorted(k for k in shared if not same_value(ma[k], mc[k]))


def strata(rows):
    an = [i for i in rows if rows[i]["category"] not in ABSTAIN_CATEGORIES]
    ab = [i for i in rows if rows[i]["category"] in ABSTAIN_CATEGORIES]
    return {"answer": an, "abstain": ab, "pooled": sorted(rows)}


def rate(rows, ids):
    return sum(hard(rows[i]) for i in ids) / len(ids) if ids else 0.0


def contrast(x, y, ids):
    w = sum(1 for i in ids if hard(x[i]) > hard(y[i]))
    l = sum(1 for i in ids if hard(x[i]) < hard(y[i]))
    return {
        "n": len(ids),
        "delta_pp": round(100.0 * sum(hard(x[i]) - hard(y[i]) for i in ids) / len(ids), 4),
        "wins": w, "losses": l,
        "p": sig6(exact_mcnemar(w, l)),
        "power_ceiling": sig6(min(1.0, 2.0 * 0.5 ** (w + l)) if w + l else 1.0),
    }


def gates(out):
    """Section 6. No verdict of any kind unless all of these hold."""
    ok = True
    print("COVERAGE GATES (prereg section 6)")
    for label, arm in (("A random-doc-sys", ARM_A), ("B oracle-support-sys", ARM_B)):
        rows, man = load(arm), manifest(arm)
        if rows is None or man is None:
            print("  {0:24s} MISSING -- arm has not been run".format(label))
            out["gates"][arm] = {"present": False}
            ok = False
            continue
        diff = manifest_diff(arm, ARM_C)
        extra = sorted(set(diff) - ALLOWED_MANIFEST_DIFFS)
        st = strata(rows)
        abstain_rate = rate(rows, st["abstain"])
        c_rows = load(ARM_C)
        c_abstain = rate(c_rows, strata(c_rows)["abstain"])
        canned = sum(1 for i in rows if CANNED in rows[i]["reply"].lower()) / len(rows)
        g = {
            "present": True,
            "cases": len(rows),
            "cases_ok": len(rows) >= MIN_CASES,
            "heavy_zero_count": man.get("heavy_zero_count"),
            "heavy_zero_ok": man.get("heavy_zero_count") == 0,
            "shared_manifest_diff": diff,
            "manifest_ok": not extra,
            "abstain_rate": round(abstain_rate, 4),
            "reference_abstain_rate": round(c_abstain, 4),
            "protection_fired": abstain_rate >= c_abstain,
            "canned_emission_rate": round(canned, 4),
        }
        out["gates"][arm] = g
        good = (g["cases_ok"] and g["heavy_zero_ok"] and g["manifest_ok"]
                and g["protection_fired"])
        ok = ok and good
        print("  {0:24s} {1} cases | heavy_zero {2} | shared diff {3} | "
              "abstain {4:.4f} vs C {5:.4f} | canned {6:.4f}  {7}".format(
                  label, g["cases"], g["heavy_zero_count"], diff or "[]",
                  abstain_rate, c_abstain, canned, "PASS" if good else "FAIL"))
    out["gates_pass"] = ok
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=os.path.join(RESULTS,
                                                   "instruction_identification.json"))
    args = ap.parse_args()

    out = {"gates": {}, "ladder": {}, "contrasts": {}, "verdict": None}
    ok = gates(out)
    print()

    # The ladder the pooled endpoint hid.
    print("ANSWER STRATUM LADDER (the preregistered primary)")
    ladder = [("random-doc  +sys  (floor)", ARM_A),
              ("UNCOMPRESSED", ARM_R),
              ("doc-bm25     no sys", ARM_BM25_BARE),
              ("doc-bm25    +sys", ARM_C),
              ("oracle       no sys", ARM_ORACLE_BARE),
              ("oracle      +sys  (ceiling)", ARM_B)]
    vals = {}
    for label, arm in ladder:
        rows = load(arm)
        if rows is None:
            print("  {0:28s} (not run)".format(label))
            continue
        st = strata(rows)
        vals[arm] = {s: round(rate(rows, st[s]), 4) for s in st}
        out["ladder"][arm] = dict(vals[arm], label=label)
        print("  {0:28s} answer {1:.4f}   abstain {2:.4f}   pooled {3:.4f}".format(
            label, vals[arm]["answer"], vals[arm]["abstain"], vals[arm]["pooled"]))

    if ARM_A in vals and ARM_B in vals:
        floor, ceil = vals[ARM_A]["answer"], vals[ARM_B]["answer"]
        got = vals[ARM_C]["answer"]
        if ceil > floor:
            frac = (got - floor) / (ceil - floor)
            out["selection_saturation"] = round(frac, 4)
            print()
            print("  BM25 captures {0:.1%} of the floor-to-ceiling selection range "
                  "({1:.4f} -> {2:.4f})".format(frac, floor, ceil))

    print()
    print("PREREGISTERED CONTRASTS")
    pairs = [("A vs C  identification", ARM_A, ARM_C),
             ("B vs C  ceiling", ARM_B, ARM_C),
             ("A vs R", ARM_A, ARM_R),
             ("B vs R", ARM_B, ARM_R),
             ("C vs R", ARM_C, ARM_R)]
    for label, xn, yn in pairs:
        x, y = load(xn), load(yn)
        if x is None or y is None:
            print("  {0:24s} (arm not run)".format(label))
            continue
        ids = sorted(set(x) & set(y))
        st = strata({i: x[i] for i in ids})
        for s in ("answer", "abstain", "pooled"):
            r = contrast(x, y, st[s])
            out["contrasts"].setdefault(label, {})[s] = r
            print("  {0:24s} {1:8s} n={2:3d}  {3:+8.2f} pp  {4:3d}W/{5:3d}L  "
                  "p={6:.3g}".format(label, s, r["n"], r["delta_pp"],
                                     r["wins"], r["losses"], r["p"]))
        print()

    # Decision rule, section 5, applied to the primary stratum only.
    print("VERDICT (prereg section 5, answer stratum)")
    if not ok:
        out["verdict"] = "NO VERDICT -- coverage gates failed"
        print("  NO VERDICT. Coverage gates failed; section 6 forbids emitting one.")
    elif "A vs C  identification" not in out["contrasts"]:
        out["verdict"] = "NO VERDICT -- arms missing"
        print("  NO VERDICT. Required arms are not on disk.")
    else:
        r = out["contrasts"]["A vs C  identification"]["answer"]
        if r["p"] <= 0.05 and r["delta_pp"] < 0:
            out["verdict"] = "SELECTION CONTRIBUTES"
            print("  SELECTION CONTRIBUTES. C beats A on the answer stratum by "
                  "{0:+.2f} pp ({1}W/{2}L, p={3:.3g}).".format(
                      -r["delta_pp"], r["losses"], r["wins"], r["p"]))
            if "selection_saturation" in out:
                print("  It is nonetheless SATURATED, not merely present: BM25 "
                      "captures {0:.1%} of the range a perfect oracle spans.".format(
                          out["selection_saturation"]))
        elif r["p"] > 0.05 and abs(r["delta_pp"]) < 2.0:
            out["verdict"] = "SELECTION IS INERT"
            print("  SELECTION IS INERT on this corpus.")
        else:
            out["verdict"] = "INDETERMINATE"
            print("  INDETERMINATE. No verdict; report the achieved MDE.")

    if args.check:
        if not os.path.exists(args.json):
            print("\nno committed analysis at " + args.json, file=sys.stderr)
            return 1
        with open(args.json, encoding="utf-8") as fh:
            old = json.load(fh)
        if old != out:
            print("\nMISMATCH against committed analysis", file=sys.stderr)
            for k in sorted(set(old) | set(out)):
                if old.get(k) != out.get(k):
                    print("  differs: " + k, file=sys.stderr)
            return 1
        print("\ncommitted identification analysis reproduces")
        return 0

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
