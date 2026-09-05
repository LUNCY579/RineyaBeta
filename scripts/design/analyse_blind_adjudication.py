"""Analyser for docs/PREREG_BLIND_ADJUDICATION.md.

Unblinds the adjudication and asks the question the substring endpoint cannot:
on the answer stratum, are the compressed arm's wins correct, legible answers,
or correct tokens inside degraded text?

Enforces the three validity gates first and refuses a verdict if any fails.

    python scripts/design/analyse_blind_adjudication.py
    python scripts/design/analyse_blind_adjudication.py --check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

RESULTS = os.path.join("experiments", "kv_significance", "results")
KEY = os.path.join(RESULTS, "adjudication_key.json")
LABELS = os.path.join(RESULTS, "adjudication_labels.json")

CLEAN = "CORRECT_CLEAN"
DEGRADED = "CORRECT_DEGRADED"

N_ALL_CORRECT = 330          # concordant answer-stratum cases in the full stratum
MIN_RATERS = 3
MIN_MAJORITY_RATE = 0.90
MIN_CONTRACT_AGREEMENT = 0.90
MIN_DISCORDANT = 6


def exact_mcnemar(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sig6(x):
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + 5)


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval by bisection, matching the repo's convention."""
    if n == 0:
        return (0.0, 1.0)
    lo, hi = 0.0, 0.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if 1 - binom_cdf(k - 1, n, m) > alpha / 2:
                b = m
            else:
                a = m
        lo = a
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if binom_cdf(k, n, m) < alpha / 2:
                b = m
            else:
                a = m
        hi = b
    else:
        hi = 1.0
    return (round(lo, 6), round(hi, 6))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=os.path.join(RESULTS, "blind_adjudication.json"))
    args = ap.parse_args()

    keyfile, labfile = load_json(KEY), load_json(LABELS)
    if keyfile is None or labfile is None:
        print("missing {0} or {1}".format(KEY, LABELS), file=sys.stderr)
        return 1
    key = keyfile["key"]
    raw = labfile["raw"]          # item_id -> [label, ...]
    final = labfile["majority"]   # item_id -> label after majority + tiebreak

    out = {"gates": {}, "verdict": None}

    # ---- gate 1: adjudication is not noise ---------------------------------
    n_items = len(key)
    with3 = [i for i in key if len(raw.get(i, [])) >= MIN_RATERS]
    decided = [i for i in key if i in final]
    unanimous = [i for i in with3
                 if len(set(raw[i])) == 1]
    majority_ok = [i for i in with3
                   if max(raw[i].count(l) for l in set(raw[i])) >= 2]
    g1_rate = len(majority_ok) / n_items if n_items else 0.0
    out["gates"]["g1_majority_rate"] = round(g1_rate, 4)
    out["gates"]["items_total"] = n_items
    out["gates"]["items_with_3_raters"] = len(with3)
    out["gates"]["items_decided"] = len(decided)
    out["gates"]["unanimous"] = len(unanimous)
    out["gates"]["g1_pass"] = (len(with3) == n_items and g1_rate >= MIN_MAJORITY_RATE)

    print("VALIDITY GATES (prereg section 5)")
    print("  1 adjudication is not noise      {0}/{1} items have {2} raters, "
          "{3} unanimous, majority on {4:.1%}  {5}".format(
              len(with3), n_items, MIN_RATERS, len(unanimous), g1_rate,
              "PASS" if out["gates"]["g1_pass"] else "FAIL"))

    # ---- gate 2: adjudicators track the contract ---------------------------
    conc = [i for i in key if key[i]["group"] == "concordant" and i in final]
    permissive = {i: 1 if final[i] in (CLEAN, DEGRADED) else 0 for i in final}
    agree = [i for i in conc if permissive[i] == key[i]["committed_endpoint"]]
    g2_rate = len(agree) / len(conc) if conc else 0.0
    out["gates"]["g2_contract_agreement"] = round(g2_rate, 4)
    out["gates"]["g2_pass"] = bool(conc) and g2_rate >= MIN_CONTRACT_AGREEMENT
    print("  2 adjudicators track contract    permissive endpoint agrees with "
          "committed on {0}/{1} concordant replies ({2:.1%})  {3}".format(
              len(agree), len(conc), g2_rate,
              "PASS" if out["gates"]["g2_pass"] else "FAIL"))

    # ---- the contrasts ------------------------------------------------------
    strict = {i: 1 if final[i] == CLEAN else 0 for i in final}

    def by_case(group):
        cases = {}
        for item, meta in key.items():
            if meta["group"] != group or item not in final:
                continue
            cases.setdefault(meta["case"], {})[meta["arm"]] = item
        return cases

    def contrast(cases, score, x, y):
        w = l = 0
        n = 0
        for arms in cases.values():
            if x not in arms or y not in arms:
                continue
            n += 1
            a, b = score[arms[x]], score[arms[y]]
            w += a > b
            l += a < b
        return {"n": n, "wins": w, "losses": l,
                "delta_pp": round(100.0 * (w - l) / n, 4) if n else 0.0,
                "p": sig6(exact_mcnemar(w, l)),
                "power_ceiling": sig6(min(1.0, 2.0 * 0.5 ** (w + l)) if w + l else 1.0)}

    disputed = by_case("disputed")
    concordant = by_case("concordant")

    print()
    print("DISPUTED SET, {0} cases".format(len(disputed)))
    for name, score in (("primary  CORRECT_CLEAN", strict),
                        ("secondary  CLEAN+DEGRADED", permissive)):
        for pair in (("C", "R"), ("N", "R"), ("C", "N")):
            r = contrast(disputed, score, *pair)
            out.setdefault("disputed", {}).setdefault(name, {})[
                "{0} vs {1}".format(*pair)] = r
            print("  {0:26s} {1} vs {2}  n={3:3d}  {4:+7.2f} pp  {5:2d}W/{6:2d}L  "
                  "p={7:.3g}".format(name, pair[0], pair[1], r["n"],
                                     r["delta_pp"], r["wins"], r["losses"], r["p"]))
        print()

    # ---- gate 3: can the primary contrast conclude? -------------------------
    prim = out.get("disputed", {}).get("primary  CORRECT_CLEAN", {}).get("C vs R")
    n_disc = (prim["wins"] + prim["losses"]) if prim else 0
    out["gates"]["g3_discordant"] = n_disc
    out["gates"]["g3_pass"] = n_disc >= MIN_DISCORDANT
    print("  3 primary contrast can conclude  {0} discordant pairs (need {1})  "
          "{2}".format(n_disc, MIN_DISCORDANT,
                       "PASS" if out["gates"]["g3_pass"] else "FAIL"))

    # ---- section 7: the conditioning bound ----------------------------------
    induced = 0
    for arms in concordant.values():
        if "C" in arms and "R" in arms and strict[arms["C"]] != strict[arms["R"]]:
            induced += 1
    n_conc = len(concordant)
    lo, hi = clopper_pearson(induced, n_conc) if n_conc else (0.0, 1.0)
    # Project onto the cases NOT yet adjudicated. The n_conc already measured
    # contribute their observed count, not their upper bound -- charging them the
    # bound as well would double-count the very cases that produced it.
    unadjudicated = N_ALL_CORRECT - n_conc
    projected_hi = hi * unadjudicated + induced
    out["conditioning"] = {
        "concordant_cases_adjudicated": n_conc,
        "induced_discordant": induced,
        "rate_ci95": [lo, hi],
        "unadjudicated_all_correct": unadjudicated,
        "projected_induced_upper_bound": round(projected_hi, 2),
        "primary_net_wins": (prim["wins"] - prim["losses"]) if prim else 0,
    }
    could_flip = projected_hi > abs(out["conditioning"]["primary_net_wins"])
    out["conditioning"]["could_flip_sign"] = bool(could_flip)
    print()
    print("CONDITIONING BOUND (prereg section 7)")
    print("  adjudication created {0} new discordant pairs in {1} concordant "
          "cases; rate 95% CI [{2:.4f}, {3:.4f}]".format(induced, n_conc, lo, hi))
    print("  upper bound projected onto all {0} all-correct cases: {1:.1f} induced "
          "pairs vs primary net {2}".format(
              N_ALL_CORRECT, projected_hi,
              out["conditioning"]["primary_net_wins"]))
    print("  {0}".format("COULD FLIP THE SIGN" if could_flip else
                         "cannot flip the sign"))

    # ---- verdict ------------------------------------------------------------
    print()
    print("VERDICT (prereg section 6)")
    gates_ok = all(out["gates"][k] for k in ("g1_pass", "g2_pass", "g3_pass"))
    if not gates_ok:
        out["verdict"] = "NO VERDICT -- validity gates failed"
        print("  NO VERDICT. Validity gates failed; section 5 forbids emitting one.")
    elif could_flip:
        out["verdict"] = "INDETERMINATE -- conditioning bound could flip the sign"
        print("  INDETERMINATE. Section 7 applies: the adjudication-induced "
              "discordance among concordant cases could change the sign, so the "
              "full stratum must be adjudicated before a verdict.")
    elif prim["delta_pp"] > 0 and prim["p"] <= 0.05:
        out["verdict"] = "SURVIVES"
        print("  SURVIVES. On the primary endpoint the gain is correct, legible "
              "answers: {0:+.2f} pp, {1}W/{2}L, p={3:.3g}.".format(
                  prim["delta_pp"], prim["wins"], prim["losses"], prim["p"]))
    elif prim["p"] > 0.05 and abs(prim["delta_pp"]) < 2.0:
        out["verdict"] = "ENDPOINT-DEPENDENT"
        print("  ENDPOINT-DEPENDENT. Withdraw the accuracy claim and restate: "
              "compression at 16.5x does not degrade answer accuracy, and the "
              "apparent gain was the endpoint crediting degraded text.")
    else:
        out["verdict"] = "INDETERMINATE"
        print("  INDETERMINATE. No verdict; report the achieved MDE.")

    if args.check:
        old = load_json(args.json)
        if old is None:
            print("\nno committed analysis at " + args.json, file=sys.stderr)
            return 1
        if old != out:
            print("\nMISMATCH against committed adjudication analysis",
                  file=sys.stderr)
            return 1
        print("\ncommitted blind-adjudication analysis reproduces")
        return 0

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
