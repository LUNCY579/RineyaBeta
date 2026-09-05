"""What could each paired cell in this repo have concluded, before it was run?

An exact McNemar test on n discordant pairs cannot produce a p-value smaller than
2 * 0.5**n, whatever the data does. So a cell's *power ceiling* is fixed by its
discordant count alone, and a cell with fewer than six discordant pairs can never
reach p < 0.05 no matter which way every pair falls.

This matters because this programme has a standing rule -- CHECK ENDPOINT POWER
BEFORE ANY SWEEP -- and a second one: never let a programme-closing verdict be the
default output of missing data. A cell whose ceiling is above 0.05 and which is then
reported as "not significant" is not evidence of absence. It is not evidence at all.

Run with no arguments to audit every arm against its dataset's uncompressed reference.
Run with --check to assert the committed ceiling table still reproduces.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from glob import glob

RESULTS = os.path.join("experiments", "kv_significance", "results")
ENDPOINT = "contract_fulfillment_score"
CEILING_TARGET = 0.05

# The committed endpoint is BINARISED at 0.999, not the raw fractional score --
# see analyse_capacity_floor.py:33 and analyse_triviality_control.py:32. Auditing
# the raw score would answer a question nobody asked, and shifts the win/loss
# ledger (the headline arm is 34/3 binarised and 34/2 raw).
HARD = 0.999

# The abstain stratum, from the response contract rather than from prose: these
# are the categories whose contract demands the literal refusal sentence.
ABSTAIN_CATEGORIES = frozenset({
    "ambiguous_evidence", "missing_answer", "conflict", "partial_support",
    "procedure_missing_input", "procedure_conflict",
})


def exact_mcnemar(wins, losses):
    """Two-sided exact McNemar, matching analyse_capacity_floor.py:40."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sig6(x):
    """Round to 6 significant figures. round(2.1e-09, 6) is 0.0, and a p-value
    printed as exactly zero is a claim no finite test can make."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + 5)


def power_ceiling(n_disc):
    """Smallest p this cell could ever have returned, over all outcomes."""
    if n_disc == 0:
        return 1.0
    return min(1.0, 2.0 * 0.5 ** n_disc)


def min_discordant_for(target=CEILING_TARGET):
    n = 1
    while power_ceiling(n) > target:
        n += 1
    return n


def exact_power(n_disc, p_win, alpha=CEILING_TARGET):
    """P(exact McNemar rejects) when a discordant pair favours the arm w.p. p_win."""
    return sum(math.comb(n_disc, k) * p_win ** k * (1 - p_win) ** (n_disc - k)
               for k in range(n_disc + 1)
               if exact_mcnemar(k, n_disc - k) <= alpha)


def required_pairs(p_win, disc_rate, alpha=CEILING_TARGET, target_power=0.80,
                   cap=4000):
    """Total paired cases needed to detect this effect, at this discordance rate.

    The ceiling test only asks whether a cell COULD ever reach alpha. This asks
    the question that actually matters: was the cell big enough to find the
    effect the programme claims exists?
    """
    if not 0.0 < disc_rate <= 1.0 or p_win <= 0.5:
        return None, None
    n_disc = min_discordant_for(alpha)
    while n_disc <= cap:
        if exact_power(n_disc, p_win, alpha) >= target_power:
            return n_disc, math.ceil(n_disc / disc_rate)
        n_disc += 1
    return None, None


def load(arm):
    """id -> (binarised endpoint, stratum)."""
    path = os.path.join(RESULTS, arm, "results.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if ENDPOINT not in row:
                continue
            hard = 1 if float(row.get(ENDPOINT) or 0.0) >= HARD else 0
            stratum = ("abstain" if row.get("category") in ABSTAIN_CATEGORIES
                       else "answer")
            out[row["id"]] = (hard, stratum)
    return out or None


def references():
    """dataset -> [(reference arm, its manifest)] for arms with eviction off."""
    by_ds = {}
    for path in sorted(glob(os.path.join(RESULTS, "*", "run_manifest.json"))):
        with open(path, encoding="utf-8") as fh:
            man = json.load(fh)
        if man.get("eviction") is False:
            arm = os.path.basename(os.path.dirname(path))
            by_ds.setdefault(man.get("dataset", "?"), []).append((arm, man))
    return by_ds


def provenance(arm_man, ref_man):
    """Is this arm/reference pair even comparable?

    A reference on a different checkpoint is not a control, it is a second
    experiment. long16k-off-baya-ad runs Rineya-Baya-1.7.0 plus a long-context
    SFT adapter and differs from long16k-off on 13 of 100 cases, so pairing a
    core-model arm against it measures the checkpoint, not the compression.
    """
    a_model, r_model = arm_man.get("model"), ref_man.get("model")
    a_ad, r_ad = arm_man.get("adapter"), ref_man.get("adapter")
    if r_model is None and r_ad is None and "model" not in ref_man:
        return "unverified"          # pre-schema manifest: provenance not recorded
    if a_model != r_model or a_ad != r_ad:
        return "MISMATCHED MODEL"
    return "matched"


def cell(ref, arm, ids):
    wins = sum(1 for i in ids if arm[i][0] > ref[i][0])
    losses = sum(1 for i in ids if arm[i][0] < ref[i][0])
    n_disc = wins + losses
    return {
        "n_paired": len(ids),
        "wins": wins,
        "losses": losses,
        "n_discordant": n_disc,
        "delta_pp": round(100.0 * sum(arm[i][0] - ref[i][0] for i in ids) / len(ids), 4)
        if ids else 0.0,
        "p_observed": sig6(exact_mcnemar(wins, losses)),
        "power_ceiling": sig6(power_ceiling(n_disc)),
        "could_reach_significance": power_ceiling(n_disc) <= CEILING_TARGET,
    }


def audit():
    refs = references()
    cells = []
    for path in sorted(glob(os.path.join(RESULTS, "*", "run_manifest.json"))):
        arm_name = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as fh:
            man = json.load(fh)
        if man.get("eviction") is not True:
            continue
        ds = man.get("dataset")
        arm = load(arm_name)
        if arm is None:
            continue
        for ref_name, ref_man in refs.get(ds, []):
            ref = load(ref_name)
            if ref is None:
                continue
            ids = sorted(set(ref) & set(arm))
            if not ids:
                continue
            rec = {
                "arm": arm_name,
                "reference": ref_name,
                "dataset": ds,
                "budget": man.get("budget"),
                "keep_frac": man.get("keep_frac"),
                "select": man.get("select"),
                "provenance": provenance(man, ref_man),
            }
            rec.update(cell(ref, arm, ids))
            for stratum in ("answer", "abstain"):
                sub = [i for i in ids if arm[i][1] == stratum]
                rec[stratum] = cell(ref, arm, sub) if sub else None
            cells.append(rec)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="assert the committed ceiling table still reproduces")
    ap.add_argument("--json", default=os.path.join(RESULTS, "power_ceiling.json"))
    ap.add_argument("--only", default="", help="substring filter on arm name")
    args = ap.parse_args()

    cells = audit()
    if args.only:
        cells = [c for c in cells if args.only in c["arm"]]

    floor_n = min_discordant_for()
    print("exact McNemar needs at least {0} discordant pairs to reach p < {1} "
          "(ceiling at {0}: {2:.5f}; at {3}: {4:.5f})".format(
              floor_n, CEILING_TARGET, power_ceiling(floor_n),
              floor_n - 1, power_ceiling(floor_n - 1)))
    print()

    comparable = [c for c in cells if c["provenance"] != "MISMATCHED MODEL"]
    crossmodel = [c for c in cells if c["provenance"] == "MISMATCHED MODEL"]
    dead = [c for c in comparable if not c["could_reach_significance"]]
    # Every arm is paired against every eviction-off arm on its dataset, so most
    # cross-checkpoint rows below are pairings this audit enumerated, not
    # comparisons the programme ever made. They are printed so that a future
    # comparison cannot reach for one of them without the label being visible.
    print("{0} enumerated arm/reference pairings; {1} cross a checkpoint boundary "
          "and are listed only to mark them unusable".format(len(cells), len(crossmodel)))
    print("of the {0} same-checkpoint pairings, {1} could NOT have reached p < {2} "
          "whatever the data did".format(len(comparable), len(dead), CEILING_TARGET))
    print()
    hdr = ("{0:44s} {1:26s} {2:>4s} {3:>3s} {4:>3s} {5:>4s} {6:>9s} {7:>8s} {8:>8s}  "
           "{9:<16s} verdict".format("arm", "ref", "n", "W", "L", "disc", "delta_pp",
                                     "p_obs", "ceiling", "provenance"))
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"], c["reference"])):
        if c["provenance"] == "MISMATCHED MODEL":
            verdict = "NOT A CONTROL"
        elif not c["could_reach_significance"]:
            verdict = "CANNOT CONCLUDE"
        else:
            verdict = ""
        print("{0:44s} {1:26s} {2:4d} {3:3d} {4:3d} {5:4d} {6:9.2f} {7:8.4f} {8:8.4f}  "
              "{9:<16s} {10}".format(
                  c["arm"][:44], c["reference"][:26], c["n_paired"], c["wins"],
                  c["losses"], c["n_discordant"], c["delta_pp"], c["p_observed"],
                  c["power_ceiling"], c["provenance"], verdict))

    # How big would an underpowered cell have to be to find the 4K effect?
    ref4k = next((c for c in cells if c["arm"].startswith("c9-sys-B256")
                  and c["reference"].startswith("c7-off-n600")), None)
    sizing = None
    if ref4k and ref4k["n_discordant"]:
        p_win = ref4k["wins"] / ref4k["n_discordant"]
        disc_rate = ref4k["n_discordant"] / ref4k["n_paired"]
        n_disc_req, n_req = required_pairs(p_win, disc_rate)
        sizing = {"anchor": ref4k["arm"], "p_win": round(p_win, 4),
                  "discordance_rate": round(disc_rate, 4),
                  "discordant_pairs_needed": n_disc_req, "paired_cases_needed": n_req,
                  "alpha": CEILING_TARGET, "power": 0.80}
        print()
        print("sizing, taking the 4K headline cell as the alternative hypothesis "
              "(p_win={0:.3f}, discordance {1:.3f}):".format(p_win, disc_rate))
        print("  {0} discordant pairs -> {1} paired cases for 80% power at "
              "alpha {2}".format(n_disc_req, n_req, CEILING_TARGET))
        for c in sorted(comparable, key=lambda c: c["dataset"]):
            if c["n_paired"] < (n_req or 0) and c["dataset"] != "rag_long4k":
                print("  {0:44s} n={1:<4d} is {2:.0%} of that".format(
                    c["arm"][:44], c["n_paired"], c["n_paired"] / n_req))

    payload = {"endpoint": ENDPOINT, "alpha": CEILING_TARGET,
               "min_discordant_for_alpha": floor_n, "sizing": sizing, "cells": cells}
    if args.check:
        if not os.path.exists(args.json):
            print("\nno committed table at " + args.json, file=sys.stderr)
            return 1
        with open(args.json, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("cells") != cells:
            oldmap = {(c["arm"], c["reference"]): c for c in old.get("cells", [])}
            newmap = {(c["arm"], c["reference"]): c for c in cells}
            for k in sorted(set(oldmap) | set(newmap)):
                if oldmap.get(k) != newmap.get(k):
                    print("\nMISMATCH {0}\n  committed: {1}\n  recomputed: {2}".format(
                        k, oldmap.get(k), newmap.get(k)), file=sys.stderr)
            return 1
        print("\ncommitted power-ceiling table reproduces")
        return 0

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print("\nwrote " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
