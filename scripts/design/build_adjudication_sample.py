"""Build the blinded sample for docs/PREREG_BLIND_ADJUDICATION.md.

Emits two files:

  adjudication_items.json   what adjudicators see: item_id, question, gold answer,
                            reply. No arm, no case id, no score, shuffled.
  adjudication_key.json     item_id -> (case id, arm). Not read until every label
                            is in.

Deterministic: seeded permutation, no RNG state carried between runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

RESULTS = os.path.join("experiments", "kv_significance", "results")
DATASET = os.path.join("artifacts", "eval_sets", "rag_long4k_d0.5.jsonl")
ENDPOINT = "contract_fulfillment_score"
HARD = 0.999
SEED = 20260826
N_CONCORDANT = 40

# Section 7 remedy, 2026-09-05. The first round returned 0 induced discordant
# pairs in 40 concordant cases, but 0/40 has a Clopper-Pearson upper bound of
# 0.0881, which projects to 25.5 possible induced pairs over the 290
# unadjudicated cases against a primary net of 7. 113 concordant cases with 0
# induced would bring the projection under 7. --extend draws the NEXT block from
# the same seeded permutation, so it is disjoint from the first 40 by
# construction, and emits only the two arms the primary contrast uses.
N_CONCORDANT_EXTENDED = 150
EXTENSION_ARMS = ("R", "C")

ARMS = {
    "R": "c7-off-n600__rag_long4k",
    "C": "c8-docbm25sys-k0.05-n600__rag_long4k",
    "N": "c9-sys-B256__rag_long4k",
}

ABSTAIN_CATEGORIES = frozenset({
    "ambiguous_evidence", "missing_answer", "conflict", "partial_support",
    "procedure_missing_input", "procedure_conflict",
})


class Rng:
    """Deterministic LCG. Same generator as cluster_bootstrap.py."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next_below(self, n):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n

    def shuffled(self, items):
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.next_below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out


def load(arm):
    path = os.path.join(RESULTS, arm, "results.jsonl")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row
    return out


def dataset():
    out = {}
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row
    return out


def hard(row):
    return 1 if float(row.get(ENDPOINT) or 0.0) >= HARD else 0


def gold_answer(case, res_row):
    """The phrase the contract requires, taken from the contract itself."""
    phrases = []
    for req in (res_row.get("response_contract") or {}).get("requirements", []):
        for ph in (req.get("match") or {}).get("any_phrases", []):
            if ph not in phrases:
                phrases.append(ph)
    return phrases or [str(case.get("reference", ""))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(RESULTS, "adjudication_items.json"))
    ap.add_argument("--key", default=os.path.join(RESULTS, "adjudication_key.json"))
    ap.add_argument("--extend", action="store_true",
                    help="build a section 7 top-up: concordant cases from the "
                         "same permutation, arms R and C only")
    ap.add_argument("--from", dest="lo", type=int, default=N_CONCORDANT)
    ap.add_argument("--to", dest="hi", type=int, default=N_CONCORDANT_EXTENDED)
    ap.add_argument("--suffix", default="ext")
    ap.add_argument("--allzero", action="store_true",
                    help="emit the endpoint-concordant cases where EVERY arm "
                         "scored 0. A substring endpoint has false negatives "
                         "(gold 'Wi-Fi 7' against a reply saying 'WiFi7'), so "
                         "these are not guaranteed concordant under adjudication "
                         "and section 7 covers them too.")
    args = ap.parse_args()
    if args.allzero:
        args.extend = True
    if args.extend:
        args.items = os.path.join(RESULTS,
                                  "adjudication_items_{0}.json".format(args.suffix))
        args.key = os.path.join(RESULTS,
                                "adjudication_key_{0}.json".format(args.suffix))

    data = dataset()
    arms = {k: load(v) for k, v in ARMS.items()}
    ids = sorted(set.intersection(*[set(a) for a in arms.values()]) & set(data))
    answer = [i for i in ids if data[i]["category"] not in ABSTAIN_CATEGORIES]

    disputed = sorted(i for i in answer
                      if hard(arms["C"][i]) != hard(arms["R"][i])
                      or hard(arms["N"][i]) != hard(arms["R"][i]))
    all_correct = sorted(i for i in answer
                         if all(hard(arms[k][i]) == 1 for k in arms))

    rng = Rng(SEED)
    permuted = rng.shuffled(all_correct)
    if args.allzero:
        allzero = sorted(i for i in answer
                         if all(hard(arms[k][i]) == 0 for k in arms))
        groups = (("concordant", allzero),)
        arm_keys = EXTENSION_ARMS
        concordant = allzero
    elif args.extend:
        concordant = sorted(permuted[args.lo:args.hi])
        groups = (("concordant", concordant),)
        arm_keys = EXTENSION_ARMS
    else:
        concordant = sorted(permuted[:N_CONCORDANT])
        groups = (("disputed", disputed), ("concordant", concordant))
        arm_keys = ("R", "C", "N")

    rows = []
    for group, cases in groups:
        for case_id in cases:
            for arm_key in arm_keys:
                res = arms[arm_key][case_id]
                rows.append({
                    "case": case_id,
                    "arm": arm_key,
                    "group": group,
                    "question": data[case_id]["question"],
                    "gold": gold_answer(data[case_id], res),
                    "category": data[case_id]["category"],
                    "reply": res.get("reply", ""),
                    "committed_endpoint": hard(res),
                })

    order = Rng(SEED + (2 if args.extend else 1)).shuffled(range(len(rows)))
    items, key = [], {}
    for slot, src in enumerate(order):
        r = rows[src]
        item_id = "{0}_{1:04d}".format(args.suffix if args.extend else "adj", slot)
        items.append({
            "item_id": item_id,
            "question": r["question"],
            "gold_answer": r["gold"],
            "reply": r["reply"],
        })
        key[item_id] = {"case": r["case"], "arm": r["arm"], "group": r["group"],
                        "category": r["category"],
                        "committed_endpoint": r["committed_endpoint"]}

    for path, payload in ((args.items, {"n": len(items), "items": items}),
                          (args.key, {"seed": SEED, "n_disputed": len(disputed),
                                      "n_concordant": len(concordant),
                                      "key": key})):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)

    print("disputed cases   {0}".format(0 if args.extend else len(disputed)))
    print("concordant cases {0} (of {1} all-correct){2}".format(
        len(concordant), len(all_correct),
        "  [section 7 top-up, disjoint from the first 40]" if args.extend else ""))
    print("replies to adjudicate {0}".format(len(items)))
    leak = [i for i in items if any(w in json.dumps(i)
                                    for w in ("c7-off", "c8-doc", "c9-sys", "arm"))]
    if leak:
        print("REFUSING: {0} items leak arm identity".format(len(leak)),
              file=sys.stderr)
        return 1
    print("wrote {0} and {1}".format(args.items, args.key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
