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
    args = ap.parse_args()

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
    concordant = sorted(rng.shuffled(all_correct)[:N_CONCORDANT])

    rows = []
    for group, cases in (("disputed", disputed), ("concordant", concordant)):
        for case_id in cases:
            for arm_key in ("R", "C", "N"):
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

    order = Rng(SEED + 1).shuffled(range(len(rows)))
    items, key = [], {}
    for slot, src in enumerate(order):
        r = rows[src]
        item_id = "adj_{0:04d}".format(slot)
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

    print("disputed cases   {0}".format(len(disputed)))
    print("concordant cases {0} (of {1} all-correct)".format(len(concordant),
                                                             len(all_correct)))
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
