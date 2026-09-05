"""Resample topics, not cases.

The 600-case rag_long4k slice is drawn from 47 distinct `topic_id`s, up to 14
cases each, and the evidence documents are reused across every case on a topic.
A paired exact McNemar over 600 rows treats them as 600 independent draws, so its
p-value is computed against a null that does not hold: if compression helps or
hurts on a topic, it does so on all fourteen of that topic's cases at once.

This reports the same contrast resampling TOPICS with replacement, which is the
unit the corpus actually varies. Where the two disagree, the cluster interval is
the one that answers the question asked.

    python scripts/design/cluster_bootstrap.py
    python scripts/design/cluster_bootstrap.py --check
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
RESAMPLES = 10000
SEED = 20260826

ABSTAIN_CATEGORIES = frozenset({
    "ambiguous_evidence", "missing_answer", "conflict", "partial_support",
    "procedure_missing_input", "procedure_conflict",
})

# (label, arm, reference). Reference must be same-checkpoint; see
# audit_power_ceiling.py for why long16k-off-baya-ad is excluded everywhere.
CONTRASTS = [
    ("headline B=256", "c9-sys-B256__rag_long4k", "c7-off-n600__rag_long4k"),
    ("headline keep0.05", "c8-docbm25sys-k0.05-n600__rag_long4k", "c7-off-n600__rag_long4k"),
    ("line-sys B=256", "c13-line-sys-B256__rag_long4k", "c7-off-n600__rag_long4k"),
    ("chunk-sys B=256", "c10-chunk-sys-B256__rag_long4k", "c7-off-n600__rag_long4k"),
    ("hybrid B=256", "c18-hybrid-B256__rag_long4k", "c7-off-n600__rag_long4k"),
    ("B=512", "c9-sys-B512__rag_long4k", "c7-off-n600__rag_long4k"),
    ("B=196", "c9-sys-B196__rag_long4k", "c7-off-n600__rag_long4k"),
    ("B=160", "c9-sys-B160__rag_long4k", "c7-off-n600__rag_long4k"),
    # HEAD-TO-HEAD. Every row above pairs an arm against uncompressed, which is
    # why the two cells carrying claim C4 -- the score used as a tie-break
    # inside a BM25-chosen set -- were invisible to this tool and to
    # audit_power_ceiling.py, which enumerates arm-vs-eviction-off only. C4 is
    # what justified retiring this programme's original mechanism, so it should
    # not have been the least-audited contrast in the repo.
    ("C4 tiebreak B=256", "c18-hybrid-B256__rag_long4k", "c9-sys-B256__rag_long4k"),
    ("C4 tiebreak B=160", "c18-hybrid-B160__rag_long4k", "c9-sys-B160__rag_long4k"),
    # C6: segmentation, head to head rather than each against uncompressed.
    ("C6 newline vs record", "c13-line-sys-B256__rag_long4k", "c9-sys-B256__rag_long4k"),
    ("C6 chunk vs record", "c10-chunk-sys-B256__rag_long4k", "c9-sys-B256__rag_long4k"),
]


class Rng:
    """Deterministic LCG, so the interval is reproducible without numpy."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next_below(self, n):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n


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
                out[row["id"]] = (1 if float(row.get(ENDPOINT) or 0.0) >= HARD
                                  else 0)
    return out


def topics():
    out = {}
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = (row.get("topic_id"), row.get("category"))
    return out


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def analyse(arm_scores, ref_scores, meta, ids, resamples=RESAMPLES, seed=SEED):
    by_topic = {}
    for i in ids:
        by_topic.setdefault(meta[i][0], []).append(i)
    keys = sorted(by_topic)

    diffs = [arm_scores[i] - ref_scores[i] for i in ids]
    point = 100.0 * sum(diffs) / len(diffs)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)

    # per-topic mean difference, the quantity the bootstrap resamples
    topic_mean = {k: sum(arm_scores[i] - ref_scores[i] for i in by_topic[k])
                     / len(by_topic[k]) for k in keys}
    topic_n = {k: len(by_topic[k]) for k in keys}

    rng = Rng(seed)
    draws = []
    n_topics = len(keys)
    for _ in range(resamples):
        num = den = 0.0
        for _ in range(n_topics):
            k = keys[rng.next_below(n_topics)]
            num += topic_mean[k] * topic_n[k]
            den += topic_n[k]
        draws.append(100.0 * num / den)
    draws.sort()

    lo, hi = percentile(draws, 0.025), percentile(draws, 0.975)
    # bootstrap two-sided p: how often does a resample cross zero?
    n_le = sum(1 for d in draws if d <= 0.0)
    p_boot = min(1.0, 2.0 * min(n_le, resamples - n_le) / resamples)
    return {
        "n_cases": len(ids),
        "n_topics": n_topics,
        "max_cases_per_topic": max(topic_n.values()),
        "delta_pp": round(point, 4),
        "wins": wins,
        "losses": losses,
        "p_mcnemar_case_level": round(exact_mcnemar(wins, losses), 8),
        "cluster_ci95_pp": [round(lo, 4), round(hi, 4)],
        "p_cluster_bootstrap": round(p_boot, 5),
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }


def run():
    meta = topics()
    ref_cache = {}
    out = {"resamples": RESAMPLES, "seed": SEED, "contrasts": {}}
    hdr = ("{0:20s} {1:8s} {2:>7s} {3:>7s} {4:>4s} {5:>4s} {6:>11s} {7:>20s} {8:>9s}"
           .format("contrast", "stratum", "n", "topics", "W", "L", "p_case",
                   "cluster 95% CI (pp)", "p_clust"))
    print(hdr)
    print("-" * len(hdr))
    for label, arm_name, ref_name in CONTRASTS:
        arm = load(arm_name)
        ref = ref_cache.setdefault(ref_name, load(ref_name))
        if arm is None or ref is None:
            print("{0:20s} MISSING".format(label))
            continue
        common = sorted(set(arm) & set(ref) & set(meta))
        for stratum, sub in (
                ("answer", [i for i in common
                            if meta[i][1] not in ABSTAIN_CATEGORIES]),
                ("abstain", [i for i in common
                             if meta[i][1] in ABSTAIN_CATEGORIES]),
                ("pooled", common)):
            if not sub:
                continue
            res = analyse(arm, ref, meta, sub)
            out["contrasts"].setdefault(label, {})[stratum] = dict(
                res, arm=arm_name, reference=ref_name)
            print("{0:20s} {1:8s} {2:7d} {3:7d} {4:4d} {5:4d} {6:11.3g} "
                  "[{7:+7.2f}, {8:+7.2f}] {9:9.4f}{10}".format(
                      label, stratum, res["n_cases"], res["n_topics"],
                      res["wins"], res["losses"], res["p_mcnemar_case_level"],
                      res["cluster_ci95_pp"][0], res["cluster_ci95_pp"][1],
                      res["p_cluster_bootstrap"],
                      "  CI CROSSES ZERO" if res["crosses_zero"] else ""))
        print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=os.path.join(RESULTS, "cluster_bootstrap.json"))
    args = ap.parse_args()

    out = run()
    if args.check:
        if not os.path.exists(args.json):
            print("no committed bootstrap at " + args.json, file=sys.stderr)
            return 1
        with open(args.json, encoding="utf-8") as fh:
            old = json.load(fh)
        if old != out:
            print("MISMATCH against committed cluster bootstrap", file=sys.stderr)
            return 1
        print("committed cluster bootstrap reproduces")
        return 0

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
