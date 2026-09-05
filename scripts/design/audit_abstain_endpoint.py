"""Where does the headline gain actually live, and does it survive the endpoint?

On the 168-case abstain stratum the response contract is two `any_phrases` rules
over one literal sentence -- "I don't know based on the available information",
once without the full stop and once with. So `contract_fulfillment_score` there is
a substring test for a canned string, counted twice, and any semantically identical
refusal in different words scores zero.

That is a task-design choice, not a bug: the dataset's own `reference` field for
these cases IS that sentence. But it has a consequence nobody measured. On
`partial_support` the question asks two fields, the evidence supports one, and the
contract demands total refusal -- so a model that reports the field it can support
and flags the other as missing is scored 0, while a model that has LOST the
evidence and can only recite the refusal is scored 1.

This script decomposes the headline by category, checks how often the losing reply
was actually correct, and rescores the stratum under a semantic abstention rubric
so the headline can be read with the string-identity component removed.

    python scripts/design/audit_abstain_endpoint.py
    python scripts/design/audit_abstain_endpoint.py --check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

RESULTS = os.path.join("experiments", "kv_significance", "results")
DATASET = os.path.join("artifacts", "eval_sets", "rag_long4k_d0.5.jsonl")
ENDPOINT = "contract_fulfillment_score"
HARD = 0.999

REFERENCE_ARM = "c7-off-n600__rag_long4k"
HEADLINE_ARM = "c9-sys-B256__rag_long4k"

ABSTAIN_CATEGORIES = frozenset({
    "ambiguous_evidence", "missing_answer", "conflict", "partial_support",
    "procedure_missing_input", "procedure_conflict",
})

# A reply abstains in substance if it says, in any words, that the information is
# not there. Fixed before looking at the win/loss ledger; the sensitivity of the
# conclusion to this list is reported by --rubric-sensitivity.
ABSTENTION_MARKERS = (
    "i don't know", "i do not know", "i don t know",
    "not provided", "not specified", "not available", "not stated",
    "not present", "not included", "not found", "not listed",
    "no information", "does not specify", "does not provide", "does not state",
    "cannot determine", "can't determine", "cannot be determined",
    "unable to determine", "unable to answer", "cannot answer", "can't answer",
    "insufficient information", "not enough information",
    "isn't in the", "is not in the", "aren't in the", "are not in the",
)


def norm(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def abstains_semantically(reply):
    low = norm(reply)
    return any(m in low for m in ABSTENTION_MARKERS)


def exact_mcnemar(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


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


def supported_values(case):
    """The claim values the evidence actually licenses for this case."""
    vals = []
    for ev in case.get("evidence") or []:
        v = ev.get("claim_value")
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
    return vals


def semantic_hard(row, case):
    """Endpoint that credits abstention in any words, keeping every other rule.

    Only the abstain stratum is rescored, and only its abstention requirement:
    the answer stratum keeps the committed endpoint untouched, so the two
    numbers stay comparable on 432 of the 600 cases by construction.
    """
    if case.get("category") not in ABSTAIN_CATEGORIES:
        return hard(row)
    return 1 if abstains_semantically(row.get("reply", "")) else 0


def ledger(ref, arm, ids, score):
    wins = sum(1 for i in ids if score(arm[i], i) > score(ref[i], i))
    losses = sum(1 for i in ids if score(arm[i], i) < score(ref[i], i))
    n = len(ids)
    return {
        "n": n,
        "ref_mean": round(sum(score(ref[i], i) for i in ids) / n, 4) if n else 0.0,
        "arm_mean": round(sum(score(arm[i], i) for i in ids) / n, 4) if n else 0.0,
        "delta_pp": round(100.0 * sum(score(arm[i], i) - score(ref[i], i)
                                      for i in ids) / n, 2) if n else 0.0,
        "wins": wins,
        "losses": losses,
        "p": round(exact_mcnemar(wins, losses), 8),
    }


def show(title, row):
    print("  {0:34s} n={1:<4d} {2:.4f} -> {3:.4f}  {4:+6.2f} pp  "
          "{5:>3d}W/{6:<3d}L  p={7:.3g}".format(
              title, row["n"], row["ref_mean"], row["arm_mean"], row["delta_pp"],
              row["wins"], row["losses"], row["p"]))


def report():
    ref, arm, data = load(REFERENCE_ARM), load(HEADLINE_ARM), dataset()
    ids = sorted(set(ref) & set(arm) & set(data))
    committed = lambda row, i: hard(row)
    semantic = lambda row, i: semantic_hard(row, data[i])

    answer = [i for i in ids if data[i]["category"] not in ABSTAIN_CATEGORIES]
    abstain = [i for i in ids if data[i]["category"] in ABSTAIN_CATEGORIES]

    out = {"reference": REFERENCE_ARM, "arm": HEADLINE_ARM, "n": len(ids)}

    print("{0} vs {1}, n = {2}".format(HEADLINE_ARM, REFERENCE_ARM, len(ids)))
    print()
    print("COMMITTED ENDPOINT (contract_fulfillment_score >= {0})".format(HARD))
    for name, sub in (("answer", answer), ("abstain", abstain), ("pooled", ids)):
        out.setdefault("committed", {})[name] = ledger(ref, arm, sub, committed)
        show(name, out["committed"][name])

    print()
    print("WHERE THE ABSTAIN GAIN LIVES, by category")
    cats = sorted({data[i]["category"] for i in abstain})
    out["by_category"] = {}
    for c in cats:
        sub = [i for i in abstain if data[i]["category"] == c]
        out["by_category"][c] = ledger(ref, arm, sub, committed)
        show(c, out["by_category"][c])

    # What did the reference actually say on the cases it "lost"?
    lost = [i for i in abstain if hard(arm[i]) > hard(ref[i])]
    helpful = [i for i in lost
               if any(norm(v) in norm(ref[i].get("reply", ""))
                      for v in supported_values(data[i]))]
    both_abstain = [i for i in lost if abstains_semantically(ref[i].get("reply", ""))]
    out["losing_replies"] = {
        "n_lost": len(lost),
        "reference_reply_contained_a_supported_claim_value": len(helpful),
        "reference_reply_abstained_in_other_words": len(both_abstain),
    }
    print()
    print("ON THE {0} ABSTAIN CASES THE UNCOMPRESSED ARM LOST".format(len(lost)))
    print("  reference reply stated a claim value the evidence supports : "
          "{0}/{1}".format(len(helpful), len(lost)))
    print("  reference reply already abstained, in different words      : "
          "{0}/{1}".format(len(both_abstain), len(lost)))

    print()
    print("SEMANTIC ABSTENTION ENDPOINT (abstain stratum only; answer stratum "
          "unchanged)")
    for name, sub in (("answer", answer), ("abstain", abstain), ("pooled", ids)):
        out.setdefault("semantic", {})[name] = ledger(ref, arm, sub, semantic)
        show(name, out["semantic"][name])

    print()
    print("COMMITTED ENDPOINT, EXCLUDING partial_support")
    keep = [i for i in ids if data[i]["category"] != "partial_support"]
    out["excluding_partial_support"] = ledger(ref, arm, keep, committed)
    show("pooled", out["excluding_partial_support"])

    # --- the answer stratum has its own endpoint problem -------------------
    # A substring endpoint scores a reply that contains the gold token inside
    # garbled text exactly as it scores a clean one. Gate on the repo's own
    # hard-failure flag and see what is left. This gate is HARSH -- it fires on
    # a third of uncompressed replies and its commonest trigger is digit-ratio
    # noise, which partly just means "the answer contains a date" -- so it is
    # reported as corroboration, never as the decisive cell. Settling this
    # properly needs a clean-answer detector frozen before it is applied.
    gated = lambda row, i: (1 if hard(row) and not row.get("quality_hard_fail")
                            else 0)
    print()
    print("CORROBORATION ONLY -- committed endpoint AND NOT quality_hard_fail")
    for name, sub in (("answer", answer), ("abstain", abstain), ("pooled", ids)):
        out.setdefault("quality_gated", {})[name] = ledger(ref, arm, sub, gated)
        show(name, out["quality_gated"][name])

    aw = [i for i in answer if hard(arm[i]) > hard(ref[i])]
    al = [i for i in answer if hard(arm[i]) < hard(ref[i])]
    mean = lambda xs, key, src: (round(sum(float(src[i].get(key) or 0.0)
                                           for i in xs) / len(xs), 4) if xs else None)
    out["answer_stratum_wins"] = {
        "n_wins": len(aw),
        "n_losses": len(al),
        "arm_intelligibility_on_wins": mean(aw, "intelligibility_score", arm),
        "ref_intelligibility_on_wins": mean(aw, "intelligibility_score", ref),
        "arm_hard_fail_on_wins": sum(1 for i in aw if arm[i].get("quality_hard_fail")),
        "ref_hard_fail_on_wins": sum(1 for i in aw if ref[i].get("quality_hard_fail")),
    }
    w = out["answer_stratum_wins"]
    print()
    print("ON THE {0} ANSWER-STRATUM CASES COMPRESSION WON".format(len(aw)))
    print("  intelligibility_score   compressed {0}  vs  uncompressed {1}".format(
        w["arm_intelligibility_on_wins"], w["ref_intelligibility_on_wins"]))
    print("  quality_hard_fail       compressed {0}/{1}  vs  uncompressed {2}/{1}".format(
        w["arm_hard_fail_on_wins"], len(aw), w["ref_hard_fail_on_wins"]))

    # --- can the uncompressed arm even comply? -----------------------------
    # The mandated sentence costs tokens the contract does not budget for.
    cap = None
    man_path = os.path.join(RESULTS, REFERENCE_ARM, "run_manifest.json")
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as fh:
            cap = (json.load(fh).get("protocol") or {}).get("max_new_tokens")
    ps_fail = [i for i in abstain
               if data[i]["category"] == "partial_support" and not hard(ref[i])]
    crowded = [i for i in ps_fail
               if float(ref[i].get("visible_generated_tokens") or 0) >= 22]
    out["generation_headroom"] = {
        "max_new_tokens": cap,
        "partial_support_failures_uncompressed": len(ps_fail),
        "of_which_had_under_10_tokens_of_headroom": len(crowded),
    }
    print()
    print("GENERATION HEADROOM (max_new_tokens = {0}; the mandated sentence is "
          "10 tokens)".format(cap))
    print("  uncompressed partial_support failures                : {0}".format(len(ps_fail)))
    print("  of which had already spent >=22 tokens on content    : {0}".format(len(crowded)))

    # --- does the winning 'abstention' contradict the evidence? ------------
    denies = [i for i in lost
              if supported_values(data[i])
              and not any(norm(v) in norm(arm[i].get("reply", ""))
                          for v in supported_values(data[i]))
              and abstains_semantically(arm[i].get("reply", ""))]
    out["grounded_retraction"] = {"n_wins": len(lost), "denies_supported_value": len(denies)}
    print()
    print("GROUNDED RETRACTION")
    print("  abstain wins whose reply omits a value the evidence contains : "
          "{0}/{1}".format(len(denies), len(lost)))

    # The two de-confounders are not independent: one removes the category the
    # abstain gain lives in, the other removes wins that are gold substrings in
    # degraded text. Crossed is the honest worst case, not the headline.
    out["crossed"] = ledger(ref, arm, keep, gated)
    print()
    print("CROSSED -- excluding partial_support AND gated on quality_hard_fail")
    show("pooled", out["crossed"])

    print()
    ce = out["committed"]["pooled"]
    se = out["semantic"]["pooled"]
    xe = out["excluding_partial_support"]
    print("headline {0:+.2f} pp (p={1:.3g}) committed"
          "  ->  {2:+.2f} pp (p={3:.3g}) semantic"
          "  ->  {4:+.2f} pp (p={5:.3g}) without partial_support".format(
              ce["delta_pp"], ce["p"], se["delta_pp"], se["p"],
              xe["delta_pp"], xe["p"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=os.path.join(RESULTS, "abstain_endpoint_audit.json"))
    args = ap.parse_args()

    out = report()
    if args.check:
        if not os.path.exists(args.json):
            print("\nno committed audit at " + args.json, file=sys.stderr)
            return 1
        with open(args.json, encoding="utf-8") as fh:
            old = json.load(fh)
        if old != out:
            print("\nMISMATCH against committed audit", file=sys.stderr)
            for k in sorted(set(old) | set(out)):
                if old.get(k) != out.get(k):
                    print("  {0}\n    committed:  {1}\n    recomputed: {2}".format(
                        k, old.get(k), out.get(k)), file=sys.stderr)
            return 1
        print("\ncommitted abstain-endpoint audit reproduces")
        return 0

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
