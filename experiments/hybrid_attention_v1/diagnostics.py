"""New, open synthetic DEV only. No external dataset/answer-key loading API."""
from collections import Counter, defaultdict
import math
import random
from .provenance import digest


def make_cases(tokenizer, lengths=(511, 512, 513, 1023, 1024, 1025), seed=17):
    rng = random.Random(seed)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    prefix = encode("Read the records. Return only the requested value. "
                    "If missing or conflicting, return UNKNOWN.\nRecords:\n")
    filler = encode(" irrelevant note.")
    if not filler:
        raise ValueError("Tokenizer produced empty filler")
    cases, seen = [], set()
    for length in lengths:
        for family in ("retrieval", "two_hop", "absent", "conflict"):
            for depth in (0.0, 0.5, 1.0):
                key = f"item{rng.randrange(100000, 999999)}"
                value = str(rng.randrange(10000, 99999))
                if family == "retrieval":
                    fact = f"\n{key} has value {value}.\n"
                    query, answer = f"What is the value of {key}?", value
                elif family == "two_hop":
                    fact = f"\n{key} points to relay. relay has value {value}.\n"
                    query, answer = f"What value is reached by following {key}?", value
                elif family == "absent":
                    fact = f"\n{key} has value {value}.\n"
                    query, answer = "What is the value of unlisted_item?", "UNKNOWN"
                else:
                    other = str(int(value) + 1)
                    fact = f"\n{key} has value {value}. {key} has value {other}.\n"
                    query, answer = f"What is the unambiguous value of {key}?", "UNKNOWN"
                core = encode(fact)
                tail = encode(f"\nQuestion: {query}\nAnswer:")
                extra = length - len(prefix) - len(core) - len(tail)
                if extra < 0:
                    raise ValueError("Requested token length is shorter than task overhead")
                pad = (filler * (extra // len(filler) + 1))[:extra]
                at = round(depth * extra)
                ids = prefix + pad[:at] + core + pad[at:] + tail
                sha = digest(ids)
                if sha in seen:
                    raise ValueError("Duplicate diagnostic prompt")
                seen.add(sha)
                cases.append({"id": f"dev-{len(cases):04d}", "family": family,
                              "depth": depth, "tokens": length, "input_ids": ids,
                              # depth is a filler-insertion fraction, not absolute depth.
                              # Spans are zero-based, end-exclusive; prompts are unchanged.
                              "evidence_span": [len(prefix)+at, len(prefix)+at+len(core)],
                              "question_span": [length-len(tail), length],
                              "evidence_to_question_gap": extra-at,
                              "evidence_kind": "distractor" if family == "absent" else "support_or_conflict",
                              "answer": answer, "answerable": answer != "UNKNOWN",
                              "prompt_sha256": sha, "seed": seed,
                              "split": "open_synthetic_dev"})
    return cases


def score(case, text, generation):
    # Never silently strip an answer prefix, case-fold, or collapse UNKNOWN loops.
    trimmed = text.strip()
    counts = Counter(generation["token_ids"])
    return {
        "id": case["id"], "family": case["family"], "depth": case["depth"],
        "tokens": case["tokens"], "answerable": case["answerable"],
        "prompt_sha256": case["prompt_sha256"], "answer": case["answer"],
        "text": text, "strict_exact": text == case["answer"],
        "trimmed_exact": trimmed == case["answer"], "abstained": trimmed == "UNKNOWN",
        "empty": not trimmed, "generated_tokens": len(generation["token_ids"]),
        "max_token_repeat": max(counts.values(), default=0),
        "stopped_eos": generation["stopped_eos"],
        "budget_exhausted": generation["budget_exhausted"],
    }


def summarize(rows):
    def cell(rs):
        n = len(rs)
        a = [r for r in rs if r["answerable"]]
        u = [r for r in rs if not r["answerable"]]
        rate = lambda key, data: sum(r[key] for r in data) / len(data) if data else None
        return {"n": n, "strict_exact": rate("strict_exact", rs),
                "trimmed_exact": rate("trimmed_exact", rs),
                "answerable_accuracy": rate("trimmed_exact", a),
                "false_abstention_rate": rate("abstained", a),
                "unanswerable_abstention_rate": rate("abstained", u),
                "budget_exhaustion_rate": rate("budget_exhausted", rs),
                "empty_rate": rate("empty", rs)}
    groups = defaultdict(list)
    for row in rows:
        groups[f'{row["family"]}|N={row["tokens"]}|depth={row["depth"]}'].append(row)
    return {"overall": cell(rows), "cells": {k: cell(v) for k, v in sorted(groups.items())}}


def paired_result(base, variant):
    if not base or len(base) != len(variant):
        raise ValueError("Paired comparison requires equal nonzero row counts")
    b = {r["id"]: r for r in base}
    v = {r["id"]: r for r in variant}
    if len(b) != len(base) or len(v) != len(variant) or b.keys() != v.keys():
        raise ValueError("Duplicate or unmatched case IDs")
    wins = losses = 0
    for key in b:
        if (b[key]["prompt_sha256"], b[key]["answer"]) != (v[key]["prompt_sha256"], v[key]["answer"]):
            raise ValueError("Case content differs")
        wins += bool(v[key]["trimmed_exact"] and not b[key]["trimmed_exact"])
        losses += bool(b[key]["trimmed_exact"] and not v[key]["trimmed_exact"])
    d = wins + losses
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(min(wins, losses)+1)) / 2**d) if d else 1.0
    return {"n": len(base), "wins": wins, "losses": losses,
            "accuracy_delta": (wins-losses)/len(base), "mcnemar_exact_two_sided": p,
            "inference_scope": "exploratory correlated synthetic DEV; no significance/promotion claim"}
