"""New, open synthetic DEV only. No external dataset/answer-key loading API.

Reporting deliberately separates registered endpoints from diagnostic-only metadata.
The latter exists to make failures attributable without changing prompts or answers.
"""
from collections import Counter, defaultdict
import math
import random
from .provenance import digest


DIAGNOSTIC_SCHEMA_VERSION = 2
# Case-set v2, registered 2026-09-09. v1 was (511, 512, 513, 1023, 1024, 1025):
# measured against the window rule, NONE of its 72 cases could put the answer
# outside a 1024-wide window, so A1_local1024 and A7_sink1024 were incapable of
# showing a window effect, and only 12-24 of 72 could show a 512-wide one.
# These lengths straddle the eviction threshold of BOTH registered windows and
# keep 512 as an all-visible control. Still 6 x 4 families x 3 depths = 72.
CASE_SET_VERSION = 2
DIAGNOSTIC_LENGTHS = (512, 1024, 1280, 1536, 1792, 2048)
SUPERSEDED_LENGTHS_V1 = (511, 512, 513, 1023, 1024, 1025)


def make_cases(tokenizer, lengths=DIAGNOSTIC_LENGTHS, seed=17):
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
                              "evidence_span": [len(prefix)+at, len(prefix)+at+len(core)],
                              "question_span": [length-len(tail), length],
                              "evidence_to_question_gap": extra-at,
                              "evidence_kind": "distractor" if family == "absent" else "support_or_conflict",
                              "answer": answer, "answerable": answer != "UNKNOWN",
                              "prompt_sha256": sha, "seed": seed,
                              "case_set_version": CASE_SET_VERSION,
                              "split": "open_synthetic_dev"})
    return cases


# Attribution metadata is mandatory: a row without it cannot be stratified by
# boundary, depth or evidence kind, which is the whole point of schema v2.
REQUIRED_CASE_FIELDS = (
    "id", "family", "depth", "tokens", "answerable", "prompt_sha256", "answer",
    "evidence_span", "question_span", "evidence_to_question_gap", "evidence_kind",
)


def score(case, text, generation):
    missing = [name for name in REQUIRED_CASE_FIELDS if name not in case]
    if missing:
        raise ValueError(
            f"Case lacks diagnostic schema v{DIAGNOSTIC_SCHEMA_VERSION} fields {missing}; "
            "scoring a case without attribution metadata is refused")
    trimmed = text.strip()
    counts = Counter(generation["token_ids"])
    if generation["stopped_eos"]:
        termination = "eos"
    elif generation["budget_exhausted"]:
        termination = "budget"
    else:
        termination = "other"
    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "id": case["id"], "family": case["family"], "depth": case["depth"],
        "tokens": case["tokens"], "answerable": case["answerable"],
        "prompt_sha256": case["prompt_sha256"], "answer": case["answer"],
        "evidence_span": case["evidence_span"], "question_span": case["question_span"],
        "evidence_to_question_gap": case["evidence_to_question_gap"],
        "evidence_kind": case["evidence_kind"],
        "text": text, "strict_exact": text == case["answer"],
        "trimmed_exact": trimmed == case["answer"], "abstained": trimmed == "UNKNOWN",
        "empty": not trimmed, "generated_tokens": len(generation["token_ids"]),
        "unique_generated_tokens": len(counts),
        "max_token_repeat": max(counts.values(), default=0),
        "stopped_eos": generation["stopped_eos"],
        "budget_exhausted": generation["budget_exhausted"],
        "termination": termination,
    }


def _cell(rows):
    n = len(rows)
    a = [r for r in rows if r["answerable"]]
    u = [r for r in rows if not r["answerable"]]
    rate = lambda key, data: sum(bool(r[key]) for r in data) / len(data) if data else None
    mean = lambda key: sum(r[key] for r in rows) / n if n else None
    repeats = [r["max_token_repeat"] for r in rows]
    return {
        "n": n,
        "strict_exact": rate("strict_exact", rows),
        "trimmed_exact": rate("trimmed_exact", rows),
        "answerable_accuracy": rate("trimmed_exact", a),
        "false_abstention_rate": rate("abstained", a),
        "unanswerable_abstention_rate": rate("abstained", u),
        "budget_exhaustion_rate": rate("budget_exhausted", rows),
        "empty_rate": rate("empty", rows),
        "stopped_eos_rate": rate("stopped_eos", rows),
        "generated_tokens_mean": mean("generated_tokens"),
        "max_token_repeat_mean": (sum(repeats) / n if n else None),
        "max_token_repeat_max": (max(repeats) if repeats else None),
    }


def summarize(rows):
    groups = defaultdict(list)
    families = defaultdict(list)
    lengths = defaultdict(list)
    depths = defaultdict(list)
    answerability = defaultdict(list)
    for row in rows:
        groups[f'{row["family"]}|N={row["tokens"]}|depth={row["depth"]}'].append(row)
        families[row["family"]].append(row)
        lengths[str(row["tokens"])].append(row)
        depths[str(row["depth"])].append(row)
        answerability["answerable" if row["answerable"] else "unanswerable"].append(row)
    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "overall": _cell(rows),
        "margins": {
            "by_family": {k: _cell(v) for k, v in sorted(families.items())},
            "by_length": {k: _cell(v) for k, v in sorted(lengths.items(), key=lambda x: int(x[0]))},
            "by_depth": {k: _cell(v) for k, v in sorted(depths.items(), key=lambda x: float(x[0]))},
            "by_answerability": {k: _cell(v) for k, v in sorted(answerability.items())},
        },
        "cells": {k: _cell(v) for k, v in sorted(groups.items())},
        "interpretation": "Margins and termination/repetition fields are diagnostic-only; registered contrasts remain authoritative and P3 remains exploratory open DEV.",
    }


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
