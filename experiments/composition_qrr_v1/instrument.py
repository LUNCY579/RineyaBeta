"""Open-DEV worlds, independently implemented oracles, and explicit scoring.

No existing training corpus or sealed asset is opened. Gold stays outside prompts.
"""
from __future__ import annotations
from collections import Counter
import random
import re
from .identity import ContractError, digest, positive_int

OPS = ("lookup", "next", "next2", "count_red", "count_blue", "missing")
RENDERERS = ("table", "prose")
ARMS = ("direct", "oracle_evidence", "oracle_binding")


def validate_world(world):
    rows = world["rows"]
    if not 4 <= len(rows) <= 16:
        raise ContractError("World size outside registered bounds")
    codes = [r["code"] for r in rows]
    if len(set(codes)) != len(codes) or world["absent"] in codes:
        raise ContractError("Codes must be unique; absent code must be absent")
    for r in rows:
        if not re.fullmatch(r"C[0-9a-f]{8}", r["code"]):
            raise ContractError("Invalid code")
        if not re.fullmatch(r"V[0-9a-f]{8}", r["vault"]):
            raise ContractError("Invalid vault")
        if r["tag"] not in ("red", "blue"):
            raise ContractError("Invalid tag")
    if not re.fullmatch(r"C[0-9a-f]{8}", world["absent"]):
        raise ContractError("Invalid absent code")
    if world["cue"] not in codes[:-2]:
        raise ContractError("Cue must have two successors")
    i = codes.index(world["cue"])
    if len({rows[j]["vault"] for j in (i, i + 1, i + 2)}) != 3:
        raise ContractError("Counterfactual operator answers must differ")
    if sum(r["tag"] == "red" for r in rows) * 2 == len(rows):
        raise ContractError("Count counterfactual must change the answer")


def make_world(seed, index):
    rng = random.Random(f"qrr-world-v1:{seed}:{index}")
    size = rng.randint(5, 12)
    def names(prefix, n):
        return [prefix + f"{x:08x}" for x in rng.sample(range(2**32), n)]
    codes, vaults = names("C", size + 1), names("V", size)
    # Independent label permutations: lexical order cannot reveal row order.
    rng.shuffle(codes)
    rng.shuffle(vaults)
    tags = [rng.choice(("red", "blue")) for _ in range(size)]
    if tags.count("red") * 2 == size:
        tags[0] = "blue" if tags[0] == "red" else "red"
    world = {"rows": [dict(code=c, vault=v, tag=t)
                      for c, v, t in zip(codes[:size], vaults, tags)],
             "cue": codes[rng.randrange(size - 2)], "absent": codes[-1]}
    validate_world(world)
    return world


def solve(world, op):
    """List-based executable specification."""
    validate_world(world)
    if op not in OPS:
        raise ContractError("Unknown operator")
    rows = world["rows"]
    if op.startswith("count_"):
        return str(sum(1 for row in rows if row["tag"] == op[6:]))
    code = world["absent"] if op == "missing" else world["cue"]
    step = {"next": 1, "next2": 2}.get(op, 0)
    for i, row in enumerate(rows):
        if row["code"] == code:
            return rows[i + step]["vault"] if i + step < len(rows) else "UNKNOWN"
    return "UNKNOWN"


def reference_solve(world, op):
    """Separate map/edge traversal implementation, not a call to solve()."""
    by_code = {r["code"]: r for r in world["rows"]}
    successor = {a["code"]: b["code"]
                 for a, b in zip(world["rows"], world["rows"][1:])}
    if op in ("count_red", "count_blue"):
        counts = Counter(r["tag"] for r in by_code.values())
        return str(counts[op.split("_")[1]])
    if op not in OPS:
        raise ContractError("Unknown operator")
    current = world["absent"] if op == "missing" else world["cue"]
    for _ in range({"next": 1, "next2": 2}.get(op, 0)):
        current = successor.get(current)
    return by_code[current]["vault"] if current in by_code else "UNKNOWN"


def question(world, op):
    if op.startswith("count_"):
        return f"How many registry rows have tag {op[6:]}?"
    cue = world["absent"] if op == "missing" else world["cue"]
    if op == "next":
        return f"What vault belongs to the row immediately after code {cue}?"
    if op == "next2":
        return f"What vault belongs to the row two rows after code {cue}?"
    return f"What vault belongs to code {cue}?"


def support(world, op):
    if op.startswith("count_") or op == "missing":
        return list(range(len(world["rows"])))
    i = next(i for i, r in enumerate(world["rows"]) if r["code"] == world["cue"])
    return list(range(i, i + {"lookup": 1, "next": 2, "next2": 3}[op]))


def prompt_for(case, arm="direct"):
    if arm not in ARMS:
        raise ContractError("Unregistered diagnostic arm")
    world, op = case["world"], case["op"]
    indices = range(len(world["rows"])) if arm == "direct" else support(world, op)
    lines = []
    for i in indices:
        r = world["rows"][i]
        if case["renderer"] == "table":
            lines.append(f"row {i + 1} | {r['code']} | {r['vault']} | {r['tag']}")
        elif case["renderer"] == "prose":
            lines.append(f"At original row {i + 1}, code {r['code']} has vault {r['vault']} and tag {r['tag']}.")
        else:
            raise ContractError("Unregistered renderer")
    extra = ""
    if arm == "oracle_binding" and op in ("lookup", "next", "next2"):
        extra = f"\nOracle binding: the target is original row {support(world, op)[-1] + 1}."
    header = "Registry columns: original row number, code, vault, tag. Row order is significant.\n"
    if arm != "direct":
        header += "Oracle diagnostic: all rows needed for this question are supplied, with original positions.\n"
    filler = [f"Background note {i}: this note is not a registry row.\n"
              for i in range(case["padding_lines"])]
    cut = {"start": 0, "middle": len(filler) // 2, "end": len(filler)}[case["placement"]]
    evidence = header + "\n".join(lines) + "\n"
    return ("Use only the registry. Reply only with a vault ID, an integer, or UNKNOWN if absent.\n"
            + "".join(filler[:cut]) + evidence + "".join(filler[cut:]) + extra
            + "\nQuestion: " + question(world, op) + "\nAnswer:")


def build_bank(worlds=8, seed=17, padding_lines=0, placement="middle"):
    positive_int(worlds, "worlds", 64)
    if type(seed) is not int or type(padding_lines) is not int or not 0 <= padding_lines <= 128:
        raise ContractError("Invalid seed or padding")
    if placement not in ("start", "middle", "end"):
        raise ContractError("Invalid placement")
    cases = []
    for i in range(worlds):
        world = make_world(seed, i)
        cluster = digest(world)
        for op in OPS:
            gold = solve(world, op)
            if gold != reference_solve(world, op):
                raise ContractError("Independent oracle disagreement")
            for renderer in RENDERERS:
                case = {"world": world, "cluster_id": cluster, "op": op,
                        "renderer": renderer, "padding_lines": padding_lines,
                        "placement": placement, "split": "open_dev", "gold": gold}
                case["case_id"] = digest({k: v for k, v in case.items() if k != "gold"})
                case["prompt_sha256"] = digest(prompt_for(case))
                cases.append(case)
    return {"schema": "qrr.bank/1", "split": "open_dev", "seed": seed,
            "cases": cases, "cases_sha256": digest(cases), "worlds": worlds}


def audit_bank(bank):
    if bank.get("schema") != "qrr.bank/1" or bank.get("split") != "open_dev":
        raise ContractError("Only this programme's open DEV schema is accepted")
    cases = bank["cases"]
    if not cases or len(cases) > 768 or digest(cases) != bank["cases_sha256"]:
        raise ContractError("Invalid or changed bank")
    ids, prompts, groups = set(), set(), {}
    for c in cases:
        if (type(c["padding_lines"]) is not int or not 0 <= c["padding_lines"] <= 128
                or c["placement"] not in ("start", "middle", "end")):
            raise ContractError("Unregistered length/position variant")
        validate_world(c["world"])
        if c["split"] != "open_dev" or c["cluster_id"] != digest(c["world"]):
            raise ContractError("Invalid world cluster or split")
        expected_id = digest({k: v for k, v in c.items()
                              if k not in ("gold", "case_id", "prompt_sha256")})
        if expected_id != c["case_id"]:
            raise ContractError("Changed case identity")
        if c["gold"] != solve(c["world"], c["op"]) or c["gold"] != reference_solve(c["world"], c["op"]):
            raise ContractError("Gold/oracle mismatch")
        p = digest(prompt_for(c))
        if p != c["prompt_sha256"] or c["case_id"] in ids or p in prompts:
            raise ContractError("Changed or duplicated prompt/case")
        ids.add(c["case_id"])
        prompts.add(p)
        groups.setdefault(c["cluster_id"], []).append((c["op"], c["renderer"]))
    expected = {(op, renderer) for op in OPS for renderer in RENDERERS}
    if len(groups) != bank["worlds"] or any(set(v) != expected or len(v) != 12 for v in groups.values()):
        raise ContractError("Incomplete world packs")
    return {"passed": True, "worlds": len(groups), "cases": len(cases),
            "oracle_agreement": True, "split": "open_dev",
            "limitations": "Two renderers; no held-out language or structural generalization claim"}


def score(case, text):
    if not isinstance(text, str):
        raise ContractError("Prediction must be text")
    # Explicit format equivalence only. No substring search or gold-dependent rescue.
    m = re.fullmatch(r"(?:(?:Answer:|vault)\s*)?(V[0-9a-f]{8}|[0-9]+|UNKNOWN)", text.strip())
    parsed = m.group(1) if m else None
    if parsed is not None and parsed.isdecimal():
        parsed = str(int(parsed))
    gold = case["gold"]
    correct = parsed == gold
    abstained = parsed == "UNKNOWN"
    cue = solve(case["world"], "lookup")
    wrong_cue = case["op"] in ("next", "next2") and parsed == cue
    category = ("correct" if correct else "false_abstention" if abstained
                else "parse_failure" if parsed is None else "cue_substitution" if wrong_cue
                else "wrong_answer")
    return {"raw_text": text, "parsed": parsed, "literal_exact": text == gold,
            "trimmed_exact": text.strip() == gold, "semantic_correct": correct,
            "answerable": gold != "UNKNOWN", "abstained": abstained,
            "cue_substitution": wrong_cue, "error_category": category}
