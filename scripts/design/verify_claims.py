#!/usr/bin/env python3
"""Cross-check every headline number in the docs against the committed data.

WHY THIS EXISTS. Over this program five published figures turned out to be wrong,
and none of them were caught by reading the code:

  allocation gain          7.4%   ->  4.2%   (fabricated rho column, bad floor)
  SWA prefill speedup      3.2x   ->  1.9x SLOWER (kernel dispatch, not shape)
  SWA cache saving         5.1x   ->  1.00x resident (counted on a view)
  significance vs SLM      +0.204 ->  -0.083 at keep 0.25 (corrupted baseline)
  MDE at k=3               9.6pp  ->  13.9pp (normal-quantile multiplicity fix)

Each survived because a number in a markdown table and a number in a JSON file
could drift apart with nothing comparing them. This compares them.

Each claim states BOTH the documented value and how to recompute it from a
committed result file, and additionally asserts that the documented value still
literally appears in the prose of the document that is supposed to state it. So
editing the doc without the data fails, regenerating the data without the doc
fails, and moving a number to a different document fails too.

Exit code is non-zero on any mismatch, so this is usable as a pre-commit check.

Usage:
    python scripts/design/verify_claims.py
    python scripts/design/verify_claims.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KVS = ROOT / "experiments/kv_significance/results"
ATS = ROOT / "experiments/attention_shape/results"

# Each claim names the document that is supposed to state it. Searching all prose
# instead would let a number pass while the document that actually quotes it
# drifts, so the mapping is explicit.
DOCS = {
    "findings": ROOT / "docs/FINDINGS_KV_PROGRAM.md",
    "negative": ROOT / "docs/NEGATIVE_RESULTS.md",
    "paper": ROOT / "docs/PAPER_DRAFT_SIGNIFICANCE.md",
    "attention_shape": ROOT / "experiments/attention_shape/README.md",
    "kv_readme": ROOT / "experiments/kv_significance/README.md",
    "p2": ROOT / "experiments/stabilizer/results/FINDINGS_P2.md",
    "c2paper": ROOT / "docs/PAPER_DRAFT_CORNERSTONE2.md",
    "ledger": ROOT / "docs/CORNERSTONE_LEDGER.md",
    # These three carry claims C1-C7 and were guarded by NOTHING until
    # 2026-09-04, which is how "(4 + 104 + 63) / 0.85 = 196" survived in a
    # headline formula when the division is 201.18.
    "improves": ROOT / "docs/FINDINGS_COMPRESSION_IMPROVES_QUALITY.md",
    "floor": ROOT / "docs/FINDINGS_CAPACITY_FLOOR.md",
    "ident": ROOT / "docs/FINDINGS_INSTRUCTION_IDENTIFICATION.md",
    "adjud": ROOT / "docs/FINDINGS_BLIND_ADJUDICATION.md",
}


def adjud(*path) -> float:
    d = json.loads((KVS / "blind_adjudication.json").read_text(encoding="utf-8"))
    for k in path:
        d = d[k]
    return float(d)


def ident(arm_tag: str, stratum: str) -> float:
    """A stratum mean from the committed identification analysis.

    Sourced from a committed JSON rather than from results.jsonl, which is
    gitignored by repo policy, so this guard still works in a fresh clone.
    """
    d = json.loads((KVS / "instruction_identification.json").read_text(
        encoding="utf-8"))
    for arm, row in d["ladder"].items():
        if arm.startswith(arm_tag):
            return float(row[stratum])
    raise KeyError(arm_tag)


def ident_contrast(label: str, stratum: str, field: str) -> float:
    d = json.loads((KVS / "instruction_identification.json").read_text(
        encoding="utf-8"))
    return float(d["contrasts"][label][stratum][field])


def floor_stat(*path) -> float:
    """A number from the committed capacity-floor analysis.

    FINDINGS_CAPACITY_FLOOR.md was listed in DOCS but had ZERO pinned claims
    until 2026-09-05, so `verify_claims` only ever checked that the file still
    existed. That is the document carrying the 196 formula whose arithmetic was
    wrong, so it is exactly the one that needed pinning.
    """
    d = json.loads((KVS / "capacity_floor.json").read_text(encoding="utf-8"))
    for k in path:
        d = d[str(k)]
    return float(d)


def abstain_audit(*path) -> float:
    d = json.loads((KVS / "abstain_endpoint_audit.json").read_text(
        encoding="utf-8"))
    for k in path:
        d = d[k]
    return float(d)


ARITHMETIC = re.compile(
    r"\(\s*([0-9][0-9\s\+\-\*/\.]*?)\s*\)\s*/\s*([0-9.]+)\s*=\s*([0-9.]+)")


def check_inline_arithmetic(text_by_doc, tol=0.51):
    """Evaluate every `(a + b + c) / d = e` written in prose.

    verify_claims compares a documented scalar to a JSON scalar. It never
    evaluated arithmetic stated inline, which is how
    `(4 + 104 + 63) / 0.85 = 196` survived in a headline formula when the
    division is 201.18. Tolerance is half a unit so a value rounded to the
    nearest integer still passes.
    """
    bad = []
    for name, text in sorted(text_by_doc.items()):
        for m in ARITHMETIC.finditer(text):
            # A retraction has to be allowed to quote the arithmetic it is
            # retracting. `(sic)` on the same line marks a formula reproduced as
            # written rather than asserted, which is the scholarly convention
            # and keeps the checker from flagging its own correction notes.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start:line_end if line_end != -1 else len(text)]
            if "(sic)" in line:
                continue
            expr, denom, stated = m.group(1), m.group(2), m.group(3)
            try:
                got = eval(expr, {"__builtins__": {}}, {}) / float(denom)  # noqa: S307
            except Exception:                                   # noqa: BLE001
                continue
            if abs(got - float(stated)) > tol:
                bad.append((name, m.group(0).strip(), got))
    return bad


def summary(base: Path, tag: str, ds: str = "rag_dev"):
    p = base / f"{tag}__{ds}" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["mean_quality_score"]


def measured(path: Path, key: str, _require: dict | None = None, **match):
    """Value from a committed result row, with the file's provenance asserted.

    Three ways this used to hand back a plausible wrong number. All are now fatal.

    THE FILE IS QUARANTINED. Results under _invalidated_* came from a build with a
    known prefill bug. A one-directory path slip read them silently. One of them
    carries score_pct_of_cache 0.781 at budget=1024 -- the same value the 0.78%
    overhead claim pins -- so the slip would not even change the answer, and
    nothing downstream would look wrong.

    MORE THAN ONE ROW MATCHES. `match` is often a budget alone, and a file may
    hold that budget at several lengths; the quarantined file above has 0.781 at
    budget=1024 in TWO rows. Returning the first made the answer depend on row
    order.

    THE FILE IS NOT THE CONFIGURATION THE CLAIM IS ABOUT. pk_128k_pool7_B256.json
    and pk_128k_pool7_B256_rf.json share length, budget and pool, differ only in
    `recent` (512 against 64), and report accuracy 0.00 against 1.00. Selecting by
    filename picked the wrong one during this program. Filenames do not carry
    configuration; the manifest does. Pass _require to assert the axes the claim
    depends on, e.g. _require={"recent": 64, "pool": 7}.
    """
    if not path.exists():
        return None
    if any(part.startswith("_invalidated") for part in path.parts):
        raise AssertionError(
            f"{path.name}: quarantined result, not admissible as evidence")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for k, want in (_require or {}).items():
        got = doc.get(k)
        if got != want:
            raise AssertionError(
                f"{path.name}: manifest {k}={got!r} but the claim requires "
                f"{want!r} -- this file is a different configuration")
    hits = [r for r in doc["results"]
            if all(r.get(k) == v for k, v in match.items())]
    if len(hits) > 1:
        raise AssertionError(
            f"{path.name}: {len(hits)} rows match {match} -- the selection is "
            f"ambiguous. Add a field that separates them.")
    return hits[0].get(key) if hits else None


def probe(fname: str, *keys):
    """Value at a nested key path inside a committed probe result."""
    path = KVS / fname
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def cla_mean():
    """Mean HELD-OUT relative error of the cross-layer KV adapter.

    Held-out, not train: the adapter is a 2048x2048 ridge fit whose train error is
    roughly half its test error, so quoting the fit would make cross-layer sharing
    look twice as viable as it is.
    """
    rows = probe("store_probe.json", "cla")
    if not rows:
        return None
    return sum(r["test"] for r in rows) / len(rows)


def pk(fname: str, depth: float | None = None, budget: int | None = None,
       length: int | None = None, require: dict | None = None):
    """Passkey accuracy from a committed arm; pooled over depths if depth is None.

    Skips cells whose acc is null -- those are trials that ran out of memory and
    measured nothing. Averaging a null as zero is the exact mistake the harness
    itself used to make.

    `require` asserts manifest fields, for the same reason measured() does: two
    files here differ only in `recent` and report 0.00 against 1.00. Any pin whose
    file has a same-(length, budget, pool) twin should state the axis that
    separates them rather than trusting the filename.

    NOTE ON n. Every depth of a cell draws the SAME key sequence -- deliberately,
    so depths are paired on needles (see passkey.py). A cell of `trials` keys at
    three depths is therefore `trials` independent needles observed three times,
    NOT 3*trials independent needles. Pooling over depth does not increase n, and
    a confidence bound computed as if it did is anti-conservative.
    """
    path = KVS / fname
    if not path.exists():
        return None
    if any(part.startswith("_invalidated") for part in path.parts):
        raise AssertionError(
            f"{path.name}: quarantined result, not admissible as evidence")
    d = json.loads(path.read_text(encoding="utf-8"))
    for k, want in (require or {}).items():
        got = d.get(k)
        if got != want:
            raise AssertionError(
                f"{path.name}: manifest {k}={got!r} but the claim requires "
                f"{want!r} -- this file is a different configuration")
    rows = [r for r in d["results"] if r.get("acc") is not None]
    if depth is not None:
        rows = [r for r in rows if r["depth"] == depth]
    if budget is not None:
        rows = [r for r in rows if r["budget"] == budget]
    if length is not None:
        rows = [r for r in rows if r["length"] == length]
    if not rows:
        return None
    return sum(r["acc"] for r in rows) / len(rows)


def lang(label: str, field: str):
    """A cornerstone-2 natural-language contrast, from the COMMITTED artifact.

    Same reasoning as identical_replies(): the per-case records these are derived
    from are gitignored, so a check that read them would pass here and fail in a
    fresh clone. scripts/design/lang_contrasts.py derives them into a few kB of
    committed JSON; run it with --check to confirm the artifact still matches the
    raw records when those are present.
    """
    path = KVS / "lang_contrasts.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))["contrasts"].get(label)
    return None if not row or row.get("status") else row[field]


def lang_quality(tag: str):
    """Mean quality of one rag_dev arm, read from its own summary.json."""
    path = KVS / f"{tag}__rag_dev" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("mean_quality_score")


def exempt_cells() -> int:
    """Committed compressed cells whose needle faced ZERO compressions.

    chunked_prefill never compresses the final chunk. A needle inside it faces
    no compression at all, so the cell measures the model rather than the
    policy -- a policy keeping NOTHING would score the same. Three such cells
    sit inside the published Halo 16K transfer claim.

    This pin is a CENSUS, not a value: it fails if the count moves in either
    direction, which forces anyone adding an exempt cell to justify it and
    anyone removing one to update the documents.
    """
    import math
    n = 0
    for path in KVS.glob("*.json"):
        if any(part.startswith("_invalidated") for part in path.parts):
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or not d.get("chunk"):
            continue
        for r in d.get("results", []):
            if not isinstance(r, dict) or r.get("acc") is None:
                continue
            rt, depth, bud = r.get("real_tokens"), r.get("depth"), r.get("budget")
            if not rt or depth is None or not bud:
                continue
            nch = math.ceil(rt / d["chunk"])
            if max(0, (nch - 1) - (int(depth * rt) // d["chunk"])) == 0:
                n += 1
    return n


def lang_quality2(tag: str, dataset: str):
    """Mean quality of one arm on a NON-rag_dev dataset (the long-context sets)."""
    path = KVS / f"{tag}__{dataset}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("mean_quality_score")


def _cat_gain(category: str):
    """Per-category quality gain of the best allocation over matched random.

    Reads the committed `per_category.json` aggregates, NOT the per-case
    `results.jsonl`. The jsonl is excluded by .gitignore (`experiments/**/
    results/**/results.jsonl`), so the earlier per-case implementation made
    these two claims verifiable only on the machine that produced them and
    MISSING on every clone -- including a clean checkout of this branch --
    even though the evidence for them is committed.

    Reading the aggregate is exact here, not an approximation: both runs cover
    the identical 752-case set and identical per-category n, so the difference
    of the per-category means equals the mean of the paired per-case
    differences. Verified across all 16 categories, max deviation 6.7e-16.

    Still returns None when the aggregates are absent, so a repository that
    genuinely lacks the evidence reports MISSING rather than a wrong number.

    The category breakdown is the evidence that the mean is the wrong summary:
    the gain is concentrated in cases where evidence competes, and is slightly
    NEGATIVE on `social`, where there is nothing to retrieve.
    """
    a = KVS / "cpu-frac0.25-random-b6f0__rag_dev" / "per_category.json"
    b = KVS / "cpu-frac0.25-rf0.15-b6f0__rag_dev" / "per_category.json"
    if not (a.exists() and b.exists()):
        return None
    A = json.loads(a.read_text(encoding="utf-8"))
    B = json.loads(b.read_text(encoding="utf-8"))
    if category not in A or category not in B:
        return None
    if A[category].get("n") != B[category].get("n"):
        return None          # unpaired: the difference of means is not the gain
    return B[category]["mean_quality_score"] - A[category]["mean_quality_score"]


def matched_pair(a: str, b: str, dataset: str) -> int:
    """How many manifest fields differ between two arms, EXCLUDING `select`.

    A contrast that attributes a difference to the score requires the two arms to
    differ in nothing else. Checking the VALUES cannot detect a violation --
    every number reproduces from its own file whether or not the comparison is
    legitimate. This checks the COMPARISON.

    It exists because the length-series table paired a random arm at
    recent=14.38 against the best significance arm at recent=8.70 and reported
    the difference as the score's contribution. Both numbers were correct; the
    contrast was not. The inflation was 0.013, in the flattering direction.

    Returns the count of differing fields, so a pin of 0 asserts a clean pair.
    """
    fields = ("keep_frac", "budget", "sinks", "recent", "pool", "pool_back",
              "pool_fwd", "grow", "window", "dataset", "case_count", "model",
              "adapter", "min_heavy_frac")
    ma = KVS / f"{a}__{dataset}" / "run_manifest.json"
    mb = KVS / f"{b}__{dataset}" / "run_manifest.json"
    if not (ma.exists() and mb.exists()):
        return None
    A = json.loads(ma.read_text(encoding="utf-8"))
    B = json.loads(mb.read_text(encoding="utf-8"))
    n = 0
    for f in fields:
        va, vb = A.get(f), B.get(f)
        if isinstance(va, float) and isinstance(vb, float):
            if abs(va - vb) > 1e-6:
                n += 1
        elif va != vb:
            n += 1
    return n


def _sel(n: int, arm: str):
    """Minimum per-compression selection time for one arm, from measure_select.

    Ratios rather than absolutes are what this pins: the absolute milliseconds
    are machine- and load-dependent, but the ratio between two arms measured in
    the same interleaved run is the quantity the claim rests on.
    """
    p = KVS / "measure_select.json"
    if not p.exists():
        return None
    for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
        if r["n"] == n and r["arm"] == arm:
            return r["ms_min"]
    return None


def _qp(section: str, *path):
    """Read a value out of measure_query_prefill.json.

    Same rationale as `_sel`: the absolute milliseconds are machine- and
    load-dependent, so what gets pinned is a RATIO or a SHARE measured inside a
    single interleaved run. The one exception is the fixed-floor fraction, which
    is itself a ratio of two fitted coefficients.

    `section` is "component_ladder" or "end_to_end"; for end_to_end the first
    path element selects a row by chunk size.
    """
    p = KVS / "measure_query_prefill.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    node = doc[section]
    if section == "end_to_end":
        chunk = path[0]
        match = [r for r in node if r["chunk"] == chunk]
        if len(match) != 1:
            raise RuntimeError(
                f"measure_query_prefill.json: {len(match)} end_to_end rows have "
                f"chunk={chunk}. A silent multi-match is how this repository "
                f"once pinned the wrong twin of a file pair.")
        node, path = match[0], path[1:]
    for key in path:
        node = node[key]
    return node


def _qp_floor_share(q: int = 16) -> float:
    """Fraction of a q-token forward that is FIXED cost, from the fitted line.

    This is the number that decides whether the `score_with_query` docstring's
    premise holds. The docstring predicts the pass costs q/chunk of a chunk,
    which assumes cost is proportional to token count. If the intercept
    dominates at q = 16, the premise is false and the prediction is wrong by
    whatever this fraction implies.
    """
    a = _qp("component_ladder", "fit_intercept_ms")
    b = _qp("component_ladder", "fit_slope_ms_per_token")
    if a is None or b is None:
        return None
    return a / (a + b * q)


def cp_bound(n: int, tail: float) -> float:
    """Clopper-Pearson bound for a cell with zero failures: tail ** (1/n).

    Closed form, so this pin checks the ARITHMETIC of a bound the prose quotes,
    not a measurement. It exists because 0.779 and 0.903 are the two numbers most
    likely to be swapped for each other -- they are the same construction at
    different n and different tails, and the pairing is easy to state backwards.
    """
    return tail ** (1.0 / n)


def retracted_fit(which: str) -> float:
    """Refit the WITHDRAWN scaling law from the B_half triple the ledger records.

    HONESTY ABOUT SCOPE. `B_half = 0.141 * L**0.535` is retracted (it was fitted
    from the centred window only). The ladders it was fitted on belong to the
    cornerstone-1 programme and are not committed on this branch, so this cannot
    be a measurement pin and must not be read as one. What it checks is that the
    coefficient and exponent still quoted in the retraction record reproduce from
    the three B_half values quoted beside them -- i.e. that a withdrawn figure
    cannot drift into a different withdrawn figure while nothing watches it.
    """
    txt = DOCS["ledger"].read_text(encoding="utf-8")
    m = re.search(r"\*\*([\d.]+) / ([\d.]+) / ([\d.]+)\*\* at 32K/64K/128K", txt)
    if not m:
        raise AssertionError("ledger no longer states the B_half triple")
    bh = [float(g) for g in m.groups()]
    ls = [math.log(L) for L in (32768, 65536, 131072)]
    lb = [math.log(b) for b in bh]
    mx = sum(ls) / 3, sum(lb) / 3
    cov = sum((x - mx[0]) * (y - mx[1]) for x, y in zip(ls, lb))
    var = sum((x - mx[0]) ** 2 for x in ls)
    slope = cov / var
    return slope if which == "exponent" else math.exp(mx[1] - slope * mx[0])


def pk_depth_pooled(depth: float):  # currently unused -- see the note in CLAIMS
    """One depth, pooled across all five decay-exponent arms (n = 30 trials)."""
    arms = ("0.0", "0.25", "0.5", "0.75", "1.0")
    vals = [pk(f"pk_128k_e{e}.json", depth) for e in arms]
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)

def identical_replies(group: str, tag: str):
    """Byte-identical reply count, read from the COMMITTED artifact.

    The per-case results.jsonl files are gitignored -- 132 MB of them -- so a
    check that read them would pass on the machine that produced them and fail in
    a fresh clone. scripts/design/identity_counts.py derives the counts into a few
    kB of committed JSON; run it with --check to confirm the artifact still
    matches the raw records when those are present.
    """
    path = ROOT / f"experiments/{group}/results/identity_counts.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    row = d.get("counts", {}).get(tag)
    return row["identical"] if row else None

def concentration(k: int):
    p = KVS / "concentration.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["rho_mean"][d["ks"].index(k)]


# (label, documented value, how to recompute, tolerance, text that must appear,
#  which document(s) must state it -- comma-separated means "all of these")
CLAIMS = [
    # --- FINDINGS_COMPRESSION_IMPROVES_QUALITY.md, previously unguarded -------
    ("improves: uncompressed answer", 0.7708,
     lambda: ident("c7-off-n600", "answer"), 5e-5, "0.7708", "improves"),
    ("improves: uncompressed abstain", 0.8214,
     lambda: ident("c7-off-n600", "abstain"), 5e-5, "0.8214", "improves"),
    ("improves: uncompressed pooled", 0.7850,
     lambda: ident("c7-off-n600", "pooled"), 5e-5, "0.7850", "improves"),
    ("improves: bm25+sys answer", 0.8032,
     lambda: ident("c8-docbm25sys", "answer"), 5e-5, "0.8032", "improves"),
    ("improves: bm25+sys abstain", 0.9226,
     lambda: ident("c8-docbm25sys", "abstain"), 5e-5, "0.9226", "improves"),
    ("improves: bm25+sys pooled", 0.8367,
     lambda: ident("c8-docbm25sys", "pooled"), 5e-5, "0.8367", "improves"),
    ("improves: bm25 no-sys answer", 0.8056,
     lambda: ident("c7-docbm25-k0.05", "answer"), 5e-5, "0.8056", "improves"),
    ("improves: bm25 no-sys pooled", 0.5800,
     lambda: ident("c7-docbm25-k0.05", "pooled"), 5e-5, "0.5800", "improves"),
    ("improves: oracle answer", 0.8102,
     lambda: ident("c7-oracle-k0.05", "answer"), 5e-5, "0.8102", "improves"),
    ("improves: oracle pooled", 0.5833,
     lambda: ident("c7-oracle-k0.05", "pooled"), 5e-5, "0.5833", "improves"),

    # --- the endpoint correction on the same document ------------------------
    ("improves: crossed de-confound", 0.53,
     lambda: abstain_audit("crossed", "delta_pp"), 5e-3, "+0.53 pp", "improves"),
    ("improves: semantic abstention", 3.33,
     lambda: abstain_audit("semantic", "pooled", "delta_pp"), 5e-3,
     "+3.33", "improves"),
    ("improves: drop partial_support", 2.67,
     lambda: abstain_audit("excluding_partial_support", "delta_pp"), 5e-3,
     "+2.67", "improves"),
    ("improves: partial_support gain", 44.74,
     lambda: abstain_audit("by_category", "partial_support", "delta_pp"), 5e-3,
     "+44.74", "improves,ident"),

    # --- FINDINGS_BLIND_ADJUDICATION.md --------------------------------------
    ("adjud: strict full stratum", 2.55,
     lambda: adjud("full_answer_stratum", "primary", "delta_pp"), 5e-3,
     "+2.55 pp", "adjud,improves"),
    ("adjud: permissive full stratum", 3.47,
     lambda: adjud("full_answer_stratum", "secondary", "delta_pp"), 5e-3,
     "+3.47 pp", "adjud"),
    ("adjud: strict wins", 12.0,
     lambda: adjud("full_answer_stratum", "primary", "wins"), 0.5, "12/1", "adjud"),
    ("adjud: contract agreement", 0.9977,
     lambda: adjud("gates", "g2_contract_agreement"), 5e-4, "99.8%", "adjud"),
    ("adjud: induced discordant", 6.0,
     lambda: adjud("conditioning", "induced_discordant"), 0.5,
     "observed 6", "adjud"),

    # --- FINDINGS_CAPACITY_FLOOR.md, listed in DOCS but never pinned ---------
    ("floor: predicted tokens", 196.0,
     lambda: floor_stat("predicted_floor_tokens"), 0.5, "196", "floor"),
    ("floor: observed knee", 196.0,
     lambda: floor_stat("observed_floor_tokens"), 0.5, "196", "floor"),
    ("floor: first budget below", 160.0,
     lambda: floor_stat("F2", "first_budget_below_floor"), 0.5, "160", "floor"),
    ("floor: B=512 reduction", 8.3,
     lambda: floor_stat("budgets", 512, "reduction_x"), 0.05, "8.3×", "floor"),
    ("floor: B=256 reduction", 16.5,
     lambda: floor_stat("budgets", 256, "reduction_x"), 0.05, "16.5×", "floor"),
    ("floor: B=196 reduction", 21.6,
     lambda: floor_stat("budgets", 196, "reduction_x"), 0.05, "21.6×", "floor"),
    ("floor: B=160 reduction", 26.4,
     lambda: floor_stat("budgets", 160, "reduction_x"), 0.05, "26.4×", "floor"),
    ("floor: B=128 reduction", 33.0,
     lambda: floor_stat("budgets", 128, "reduction_x"), 0.05, "33.0×", "floor"),
    ("floor: B=96 reduction", 44.0,
     lambda: floor_stat("budgets", 96, "reduction_x"), 0.05, "44.0×", "floor"),
    ("floor: B=256 answer", 0.8056,
     lambda: floor_stat("budgets", 256, "answer", "arm"), 5e-5, "0.8056", "floor"),

    # --- FINDINGS_INSTRUCTION_IDENTIFICATION.md ------------------------------
    ("ident: random floor answer", 0.0486,
     lambda: ident("c19-randomdocsys", "answer"), 5e-5, "0.0486", "ident"),
    ("ident: oracle+sys answer", 0.8009,
     lambda: ident("c19-oraclesys", "answer"), 5e-5, "0.8009", "ident"),
    ("ident: A vs C answer", -75.46,
     lambda: ident_contrast("A vs C  identification", "answer", "delta_pp"),
     5e-3, "75.46", "ident"),
    ("ident: B vs C ceiling", -0.23,
     lambda: ident_contrast("B vs C  ceiling", "answer", "delta_pp"),
     5e-3, "0.23", "ident"),

    ("quality: no eviction", 0.66939,
     lambda: summary(KVS, "off"), 5e-6, "0.66939", "findings"),
    ("quality: keep 0.75", 0.65677,
     lambda: summary(KVS, "frac0.75"), 5e-6, "0.65677", "findings"),
    ("quality: keep 0.50 pooled", 0.63371,
     lambda: summary(KVS, "frac0.5-pool7-fixed"), 5e-6, "0.63371", "findings"),
    ("quality: keep 0.50", 0.61401,
     lambda: summary(KVS, "frac0.5"), 5e-6, "0.61401", "findings"),
    ("quality: keep 0.25 recent-frac", 0.41932,
     lambda: summary(KVS, "frac0.25-rf0.25"), 5e-6, "0.41932", "findings"),
    ("quality: keep 0.25", 0.33863,
     lambda: summary(KVS, "frac0.25"), 5e-6, "0.33863", "findings"),

    ("control: StreamingLLM keep 0.75", 0.47545,
     lambda: summary(KVS, "frac0.75-rf1.0"), 5e-6, "0.47545", "findings"),
    ("control: StreamingLLM keep 0.50", 0.47682,
     lambda: summary(KVS, "frac0.5-rf1.0"), 5e-6, "0.47682", "findings"),
    ("control: StreamingLLM keep 0.25", 0.42208,
     lambda: summary(KVS, "frac0.25-rf1.0"), 5e-6, "0.42208", "findings"),

    ("chunked: peak 32K", 1744.1,
     lambda: measured(KVS / "measure_chunked_long.json", "peak_mib", length=32768),
     0.05, "1744.1", "findings"),
    ("chunked: peak 128K", 1744.9,
     lambda: measured(KVS / "measure_chunked_long.json", "peak_mib", length=131072),
     0.05, "1744.9", "findings"),
    ("chunked: resident cache 128K", 672.0,
     lambda: measured(KVS / "measure_chunked_long.json", "cache_resident_mib",
                      length=131072), 0.05, "672", "findings"),

    ("resident: swa-w1024-g4 @16K", 1792.0,
     lambda: measured(ATS / "measure_fixed.json", "cache_resident_mib",
                      variant="swa-w1024-g4", length=16384), 0.05, "1792.0", "findings"),
    ("resident: gqa-1KV @16K", 224.0,
     lambda: measured(ATS / "measure_fixed.json", "cache_resident_mib",
                      variant="gqa-1KV", length=16384), 0.05, "224.0", "findings"),

    ("concentration: rho(32)", 0.951,
     lambda: concentration(32), 1e-3, "0.951", "findings"),
    ("concentration: rho(64)", 0.980,
     lambda: concentration(64), 1e-3, "0.980", "findings"),

    # --- store: what to keep each retained position AS ----------------------
    # The first three are the calibration points that give attention relerr a
    # scale. They are quoted in NEGATIVE_RESULTS.md beside the candidates they
    # judge, so an edit that moves a candidate without its reference fails here.
    ("calib: keep 0.75 relerr", 0.00937,
     lambda: probe("store_probe.json", "calibration", "keep0.75", "mean"),
     5e-6, "0.00937", "findings,negative"),
    ("calib: keep 0.50 relerr", 0.03729,
     lambda: probe("store_probe.json", "calibration", "keep0.50", "mean"),
     5e-6, "0.03729", "findings,negative"),
    ("calib: keep 0.25 relerr", 0.11381,
     lambda: probe("store_probe.json", "calibration", "keep0.25", "mean"),
     5e-6, "0.11381", "findings,negative"),

    ("store: int8 K per-channel", 0.00774,
     lambda: probe("store_probe.json", "calibration", "int8", "mean"),
     5e-6, "0.00774", "findings,negative"),
    ("store: int8 both per-token", 0.04134,
     lambda: probe("store_probe.json", "calibration", "int8_tokenwise", "mean"),
     5e-6, "0.04134", "findings,negative"),
    ("store: int4 sinks quantised too", 0.12804,
     lambda: probe("store_probe.json", "calibration", "int4_notsink", "mean"),
     5e-6, "0.12804", "negative"),
    ("store: low-rank 512 (oracle)", 0.00147,
     lambda: probe("store_probe.json", "calibration", "lowrank512", "mean"),
     5e-6, "0.00147", "negative"),
    ("store: low-rank 256 (oracle)", 0.01236,
     lambda: probe("store_probe.json", "calibration", "lowrank256", "mean"),
     5e-6, "0.01236", "negative"),
    ("store: rank-128 K only", 0.08326,
     lambda: probe("store_probe.json", "calibration", "lowrank256_Konly", "mean"),
     5e-6, "0.08326", "negative"),
    ("store: rank-128 V only", 0.19383,
     lambda: probe("store_probe.json", "calibration", "lowrank256_Vonly", "mean"),
     5e-6, "0.19383", "negative"),
    ("store: unhealed 8->2 KV merge", 1.0206,
     lambda: probe("store_probe.json", "calibration", "gqa2kv_merge", "mean"),
     5e-5, "1.0206", "negative"),
    ("store: CLA held-out relerr", 0.1720, cla_mean, 5e-5, "0.1720", "findings,negative"),

    # --- significance-gated mixed precision (the survivor) ------------------
    ("mixed: pure int8 @4K relerr", 0.00762,
     lambda: probe("mixed_precision.json", "pure_int8@4K", "relerr"),
     5e-6, "0.00762", "findings,negative"),
    ("mixed: pure int4 @4K relerr", 0.11179,
     lambda: probe("mixed_precision.json", "pure_int4@4K", "relerr"),
     5e-6, "0.11179", "findings,negative"),
    ("mixed: top-25% @4K relerr", 0.00693,
     lambda: probe("mixed_precision.json", "mix_p0.25@4K", "relerr"),
     5e-6, "0.00693", "findings,negative"),
    ("mixed: top-25% @4K factor", 3.041,
     lambda: probe("mixed_precision.json", "mix_p0.25@4K", "factor"),
     5e-4, "3.04", "findings,negative"),
    ("mixed: pure int8 @4K factor", 1.936,
     lambda: probe("mixed_precision.json", "pure_int8@4K", "factor"),
     5e-4, "1.94", "findings,negative"),
    ("mixed: pure int4 @4K factor", 3.755,
     lambda: probe("mixed_precision.json", "pure_int4@4K", "factor"),
     5e-4, "3.75", "findings,negative"),

    # --- the decay-exponent sweep at 128K -----------------------------------
    # Pinned in both documents. These are what the withdrawn 0.44 figure was
    # competing against; see results/_superseded_unmatched/NOTE.md.
    ("decay: e=0 pooled @128K", 0.2222, lambda: pk("pk_128k_e0.0.json"),
     5e-4, "0.22", "findings"),
    ("decay: e=1 pooled @128K", 0.3889, lambda: pk("pk_128k_e1.0.json"),
     5e-4, "0.39", "findings"),
    # The 6-trial arms these used to pin are superseded by the 12-needle paired
    # sweep; the old cells are still on disk but their prose is gone from both
    # documents, which is exactly what this check is for.
    ("decay paired: e=0 at depth 0.9", 0.0000,
     lambda: pk("pk_128k_pe0.0.json", 0.9), 5e-4, "0.00", "findings"),
    ("decay paired: e=1 at depth 0.9", 0.8333,
     lambda: pk("pk_128k_pe1.0.json", 0.9), 5e-4, "0.83", "findings"),
    ("decay paired: e=0 mean", 0.2222,
     lambda: pk("pk_128k_pe0.0.json"), 5e-4, "0.22", "findings"),
    ("decay paired: e=1 mean", 0.5278,
     lambda: pk("pk_128k_pe1.0.json"), 5e-4, "0.53", "findings"),
    # The pooled depth curve that used to be checked here is WITHDRAWN: the five
    # arms share one key sequence, so "n = 30 per depth" was five repetitions of
    # six needles, and depth was confounded with needle identity. A withdrawn
    # figure must not sit in a list of numbers that "reproduce". The retraction
    # still quotes 0.367 / 0.133 / 0.500 as history; it is not a claim.

    # --- how a quantised cache degrades under repeated compression ----------
    # These two arms used to print and not serialise, so they could not be
    # checked. They constrain the implementation: round-tripping per compression
    # round is what makes int8 lose the advantage it starts with.
    ("requant: int8 after 1 round", 0.00276,
     lambda: probe("store_probe.json", "requant", "int8")[0], 5e-6, "0.00276", "negative"),
    ("requant: int8 after 5 rounds", 0.00637,
     lambda: probe("store_probe.json", "requant", "int8")[4], 5e-6, "0.00637", "negative"),
    ("requant: int4 after 5 rounds", 0.10396,
     lambda: probe("store_probe.json", "requant", "int4")[4], 5e-6, "0.10396", "negative"),
    ("scale tail: min int8 K scale @4K", 1.156e-04,
     lambda: next((r["min"] for r in probe("store_probe.json", "scale_tail") or []
                   if r["n"] == 4096 and r["tensor"] == "K" and r["bits"] == 8), None),
     5e-7, "1.156e-04", "negative"),

    # --- 128K under the PAIRED design (12 needles, identical in every cell) --
    # The headline of the program: at a constant keep fraction, accuracy does not
    # fall with length. Everything before pk_128k_grid12.json compared cells that
    # saw different needles.
    ("128K B=2048 depth 0.1", 0.4167, lambda: pk("pk_128k_grid12.json", 0.1, 2048),
     5e-4, "0.42", "findings,negative"),
    ("128K B=8192 depth 0.1", 1.0000, lambda: pk("pk_128k_grid12.json", 0.1, 8192),
     5e-4, "1.00", "findings"),
    ("128K B=2048 depth 0.5", 0.3333, lambda: pk("pk_128k_grid12.json", 0.5, 2048),
     5e-4, "0.33", "findings,negative"),
    ("128K B=8192 depth 0.5", 0.8333, lambda: pk("pk_128k_grid12.json", 0.5, 8192),
     5e-4, "0.83", "findings,negative"),
    ("128K B=2048 depth 0.9", 0.8333, lambda: pk("pk_128k_grid12.json", 0.9, 2048),
     5e-4, "0.83", "findings,negative"),
    ("128K B=8192 depth 0.9", 0.9167, lambda: pk("pk_128k_grid12.json", 0.9, 8192),
     5e-4, "0.92", "findings,negative"),
    ("128K B=8192 mean over depths", 0.9167,
     lambda: pk("pk_128k_grid12.json", None, 8192), 5e-4, "0.92", "findings"),
    ("128K B=2048 mean over depths", 0.5278,
     lambda: pk("pk_128k_grid12.json", None, 2048), 5e-4, "0.53", "findings"),
    ("32K uncompressed reference", 1.0000,
     lambda: pk("passkey_ceiling.json", None, 0, 32768), 5e-4, "1.00", "findings"),

    # --- 64K ladder: the keep fraction replicated at a second length --------
    # The point of these is the CONTRAST with the 128K rows above at matched keep:
    # 1.6% -> 0.61 / 0.53 and 6.5% -> 0.97 / 0.92. If either pair drifts apart the
    # claim that length is nearly a free parameter has to be restated.
    ("64K B=1024 mean over depths", 0.6111,
     lambda: pk("pk_64k_ladder.json", None, 1024), 5e-4, "0.61", "findings"),
    ("64K B=2048 mean over depths", 0.8056,
     lambda: pk("pk_64k_ladder.json", None, 2048), 5e-4, "0.81", "findings"),
    ("64K B=4096 mean over depths", 0.9722,
     lambda: pk("pk_64k_ladder.json", None, 4096), 5e-4, "0.97", "findings"),
    ("64K B=4096 depth 0.1", 1.0000,
     lambda: pk("pk_64k_ladder.json", 0.1, 4096), 5e-4, "1.00", "findings"),
    ("64K B=4096 depth 0.9", 0.9167,
     lambda: pk("pk_64k_ladder.json", 0.9, 4096), 5e-4, "0.92", "findings"),
    ("64K B=1024 depth 0.9", 0.5833,
     lambda: pk("pk_64k_ladder.json", 0.9, 1024), 5e-4, "0.58", "findings"),

    # --- counts that were hand-written and wrong -----------------------------
    # The attention_shape README claimed "60/60 byte-identical ... with it, 51/60"
    # for a 510-case set. Recomputed here so the scale cannot drift again.
    ("compat: identical vs gate baseline", 393,
     lambda: identical_replies("attention_shape", "base-8KV__chat_dev"),
     0.5, "393/510", "attention_shape"),
    ("pooling: identical replies", 631,
     lambda: identical_replies("kv_significance", "frac0.5-pool7-fixed__rag_dev"),
     0.5, "631/752", "kv_readme"),
    ("quality: identical at keep 0.75", 711,
     lambda: identical_replies("kv_significance", "frac0.75__rag_dev"),
     0.5, "711/752", "findings"),
    ("quality: identical at keep 0.25", 159,
     lambda: identical_replies("kv_significance", "frac0.25__rag_dev"),
     0.5, "159/752", "findings"),

    # --- the cap-vs-chunk grid, which decides where the reference ladder stops -
    # These four cells were run from scratch files that were then deleted, so the
    # 2x2 in FINDINGS 4b had no source for a while. Re-run with cap_mib in the
    # manifest and pinned here. The claim is that the CAP decides feasibility and
    # the chunk size does not, so both diagonals matter.
    ("mem: 32K cap6144 chunk1024 peak", 4571.0,
     lambda: measured(KVS / "mem_cap6144_chunk1024.json", "peak_mib", length=32768),
     1.0, "4571", "findings"),
    ("mem: 32K cap6144 chunk4096 peak", 4897.0,
     lambda: measured(KVS / "mem_cap6144_chunk4096.json", "peak_mib", length=32768),
     1.0, "4897", "findings"),
    ("mem: 32K cap7800 chunk1024 peak", 4699.0,
     lambda: measured(KVS / "mem_cap7800_chunk1024.json", "peak_mib", length=32768),
     1.0, "4699", "findings"),
    ("mem: 32K cap7800 chunk4096 peak", 4973.0,
     lambda: measured(KVS / "mem_cap7800_chunk4096.json", "peak_mib", length=32768),
     1.0, "4973", "findings"),
    ("mem: 64K full cache peak at OOM", 6426.0,
     lambda: measured(KVS / "mem_64k_full.json", "peak_mib", length=65536),
     1.0, "6,426", "findings"),

    # --- decode throughput and the latency floor ----------------------------
    # The operating point rests on these: B=2048 and B=4096 sit on the same floor,
    # so if the floor or the knee moves, the recommendation moves with it.
    ("decode: floor ms", 34.70,
     lambda: probe("probe_decode_floor.json", "fit", "floor_ms"),
     5e-3, "34.70", "findings"),
    ("decode: knee tokens", 4096,
     lambda: probe("probe_decode_floor.json", "fit", "knee_tokens"),
     0.5, "4,096 tokens (448 MiB)", "findings"),
    ("decode: 16K uncompressed", 76.36,
     lambda: next((r["ms_per_token"] for r in
                   (probe("probe_decode_floor.json", "results") or [])
                   if r["cache_tokens"] == 16384), None),
     5e-3, "76.36", "findings"),
    ("128K B=4096 mean", 0.7778,
     lambda: pk("pk_128k_B4096.json"), 5e-4, "0.78", "findings"),
    ("128K B=4096 depth 0.1", 0.7500,
     lambda: pk("pk_128k_B4096.json", 0.1), 5e-4, "0.75", "findings"),
    ("128K B=4096 depth 0.5", 0.6667,
     lambda: pk("pk_128k_B4096.json", 0.5), 5e-4, "0.67", "findings"),

    # --- the matched 32K control: the only measured eviction cost -----------
    # Reference and eviction arm share seed, needles, trial count AND RoPE
    # scaling. The older 32K pair did not share the last of those, so its
    # difference was a model change plus an eviction change.
    ("32K matched reference (YaRN)", 0.9722,
     lambda: pk("pk_32k_reference_yarn.json"), 5e-4, "0.97", "findings"),
    ("32K B=2048 paired", 0.7778,
     lambda: pk("pk_32k_paired.json"), 5e-4, "0.78", "findings"),
    ("32K reference depth 0.5", 0.9167,
     lambda: pk("pk_32k_reference_yarn.json", 0.5), 5e-4, "0.92", "findings"),
    ("32K B=2048 depth 0.5", 0.5833,
     lambda: pk("pk_32k_paired.json", 0.5), 5e-4, "0.58", "findings"),

    # --- the mechanism's own memory overhead --------------------------------
    # Unpinned until now, which is how 0.25% survived in the paper abstract while
    # 74/74 passed. The committed measurement is 0.78% for the score tensor; the
    # mechanism as used also carries _COUNTS, so 1.56% is the honest figure.
    ("overhead: score tensor pct", 0.781,
     lambda: measured(KVS / "measure_sig.json", "score_pct_of_cache", budget=1024),
     5e-4, "0.78%", "findings"),
    ("overhead: score MiB at B=1024", 0.929,
     lambda: measured(KVS / "measure_sig.json", "score_mib", budget=1024),
     5e-4, "0.929", "paper"),
    ("overhead: full matrix pct", 1.5625,
     lambda: 2 * (measured(KVS / "measure_sig.json", "score_pct_of_cache",
                           budget=1024) or 0), 1e-3, "1.56%", "paper"),

    # --- 64K saturation: the bound that replaces an impossible baseline -----
    ("64K B=8192 mean (saturated)", 1.0000,
     lambda: pk("pk_ladder_65536_B8192.json"), 5e-4, "1.00", "findings"),
    ("64K B=8192 depth 0.9", 1.0000,
     lambda: pk("pk_ladder_65536_B8192.json", 0.9), 5e-4, "1.00", "findings"),
    ("128K B=16384 mean (saturated)", 1.0000,
     lambda: pk("pk_ladder_131072_B16384.json"), 5e-4, "1.00", "findings"),
    ("128K B=16384 depth 0.5", 1.0000,
     lambda: pk("pk_ladder_131072_B16384.json", 0.5), 5e-4, "1.00", "findings"),

    # --- span pooling: the setting that was off ------------------------------
    # 4f. Pinned at both widths because the claim is that the effect is robust to
    # pool width, not tuned to one. If either drifts, that claim goes with it.
    ("pool=7 @128K B=2048 mean", 1.0000,
     lambda: pk("pk_128k_pool7.json"), 5e-4, "1.00", "findings"),
    ("pool=15 @128K B=2048 mean", 1.0000,
     lambda: pk("pk_128k_pool15.json"), 5e-4, "1.00", "findings"),
    ("pool=7 @128K depth 0.5", 1.0000,
     lambda: pk("pk_128k_pool7.json", 0.5), 5e-4, "1.00", "findings"),
    ("pool=7 floor: B=256 mean", 1.0000,
     lambda: pk("pk_128k_pool7_B256_rf.json",
                require={"recent": 64, "pool": 7}), 5e-4, "1.00", "findings"),
    ("pool=7 floor: B=128 mean", 0.9722,
     lambda: pk("pk_128k_pool7_B128_rf.json",
                require={"recent": 32, "pool": 7}), 5e-4, "0.972", "findings"),
    ("pool=7 exponent: e=0 mean", 0.5000,
     lambda: pk("pk_128k_pool7_e0.0.json"), 5e-4, "0.500", "findings"),
    ("pool=7 exponent: e=1 mean", 1.0000,
     lambda: pk("pk_128k_pool7_e1.0.json"), 5e-4, "1.000", "findings"),
    # --- O2 resolution, the span gate, and the baselines ---------------------
    ("O2: d=7 back=7", 0.9444,
     lambda: pk("pk_128k_B64_d7_b7f0.json",
                require={"key_digits": 7, "pool_back": 7}), 5e-4, "0.944", "p2"),
    ("O2: d=7 back=8", 1.0000,
     lambda: pk("pk_128k_B64_d7_b8f0.json",
                require={"key_digits": 7, "pool_back": 8}), 5e-4, "1.000", "p2"),
    ("baseline: random @B=16", 0.0000,
     lambda: pk("pk_128k_B16_base_random.json",
                require={"select": "random"}), 5e-4, "0/36", "p2"),
    ("baseline: streaming @B=16", 0.0000,
     lambda: pk("pk_128k_B16_base_streaming.json",
                require={"select": "streaming"}), 5e-4, "0/36", "p2"),
    ("S3 gate: 2 hops uncompressed", 0.8333,
     lambda: pk("pk_8k_hops2_full.json", require={"hops": 2}), 5e-4, "0.833", "p2"),
    ("S3 gate: 3 hops uncompressed", 0.7222,
     lambda: pk("pk_8k_hops3_full.json", require={"hops": 3}), 5e-4, "0.722", "p2"),
    ("O8 gate: W=3 full cache", 0.8889,
     lambda: pk("pk_16k_w3_full.json", require={"needle_words": 3}), 5e-4, "0.889", "p2"),
    ("O8 gate: W=12 full cache", 0.8056,
     lambda: pk("pk_16k_w12_full.json", require={"needle_words": 12}), 5e-4, "0.806", "p2"),
    ("O8 gate: W=20 full cache", 0.9444,
     lambda: pk("pk_16k_w20_full.json", require={"needle_words": 20}), 5e-4, "0.944", "p2"),
    ("stage 2: W=12 compressed", 0.8611,
     lambda: pk("pk_128k_B256_w12_grow.json",
                require={"needle_words": 12}), 5e-4, "0.861", "p2"),
    ("stage 2: W=20 compressed", 0.5278,
     lambda: pk("pk_128k_B256_w20_grow.json",
                require={"needle_words": 20}), 5e-4, "0.528", "p2"),
    # --- stage 1: retrieval under competition --------------------------------
    # P1 was pre-registered in docs/PLAN_LONG_REASONING.md before these ran. The
    # k=16 contrast is the whole result, so it must not drift unpinned.
    ("stage1: k=16 framed", 0.6111,
     lambda: pk("pk_128k_B64_dist16_framed.json",
                require={"distractors": 16, "distractor_style": "framed"}),
     5e-4, "0.611", "p2"),
    ("stage1: k=16 bare", 0.9722,
     lambda: pk("pk_128k_B64_dist16_bare.json",
                require={"distractors": 16, "distractor_style": "bare"}),
     5e-4, "0.972", "p2"),
    ("stage1: k=4 framed", 0.8889,
     lambda: pk("pk_128k_B64_dist4_framed.json",
                require={"distractors": 4, "distractor_style": "framed"}),
     5e-4, "0.889", "p2"),
    ("stage1: k=4 bare", 0.9444,
     lambda: pk("pk_128k_B64_dist4_bare.json",
                require={"distractors": 4, "distractor_style": "bare"}),
     5e-4, "0.944", "p2"),
    ("stage1: k=16 framed depth 0.9 unaffected", 1.0000,
     lambda: pk("pk_128k_B64_dist16_framed.json", 0.9,
                require={"distractors": 16}), 5e-4, "1.00", "p2"),
    # --- cornerstone 2: the pool window -------------------------------------
    # These were stated in FINDINGS_P2.md for hours with nothing checking them.
    # Twelve of eighteen figures in that file were unpinned, which is the same gap
    # that let a 3x memory error survive earlier in this program.
    ("causal: 128K B=64 centred (3,3)", 0.2778,
     lambda: pk("pk_128k_B64_pool_b3f3.json",
                require={"pool_back": 3, "pool_fwd": 3}), 5e-4, "0.278", "p2"),
    ("causal: 128K B=64 causal (6,0)", 1.0000,
     lambda: pk("pk_128k_B64_pool_b6f0.json",
                require={"pool_back": 6, "pool_fwd": 0}), 5e-4, "1.000", "p2"),
    ("causal: 128K B=16 causal (6,0)", 1.0000,
     lambda: pk("pk_128k_B16_pool_b6f0.json",
                require={"pool_back": 6, "pool_fwd": 0}), 5e-4, "1.000", "p2"),
    ("causal: 32K B=16 centred", 0.0278,
     lambda: pk("pk_32768_pool7_B16.json"), 5e-4, "0.028", "p2"),
    ("causal: 32K B=32 centred", 0.3889,
     lambda: pk("pk_32768_pool7_B32.json"), 5e-4, "0.389", "p2"),
    ("causal: 64K B=16 causal (6,0)", 0.9722,
     lambda: pk("pk_65536_B16_pool_b6f0.json",
                require={"pool_back": 6, "pool_fwd": 0}), 5e-4, "0.972", "p2"),
    # forward reach is inert: matched-back pairs must agree
    ("reach: back=4 fwd=0", 1.0000,
     lambda: pk("pk_128k_B64_pool_b4f0.json",
                require={"pool_back": 4, "pool_fwd": 0}), 5e-4, "1.000", "p2"),
    ("reach: back=4 fwd=4", 1.0000,
     lambda: pk("pk_128k_B64_pool_b4f4.json",
                require={"pool_back": 4, "pool_fwd": 4}), 5e-4, "1.000", "p2"),
    ("reach: back=2 fwd=0", 0.0278,
     lambda: pk("pk_128k_B64_pool_b2f0.json",
                require={"pool_back": 2, "pool_fwd": 0}), 5e-4, "0.028", "p2"),
    ("reach: back=2 fwd=2", 0.1389,
     lambda: pk("pk_128k_B64_pool_b2f2.json",
                require={"pool_back": 2, "pool_fwd": 2}), 5e-4, "0.139", "p2"),
    ("reach: back=5 fwd=0", 0.9722,
     lambda: pk("pk_128k_B64_pool_b5f0.json",
                require={"pool_back": 5, "pool_fwd": 0}), 5e-4, "0.972", "p2"),
    # the span law: the discriminating cell is d=7 at back=4
    ("span law: d=7 back=4", 0.0000,
     lambda: pk("pk_128k_B64_d7_b4f0.json",
                require={"key_digits": 7, "pool_back": 4}), 5e-4, "0.000", "p2"),
    ("span law: d=7 back=6", 0.7778,
     lambda: pk("pk_128k_B64_d7_b6f0.json",
                require={"key_digits": 7, "pool_back": 6}), 5e-4, "0.778", "p2"),
    ("span law: d=4 back=2", 0.4444,
     lambda: pk("pk_128k_B64_d4_b2f0.json",
                require={"key_digits": 4, "pool_back": 2}), 5e-4, "0.444", "p2"),
    ("span law: d=4 back=3", 0.9722,
     lambda: pk("pk_128k_B64_d4_b3f0.json",
                require={"key_digits": 4, "pool_back": 3}), 5e-4, "0.972", "p2"),
    ("pool=7 exponent: e=0 depth 0.9", 0.1667,
     lambda: pk("pk_128k_pool7_e0.0.json", 0.9), 5e-4, "0.17", "findings"),

    # --- A2: the sinks are load-bearing -------------------------------------
    # Sinks were fixed at four in every earlier result and never ablated, so the
    # headline read as "eight score-chosen positions retrieve a needle". These
    # two cells are why it now reads "four sinks PLUS eight". sinks=0 gets the
    # freed slots -- 12 score-chosen against 8 -- and still scores zero.
    ("A2: sinks=0 with 12 score-chosen slots", 0.0000,
     lambda: pk("pk_128k_B16_sinks0.json",
                require={"sinks": 0, "key_digits": 5}), 5e-4, "0.000", "c2paper"),
    ("A2: sinks=2 with 10 score-chosen slots", 1.0000,
     lambda: pk("pk_128k_B16_sinks2.json",
                require={"sinks": 2, "key_digits": 5}), 5e-4, "1.000", "c2paper"),

    # --- O13: the chunk axis is non-monotonic, at a second seed --------------
    # One bad cell between two perfect neighbours is a run; two seeds is a fact.
    # The control pins that the seed itself is not the cause.
    ("O13: chunk 2048, seed B", 0.8333,
     lambda: pk("pk_128k_B16_chunk2048_seedB.json",
                require={"chunk": 2048, "sinks": 4}), 5e-4, "0.833", "c2paper"),
    ("O13: chunk 4096 control, seed B", 1.0000,
     lambda: pk("pk_128k_B16_chunk4096_seedB.json",
                require={"chunk": 4096, "sinks": 4}), 5e-4, "1.000", "c2paper"),

    # --- A4: the span axis is capped by the MODEL, not by the mechanism ------
    # budget=0 is no compression at all. A fall here is recall, not eviction,
    # and it is why section 4.5's reach law stops at d=9.
    ("A4: 12-digit key, full cache, no eviction", 0.0000,
     lambda: pk("pk_16k_d12_full.json", budget=0,
                require={"key_digits": 12}), 5e-4, "0.000", "c2paper"),
    ("A4: 20-digit key, full cache", 0.0000,
     lambda: pk("pk_16k_d20_full.json", budget=0,
                require={"key_digits": 20}), 5e-4, "0.000", "c2paper"),
    ("A4: 30-digit key, full cache", 0.0000,
     lambda: pk("pk_16k_d30_full.json", budget=0,
                require={"key_digits": 30}), 5e-4, "0.000", "c2paper"),

    # --- figures the prose derives rather than measures ----------------------
    # These five were quoted in the documents with nothing checking them. Three
    # are arithmetic on pinned quantities; two are the record of a retraction.
    ("decoy gap: bare minus framed", 0.3611,
     lambda: pk("pk_128k_B64_dist16_bare.json",
                require={"distractors": 16, "distractor_style": "bare"})
             - pk("pk_128k_B64_dist16_framed.json",
                  require={"distractors": 16, "distractor_style": "framed"}),
     5e-4, "0.361", "c2paper"),
    ("CP lower bound, n=12, one-sided 95%", 0.7791,
     lambda: cp_bound(12, 0.05), 5e-4, "0.779", "ledger,c2paper"),
    ("CP bound, n=36, two-sided 2.5% tail", 0.9026,
     lambda: cp_bound(36, 0.025), 5e-4, "0.903", "ledger,c2paper"),
    ("RETRACTED fit: coefficient", 0.1407,
     lambda: retracted_fit("coef"), 5e-4, "0.141", "ledger,c2paper"),
    ("RETRACTED fit: exponent", 0.5350,
     lambda: retracted_fit("exponent"), 5e-4, "0.535", "ledger,c2paper"),

    # --- a RELATIONAL pin, and why this class of pin had to be invented -----
    # "Derived reach matches or beats the best hand-tuned reach at every span
    # length" sat in FOUR documents and was false: at d=7 derived scores 0.889
    # against a tuned back=8's 1.000.
    #
    # The instructive part is HOW it survived a checker at 136/136. Both
    # falsifying arms were ALREADY PINNED, at their true values, as "O2: d=7
    # back=7" (0.944) and "O2: d=7 back=8" (1.000). Nothing was stale and no
    # number had drifted. Every pin above verifies ONE number against ONE file,
    # but the false claim was a RELATION BETWEEN numbers -- and no pin has ever
    # checked a relation, so a comparative claim could contradict two pins that
    # both passed.
    #
    # This pin therefore asserts the ORDERING, not a value. It fails if anyone
    # ever restores the withdrawn claim, and it fails if a future arm changes
    # which side wins. The lesson generalises: a claim of the form "A beats the
    # best B" needs a pin over max(B), not a pin on each B.
    ("RELATION: derived reach LOSES to the best tuned reach at d=7", -0.1111,
     lambda: pk("pk_128k_B64_d7_grow2.0.json", require={"key_digits": 7, "grow": 2.0})
             - max(pk("pk_128k_B64_d7_b6f0.json", require={"pool_back": 6}),
                   pk("pk_128k_B64_d7_b7f0.json", require={"pool_back": 7}),
                   pk("pk_128k_B64_d7_b8f0.json", require={"pool_back": 8})),
     5e-4, "−0.111", "p2"),
    # The deployment-relevant relation, which is the one that DOES survive:
    # derived beats a single fixed reach held constant across span lengths.
    ("RELATION: derived beats a CONSTANT back=6 at d=7", 0.1111,
     lambda: pk("pk_128k_B64_d7_grow2.0.json", require={"key_digits": 7, "grow": 2.0})
             - pk("pk_128k_B64_d7_b6f0.json", require={"key_digits": 7, "pool_back": 6}),
     5e-4, "0.889", "p2"),
    ("d=9 tuned back=8 (no derived arm exists at d=9)", 0.8333,
     lambda: pk("pk_128k_B64_d9_b8f0.json",
                require={"key_digits": 9, "pool_back": 8}), 5e-4, "0.833", "p2"),

    # --- the FIRST natural-language cells cornerstone 2 ever produced --------
    # Until 2026-08-25 every pool_back result file was passkey and the language
    # runner had no --pool-back flag. These pin the arms and the paired
    # contrasts between them. The contrasts come from a derived artifact for the
    # same reason identical_replies() does: the per-case records are gitignored,
    # so a check reading them would pass here and fail in a fresh clone.
    ("LANG keep0.25 centred window", 0.26096,
     lambda: lang_quality("frac0.25-pool7-fixed"), 5e-6, "0.26096", "p2"),
    ("LANG keep0.25 CAUSAL window", 0.41286,
     lambda: lang_quality("frac0.25-b6f0"), 5e-6, "0.41286", "p2"),
    ("LANG keep0.75 centred window", 0.65999,
     lambda: lang_quality("frac0.75-pool7-fixed"), 5e-6, "0.65999", "p2"),
    ("LANG keep0.75 CAUSAL window", 0.65136,
     lambda: lang_quality("frac0.75-b6f0"), 5e-6, "0.65136", "p2"),
    ("LANG keep0.50 CAUSAL window", 0.64858,
     lambda: lang_quality("frac0.5-b6f0"), 5e-6, "0.64858", "p2"),
    ("LANG keep0.50 derived (grow=2.0)", 0.61346,
     lambda: lang_quality("frac0.5-grow2"), 5e-6, "0.61346", "p2"),

    # The crossover: direction helps under pressure and HURTS without it. Both
    # signs are pinned, because a one-sided pin would let the inconvenient half
    # of the finding drift away unnoticed.
    ("LANG contrast keep0.25 causal-minus-centred", 0.15190,
     lambda: lang("keep0.25_centred_vs_causal", "paired_mean_diff"),
     5e-6, "0.15190", "p2"),
    ("LANG contrast keep0.75 causal-minus-centred (NEGATIVE)", -0.00863,
     lambda: lang("keep0.75_centred_vs_causal", "paired_mean_diff"),
     5e-6, "-0.00863", "p2"),
    ("LANG contrast keep0.50 derived-minus-causal (NEGATIVE)", -0.03513,
     lambda: lang("keep0.50_causal_vs_derived", "paired_mean_diff"),
     5e-6, "-0.03513", "p2"),
    # RETRACTED 2026-08-26 -- the two keep-0.25 derived pins that used to sit
    # here read from frac0.25-grow2__rag_dev, now quarantined under
    # _invalidated_grow_noisefloor. At 59% masked that arm's grow gate was FULLY
    # OPEN (24 of 24 eligible positions admitted against a true median admitting
    # 2), so 0.33833 and the -0.07452 contrast measured a degenerate positional
    # policy rather than derived reach. Deleting the pins rather than adjusting
    # them: there is no correct value to pin until the arm is re-run, and a pin
    # that silently tracks a moved number is worse than no pin.
    #
    # The keep-0.50 pin above is RETAINED with a stated caveat: that arm sat at
    # 30% masked -- the floor was depressed to roughly the 27th percentile, not
    # collapsed to zero -- so it is degraded rather than void. It is a separate
    # question from the collapsed arm and is flagged in the documents.
    ("LANG keep0.25 discordant pairs favouring causal", 235,
     lambda: lang("keep0.25_centred_vs_causal", "b_better"), 0.5, "235", "p2"),
    ("LANG keep0.75 discordant pairs favouring CENTRED", 23,
     lambda: lang("keep0.75_centred_vs_causal", "b_worse"), 0.5, "23", "p2"),

    # --- BUDGET ALLOCATION: the largest language effect measured -------------
    # The keep-0.25 arms had `recent` pinned at a fixed 32 on a ~232-token
    # prompt, so sinks+recent ate 61% of the budget and heavy_zero_count fired
    # 17,360 times -- the policy silently degenerated to StreamingLLM. The
    # heavy_zero_count pins matter more than the quality ones: they are what
    # distinguishes "the mechanism scored badly" from "the mechanism did not run".
    ("LANG keep0.25 fair-share allocation", 0.60617,
     lambda: lang_quality("frac0.25-rf0.25-b6f0"), 5e-6, "0.60617", "p2"),
    ("LANG allocation effect (largest on language)", 0.19331,
     lambda: lang("keep0.25_allocation_fixed_vs_fair", "paired_mean_diff"),
     5e-6, "0.19331", "p2"),
    ("LANG heavy_zero under FIXED recent=32", 17360,
     lambda: json.loads((KVS / "frac0.25-b6f0__rag_dev" / "run_manifest.json")
                        .read_text(encoding="utf-8"))["heavy_zero_count"],
     0.5, "17,360", "p2"),
    # --- THE MATCHED CONTROL, which the language claim never had -------------
    # All CPU, keep 0.25, causal window, sinks=4, recent-frac 0.25. `random`
    # differs from `significance` in NOTHING except how the heavy slots are
    # chosen, so this is the first language number that can be attributed to the
    # SCORE. The previously published "+0.184 over the positional control"
    # cannot: its control also carried a 3.7x larger recent window.
    ("LANG the score vs matched RANDOM (the clean control)", 0.25184,
     lambda: lang("keep0.25_cpu_random_vs_significance", "paired_mean_diff"),
     5e-6, "0.25184", "p2"),
    ("LANG the score vs matched-budget streaming", 0.16747,
     lambda: lang("keep0.25_cpu_streaming_vs_significance", "paired_mean_diff"),
     5e-6, "0.16747", "p2"),
    ("LANG matched random arm", 0.34304,
     lambda: lang_quality("cpu-frac0.25-random-b6f0"), 5e-6, "0.34304", "p2"),
    ("LANG matched streaming arm", 0.42740,
     lambda: lang_quality("cpu-frac0.25-streaming-b6f0"), 5e-6, "0.42740", "p2"),
    ("LANG significance arm, CPU (device anchor)", 0.59488,
     lambda: lang_quality("cpu-frac0.25-rf0.25-b6f0"), 5e-6, "0.59488", "p2"),
    # --- O13 RESOLVED: marginal capacity, not the chunk schedule -------------
    # Registered with kill conditions before the runs. Prediction 1 held,
    # prediction 2 was refuted -- and BOTH are pinned, because a programme that
    # pins only its confirmed predictions is not keeping a record, it is
    # curating one.
    ("O13: B=32 rescues chunk 2048", 1.0000,
     lambda: pk("pk_128k_B32_chunk2048.json", require={"chunk": 2048}),
     5e-4, "1.000", "p2"),
    ("O13: B=32 chunk 4096 matched control", 1.0000,
     lambda: pk("pk_128k_B32_chunk4096.json", require={"chunk": 4096}),
     5e-4, "1.000", "p2"),
    ("O13: back=8 REFUTED prediction 2 (worse, not better)", 0.5556,
     lambda: pk("pk_128k_B16_chunk2048_b8.json",
                require={"chunk": 2048, "pool_back": 8}), 5e-4, "0.556", "p2"),

    # --- LONG CONTEXT: the model is the constraint, not the cache ------------
    # The uncompressed arms are the load-bearing pins here. Without them a
    # reader could take "4x compression retains quality at 16K" for a triumph
    # rather than a statement about a regime the model has already lost.
    ("LONG 4K uncompressed", 0.39102,
     lambda: lang_quality2("long4k-off", "rag_long4k"), 5e-6, "0.39102", "p2"),
    ("LONG 16K uncompressed", 0.18571,
     lambda: lang_quality2("long16k-off", "rag_long16k"), 5e-6, "0.18571", "p2"),
    ("LONG 32K uncompressed", 0.12827,
     lambda: lang_quality2("long32k-off", "rag_long32k"), 5e-6, "0.12827", "p2"),
    ("LONG 16K keep 0.25 (indistinguishable from uncompressed)", 0.19559,
     lambda: lang_quality2("long16k-b6f0-k0.25-rf0.25", "rag_long16k"),
     5e-6, "0.19559", "p2"),
    ("LONG 4K keep 0.50", 0.39126,
     lambda: lang_quality2("long4k-b6f0-k0.5", "rag_long4k"), 5e-6, "0.39126", "p2"),

    ("LONG 32K keep 0.25 (4.0x, indistinguishable)", 0.10851,
     lambda: lang_quality2("long32k-b6f0-k0.25-rf0.25", "rag_long32k"),
     5e-6, "0.10851", "p2"),
    # --- the allocation curve: an INTERIOR optimum, endpoint self-validating -
    ("ALLOC rf=0.10 (83.1% under the score)", 0.59788,
     lambda: lang_quality("cpu-frac0.25-rf0.10-b6f0"), 5e-6, "0.59788", "p2"),
    ("ALLOC rf=0.15 OPTIMUM", 0.60834,
     lambda: lang_quality("cpu-frac0.25-rf0.15-b6f0"), 5e-6, "0.60834", "p2"),
    ("ALLOC rf=0.35", 0.53656,
     lambda: lang_quality("cpu-frac0.25-rf0.35-b6f0"), 5e-6, "0.53656", "p2"),
    # The endpoint check: at rf=0.50 only 43.2% sits under the score, so the
    # policy must approach StreamingLLM -- and it lands within 0.010 of the
    # separately-run streaming control. Pinned because an agreement nothing
    # forced is worth as much as a headline.
    ("ALLOC rf=0.50 approaches the streaming control", 0.43732,
     lambda: lang_quality("cpu-frac0.25-rf0.50-b6f0"), 5e-6, "0.43732", "p2"),
    ("ALLOC best-vs-random gap", 0.26530,
     lambda: lang_quality("cpu-frac0.25-rf0.15-b6f0")
             - lang_quality("cpu-frac0.25-random-b6f0"), 5e-6, "0.26530", "p2"),
    # --- UN-RETRACTED: re-run with the fixed gate, moved by 0.00014 ----------
    # Pinned against the FIXED-gate arm, not the quarantined one. The withdrawn
    # figures were 0.33833 and -0.07452; the honest replacements are these.
    ("LANG keep0.25 derived reach (fixed gate)", 0.33820,
     lambda: lang_quality("frac0.25-grow2-fixedgate"), 5e-6, "0.33820", "p2"),
    ("LANG keep0.25 derived-minus-causal, fixed gate (NEGATIVE)", -0.07466,
     lambda: lang_quality("frac0.25-grow2-fixedgate")
             - lang_quality("frac0.25-b6f0"), 5e-6, "0.07466", "p2"),
    # --- the audit's re-runs: what the defects actually cost ----------------
    # Pinned at the FIXED gate. Every grow=2.0 arm is unchanged to four decimals,
    # so section 4.6 is untouched; only grow=5.0 moved, and it moved the way the
    # mechanism predicts (a stricter multiplier depends more on a correct floor).
    ("REGROW d=4 grow2.0 fixed gate (unchanged)", 1.0000,
     lambda: pk("pk_128k_B64_d4_grow2.0_fixedgate.json",
                require={"key_digits": 4, "grow": 2.0}), 5e-4, "1.0000", "p2"),
    ("REGROW d=7 grow2.0 fixed gate (unchanged)", 0.8889,
     lambda: pk("pk_128k_B64_d7_grow2.0_fixedgate.json",
                require={"key_digits": 7, "grow": 2.0}), 5e-4, "0.8889", "p2"),
    ("REGROW d=7 grow5.0 fixed gate (the one that MOVED)", 0.7222,
     lambda: pk("pk_128k_B64_d7_grow5.0_fixedgate.json",
                require={"key_digits": 7, "grow": 5.0}), 5e-4, "0.7222", "p2"),
    # Halo depth-0.9 re-measured at chunk 1024, where it faces 2 compressions
    # rather than 0. Same answer -- the claim was true, it just had not been
    # tested. The B=32 depth-0.1 cell is the honest cost of measuring properly.
    ("REGROW halo 16K B=32 chunk1024 d0.1 (the one real failure)", 0.9167,
     lambda: pk("halo_16k_B32_grow_c1024.json", 0.1,
                require={"chunk": 1024}), 5e-4, "0.917", "p2"),
    ("REGROW halo 16K B=32 chunk1024 d0.9 (was vacuous)", 1.0000,
     lambda: pk("halo_16k_B32_grow_c1024.json", 0.9,
                require={"chunk": 1024}), 5e-4, "1.000", "p2"),
    # --- B1: a NULL RESULT against a pre-registered bar ---------------------
    # Pinned because a programme that pins only its wins is not keeping a record.
    ("B1 best floor (0.85) ties the tuned optimum", 0.60297,
     lambda: lang_quality("cpu-frac0.25-B1f0.85-b6f0"), 5e-6, "0.60297", "p2"),
    ("B1 floor 0.78", 0.59805,
     lambda: lang_quality("cpu-frac0.25-B1f0.78-b6f0"), 5e-6, "0.59805", "p2"),
    ("B1 floor 0.70", 0.59272,
     lambda: lang_quality("cpu-frac0.25-B1f0.70-b6f0"), 5e-6, "0.59272", "p2"),
    ("B1 minus tuned optimum (indistinguishable)", -0.00537,
     lambda: lang_quality("cpu-frac0.25-B1f0.85-b6f0")
             - lang_quality("cpu-frac0.25-rf0.15-b6f0"), 5e-6, "0.00537", "p2"),
    # the keep-0.5 matched pair, completed today
    ("LANG keep0.5 matched significance", 0.65338,
     lambda: lang_quality("cpu-frac0.5-rf0.25-b6f0"), 5e-6, "0.65338", "p2"),
    ("LANG keep0.5 score-vs-random gap", 0.21147,
     lambda: lang_quality("cpu-frac0.5-rf0.25-b6f0")
             - lang_quality("cpu-frac0.5-random-b6f0"), 5e-6, "0.21147", "p2"),
    # --- the gain is CONCENTRATED, which changes the deployment claim -------
    # The mean (+0.2653) and the median (+0.0105) describe different worlds.
    # Pinning both, because quoting only the mean would be true and misleading.
    ("SHAPE: mean per-case gain over matched random", 0.26530,
     lambda: lang_quality("cpu-frac0.25-rf0.15-b6f0")
             - lang_quality("cpu-frac0.25-random-b6f0"), 5e-6, "0.26530", "p2"),
    ("SHAPE: social category, where the score has no job", -0.0529,
     lambda: _cat_gain("social"), 5e-4, "0.0529", "p2"),
    ("SHAPE: ambiguous_evidence, where it has the most job", 0.6461,
     lambda: _cat_gain("ambiguous_evidence"), 5e-4, "0.6461", "p2"),
    # --- A3: the certification the word "lossless" needed ------------------
    ("A3: 100-needle bound", 0.9705,
     lambda: cp_bound(100, 0.05), 5e-5, "0.9705", "p2,c2paper"),
    ("A3: pooled accuracy over 5 shards", 1.0000,
     lambda: (lambda fs: sum(pk(f) for f in fs) / len(fs))(
         ["pk_128k_B16_power_s%s.json" % s for s in
          ("20260820", "771103", "404411", "918273", "552146")]),
     5e-4, "300/300", "p2"),
    # --- the length series: random/streaming collapse, the score does not ---
    ("LEN 4K random (19% of uncompressed)", 0.07366,
     lambda: lang_quality2("long4k-random-k0.25", "rag_long4k"), 5e-6, "0.07366", "p2"),
    ("LEN 4K streaming (11%)", 0.04110,
     lambda: lang_quality2("long4k-streaming-k0.25", "rag_long4k"), 5e-6, "0.04110", "p2"),
    ("LEN 4K significance (99%)", 0.38730,
     lambda: lang_quality2("long4k-sig-k0.25-rf0.15", "rag_long4k"), 5e-6, "0.38730", "p2"),
    ("LEN 16K random", 0.04969,
     lambda: lang_quality2("long16k-random-k0.25", "rag_long16k"), 5e-6, "0.04969", "p2"),
    ("LEN 16K significance EXCEEDS the full cache", 0.19637,
     lambda: lang_quality2("long16k-sig-k0.25-rf0.15", "rag_long16k"), 5e-6, "0.19637", "p2"),
    ("LEN 32K random", 0.03935,
     lambda: lang_quality2("long32k-random-k0.25", "rag_long32k"), 5e-6, "0.03935", "p2"),
    # --- the adapter, DISENTANGLED from the model it sits on -----------------
    ("ADAPTER: Halo base at 4K (the control that split the confound)", 0.49124,
     lambda: lang_quality2("long4k-off-halo", "rag_long4k"), 5e-6, "0.49124", "p2"),
    ("ADAPTER: Halo + 250-step adapter at 4K", 0.68977,
     lambda: lang_quality2("long4k-off-adapter250", "rag_long4k"), 5e-6, "0.68977", "p2"),
    ("ADAPTER: gain attributable to the fine-tune alone", 0.19853,
     lambda: lang_quality2("long4k-off-adapter250", "rag_long4k")
             - lang_quality2("long4k-off-halo", "rag_long4k"), 5e-6, "0.19853", "p2"),
    # --- MATCHING pins: assert the CONTRAST, not just the numbers -----------
    # Every value pin below passed while the length-series table compared arms at
    # different allocations. Values reproduce from their own files whether or not
    # the comparison is legitimate, so no value pin can catch a bad pairing.
    # These assert that the two arms of each headline contrast differ in NOTHING
    # except `select`. 0 = clean pair.
    ("MATCH: 4K random vs significance differ only in select", 0,
     lambda: matched_pair("long4k-random-k0.25", "long4k-sig-k0.25-rf0.15",
                          "rag_long4k"), 0.5, "158.56", "p2"),
    ("MATCH: 16K random vs significance differ only in select", 0,
     lambda: matched_pair("long16k-random-k0.25", "long16k-sig-k0.25-rf0.15",
                          "rag_long16k"), 0.5, "619.57", "p2"),
    ("MATCH: native random vs significance differ only in select", 0,
     lambda: matched_pair("cpu-frac0.25-random-b6f0", "cpu-frac0.25-rf0.25-b6f0",
                          "rag_dev"), 0.5, "0.25184", "p2"),
    ("LEN 32K significance (matched arm)", 0.11839,
     lambda: lang_quality2("long32k-sig-k0.25-rf0.15", "rag_long32k"), 5e-6, "0.11839", "p2"),
    ("LEN 32K streaming", 0.02795,
     lambda: lang_quality2("long32k-streaming-k0.25", "rag_long32k"), 5e-6, "0.02795", "p2"),
    ("LEN 32K score minus random (matched)", 0.07904,
     lambda: lang_quality2("long32k-sig-k0.25-rf0.15", "rag_long32k")
             - lang_quality2("long32k-random-k0.25", "rag_long32k"), 5e-6, "0.07904", "p2"),
    ("MATCH: 32K random vs significance differ only in select", 0,
     lambda: matched_pair("long32k-random-k0.25", "long32k-sig-k0.25-rf0.15",
                          "rag_long32k"), 0.5, "0.07904", "p2"),
    ("REGROW keep0.50 language, fixed gate", 0.61437,
     lambda: lang_quality("frac0.5-grow2-fixedgate"), 5e-6, "0.61437", "p2"),
    # --- the remaining headline-table figures, pinned in BOTH documents -----
    # An audit found these four asserted in the paper's length-series table with
    # nothing checking them. A number in a headline table is the last place an
    # unpinned figure should survive.
    ("LEN 16K streaming", 0.03004,
     lambda: lang_quality2("long16k-streaming-k0.25", "rag_long16k"),
     5e-6, "0.03004", "p2,c2paper"),
    ("LEN 16K score minus random", 0.14669,
     lambda: lang_quality2("long16k-sig-k0.25-rf0.15", "rag_long16k")
             - lang_quality2("long16k-random-k0.25", "rag_long16k"),
     5e-6, "0.14669", "p2,c2paper"),
    ("LEN 4K score minus random", 0.31363,
     lambda: lang_quality2("long4k-sig-k0.25-rf0.15", "rag_long4k")
             - lang_quality2("long4k-random-k0.25", "rag_long4k"),
     5e-6, "0.31363", "p2,c2paper"),
    ("LANG keep0.5 matched random arm", 0.44191,
     lambda: lang_quality("cpu-frac0.5-random-b6f0"), 5e-6, "0.44191", "p2,c2paper"),
    # the two long-context compression deltas the findings state
    ("LONG 16K compression delta vs uncompressed", 0.00988,
     lambda: lang_quality2("long16k-b6f0-k0.25-rf0.25", "rag_long16k")
             - lang_quality2("long16k-off", "rag_long16k"), 5e-6, "0.00988", "p2"),
    ("LONG 32K compression delta vs uncompressed", -0.01976,
     lambda: lang_quality2("long32k-b6f0-k0.25-rf0.25", "rag_long32k")
             - lang_quality2("long32k-off", "rag_long32k"), 5e-6, "0.01976", "p2"),
    # --- BAYA: the mechanism on a model that can use long context ----------
    # The generalisation this campaign was missing. Every earlier language
    # result came from a model failing the task, so "the score recovers a bad
    # eviction" could not be told apart from "the score props up a weak model".
    ("BAYA 4K base uncompressed", 0.68905,
     lambda: lang_quality2("long4k-off-baya", "rag_long4k"), 5e-6, "0.68905", "p2"),
    ("BAYA 4K +adapter uncompressed", 0.65777,
     lambda: lang_quality2("long4k-off-baya-ad", "rag_long4k"), 5e-6, "0.65777", "p2"),
    ("BAYA 4K +adapter keep0.25 significance", 0.65970,
     lambda: lang_quality2("long4k-sig-baya-ad", "rag_long4k"), 5e-6, "0.65970", "p2"),
    ("BAYA 4K +adapter keep0.25 random", 0.45294,
     lambda: lang_quality2("long4k-random-baya-ad", "rag_long4k"), 5e-6, "0.45294", "p2"),
    ("BAYA the score is worth this at 4x on a capable model", 0.20676,
     lambda: lang_quality2("long4k-sig-baya-ad", "rag_long4k")
             - lang_quality2("long4k-random-baya-ad", "rag_long4k"),
     5e-6, "0.20676", "p2"),
    ("BAYA compression costs this much (indistinguishable from 0)", 0.00193,
     lambda: lang_quality2("long4k-sig-baya-ad", "rag_long4k")
             - lang_quality2("long4k-off-baya-ad", "rag_long4k"),
     5e-6, "0.00193", "p2"),
    ("BAYA the 1200-step adapter COST this at 4K", -0.03129,
     lambda: lang_quality2("long4k-off-baya-ad", "rag_long4k")
             - lang_quality2("long4k-off-baya", "rag_long4k"), 5e-6, "0.03129", "p2"),
    ("MATCH: Baya 4K random vs significance differ only in select", 0,
     lambda: matched_pair("long4k-random-baya-ad", "long4k-sig-baya-ad",
                          "rag_long4k"), 0.5, "0.20676", "p2"),
    ("BAYA 16K uncompressed", 0.66108,
     lambda: lang_quality2("long16k-off-baya-ad", "rag_long16k"), 5e-6, "0.66108", "p2"),
    ("BAYA 16K keep0.25 significance -- IDENTICAL to uncompressed", 0.0,
     lambda: lang_quality2("long16k-sig-baya-ad", "rag_long16k")
             - lang_quality2("long16k-off-baya-ad", "rag_long16k"),
     1e-9, "0.66108", "p2"),
    ("BAYA 16K keep0.25 random", 0.42022,
     lambda: lang_quality2("long16k-random-baya-ad", "rag_long16k"), 5e-6, "0.42022", "p2"),
    ("BAYA the score is worth this at 16K", 0.24086,
     lambda: lang_quality2("long16k-sig-baya-ad", "rag_long16k")
             - lang_quality2("long16k-random-baya-ad", "rag_long16k"),
     5e-6, "0.24086", "p2"),
    ("MATCH: Baya 16K random vs significance differ only in select", 0,
     lambda: matched_pair("long16k-random-baya-ad", "long16k-sig-baya-ad",
                          "rag_long16k"), 0.5, "0.24086", "p2"),
    # --- multi-hop: compression free, length breaks the chain ---------------
    # 8K hop arms are pinned on depths 0.1 and 0.5 ONLY: the depth-0.9 cell
    # faces zero compressions at 8K/chunk 4096 and measures nothing. The census
    # pin caught this in fresh data before it reached a conclusion.
    ("HOP 8K 2-hop significance (valid depths)", 0.7917,
     lambda: (pk("pk_8k_B64_hop2_sig.json", 0.1) + pk("pk_8k_B64_hop2_sig.json", 0.5)) / 2,
     5e-4, "0.7917", "p2"),
    ("HOP 8K 2-hop random (valid depths)", 0.3333,
     lambda: (pk("pk_8k_B64_hop2_random.json", 0.1) + pk("pk_8k_B64_hop2_random.json", 0.5)) / 2,
     5e-4, "0.3333", "p2"),
    ("HOP 8K 3-hop significance (valid depths)", 0.5417,
     lambda: (pk("pk_8k_B64_hop3_sig.json", 0.1) + pk("pk_8k_B64_hop3_sig.json", 0.5)) / 2,
     5e-4, "0.5417", "p2"),
    ("HOP 8K 3-hop random -- nominally HIGHER (valid depths)", 0.7500,
     lambda: (pk("pk_8k_B64_hop3_random.json", 0.1) + pk("pk_8k_B64_hop3_random.json", 0.5)) / 2,
     5e-4, "0.7500", "p2"),
    ("HOP 128K 2-hop significance (length, not compression)", 0.4722,
     lambda: pk("pk_128k_B64_hop2_sig.json", require={"hops": 2}), 5e-4, "0.4722", "p2"),
    ("HOP 128K 3-hop significance", 0.0278,
     lambda: pk("pk_128k_B64_hop3_sig.json", require={"hops": 3}), 5e-4, "0.0278", "p2"),
    # --- O11: the score is QUERY-CONDITIONED, not an importance measure -----
    # The reference and three controls differ in ONE field: the scoring query.
    # Both wrong queries collapse retrieval, so a single unlucky phrasing cannot
    # carry the result. Pinned because this is the finding that most changes how
    # the mechanism must be described.
    ("O11 reference: scored with the real question", 1.0000,
     lambda: pk("pk_8k_o11_realq.json", require={"query_aware": True}),
     5e-4, "1.0000", "p2"),
    ("O11 scored with a wrong question (weather)", 0.0556,
     lambda: pk("pk_8k_o11_wrongq.json"), 5e-4, "0.0556", "p2"),
    ("O11 scored with a second wrong question (sky)", 0.0000,
     lambda: pk("pk_8k_o11_wrongq2.json"), 5e-4, "0.0000", "p2"),
    ("O11 no query pass at all", 0.0000,
     lambda: pk("pk_8k_o11_noqa.json", require={"query_aware": False}),
     5e-4, "0.0000", "p2"),
    # --- O12 decode half, from data already committed ----------------------
    ("O12 scoring cost at decode, 16K", 0.2025,
     lambda: (measured(KVS / "measure_decode_full_16384.json", "ms_per_token", arm="evict")
              / measured(KVS / "measure_decode_full_16384.json", "ms_per_token",
                         arm="evict_noscore")) - 1.0, 5e-4, "20.3%", "p2"),
    ("O12 compression is SLOWER at 4K", 0.8259,
     lambda: measured(KVS / "measure_decode_full_4096.json", "ms_per_token", arm="full")
             / measured(KVS / "measure_decode_full_4096.json", "ms_per_token", arm="evict"),
     5e-4, "0.83", "p2"),
    # --- O12: per-compression selection cost, measured without a model ------
    ("O12 directional pooling is FREE vs centred at 32K", 1.02,
     lambda: _sel(32768, "centred (3,3)") / _sel(32768, "causal (6,0)"),
     0.02, "1.02", "p2"),
    ("O12 derived reach costs 2.14x at 2K", 2.14,
     lambda: _sel(2048, "derived reach grow=2.0") / _sel(2048, "causal (6,0)"),
     0.02, "2.14", "p2"),
    ("O12 derived reach costs only 1.12x at 128K", 1.12,
     lambda: _sel(131072, "derived reach grow=2.0") / _sel(131072, "causal (6,0)"),
     0.02, "1.12", "p2"),
    # --- O12 final: the query-aware prefill pass, CPU-only ------------------
    # The structural claim. The docstring predicted cost proportional to token
    # count; this is the fraction that is NOT, and it is what makes the naive
    # prediction wrong by ~8x. Pinned to three documents because a number that
    # carries an argument in three places can drift in two of them.
    ("O12 fixed floor is 83% of a 16-token forward", 0.8297,
     _qp_floor_share, 5e-4, "82.97%", "p2,c2paper"),
    ("O12 forward-cost fit R^2", 0.99721,
     lambda: _qp("component_ladder", "fit_r2"), 5e-5, "0.9972", "p2,c2paper"),
    ("O12 query pass share of prefill, chunk 512", 0.1549,
     lambda: _qp("end_to_end", 512, "query_share_of_aware"), 5e-4,
     "15.49%", "p2,c2paper"),
    ("O12 query pass share of prefill, chunk 2048", 0.0554,
     lambda: _qp("end_to_end", 2048, "query_share_of_aware"), 5e-4,
     "5.54%", "p2,c2paper"),
    ("O12 docstring overstated, chunk 512", 7.9,
     lambda: _qp("end_to_end", 512, "overstatement_factor"), 0.05, "7.9", "p2,c2paper"),
    ("O12 docstring overstated, chunk 2048", 11.3,
     lambda: _qp("end_to_end", 2048, "overstatement_factor"), 0.05,
     "11.3", "p2,c2paper"),
    # RELATIONAL pin. The claim is not "restore is slow", it is "restore scales
    # with the CACHE and not with the question". A pin on the absolute ms cannot
    # catch that being stated backwards; this one can. 8x length -> 8x cost.
    ("O12 _restore scales linearly with cache length", 8.0,
     lambda: _qp("component_ladder", "restore", 3, "ms_min")
             / _qp("component_ladder", "restore", 0, "ms_min"),
     0.3, "99.47", "p2,c2paper"),
    # THE DIRECTION THAT MATTERS. Cross-arm ratios carry ~0.05 of run-to-run
    # spread, so the tolerance is deliberately loose: what is being pinned is
    # that the mechanism is FASTER than no mechanism, not the third digit.
    ("O12 full mechanism is FASTER than no compression, chunk 2048", 0.844,
     lambda: _qp("end_to_end", 2048, "aware_over_plain"), 5e-3, "0.844", "p2,c2paper"),
    ("O12 the GPU was never touched", 0.0,
     lambda: float(bool(json.loads(
         (KVS / "measure_query_prefill.json").read_text(encoding="utf-8")
     )["cuda_initialized"])), 1e-9, "no GPU touched", "p2"),
    ("CENSUS: committed cells that faced ZERO compressions", 12,
     exempt_cells, 0.5, "8 faced zero compressions", "p2"),
    ("LANG heavy_zero under PROPORTIONAL recent", 0,
     lambda: json.loads((KVS / "frac0.25-rf0.25-b6f0__rag_dev" / "run_manifest.json")
                        .read_text(encoding="utf-8"))["heavy_zero_count"],
     0.5, "zero", "p2"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    text = {}
    for name, path in DOCS.items():
        if not path.exists():
            raise SystemExit(f"missing {path}")
        text[name] = path.read_text(encoding="utf-8")

    print(f"{'claim':<34}{'documented':>12}{'committed':>12}  {'in doc':>7}  verdict")
    bad = 0
    for label, doc_val, fn, tol, needle, which in CLAIMS:
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001
            got = None
            if args.verbose:
                print(f"  {label}: {type(e).__name__}: {e}")
        # `which` may name several documents; the number has to appear in each,
        # so a figure duplicated across docs cannot drift in one copy only.
        in_doc = all(needle in text[w] for w in which.split(","))
        ok = got is not None and abs(doc_val - got) <= tol and in_doc
        bad += not ok
        shown = f"{got:>12.5f}" if isinstance(got, (int, float)) else f"{'MISSING':>12}"
        print(f"{label:<34}{doc_val:>12.5f}{shown}  {str(in_doc):>7}  "
              f"{'OK' if ok else 'MISMATCH'}")

    # Coverage: a document listed in DOCS but carrying no pinned claim is only
    # protected against deletion. FINDINGS_CAPACITY_FLOOR.md sat in that state
    # while a wrong division survived in its headline formula, and the 270/270
    # pass rate said nothing about it, because the pass rate averages over a
    # document set whose per-document coverage included zeroes.
    pinned = set()
    for _lbl, _v, _fn, _tol, _needle, which in CLAIMS:
        pinned.update(which.split(","))
    uncovered = sorted(d for d in DOCS if d not in pinned)
    if uncovered:
        print()
        for d in uncovered:
            print(f"NO PINNED CLAIM guards {DOCS[d].name} (tag {d!r})")

    arith = check_inline_arithmetic(text)
    if arith:
        print()
        for name, expr, got in arith:
            print(f"ARITHMETIC MISMATCH in {name}: {expr}  evaluates to {got:.4f}")
    n = len(CLAIMS)
    print(f"\n{n - bad}/{n} documented numbers reproduce from committed data")
    print(f"{len(arith)} inline formulas in docs/ disagree with their own "
          f"arithmetic")
    bad += len(arith)
    if bad:
        print("A mismatch means the prose and the data disagree. Fix whichever is")
        print("wrong -- and if the data is right, state the retraction rather than")
        print("silently overwriting the old figure.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
