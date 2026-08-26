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
}


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
    ("CENSUS: committed cells that faced ZERO compressions", 8,
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

    n = len(CLAIMS)
    print(f"\n{n - bad}/{n} documented numbers reproduce from committed data")
    if bad:
        print("A mismatch means the prose and the data disagree. Fix whichever is")
        print("wrong -- and if the data is right, state the retraction rather than")
        print("silently overwriting the old figure.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
