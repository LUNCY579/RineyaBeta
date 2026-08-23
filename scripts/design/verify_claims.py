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
    ("pool=7 exponent: e=0 depth 0.9", 0.1667,
     lambda: pk("pk_128k_pool7_e0.0.json", 0.9), 5e-4, "0.17", "findings"),
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
