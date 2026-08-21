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
}


def summary(base: Path, tag: str, ds: str = "rag_dev"):
    p = base / f"{tag}__{ds}" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["mean_quality_score"]


def measured(path: Path, key: str, **match):
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            return r.get(key)
    return None


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
       length: int | None = None):
    """Passkey accuracy from a committed arm; pooled over depths if depth is None.

    Skips cells whose acc is null -- those are trials that ran out of memory and
    measured nothing. Averaging a null as zero is the exact mistake the harness
    itself used to make.
    """
    path = KVS / fname
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
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
    ("decay: e=1 at depth 0.9", 0.8333, lambda: pk("pk_128k_e1.0.json", 0.9),
     5e-4, "5/6", "findings,negative"),
    ("decay: e=0 at depth 0.9", 0.1667, lambda: pk("pk_128k_e0.0.json", 0.9),
     5e-4, "1/6", "findings,negative"),
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
