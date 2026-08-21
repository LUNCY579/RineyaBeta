#!/usr/bin/env python3
"""Derive byte-identical reply counts into a small COMMITTED artifact.

WHY THIS EXISTS. Every "N/752 byte-identical" figure in this program is computed
from per-case `results.jsonl` files, and those are gitignored -- 132 MB of them.
So the counts were reproducible on this machine and nowhere else: a fresh clone
has the summary scores but not the records they were derived from, and
`verify_claims.py` checks that read the raw records would fail there while passing
here. That is the worst kind of check, one that only works for the person who
already knows the answer.

This walks the result directories, computes the counts once, and writes a few
kilobytes of JSON that IS committed. verify_claims then reads the committed
counts, and re-derives from the raw records only when they happen to be present --
disagreeing loudly if the two ever differ.

Byte-identical means the greedy reply string is unchanged against the baseline
run. It is the strictest available signal and the one that survives a scorer
being wrong: two of this program's aggregate metrics moved in the wrong direction
under damage, and the identity count is what exposed that.

Usage:
    python scripts/design/identity_counts.py            # write the artifacts
    python scripts/design/identity_counts.py --check    # verify, non-zero on drift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (output file, baseline dir, [comparison dirs]) -- baselines named explicitly
# because "the obvious baseline" differs per experiment.
GROUPS = [
    (ROOT / "experiments/kv_significance/results/identity_counts.json",
     ROOT / "experiments/kv_significance/results",
     "off__rag_dev",
     ["frac0.75__rag_dev", "frac0.5-pool7-fixed__rag_dev", "frac0.5__rag_dev",
      "frac0.25-rf0.25__rag_dev", "frac0.25__rag_dev", "frac0.75-rf1.0__rag_dev",
      "frac0.5-rf1.0__rag_dev", "frac0.25-rf1.0__rag_dev"]),
    (ROOT / "experiments/attention_shape/results/identity_counts.json",
     ROOT / "experiments/attention_shape/results",
     None,   # baseline lives outside this directory; see BASE_OVERRIDE
     ["base-8KV__chat_dev", "gqa-2KV__chat_dev", "gqa-1KV__chat_dev",
      "swa-w512-g4__chat_dev"]),
]

# The attention_shape comparison is against the stored gate baseline, which is a
# different harness entirely -- that is the whole point of the compat check.
BASE_OVERRIDE = {
    ROOT / "experiments/attention_shape/results": ROOT / "artifacts/gate_runs/r2_base_chat_dev",
}


def replies(d: Path) -> dict[str, str] | None:
    f = d / "results.jsonl"
    if not f.exists():
        return None
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r["reply"]
    return out


def compute(base_dir: Path, names: list[str], base_name: str | None):
    base_path = BASE_OVERRIDE.get(base_dir, base_dir / base_name if base_name else None)
    base = replies(base_path) if base_path else None
    if base is None:
        return None, f"baseline records absent: {base_path}"
    rows = {}
    for name in names:
        r = replies(base_dir / name)
        if r is None:
            continue
        keys = [k for k in base if k in r]
        if not keys:
            continue
        rows[name] = {"identical": sum(1 for k in keys if base[k] == r[k]),
                      "compared": len(keys)}
    return {"baseline": str(base_path.relative_to(ROOT)).replace("\\", "/"),
            "counts": rows}, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compare committed counts against a fresh derivation")
    args = ap.parse_args()

    bad = 0
    for out, base_dir, base_name, names in GROUPS:
        fresh, why = compute(base_dir, names, base_name)
        if fresh is None:
            # No raw records here. That is the normal state of a fresh clone, and
            # it is exactly why the committed artifact exists.
            print(f"{out.name}: cannot derive ({why})")
            if args.check and out.exists():
                print("  committed artifact present; nothing to compare against")
            continue

        if args.check:
            if not out.exists():
                print(f"MISSING {out}")
                bad += 1
                continue
            old = json.loads(out.read_text(encoding="utf-8"))
            if old.get("counts") != fresh["counts"]:
                print(f"DRIFT in {out}")
                for k in sorted(set(old.get("counts", {})) | set(fresh["counts"])):
                    a, b = old.get("counts", {}).get(k), fresh["counts"].get(k)
                    if a != b:
                        print(f"  {k}: committed {a} -> derived {b}")
                bad += 1
            else:
                print(f"{out.name}: {len(fresh['counts'])} entries match")
        else:
            out.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)}  ({len(fresh['counts'])} entries)")
            for k, v in fresh["counts"].items():
                print(f"  {k:<32} {v['identical']}/{v['compared']}")

    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
