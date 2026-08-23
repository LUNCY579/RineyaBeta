#!/usr/bin/env python3
"""Import every benchmark declared in configs/eval_registry.yaml.

`import_eval_assets.py` handles the two sets that live in the upstream Chat-Bubble
tree plus the vendored scorer. This handles the three that ship inside this
repository and were imported-but-unused, which is why every verdict in
COMPRESSION_FINDINGS.md rested on two benchmarks.

Row counts are checked against the registry rather than trusted, and each entry gets
a sha256 recorded. Sealed splits are imported so they exist and are hashed, but
`run_gates.py` refuses to evaluate them.

The puzzle sets need conversion: they ship raw (`prompt` + `final_answer`, no chat
template). They are converted `short_sft` style -- assistant target is the bare
answer -- matching MINI_MODEL_ROADMAP §12.5, where `think_sft` was rejected for
colliding with the no-CoT target rule and `enable_thinking=false`.

Usage: python scripts/eval/import_from_registry.py [--only name] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "eval_registry.yaml"

PUZZLE_SYSTEM = "Solve the problem and return only the short final answer requested by the user."

# Where each in-repo set comes from, and how to shape it. Entries already handled by
# import_eval_assets.py (chat_dev, rag_dev, rag_test_SEALED) are skipped.
IN_REPO = {
    "longctx_demo": ("Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl", None, None),
    "puzzle_validation": ("Rineya-Puzzle-Reasoning-51k/data/validation.jsonl", "puzzle", None),
    "puzzle_test_SEALED": ("Rineya-Puzzle-Reasoning-51k/data/test.jsonl", "puzzle", None),
    "puzzle_challenge_SEALED": ("Rineya-Puzzle-Reasoning-51k/data/challenge.jsonl", "puzzle", None),
    "lcr_dev": ("Rineya_LCR_SFT_v1/Rineya_LCR_SFT_v1/data/lcr_sft_v1.jsonl", None, ("split", "dev")),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_puzzle(row: dict) -> dict | None:
    """Raw puzzle row -> chat form, short_sft style (bare answer, no CoT)."""
    prompt = str(row.get("prompt") or "").strip()
    answer = str(row.get("final_answer") or "").strip()
    if not prompt or not answer:
        return None
    return {
        "id": row["id"],
        "family": row.get("family", "unlabelled"),
        "difficulty": row.get("difficulty"),
        "accepted_answers": row.get("accepted_answers"),
        "messages": [
            {"role": "system", "content": PUZZLE_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    out_manifest: dict = {}
    print(f"{'benchmark':26s} {'rows':>6s} {'expected':>9s} {'role':>7s}  status")

    for name, entry in registry.items():
        if args.only and name != args.only:
            continue
        if name not in IN_REPO:
            print(f"{name:26s} {'-':>6s} {entry.get('rows', '?'):>9} "
                  f"{entry.get('role', '?'):>7s}  handled by import_eval_assets.py")
            continue

        rel, convert, split_filter = IN_REPO[name]
        src = ROOT / rel
        if not src.exists():
            print(f"{name:26s} {'-':>6s} {entry.get('rows','?'):>9} "
                  f"{entry.get('role','?'):>7s}  MISSING {rel}")
            continue
        dst = ROOT / entry["path"]

        rows, skipped = [], 0
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if split_filter and str(row.get(split_filter[0])) != split_filter[1]:
                    continue
                if convert == "puzzle":
                    row = convert_puzzle(row)
                    if row is None:
                        skipped += 1
                        continue
                rows.append(row)

        expected = int(entry["rows"])
        ok = len(rows) == expected
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("w", encoding="utf-8", newline="\n") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        status = "ok" if ok else f"COUNT MISMATCH (expected {expected})"
        if skipped:
            status += f", {skipped} skipped for missing prompt/answer"
        print(f"{name:26s} {len(rows):>6,d} {expected:>9,d} "
              f"{entry.get('role','?'):>7s}  {status}")

        out_manifest[name] = {
            "path": entry["path"], "source": rel, "rows": len(rows),
            "expected_rows": expected, "rows_match": ok, "role": entry.get("role"),
            "converted": convert, "split_filter": split_filter,
            "sha256": sha256_file(dst) if (dst.exists() and not args.dry_run) else None,
        }

    if not args.dry_run and out_manifest:
        mf = ROOT / "manifests" / "eval_registry_import.json"
        mf.parent.mkdir(parents=True, exist_ok=True)
        mf.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
        print(f"\nwrote {mf.relative_to(ROOT)}")

    gates = [n for n, e in registry.items() if e.get("role") == "gate"]
    sealed = [n for n, e in registry.items() if e.get("role") == "sealed"]
    print(f"\ngates ({len(gates)}): {', '.join(gates)}")
    print(f"sealed ({len(sealed)}): {', '.join(sealed)}  -- run_gates refuses these")


if __name__ == "__main__":
    main()
