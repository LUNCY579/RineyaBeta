#!/usr/bin/env python3
"""Import the frozen evaluation sets and the upstream quality scorer.

Why vendor rather than rewrite
------------------------------
The upstream tree already contains a versioned quality evaluator (v4.5) whose
output is what every prior Rineya artifact was scored with -- `quality_score`,
`intelligibility_diagnostics` (which is the degeneration screen C7 needs, already
built), `response_contract` brevity checks, and the per-category machinery C6
needs. Writing a fresh scorer would make R2's numbers incomparable with the entire
prior body of work, which is worse than useless: it would look like a measurement.

So the scorer is copied **byte-identically** and the upstream directory layout is
mirrored (`scripts/common/` + `configs/`), which lets every internal path in those
files resolve without a single edit. Provenance and sha256 for each file land in
manifests/eval_assets.json.

Evaluation sets imported
------------------------
  RAG DEV      752 rows, the gate set for C6
  RAG TEST     738 rows, sealed -- imported for completeness, NOT to be used
               until a rung is frozen
  chat DEV     510 rows, reconstructed from a prior run's results.jsonl by
               dropping every model-produced field and keeping only the inputs
               (id, category, messages, reference, response_contract)

The chat DEV inputs do not exist upstream as a standalone file; they are only
recoverable from result dumps. The reconstruction is explicit about which fields
are inputs so it cannot silently carry a previous model's answers into a gate.

Usage: python scripts/eval/import_eval_assets.py [--upstream PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble")

# Fields in a chat_dev results.jsonl that are INPUTS, not model output.
CHAT_DEV_INPUT_FIELDS = ("id", "category", "messages", "reference", "response_contract")

# (upstream relative path, destination relative path)
SCORER_FILES = (
    ("short-response-model-phase-0-1-2/scripts/common/quality_evaluation.py",
     "scripts/common/quality_evaluation.py"),
    ("short-response-model-phase-0-1-2/scripts/common/observable_quality.py",
     "scripts/common/observable_quality.py"),
    # Supplies finalize_response / build_stopping_criteria / infer_stop_reason.
    # For mode "eos_only" these reduce to "no extra criteria, no trimming", but the
    # real functions are used rather than that shortcut reimplemented.
    ("short-response-model-phase-0-1-2/scripts/common/response_control.py",
     "scripts/common/response_control.py"),
    ("short-response-model-phase-0-1-2/configs/quality_evaluation.json",
     "configs/quality_evaluation.json"),
    ("short-response-model-phase-0-1-2/configs/quality_issue_taxonomy_v4_5.json",
     "configs/quality_issue_taxonomy_v4_5.json"),
)

EVAL_SETS = (
    ("rag_dev_752",
     "short-response-model-phase-0-1-2/artifacts/phase_9_grounded_recovery_v1_7_5_qwen/dataset/dev.jsonl",
     "artifacts/eval_sets/rag_dev.jsonl", 752, "copy", "C6 gate set"),
    ("rag_test_738_SEALED",
     "short-response-model-phase-0-1-2/artifacts/phase_9_grounded_recovery_v1_7_5_qwen/dataset/test.jsonl",
     "artifacts/eval_sets/rag_test_SEALED.jsonl", 738, "copy",
     "sealed; do not use until a rung is frozen"),
    # The real benchmark file, named by the upstream evaluation_manifest.json and
    # verified against the sha256 it records. No reconstruction from result dumps.
    ("chat_dev_510",
     "short-response-model-phase-0-1-2/data/dev/dev.jsonl",
     "artifacts/eval_sets/chat_dev.jsonl", 510, "copy",
     "C5 gate set; upstream sha256 e059e548f684c76e1520d98fb4184ea7b8eda2053c3c23eed46794bcadfef954"),
)

# The generation protocol the anchors were produced under. Any comparison against
# them is only valid if R2 is generated the same way, so it is recorded here rather
# than left to be rediscovered.
GENERATION_PROTOCOL = {
    "max_new_tokens": 32,
    "response_control": "eos_only",
    "min_new_tokens_before_boundary": 1,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "source": "artifacts/teacher_dual_sft_v1/baseline_chat_dev/evaluation_manifest.json",
}

# How the scorer must be called, taken from the upstream reference implementation
# (Rineya_Layerwise_Soup_Lab_v1.0/src/rineya_soup_lab/evaluation/native_v45.py):
#   corpus_profile = build_corpus_profile(dataset_records)   # inputs, not results
#   evaluate_response(record, reply, visible_token_count,
#                     visible_model_tokens=visible_token_count, ...)
# Getting either detail wrong shifts quality_score by ~3e-3 on a fifth of the rows.
SCORER_INVOCATION = {
    "corpus_profile_from": "dataset records (inputs), not result rows",
    "token_count_arg": "visible_generated_tokens",
    "visible_model_tokens_arg": "visible_generated_tokens",
    "reference": "Rineya_Layerwise_Soup_Lab_v1.0/src/rineya_soup_lab/evaluation/native_v45.py",
}

# Prior scores on the same 510 chat DEV cases, kept as historical reference points.
# Both gates were scored with the same v4.5 evaluator, so chat and RAG anchors are
# directly comparable to what R2 will produce.
ANCHORS = (
    ("chat_dev__teacher_baseline",
     "short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/baseline_chat_dev/summary.json"),
    ("chat_dev__teacher_post_sft",
     "short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/post_sft_chat_dev/summary.json"),
    ("rag_dev__teacher_baseline",
     "short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/baseline_rag_dev/summary.json"),
    ("rag_dev__teacher_post_sft",
     "short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/post_sft_rag_dev/summary.json"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_outputs(src: Path, dst: Path) -> tuple[int, list[str]]:
    """Keep only input fields, so no prior model's reply can leak into a gate."""
    kept, dropped = 0, set()
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            dropped |= set(row) - set(CHAT_DEV_INPUT_FIELDS)
            fout.write(json.dumps({k: row[k] for k in CHAT_DEV_INPUT_FIELDS if k in row},
                                  ensure_ascii=False, sort_keys=True) + "\n")
            kept += 1
    return kept, sorted(dropped)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    args = ap.parse_args()
    up = Path(args.upstream)
    if not up.exists():
        raise SystemExit(f"upstream tree not found: {up}")

    manifest: dict = {"upstream_root": str(up), "scorer": {}, "eval_sets": {},
                      "anchors": {}, "notes": {},
                      "generation_protocol": GENERATION_PROTOCOL,
                      "scorer_invocation": SCORER_INVOCATION}

    print("=== scorer (vendored byte-identical; upstream layout mirrored so its "
          "internal paths resolve unmodified) ===")
    for rel_src, rel_dst in SCORER_FILES:
        src, dst = up / rel_src, ROOT / rel_dst
        if not src.exists():
            raise SystemExit(f"missing upstream scorer file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
        if digest != sha256_file(src):
            raise SystemExit(f"copy mismatch for {rel_dst}")
        manifest["scorer"][rel_dst] = {"upstream": rel_src, "sha256": digest,
                                       "bytes": dst.stat().st_size}
        print(f"  {rel_dst:52s} {digest[:16]}  {dst.stat().st_size:>7,d} B")

    # Make scripts/ and scripts/common/ importable so the vendored module's
    # `from scripts.common.observable_quality import ...` works untouched.
    for pkg in (ROOT / "scripts" / "__init__.py", ROOT / "scripts" / "common" / "__init__.py"):
        if not pkg.exists():
            pkg.write_text("", encoding="utf-8")
            print(f"  created {pkg.relative_to(ROOT)} (package marker)")

    print("\n=== evaluation sets ===")
    for name, rel_src, rel_dst, expect, mode, note in EVAL_SETS:
        src, dst = up / rel_src, ROOT / rel_dst
        if not src.exists():
            print(f"  {name:22s} MISSING upstream: {rel_src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "strip_outputs":
            rows, dropped = strip_outputs(src, dst)
            manifest["notes"][name] = (
                f"reconstructed from a results dump; kept {list(CHAT_DEV_INPUT_FIELDS)}, "
                f"dropped {len(dropped)} model-produced fields")
        else:
            shutil.copy2(src, dst)
            rows = sum(1 for l in dst.open(encoding="utf-8") if l.strip())
            dropped = []
        ok = rows == expect
        manifest["eval_sets"][name] = {
            "upstream": rel_src, "path": rel_dst, "rows": rows,
            "expected_rows": expect, "rows_match": ok,
            "sha256": sha256_file(dst), "note": note,
            "dropped_output_fields": dropped}
        print(f"  {name:22s} {rows:>4,d} rows {'ok' if ok else f'EXPECTED {expect}'}"
              f"  -> {rel_dst}")
        if dropped:
            print(f"    dropped {len(dropped)} model-produced fields, e.g. "
                  f"{', '.join(dropped[:5])}")

    print("\n=== historical anchors (prior scores on the same 510 cases) ===")
    for name, rel in ANCHORS:
        src = up / rel
        if not src.exists():
            print(f"  {name:34s} missing")
            continue
        payload = json.loads(src.read_text(encoding="utf-8"))
        manifest["anchors"][name] = {"upstream": rel, "summary": payload}
        keys = [k for k in ("mean_quality_score", "quality_pass_rate",
                            "intelligibility_pass_rate", "case_count") if k in payload]
        print(f"  {name:34s} " + "  ".join(f"{k}={payload[k]}" for k in keys[:3]))

    out = ROOT / "manifests" / "eval_assets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
