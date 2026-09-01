#!/usr/bin/env python3
"""Register the historical 64-row long-context evaluator and its known anchors.

CRCC's preregistration (docs/PLAN_CATASTROPHIC_REMOVAL_CORE.md 6.4) blocks any
Halo-comparable ``30/64`` claim until the exact historical protocol is pinned and
the known anchors reproduce under one executable path.  This module does that and
nothing else: it never merges a donor, never trains, and never opens a sealed set.

The suite itself lives outside this repository, so the registry pins content
hashes rather than vendoring 256 long-context prompts.  ``score_predictions``
re-implements the suite's ``strict_exact`` rule and is proven against the suite's
own stored reports before any new number is trusted.

Usage:
  python scripts/critical_core/register_longctx64.py \
      --suite-root <path to Rineya_Long_Context_Demo_v1> \
      --prediction qwen3_0_6b=<preds.jsonl> \
      --out experiments/catastrophic_removal_core/evaluators/longctx_showcase64_registry.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    assert_unsealed_path,
    atomic_write_json,
    object_sha256,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)

REGISTRY_SCHEMA = "crcc.longctx64-registry.v1"
EXPECTED_CASE_COUNT = 64

# Files whose bytes decide a score. Drivers are deliberately excluded: they are
# executable convenience, not evidence, and the suite's own MANIFEST already
# disagrees with them on disk.
EVIDENCE_FILES = (
    "data/showcase_prompts.jsonl",
    "data/showcase_answer_key.jsonl",
    "scripts/evaluate_predictions.py",
    "PROMPT_TEMPLATE.txt",
    "schema.json",
)

# Frozen generation protocol, recovered from the stored anchor artifacts.
PROTOCOL = {
    "max_new_tokens": 24,
    "do_sample": False,
    "decoding": "greedy",
    "use_cache": True,
    "seed": 17,
    "add_special_tokens": False,
    "chat_template": "apply_chat_template(add_generation_prompt=True)",
    "roles_kept": ["system", "user"],
    "max_context_tokens": 4096,
    "on_overflow": "error",
    "truncation": "never",
    "scoring": "strict_exact: normalized prediction in normalized accepted_answers",
}

_PREDICTION_FIELDS = ("prediction", "response", "output", "generated_text", "answer")

_QUOTE_CHARACTERS = "[" + "“”" + "\"'" + "`" + "]"


def normalize_answer(text: str) -> str:
    """Exact re-implementation of the suite scorer's normalisation."""

    value = str(text).strip().lower()
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", value)
    value = re.sub(_QUOTE_CHARACTERS, "", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _prediction_text(row: Mapping[str, Any]) -> str:
    for field in _PREDICTION_FIELDS:
        if row.get(field) is not None:
            return str(row[field])
    return ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def score_predictions(
    key_rows: Sequence[Mapping[str, Any]], prediction_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return strict-exact outcomes plus the per-case vector used for paired tests."""

    key_by_id = {str(row["id"]): row for row in key_rows}
    if len(key_by_id) != len(key_rows):
        raise ContractError("answer key contains duplicate ids")
    predictions: dict[str, Mapping[str, Any]] = {}
    for row in prediction_rows:
        if "id" not in row:
            continue
        predictions[str(row["id"])] = row

    outcomes: dict[str, bool] = {}
    covered = 0
    for case_id, key in sorted(key_by_id.items()):
        row = predictions.get(case_id)
        if row is None:
            outcomes[case_id] = False
            continue
        covered += 1
        accepted = {
            normalize_answer(value)
            for value in key.get("accepted_answers", [key.get("answer", "")])
        }
        outcomes[case_id] = normalize_answer(_prediction_text(row)) in accepted

    correct = sum(1 for value in outcomes.values() if value)
    return {
        "case_count": len(key_by_id),
        "covered": covered,
        "correct": correct,
        "strict_exact": correct / len(key_by_id) if key_by_id else 0.0,
        "per_case": outcomes,
    }


def _self_check(suite_root: Path, key_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prove this scorer reproduces the suite's own stored reports before use."""

    checks: list[dict[str, Any]] = []
    outputs = suite_root / "outputs"
    for report_path in sorted(outputs.glob("*_showcase_report.json")):
        stem = report_path.name[: -len("_report.json")]
        predictions_path = outputs / f"{stem}.jsonl"
        if not predictions_path.is_file():
            continue
        stored = json.loads(report_path.read_text(encoding="utf-8"))["runs"][0]
        recomputed = score_predictions(key_rows, read_jsonl(predictions_path))
        matched = abs(float(stored["strict_exact"]) - recomputed["strict_exact"]) < 1e-12
        checks.append(
            {
                "run": str(stored["run"]),
                "stored_strict_exact": float(stored["strict_exact"]),
                "recomputed_strict_exact": recomputed["strict_exact"],
                "recomputed_correct_of_64": recomputed["correct"],
                "reproduces": bool(matched),
            }
        )
        if not matched:
            raise ContractError(
                f"scorer disagrees with the suite's stored report for {stored['run']}: "
                f"{stored['strict_exact']} vs {recomputed['strict_exact']}"
            )
    if not checks:
        raise ContractError("no stored suite reports were available to validate the scorer")
    return checks


def build_registry(
    *,
    suite_root: Path,
    predictions: Mapping[str, Path],
    anchors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    assert_unsealed_path(suite_root, purpose="CRCC long-context registration")
    files: dict[str, dict[str, Any]] = {}
    for relative in EVIDENCE_FILES:
        path = suite_root / relative
        if not path.is_file():
            raise ContractError(f"suite is missing evidence file {relative}")
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    key_rows = read_jsonl(suite_root / "data/showcase_answer_key.jsonl")
    prompt_rows = read_jsonl(suite_root / "data/showcase_prompts.jsonl")
    if len(key_rows) != EXPECTED_CASE_COUNT or len(prompt_rows) != EXPECTED_CASE_COUNT:
        raise ContractError(
            f"showcase split must contain exactly {EXPECTED_CASE_COUNT} rows; "
            f"found prompts={len(prompt_rows)} key={len(key_rows)}"
        )
    if {str(row["id"]) for row in key_rows} != {str(row["id"]) for row in prompt_rows}:
        raise ContractError("showcase prompt and answer-key ids disagree")

    scorer_self_check = _self_check(suite_root, key_rows)

    reproductions: dict[str, Any] = {}
    for label, path in sorted(predictions.items()):
        scored = score_predictions(key_rows, read_jsonl(path))
        anchor = dict(anchors.get(label, {}))
        entry: dict[str, Any] = {
            "predictions_path": repo_relative(path),
            "predictions_sha256": sha256_file(path),
            "covered": scored["covered"],
            "correct_of_64": scored["correct"],
            "strict_exact": scored["strict_exact"],
            "per_case": scored["per_case"],
        }
        if "historical_correct_of_64" in anchor:
            historical = int(anchor["historical_correct_of_64"])
            entry["historical_correct_of_64"] = historical
            entry["delta_vs_historical"] = scored["correct"] - historical
        entry.update({k: v for k, v in anchor.items() if k != "historical_correct_of_64"})
        reproductions[label] = entry

    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "evaluator": "Rineya Long-Context Demonstration Suite v1.0.0 -- showcase split",
        "status": "REGISTERED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_root_is_external": True,
        "suite_root_observed_at": str(suite_root),
        "case_count": EXPECTED_CASE_COUNT,
        "evidence_files": files,
        "protocol": dict(PROTOCOL),
        "scorer_self_check": scorer_self_check,
        "reproductions": reproductions,
        "claim_boundary": (
            "Registration and anchor reproduction only. No CRCC donor was built, "
            "merged or evaluated on these cases, and no sealed set was opened."
        ),
    }
    registry["registry_sha256"] = object_sha256(
        {
            key: value
            for key, value in registry.items()
            if key not in {"created_at", "suite_root_observed_at"}
        }
    )
    return registry


def _split_pairs(items: Iterable[str]) -> dict[str, str]:
    table: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ContractError(f"expected LABEL=VALUE, got {item!r}")
        label, value = item.split("=", 1)
        table[label.strip()] = value.strip()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="reproduced prediction JSONL, repeatable",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        metavar="LABEL=N",
        help="historical strict-exact hit count out of 64, repeatable",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    predictions = {
        label: resolve_repo_path(value, must_exist=True)
        for label, value in _split_pairs(args.prediction).items()
    }
    anchors = {
        label: {"historical_correct_of_64": int(value)}
        for label, value in _split_pairs(args.anchor).items()
    }
    registry = build_registry(
        suite_root=Path(args.suite_root).expanduser().resolve(),
        predictions=predictions,
        anchors=anchors,
    )
    output = resolve_repo_path(args.out)
    atomic_write_json(output, registry)

    print(f"registered evaluator with {registry['case_count']} cases")
    for check in registry["scorer_self_check"]:
        print(
            f"  scorer self-check {check['run']:28s} "
            f"{check['recomputed_correct_of_64']:>2d}/64 reproduces={check['reproduces']}"
        )
    for label, row in sorted(registry["reproductions"].items()):
        historical = row.get("historical_correct_of_64")
        suffix = (
            f" (historical {historical}/64, delta {row['delta_vs_historical']:+d})"
            if historical is not None
            else ""
        )
        print(f"  reproduced {label:20s} {row['correct_of_64']:>2d}/64{suffix}")
    print(f"wrote {repo_relative(output)}")


if __name__ == "__main__":
    main()
