#!/usr/bin/env python3
"""Build selected CRCC merge candidates and run unsealed development gates.

Execution is opt-in through ``--execute``.  The runner uses argv lists (never a
shell), writes an atomic ledger after every step, and refuses any dataset whose
name or path is marked SEALED.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    SCHEMA_VERSION,
    assert_unsealed_path,
    atomic_write_json,
    object_sha256,
    read_json,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], *, execute: bool) -> dict[str, Any]:
    print(" ".join(command))
    if not execute:
        return {"status": "PLANNED", "command": command}
    started = _utc_now()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    result = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "started_at": started,
        "completed_at": _utc_now(),
        "command": command,
    }
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return result


def _selected(candidate: Mapping[str, Any], requested: list[str]) -> bool:
    if not requested:
        return candidate["analysis_role"] == "PRIMARY_MATCHED_CONTROL"
    cid = str(candidate["candidate_id"])
    return any(cid == value or cid.startswith(value) for value in requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if not args.build and not args.evaluate:
        raise ContractError("select --build, --evaluate, or both")
    matrix_path = resolve_repo_path(args.matrix, must_exist=True)
    matrix = read_json(matrix_path)
    if matrix.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("merge matrix schema is not CRCC v1")
    datasets = list(
        args.datasets
        if args.datasets is not None
        else matrix["development_evaluation"]["datasets"]
    )
    for dataset in datasets:
        assert_unsealed_path(Path(str(dataset)), purpose="CRCC development selection")
        if "SEALED" in str(dataset).upper():
            raise ContractError(f"sealed dataset refused: {dataset}")

    ledger_path = resolve_repo_path(
        args.ledger or str(matrix_path.parent / "execution_ledger.json")
    )
    ledger: dict[str, Any]
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        if ledger.get("matrix_sha256") != matrix["matrix_sha256"]:
            raise ContractError("existing ledger belongs to a different merge matrix")
    else:
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "status": "DEVELOPMENT_EXECUTION_ONLY",
            "created_at": _utc_now(),
            "matrix": repo_relative(matrix_path),
            "matrix_file_sha256": sha256_file(matrix_path),
            "matrix_sha256": matrix["matrix_sha256"],
            "sealed_evaluation_opened": False,
            "candidates": {},
        }

    chosen = [
        candidate
        for candidate in matrix["candidates"]
        if _selected(candidate, args.candidate)
    ]
    if not chosen:
        raise ContractError("candidate filter selected nothing")

    for candidate in chosen:
        cid = str(candidate["candidate_id"])
        record = ledger["candidates"].setdefault(cid, {"steps": []})
        if not candidate["ready"]:
            record["status"] = "SKIPPED_NOT_READY"
            print(f"skip {cid}: target or donor artifact not ready; rerun plan_merge_grid.py")
            atomic_write_json(ledger_path, ledger)
            continue
        try:
            if args.build:
                output_model = resolve_repo_path(candidate["output_model"])
                if (output_model / "config.json").is_file():
                    step = {
                        "status": "REUSED_EXISTING_MODEL",
                        "output_model": repo_relative(output_model),
                    }
                else:
                    step = _run(
                        [
                            sys.executable,
                            "scripts/critical_core/apply_layer_deltas.py",
                            "--plan",
                            candidate["plan"],
                        ],
                        execute=args.execute,
                    )
                step["kind"] = "build"
                record["steps"].append(step)
                atomic_write_json(ledger_path, ledger)

            if args.evaluate:
                model_path = resolve_repo_path(candidate["output_model"])
                if args.execute and not (model_path / "config.json").is_file():
                    raise FileNotFoundError(
                        f"candidate model is not built: {model_path}; add --build"
                    )
                for dataset in datasets:
                    eval_dir = matrix_path.parent / "eval" / cid / str(dataset)
                    command = [
                        sys.executable,
                        "scripts/eval/run_gates.py",
                        "--dataset",
                        str(dataset),
                        "--model",
                        repo_relative(model_path),
                        "--out",
                        repo_relative(eval_dir),
                        "--cap-mib",
                        str(matrix["development_evaluation"].get("cap_mib", 4096)),
                    ]
                    if args.limit is not None:
                        command.extend(["--limit", str(args.limit)])
                    step = _run(command, execute=args.execute)
                    step.update({"kind": "development_eval", "dataset": dataset})
                    summary_path = eval_dir / "summary.json"
                    if summary_path.exists():
                        step["summary"] = read_json(summary_path)
                        step["summary_sha256"] = sha256_file(summary_path)
                    record["steps"].append(step)
                    atomic_write_json(ledger_path, ledger)
            record["status"] = "PASS" if args.execute else "PLANNED"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "FAIL"
            record["error"] = f"{type(exc).__name__}: {exc}"
            atomic_write_json(ledger_path, ledger)
            if not args.continue_on_error:
                raise

    ledger["updated_at"] = _utc_now()
    ledger["ledger_sha256"] = object_sha256(
        {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    )
    atomic_write_json(ledger_path, ledger)
    print(f"wrote {repo_relative(ledger_path)}")


if __name__ == "__main__":
    main()
