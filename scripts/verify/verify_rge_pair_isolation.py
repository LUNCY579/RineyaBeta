#!/usr/bin/env python3
"""CPU-only negative controls for RGE pair execution isolation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiment"))
import rge_optimizer_pair as pair  # noqa: E402


def expect_exit(function, needle: str) -> None:
    try:
        function()
    except SystemExit as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected {needle!r}, got {exc!r}") from exc
    else:
        raise AssertionError(f"expected SystemExit containing {needle!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rge_pair_isolation_") as tmp:
        root = Path(tmp)
        pair.refuse_nonempty_run_dir(root / "missing", arm="control")
        checks.append("missing fresh directory accepted")

        empty = root / "empty"
        empty.mkdir()
        pair.refuse_nonempty_run_dir(empty, arm="control")
        checks.append("empty fresh directory accepted")

        stale = root / "stale"
        stale.mkdir()
        (stale / "train_log.jsonl").write_text("{}\n", encoding="utf-8")
        expect_exit(lambda: pair.refuse_nonempty_run_dir(stale, arm="control"),
                    "requires a fresh run directory")
        checks.append("stale log directory rejected")

        file_path = root / "not_a_dir"
        file_path.write_text("x", encoding="utf-8")
        expect_exit(lambda: pair.refuse_nonempty_run_dir(file_path, arm="candidate"),
                    "not a directory")
        checks.append("non-directory run path rejected")

        control = root / "control"
        candidate = root / "candidate"
        control.mkdir(); candidate.mkdir()
        (control / "summary.json").write_text("{}", encoding="utf-8")
        (candidate / "summary.json").write_text("{}", encoding="utf-8")
        original = pair.summarize_run
        pair.summarize_run = lambda path: {"arm": path.name}
        try:
            selected = pair.summarize_selected(
                {"control": control, "candidate": candidate}, ("control",))
        finally:
            pair.summarize_run = original
        if selected != {"control": {"arm": "control"}}:
            raise AssertionError(selected)
        checks.append("single-arm invocation ignores stale opposite-arm summary")

        if set(pair.parse_order("control-candidate")) != {"control", "candidate"}:
            raise AssertionError("pair order lost an arm")
        checks.append("two-arm order retains both arms")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "checks": len(checks),
        "check_names": checks,
        "sealed_data_accessed": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.evidence:
        path = Path(args.evidence)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
