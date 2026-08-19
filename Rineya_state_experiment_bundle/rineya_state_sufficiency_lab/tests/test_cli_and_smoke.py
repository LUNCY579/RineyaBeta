from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rineya_lab.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parser_exposes_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    commands = set(actions[0].choices)
    assert commands == {"validate-data", "plan", "budget", "train", "evaluate", "compare",
                        "aggregate", "smoke", "info"}


def test_print_config_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["plan", "-c", str(REPO_ROOT / "configs" / "train_full.yaml"), "--print-config"])
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment"]["condition"] == "FULL"


def test_bad_config_returns_exit_code_two(capsys):
    code = main(["plan", "--set", "lora.r=nonsense"])
    assert code == 2
    assert "Config error" in capsys.readouterr().err


def test_missing_dataset_returns_exit_code_three(tmp_path, capsys):
    """An explicit root that does not exist must fail, never silently fall back."""
    code = main(["plan", "--dataset-root", str(tmp_path / "nowhere")])
    assert code == 3
    stderr = capsys.readouterr().err
    assert "Not found" in stderr
    assert "explicit dataset root" in stderr


def test_plan_subcommand(dataset_root, tmp_path, capsys):
    code = main([
        "plan",
        "-c", str(REPO_ROOT / "configs" / "train_split_stateful.yaml"),
        "--dataset-root", str(dataset_root),
        "--set", "dataset.verify_hashes=false",
        "--set", "dataset.train_split=dev",
        "--set", "data_view.max_records=50",
        "-o", str(tmp_path / "plan.json"),
    ])
    assert code == 0
    payload = json.loads((tmp_path / "plan.json").read_text())
    assert payload["plan"]["view"] == "split_stateful"
    diagnostics = payload["order_diagnostics"]["epoch_0"]
    ref = diagnostics["uniform_shuffle_reference"]
    # Ceiling, not equality -- see test_views.test_shuffled_order_scatters_same_problem_rows.
    assert diagnostics["adjacent_same_problem"] <= ref["adjacent_same_problem_upper"]
    assert diagnostics["adjacent_consecutive_steps"] <= ref["adjacent_consecutive_steps_upper"]


def test_validate_data_subcommand(dataset_root, tmp_path):
    code = main([
        "validate-data",
        "--dataset-root", str(dataset_root),
        "--set", "dataset.verify_hashes=false",
        "--splits", "dev",
        "--skip-hashes",
        "-o", str(tmp_path / "validation.json"),
    ])
    assert code == 0
    payload = json.loads((tmp_path / "validation.json").read_text())
    assert payload["status"] == "PASS"


def test_budget_subcommand_matches_two_arms(dataset_root, tmp_path):
    code = main([
        "budget",
        "-c", str(REPO_ROOT / "configs" / "dry_run_full.yaml"),
        "--config2", str(REPO_ROOT / "configs" / "dry_run_split.yaml"),
        "--dataset-root", str(dataset_root),
        "--sample", "60",
        "-o", str(tmp_path / "budget.json"),
    ])
    assert code == 0
    payload = json.loads((tmp_path / "budget.json").read_text())
    matched = payload["matched_budget"]
    assert matched["matched_target_tokens"] <= payload["primary"]["projected_target_tokens"]
    assert matched["matched_target_tokens"] <= payload["secondary"]["projected_target_tokens"]


def test_evaluate_subcommand_with_reference_backend(dataset_root, tmp_path):
    code = main([
        "evaluate",
        "-c", str(REPO_ROOT / "configs" / "eval_full.yaml"),
        "--dataset-root", str(dataset_root),
        "--output-dir", str(tmp_path / "eval_full"),
        "--set", "dataset.verify_hashes=false",
        "--set", "evaluation.backend.kind=reference",
        "--set", "evaluation.backend.adapter_path=null",
        "--set", "evaluation.limit=6",
    ])
    assert code == 0
    assert (tmp_path / "eval_full" / "eval_report.json").exists()
    assert (tmp_path / "eval_full" / "resolved_config.json").exists()


def test_info_subcommand(dataset_root, capsys):
    assert main(["info", "--dataset-root", str(dataset_root), "--set", "dataset.verify_hashes=false"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "environment" in payload and "lab_version" in payload


def test_smoke_subcommand_passes(dataset_root, tmp_path, capsys):
    code = main(["smoke", "--dataset-root", str(dataset_root), "--output-dir", str(tmp_path / "smoke")])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS", payload
    assert code == 0
    assert payload["steps"]["failure_detection"]["ok"]


def test_console_entry_point_is_installed():
    result = subprocess.run([sys.executable, "-m", "rineya_lab.cli", "--version"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "rineya-lab" in result.stdout


# -- log file handles: one at a time, and released ---------------------------------------


def test_setup_logging_keeps_one_log_file_and_releases_the_previous(tmp_path):
    """Per-run log files must be per-run, and must not stay open.

    setup_logging() used to *add* a FileHandler on every call and remove none.
    Two consequences, both reproduced by this test before the fix:

      * a line logged during run B also landed in run A's log file, so the
        per-run artifacts in docs/READINESS_ASSESSMENT.md 6.5 were not per-run;
      * the handle stayed open, and on Windows that made the directory
        undeletable -- `smoke --all` passed every probe and then died in
        TemporaryDirectory.cleanup() with WinError 32 on evaluate.log.
    """
    import logging
    import shutil

    from rineya_lab.utils.logging import close_log_files, get_logger, setup_logging

    root = logging.getLogger("rineya_lab")
    logger = get_logger("test.logfiles")

    # Earlier tests in this module run evaluate/smoke, which leave a tracked
    # handler attached. Start from a clean slate so the count below measures
    # this test's own calls rather than ambient session state.
    close_log_files()
    before = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    try:
        first = tmp_path / "run_a" / "train.log"
        second = tmp_path / "run_b" / "evaluate.log"

        setup_logging(log_file=first)
        logger.info("belongs to run A")
        setup_logging(log_file=second)
        logger.info("belongs to run B")

        # Exactly one file handler is attached, not one per call.
        attached = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(attached) == len(before) + 1

        # Run A's log did not absorb run B's output.
        assert "belongs to run A" in first.read_text(encoding="utf-8")
        assert "belongs to run B" not in first.read_text(encoding="utf-8")
        assert "belongs to run B" in second.read_text(encoding="utf-8")

        # And once closed, the directory can actually be deleted -- this is the
        # assertion that fails on Windows with a leaked handle.
        close_log_files()
        shutil.rmtree(second.parent)
        assert not second.parent.exists()
    finally:
        close_log_files()
        for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler) and h not in before]:
            root.removeHandler(handler)
            handler.close()
