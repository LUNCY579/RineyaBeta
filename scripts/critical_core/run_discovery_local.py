#!/usr/bin/env python3
"""Run the complete CRCC v1 unsealed discovery pipeline on a local/private machine.

Execution order:
  1. clean branch + pinned runtime verification;
  2. runtime doctor + frozen discovery contract;
  3. official pinned checkpoint SHA-256 verification;
  4. integrity validation of any resumable artifacts;
  5. intact baseline + all 28 one-layer bypasses;
  6. independent resume validation after inference;
  7. audited discovery evidence assembly and G0 decision.

The script never invokes DEV, the historical 64-row evaluator, recovery training,
donor merging, or sealed evaluation. Interrupted scans are resumable after integrity
validation. A machine-readable execution_state.json is updated at every stage.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import atomic_write_json, read_json, resolve_repo_path
from scripts.critical_core.resume_integrity import validate_scan_directory

PREREG_HEAD = "242613d1b4b5e0a5a4a7154c60337115ca394bd7"
DEFAULT_CONFIG = "configs/critical_core_scan_v1.yaml"
DEFAULT_RAW = "runs/critical_core/crcc-v1/scan"
DEFAULT_DIAGNOSTICS = "runs/critical_core/crcc-v1/diagnostics"
DEFAULT_EVIDENCE = "experiments/catastrophic_removal_core/results/discovery_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


class ExecutionLedger:
    def __init__(self, path: Path, *, config: str, raw_dir: Path, evidence_dir: Path) -> None:
        self.path = path
        self.payload: dict[str, Any] = {
            "schema_version": "crcc.local-execution.v1",
            "program": "CRCC v1 unsealed discovery",
            "status": "INITIALIZING",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "preregistration_head": PREREG_HEAD,
            "execution_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "config": config,
            "raw_dir": str(raw_dir),
            "evidence_dir": str(evidence_dir),
            "scientific_evidence_complete": False,
            "membership_frozen": False,
            "sealed_evaluation_opened": False,
            "historical_long_context_64_executed": False,
            "stages": [],
        }
        self.write()

    def write(self) -> None:
        self.payload["updated_at"] = utc_now()
        atomic_write_json(self.path, self.payload)

    def begin(self, name: str, command: list[str] | None = None) -> None:
        self.payload["status"] = f"RUNNING__{name.upper()}"
        self.payload["stages"].append(
            {
                "name": name,
                "status": "RUNNING",
                "started_at": utc_now(),
                "command": command,
            }
        )
        self.write()

    def finish(self, name: str, **metadata: Any) -> None:
        for stage in reversed(self.payload["stages"]):
            if stage["name"] == name and stage["status"] == "RUNNING":
                stage["status"] = "PASS"
                stage["completed_at"] = utc_now()
                stage.update(metadata)
                break
        self.write()

    def fail(self, name: str, exc: BaseException) -> None:
        for stage in reversed(self.payload["stages"]):
            if stage["name"] == name and stage["status"] == "RUNNING":
                stage["status"] = "FAIL"
                stage["completed_at"] = utc_now()
                stage["failure"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                break
        self.payload["status"] = f"FAILED__{name.upper()}"
        self.write()


def run_checked(command: list[str], *, stage: str, ledger: ExecutionLedger) -> None:
    ledger.begin(stage, command)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except Exception as exc:
        ledger.fail(stage, exc)
        raise
    ledger.finish(stage)


def expected_ids_from_contract(diagnostics_dir: Path) -> list[str]:
    contract_path = diagnostics_dir / "discovery_contract.json"
    if not contract_path.exists():
        raise RuntimeError(f"runtime doctor did not write {contract_path}")
    contract = read_json(contract_path)
    ids = contract.get("diagnostic_case_ids")
    if not isinstance(ids, list) or len(ids) != 32:
        raise RuntimeError("discovery contract does not contain the frozen 32 case IDs")
    return [str(value) for value in ids]


def validate_resume_stage(
    *,
    raw_dir: Path,
    expected_ids: list[str],
    ledger: ExecutionLedger,
    stage: str,
) -> dict[str, Any]:
    ledger.begin(stage)
    try:
        report = validate_scan_directory(
            raw_dir,
            expected_ids=expected_ids,
            expected_layers=28,
        )
    except Exception as exc:
        ledger.fail(stage, exc)
        raise
    ledger.finish(
        stage,
        baseline_present=report["baseline_present"],
        present_layers=report["present_layers"],
        present_layer_count=report["present_layer_count"],
        complete=report["complete"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW)
    parser.add_argument("--diagnostics-dir", default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--offline", action="store_true", help="refuse checkpoint download")
    parser.add_argument("--skip-model-smoke", action="store_true", help="skip doctor's real-model intact/bypass forward smoke")
    parser.add_argument("--doctor-only", action="store_true", help="stop after preflight and resume validation")
    args = parser.parse_args()

    raw_dir = resolve_repo_path(args.raw_dir)
    diagnostics_dir = resolve_repo_path(args.diagnostics_dir)
    evidence_dir = resolve_repo_path(args.evidence_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ledger = ExecutionLedger(
        diagnostics_dir / "execution_state.json",
        config=args.config,
        raw_dir=raw_dir,
        evidence_dir=evidence_dir,
    )

    try:
        context_cmd = [
            sys.executable,
            "scripts/verify/verify_crcc_execution_context.py",
        ]
        run_checked(context_cmd, stage="execution_context", ledger=ledger)

        doctor_cmd = [
            sys.executable,
            "scripts/verify/doctor_crcc_runtime.py",
            "--config",
            args.config,
            "--report",
            str((diagnostics_dir / "runtime_doctor.json").relative_to(ROOT)),
        ]
        if args.offline:
            doctor_cmd.append("--offline")
        if not args.skip_model_smoke:
            doctor_cmd.append("--model-smoke")
        run_checked(doctor_cmd, stage="runtime_doctor", ledger=ledger)

        checkpoint_cmd = [
            sys.executable,
            "scripts/verify/verify_crcc_checkpoint.py",
            "--config",
            args.config,
        ]
        run_checked(checkpoint_cmd, stage="official_checkpoint_hash", ledger=ledger)

        expected_ids = expected_ids_from_contract(diagnostics_dir)
        before = validate_resume_stage(
            raw_dir=raw_dir,
            expected_ids=expected_ids,
            ledger=ledger,
            stage="resume_integrity_before_scan",
        )
        if args.doctor_only:
            ledger.payload["status"] = "DOCTOR_COMPLETE__NO_MODEL_EVIDENCE_RUN"
            ledger.payload["resume_state"] = before
            ledger.write()
            print("CRCC local doctor complete; discovery inference was not run.")
            return

        scanner_cmd = [
            sys.executable,
            "scripts/critical_core/scan_layer_ablation.py",
            "--config",
            args.config,
            "--out",
            str(raw_dir.relative_to(ROOT)),
            "--hash-weights",
        ]
        run_checked(scanner_cmd, stage="baseline_and_28_layer_scan", ledger=ledger)

        after = validate_resume_stage(
            raw_dir=raw_dir,
            expected_ids=expected_ids,
            ledger=ledger,
            stage="resume_integrity_after_scan",
        )
        if not after["complete"]:
            raise RuntimeError(
                f"scanner exited without complete evidence: baseline={after['baseline_present']} "
                f"layers={after['present_layer_count']}/28"
            )

        execution_commit = git_output("rev-parse", "HEAD") or "LOCAL_WORKTREE_UNKNOWN"
        session_id = f"local-{os.getpid()}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        assembler_cmd = [
            sys.executable,
            "scripts/critical_core/assemble_scan_evidence.py",
            "--config",
            args.config,
            "--run-manifest",
            str((raw_dir / "run_manifest.json").relative_to(ROOT)),
            "--baseline",
            str((raw_dir / "baseline.json").relative_to(ROOT)),
            "--layers-dir",
            str((raw_dir / "layers").relative_to(ROOT)),
            "--out-dir",
            str(evidence_dir.relative_to(ROOT)),
            "--source-head",
            PREREG_HEAD,
            "--execution-commit",
            execution_commit,
            "--workflow-run-id",
            session_id,
            "--raw-artifact-name",
            f"local-untracked-crcc-v1-{session_id}",
        ]
        run_checked(assembler_cmd, stage="assemble_and_g0", ledger=ledger)

        evidence = read_json(evidence_dir / "evidence_manifest.json")
        shutil.copy2(evidence_dir / "scan_summary.json", raw_dir / "scan_summary.json")
        shutil.copy2(evidence_dir / "evidence_manifest.json", raw_dir / "evidence_manifest.json")

        g0 = evidence.get("g0", {})
        g0_pass = bool(g0.get("pass"))
        ledger.payload["status"] = str(evidence.get("status", "DISCOVERY_COMPLETE_UNKNOWN"))
        ledger.payload["scientific_evidence_complete"] = True
        ledger.payload["membership_frozen"] = g0_pass
        ledger.payload["g0_pass"] = g0_pass
        ledger.payload["automatic_critical_layers"] = evidence.get("automatic_critical_layers", [])
        ledger.payload["membership_sha256"] = evidence.get("membership_sha256")
        ledger.payload["scan_summary_sha256"] = evidence.get("scan_summary_sha256")
        ledger.payload["sealed_evaluation_opened"] = False
        ledger.payload["historical_long_context_64_executed"] = False
        ledger.payload["completed_at"] = utc_now()
        ledger.write()

        print("\nCRCC local discovery execution complete.")
        print(f"status: {ledger.payload['status']}")
        print(f"G0: {'PASS' if g0_pass else 'FAIL'}")
        print("automatic CRCC layers:", ledger.payload["automatic_critical_layers"])
        print(f"evidence: {evidence_dir}")
        if not g0_pass:
            print("G0 failed: stop for instrument review; do not design matched recovery arms.")
    except Exception as exc:
        if not str(ledger.payload.get("status", "")).startswith("FAILED__"):
            ledger.payload["status"] = "FAILED__UNCLASSIFIED"
            ledger.payload["failure"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            ledger.write()
        raise


if __name__ == "__main__":
    main()