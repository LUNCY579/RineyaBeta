#!/usr/bin/env python3
"""Prove the ordinary evaluator refuses RGE access to the sealed test set."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "verify_rge_sealed_boundary"
EVIDENCE = ROOT / "experiments" / "rge_peft" / "evidence" / "m1_sealed_boundary.json"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    command = [sys.executable, str(ROOT / "scripts/eval/run_gates.py"),
               "--dataset", "rag_test_SEALED", "--out", str(OUT), "--limit", "1"]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    output = proc.stdout + proc.stderr
    result_files = ([path.relative_to(OUT).as_posix() for path in OUT.rglob("*")
                    if path.is_file()] if OUT.exists() else [])
    refused = proc.returncode != 0 and "refusing to evaluate the sealed test set" in output
    passed = refused and not result_files
    evidence = {
        "schema_version": 1, "status": "PASS" if passed else "FAIL",
        "command": "scripts/eval/run_gates.py --dataset rag_test_SEALED --limit 1",
        "returncode": proc.returncode, "refusal_observed": refused,
        "result_files_created": result_files, "sealed_records_loaded": False,
        "output_tail": output.strip().splitlines()[-3:],
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2, sort_keys=True),
                        encoding="utf-8", newline="\n")
    print(json.dumps(evidence, indent=2))
    shutil.rmtree(OUT, ignore_errors=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
