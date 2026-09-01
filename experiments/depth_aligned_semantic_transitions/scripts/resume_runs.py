"""Resume a partially-completed batch of DAST training runs.

A run is COMPLETE only when both `dev_summary.json` and `generation_summary.json`
exist. Anything else — an absent directory, or one holding only `metrics.jsonl` and
`run_manifest.json` because the process was killed mid-generation — is re-run from
scratch.

Partial directories are deleted before re-running rather than resumed in place. The
trainer's checkpoint/resume path is explicitly marked engineering beta in
RUNBOOK section 10 ("do not promote resume as evidence until an interrupted-run
equivalence check has compared resumed and uninterrupted outputs"), so re-running is
the only option that yields a result anyone may cite.

Usage
-----
    python scripts/resume_runs.py --tags g3_s29_state_ordered g3_s43_...   # report
    python scripts/resume_runs.py --tags ... --run                        # execute
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"


def is_complete(tag: str) -> bool:
    directory = RUNS / tag
    return (directory / "dev_summary.json").exists() and (
        directory / "generation_summary.json"
    ).exists()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume an interrupted DAST run batch")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--run", action="store_true", help="execute instead of reporting")
    args = parser.parse_args(argv)

    complete, pending, missing_config = [], [], []
    for tag in args.tags:
        if is_complete(tag):
            complete.append(tag)
        elif not (CONFIGS / f"{tag}.yaml").exists():
            missing_config.append(tag)
        else:
            pending.append(tag)

    print(f"complete : {len(complete)}/{len(args.tags)}")
    for tag in complete:
        print(f"    ok      {tag}")
    for tag in pending:
        partial = (RUNS / tag).is_dir()
        print(f"    PENDING {tag}{'  (partial dir will be removed)' if partial else ''}")
    for tag in missing_config:
        print(f"    NO CONFIG {tag}  -> configs/{tag}.yaml does not exist")

    if missing_config:
        print("\nrefusing to proceed: a requested tag has no config", file=sys.stderr)
        return 2
    if not pending:
        print("\nnothing to do")
        return 0
    if not args.run:
        print(f"\n{len(pending)} run(s) would be executed. Re-run with --run to execute.")
        return 0

    failures = []
    for index, tag in enumerate(pending, start=1):
        directory = RUNS / tag
        if directory.is_dir():
            shutil.rmtree(directory)
        started = time.strftime("%H:%M:%S")
        print(f"### [{index}/{len(pending)}] {tag} start {started}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-u", "-m", "dast.cli", "train", "--config",
             str(CONFIGS / f"{tag}.yaml")],
            cwd=ROOT,
        )
        print(
            f"### [{index}/{len(pending)}] {tag} EXIT={completed.returncode} "
            f"{time.strftime('%H:%M:%S')}",
            flush=True,
        )
        if completed.returncode != 0 or not is_complete(tag):
            failures.append(tag)

    if failures:
        print(f"\nFAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nall pending runs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
