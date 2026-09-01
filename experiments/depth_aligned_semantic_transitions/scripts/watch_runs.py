"""Live watchdog for a batch of DAST training runs.

Emits ONE stdout line per event and exits when the batch is complete, so it can be
driven by a Monitor. Designed around a failure this program actually hit: a
long-running batch was stopped externally and left no completion record, so silence
looked identical to progress.

**Silence is not success**, so this watches for every terminal state, not just the
happy path:

  COMPLETE  a run finished and its summaries exist (accuracy reported)
  FAILED    a run's directory exists but training died without summaries
  DIED      no `dast.cli train` process is alive while runs remain incomplete
  STALE     the accelerator has been idle for several consecutive polls
  DONE      every expected run is complete (exit 0)

Liveness is judged by process existence and accelerator occupancy rather than by
file mtime, because `metrics.jsonl` is written only during training: a run in its
~35-minute generation phase writes nothing and would look stale to an mtime check.

Usage
-----
    python scripts/watch_runs.py --tags g3_s17_state_final_repeated g3_s29_... \\
        [--interval 60] [--stale-polls 3]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def emit(message: str) -> None:
    print(message, flush=True)


def _run_state(tag: str) -> str:
    """complete | training | absent"""

    directory = RUNS / tag
    if not directory.is_dir():
        return "absent"
    if (directory / "dev_summary.json").exists() and (
        directory / "generation_summary.json"
    ).exists():
        return "complete"
    return "training"


def _accuracy(tag: str) -> str:
    try:
        payload = json.loads(
            (RUNS / tag / "generation_summary.json").read_text(encoding="utf-8")
        )
        return f"{payload['accuracy']:.4f}"
    except Exception:  # noqa: BLE001
        return "?"


def _trainer_alive() -> bool:
    """Is a `dast.cli train` process running? Windows-friendly, no psutil."""

    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |"
                " Where-Object { $_.CommandLine -like '*dast.cli*train*' } |"
                " Measure-Object).Count",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:  # noqa: BLE001 - a failed probe must not kill the watch
        return True  # fail safe: assume alive rather than raise a false DIED


def _accelerator_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return int((out.stdout or "0").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return -1  # unknown; never counted as idle


_EXIT_LINE = re.compile(r"^### \[\d+/\d+\]\s+(?P<tag>\S+)\s+EXIT=(?P<code>-?\d+)")


def _batch_exits(log_path: str | None) -> list[tuple[str, int]]:
    """Per-run exit codes recorded by resume_runs.py, oldest first."""

    if not log_path:
        return []
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        match = _EXIT_LINE.match(line.strip())
        if match:
            out.append((match["tag"], int(match["code"])))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live watchdog for DAST run batches")
    parser.add_argument("--tags", nargs="+", required=True, help="run directory names under runs/")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--stale-polls", type=int, default=3, help="idle polls before STALE")
    parser.add_argument("--idle-mib", type=int, default=1500, help="below this the GPU is idle")
    parser.add_argument(
        "--batch-log",
        help="resume_runs.py log to scan for nonzero EXIT lines. Without this, a run "
             "that fails while the batch CONTINUES is invisible: the FAILED check "
             "below only fires when no trainer is alive at all.",
    )
    args = parser.parse_args(argv)

    seen_complete: set[str] = set()
    seen_failed: set[str] = set()
    seen_exit: set[str] = set()
    idle_streak = 0
    stale_reported = False
    started = time.monotonic()

    total = len(args.tags)
    already = [t for t in args.tags if _run_state(t) == "complete"]
    seen_complete.update(already)
    emit(f"WATCH armed: {len(already)}/{total} already complete, interval {args.interval:.0f}s")

    while True:
        states = {tag: _run_state(tag) for tag in args.tags}

        for tag, state in states.items():
            if state == "complete" and tag not in seen_complete:
                seen_complete.add(tag)
                elapsed = (time.monotonic() - started) / 60
                emit(
                    f"COMPLETE {tag} acc={_accuracy(tag)} "
                    f"({len(seen_complete)}/{total}, {elapsed:.0f} min into watch)"
                )

        if len(seen_complete) == total:
            emit(f"DONE all {total} runs complete")
            return 0

        # A run can die while the batch moves on to the next one. That leaves a
        # trainer alive, so the liveness check below stays quiet and the failure is
        # never reported. The batch log is the authoritative record of per-run exit
        # codes, so it is scanned every poll.
        for tag, code in _batch_exits(args.batch_log):
            key = f"{tag}:{code}"
            if code != 0 and key not in seen_exit:
                seen_exit.add(key)
                emit(
                    f"FAILED {tag} exited {code} while the batch continued -- "
                    f"re-queue it with: python scripts/resume_runs.py --run --tags {tag}"
                )

        alive = _trainer_alive()
        mib = _accelerator_mib()

        if not alive:
            # A run directory with no summaries and no live trainer is a dead run.
            for tag, state in states.items():
                if state == "training" and tag not in seen_failed:
                    seen_failed.add(tag)
                    emit(f"FAILED {tag} has no summaries and no live trainer")
            emit(
                f"DIED no dast.cli train process alive, {len(seen_complete)}/{total} complete "
                f"-- resume with: python scripts/resume_runs.py --tags <the same tags>"
            )
            return 1

        if 0 <= mib < args.idle_mib:
            idle_streak += 1
            if idle_streak >= args.stale_polls and not stale_reported:
                stale_reported = True
                emit(
                    f"STALE accelerator under {args.idle_mib} MiB for {idle_streak} polls "
                    f"while {total - len(seen_complete)} runs remain -- process alive but "
                    "possibly hung"
                )
        else:
            idle_streak = 0
            stale_reported = False

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
