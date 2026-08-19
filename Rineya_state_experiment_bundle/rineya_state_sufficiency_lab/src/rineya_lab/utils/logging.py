"""Minimal logging setup shared by every entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_HANDLERS: list[logging.FileHandler] = []


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the shared logger, keeping at most one open log file.

    ``train`` and ``evaluate`` each call this with their own ``log_file``, and a
    single process can invoke them many times -- ``smoke --all`` does so around
    fifteen times. Previously every call *added* a FileHandler and none were
    ever removed, with two consequences:

    * every log line was duplicated into every log file opened earlier in the
      process, so ``runs/<train>/train.log`` also accumulated the output of
      every later evaluation -- the per-run artifacts promised in
      ``docs/READINESS_ASSESSMENT.md`` 6.5 were not per-run at all;
    * the handles stayed open, which on Windows makes the directory
      undeletable. ``smoke --all`` passed every probe and then died in
      ``TemporaryDirectory.cleanup()`` with WinError 32 on ``evaluate.log``.

    Attaching a new log file therefore closes and detaches the previous one.
    """
    global _CONFIGURED
    root = logging.getLogger("rineya_lab")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not _CONFIGURED:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(stream)
        root.propagate = False
        _CONFIGURED = True
    if log_file is not None:
        close_log_files()
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
        _FILE_HANDLERS.append(file_handler)


def close_log_files() -> None:
    """Detach and close every log file this module opened.

    Call before deleting or moving a directory that holds one, and at the end of
    a long-lived process. Safe to call when nothing is open.
    """
    root = logging.getLogger("rineya_lab")
    while _FILE_HANDLERS:
        handler = _FILE_HANDLERS.pop()
        root.removeHandler(handler)
        handler.close()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rineya_lab.{name}")
