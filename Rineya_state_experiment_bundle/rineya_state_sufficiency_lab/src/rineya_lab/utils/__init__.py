"""Cross-cutting helpers: IO, hashing, determinism, environment capture, logging."""

from rineya_lab.utils.hashing import sha256_bytes, sha256_file, sha256_text, stable_shuffle_key
from rineya_lab.utils.io import (
    atomic_write_text,
    count_lines,
    iter_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from rineya_lab.utils.logging import close_log_files, get_logger, setup_logging

__all__ = [
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "stable_shuffle_key",
    "atomic_write_text",
    "count_lines",
    "iter_jsonl",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
    "get_logger",
    "setup_logging",
    "close_log_files",
]
