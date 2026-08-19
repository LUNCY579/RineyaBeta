"""Deterministic hashing helpers.

Every hash in this repository is SHA-256 over UTF-8 bytes. Python's builtin
``hash`` is never used for anything that must be reproducible across processes,
because it is randomized per interpreter unless ``PYTHONHASHSEED`` is pinned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a canonical JSON encoding (sorted keys, compact separators)."""
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def stable_shuffle_key(identifier: str, salt: str) -> str:
    """Deterministic sort key used for the global shuffle of transition rows.

    Shuffling by a salted hash rather than ``random.shuffle`` keeps the ordering
    reproducible across machines, Python versions and process restarts, which is
    a requirement for the order ablation (condition B versus condition C).
    """
    return sha256_text(f"{salt}|{identifier}")
