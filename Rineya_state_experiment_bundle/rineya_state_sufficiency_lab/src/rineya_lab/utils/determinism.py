"""Deterministic seeding across python / numpy / torch.

Determinism is a stated requirement of the protocol: the FULL and SPLIT arms
must differ only in training representation, so the seed schedule has to be
identical and reproducible. ``seed_everything`` is safe to call when torch is
absent -- it simply seeds what exists and reports what it seeded.
"""

from __future__ import annotations

import os
import random
from typing import Any


def derive_seed(base_seed: int, *tags: str | int) -> int:
    """Derive a stable child seed from a base seed and a tuple of tags.

    Used so that, for example, the data-order RNG and the dropout RNG are
    independent yet both reproducible from one recorded ``experiment.seed``.
    """
    from rineya_lab.utils.hashing import sha256_text

    payload = "|".join([str(base_seed), *[str(tag) for tag in tags]])
    return int(sha256_text(payload)[:16], 16) % (2**31 - 1)


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed every RNG we can reach. Returns a record for the run manifest."""
    record: dict[str, Any] = {"seed": seed, "python_hash_seed": os.environ.get("PYTHONHASHSEED")}
    random.seed(seed)
    record["random"] = True

    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
        record["numpy"] = True
    except Exception:  # pragma: no cover - numpy is a hard dependency in practice
        record["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.manual_seed_all(seed)
        record["torch"] = True
        record["torch_deterministic_requested"] = deterministic_torch
        if deterministic_torch:  # pragma: no cover - GPU only
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:
                record["torch_deterministic_error"] = str(exc)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        record["torch"] = False
    return record


def warn_if_hash_seed_unpinned() -> str | None:
    """The dataset shuffle uses SHA-256, but user extensions may not.

    Returns a warning string when ``PYTHONHASHSEED`` is unset so the runbook can
    surface it rather than silently producing an unreproducible ordering.
    """
    if os.environ.get("PYTHONHASHSEED") is None:
        return (
            "PYTHONHASHSEED is not set. Built-in hashing is randomised per process. "
            "All ordering in this repository uses SHA-256 and is unaffected, but set "
            "PYTHONHASHSEED=0 before adding any set/dict-iteration-dependent ordering."
        )
    return None
