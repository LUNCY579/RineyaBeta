"""Learning-rate schedules implemented without a Trainer dependency."""

from __future__ import annotations

import math


def resolve_warmup_steps(total_updates: int, warmup_ratio: float, warmup_steps: int | None) -> int:
    if warmup_steps is not None:
        return max(0, min(warmup_steps, total_updates))
    return max(0, min(int(round(total_updates * warmup_ratio)), total_updates))


def lr_multiplier(step: int, total_updates: int, warmup: int, kind: str) -> float:
    """Multiplier in [0, 1] applied to the peak learning rate at ``step`` (0-based)."""
    if total_updates <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return (step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total_updates - warmup)
    progress = min(max(progress, 0.0), 1.0)
    if kind == "constant":
        return 1.0
    if kind == "linear":
        return 1.0 - progress
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown scheduler {kind!r}; expected constant, linear or cosine")


def build_lambda(total_updates: int, warmup: int, kind: str):
    def _fn(step: int) -> float:
        return lr_multiplier(step, total_updates, warmup, kind)

    return _fn
