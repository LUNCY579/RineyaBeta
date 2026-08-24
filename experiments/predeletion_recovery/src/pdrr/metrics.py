"""Behavioral and functional recovery metrics."""

from __future__ import annotations

import math
from typing import Iterable


def _utility(value: float, higher_is_better: bool) -> float:
    return float(value) if higher_is_better else -float(value)


def recovery_ratio(
    full: float,
    hard: float,
    method: float,
    *,
    higher_is_better: bool = True,
    eps: float = 1e-12,
) -> float:
    """Fraction of pruning damage recovered.

    Raises instead of emitting a numerically meaningless value when hard pruning
    did not measurably damage the selected metric.
    """
    u_full = _utility(full, higher_is_better)
    u_hard = _utility(hard, higher_is_better)
    u_method = _utility(method, higher_is_better)
    denom = u_full - u_hard
    if abs(denom) <= eps:
        raise ValueError("Recovery Ratio is undefined: full and hard anchors coincide")
    return (u_method - u_hard) / denom


def handoff_completeness(
    divergence_hard: float,
    divergence_recovered: float,
    *,
    eps: float = 1e-12,
) -> float:
    if divergence_hard < -eps or divergence_recovered < -eps:
        raise ValueError("divergences must be non-negative")
    if divergence_hard <= eps:
        raise ValueError("handoff is undefined: hard-pruning divergence is zero")
    return 1.0 - divergence_recovered / divergence_hard


def scaffold_dependence(with_scaffold: float, without_scaffold: float) -> float:
    return float(with_scaffold) - float(without_scaffold)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")
