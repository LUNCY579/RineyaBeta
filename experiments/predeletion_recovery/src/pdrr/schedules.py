"""Pure, checkpoint-free recovery schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ScheduleKind = Literal["linear", "cosine", "sigmoid", "staircase"]


@dataclass(frozen=True)
class RecoverySchedule:
    total_steps: int
    notice_fraction: float = 0.20
    withdrawal_end_fraction: float = 0.70
    recovery_end_fraction: float = 0.78
    kind: ScheduleKind = "cosine"

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= self.notice_fraction < self.withdrawal_end_fraction <= 1.0:
            raise ValueError("require 0 <= notice < withdrawal_end <= 1")
        if not self.withdrawal_end_fraction <= self.recovery_end_fraction <= 1.0:
            raise ValueError("recovery_end must be between withdrawal_end and 1")

    @property
    def notice_steps(self) -> int:
        return round(self.total_steps * self.notice_fraction)

    @property
    def withdrawal_end_step(self) -> int:
        return round(self.total_steps * self.withdrawal_end_fraction)

    @property
    def recovery_end_step(self) -> int:
        return round(self.total_steps * self.recovery_end_fraction)

    def _progress(self, step: int) -> float:
        start = self.notice_steps
        end = self.withdrawal_end_step
        if step <= start:
            return 0.0
        if step >= end:
            return 1.0
        return (step - start) / max(end - start, 1)

    def doomed_gate(self, step: int) -> float:
        """Participation of the component scheduled for deletion."""
        p = self._progress(step)
        if self.kind == "linear":
            return 1.0 - p
        if self.kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * p))
        if self.kind == "sigmoid":
            # Rescale a logistic curve so endpoints are exactly 1 and 0.
            raw = 1.0 / (1.0 + math.exp(12.0 * (p - 0.5)))
            lo = 1.0 / (1.0 + math.exp(6.0))
            hi = 1.0 / (1.0 + math.exp(-6.0))
            return min(max((raw - lo) / (hi - lo), 0.0), 1.0)
        # Four deterministic plateaus; no hidden state to checkpoint.
        if p < 0.25:
            return 1.0
        if p < 0.50:
            return 0.75
        if p < 0.75:
            return 0.35
        return 0.0

    def recovery_weight(self, step: int) -> float:
        """Recovery scaffold is forced to zero before final consolidation."""
        if step >= self.recovery_end_step:
            return 0.0
        if step <= self.notice_steps:
            return 1.0
        span = max(self.recovery_end_step - self.notice_steps, 1)
        p = (step - self.notice_steps) / span
        return 0.5 * (1.0 + math.cos(math.pi * p))

    def compact_only(self, step: int) -> bool:
        return step >= self.withdrawal_end_step

    def phase(self, step: int) -> str:
        if step < self.notice_steps:
            return "notice"
        if step < self.withdrawal_end_step:
            return "withdrawal"
        return "consolidation"
