"""Statistical comparison of two runs, paired at the source-problem level."""

from rineya_lab.analysis.multiseed import aggregate_reports
from rineya_lab.analysis.stats import (
    ComparisonDecision,
    exact_mcnemar,
    paired_bootstrap_ci,
    sign_flip_test,
)

__all__ = [
    "ComparisonDecision",
    "exact_mcnemar",
    "paired_bootstrap_ci",
    "sign_flip_test",
    "aggregate_reports",
]
