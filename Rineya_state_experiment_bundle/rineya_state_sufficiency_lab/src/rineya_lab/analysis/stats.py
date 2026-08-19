"""Paired statistics for the FULL-versus-SPLIT comparison.

All tests are paired on the *source problem*, never on the expanded transition
row. Transitions from one problem are not independent observations, so treating
them as such would understate the variance and inflate significance.

Three complementary procedures are reported because they fail differently:

* **Exact McNemar** -- discordant-pair test, exact binomial, no normal
  approximation. Applies only to binary outcomes.
* **Paired cluster bootstrap** -- resamples problems with replacement, giving a
  confidence interval for the accuracy difference that also covers non-binary
  per-problem scores.
* **Sign-flip randomization** -- an exact-in-distribution permutation test under
  the sharp null of exchangeable signs; assumption-light and robust to ties.

Decisions (superiority / non-inferiority / equivalence) are read off the
bootstrap interval against a *pre-registered* margin.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rineya_lab.utils.hashing import sha256_text

# --------------------------------------------------------------------------------------
# Deterministic RNG (independent of numpy's global state)
# --------------------------------------------------------------------------------------


class _DeterministicRNG:
    """Small xorshift-based RNG seeded from a string, for reproducible resampling."""

    def __init__(self, seed: int | str) -> None:
        digest = sha256_text(str(seed))
        self.state = int(digest[:16], 16) or 0x9E3779B97F4A7C15

    def next_u64(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return self.state

    def randint(self, upper: int) -> int:
        return self.next_u64() % upper

    def random(self) -> float:
        return self.next_u64() / float(1 << 64)


# --------------------------------------------------------------------------------------
# Exact binomial helpers
# --------------------------------------------------------------------------------------


def binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return min(1.0, total)


@dataclass
class McNemarResult:
    applicable: bool
    b: int = 0  # baseline correct, candidate wrong
    c: int = 0  # baseline wrong, candidate correct
    n_discordant: int = 0
    p_value: float | None = None
    p_value_one_sided: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "baseline_only_correct": self.b,
            "candidate_only_correct": self.c,
            "discordant_pairs": self.n_discordant,
            "p_value_two_sided": self.p_value,
            "p_value_one_sided_candidate_better": self.p_value_one_sided,
            "note": self.note,
        }


def exact_mcnemar(baseline: Sequence[float], candidate: Sequence[float]) -> McNemarResult:
    """Exact McNemar test on paired binary outcomes."""
    if len(baseline) != len(candidate):
        raise ValueError("paired sequences must have equal length")
    if not all(value in (0, 1, 0.0, 1.0) for value in [*baseline, *candidate]):
        return McNemarResult(
            applicable=False,
            note="McNemar requires binary per-problem outcomes; scores are fractional. "
                 "Use the bootstrap interval and the sign-flip test instead.",
        )
    b = sum(1 for x, y in zip(baseline, candidate) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(baseline, candidate) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return McNemarResult(applicable=True, b=0, c=0, n_discordant=0, p_value=1.0, p_value_one_sided=1.0,
                             note="no discordant pairs")
    two_sided = min(1.0, 2 * binomial_cdf(min(b, c), n, 0.5))
    one_sided = binomial_cdf(b, n, 0.5)  # P(candidate at least this much better)
    return McNemarResult(applicable=True, b=b, c=c, n_discordant=n, p_value=two_sided, p_value_one_sided=one_sided)


# --------------------------------------------------------------------------------------
# Bootstrap and randomization
# --------------------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    observed_difference: float
    ci_low: float
    ci_high: float
    alpha: float
    iterations: int
    baseline_mean: float
    candidate_mean: float
    standard_error: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_difference": self.observed_difference,
            "observed_difference_pp": round(100 * self.observed_difference, 4),
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_low_pp": round(100 * self.ci_low, 4),
            "ci_high_pp": round(100 * self.ci_high, 4),
            "alpha": self.alpha,
            "iterations": self.iterations,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "standard_error": self.standard_error,
        }


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = 10000,
    alpha: float = 0.05,
    seed: int | str = 20260818,
) -> BootstrapResult:
    """Percentile CI for ``mean(candidate) - mean(baseline)`` over paired problems."""
    if len(baseline) != len(candidate):
        raise ValueError("paired sequences must have equal length")
    n = len(baseline)
    if n == 0:
        raise ValueError("cannot bootstrap an empty comparison")

    differences = [c - b for b, c in zip(baseline, candidate)]
    observed = sum(differences) / n

    rng = _DeterministicRNG(f"bootstrap|{seed}|{n}|{iterations}")
    samples: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += differences[rng.randint(n)]
        samples.append(total / n)
    samples.sort()

    low_index = max(0, int(math.floor((alpha / 2) * iterations)))
    high_index = min(iterations - 1, int(math.ceil((1 - alpha / 2) * iterations)) - 1)
    mean_sample = sum(samples) / iterations
    variance = sum((s - mean_sample) ** 2 for s in samples) / max(1, iterations - 1)

    return BootstrapResult(
        observed_difference=round(observed, 8),
        ci_low=round(samples[low_index], 8),
        ci_high=round(samples[high_index], 8),
        alpha=alpha,
        iterations=iterations,
        baseline_mean=round(sum(baseline) / n, 8),
        candidate_mean=round(sum(candidate) / n, 8),
        standard_error=round(math.sqrt(variance), 8),
    )


@dataclass
class RandomizationResult:
    observed_difference: float
    p_value_two_sided: float
    p_value_one_sided: float
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_difference": self.observed_difference,
            "p_value_two_sided": self.p_value_two_sided,
            "p_value_one_sided_candidate_better": self.p_value_one_sided,
            "iterations": self.iterations,
        }


def sign_flip_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = 10000,
    seed: int | str = 20260818,
) -> RandomizationResult:
    """Paired sign-flip randomization test on the per-problem differences."""
    differences = [c - b for b, c in zip(baseline, candidate)]
    n = len(differences)
    if n == 0:
        raise ValueError("cannot randomize an empty comparison")
    observed = sum(differences) / n

    rng = _DeterministicRNG(f"signflip|{seed}|{n}|{iterations}")
    at_least_as_extreme = 0
    at_least_as_large = 0
    for _ in range(iterations):
        total = 0.0
        for value in differences:
            total += value if rng.next_u64() & 1 else -value
        statistic = total / n
        if abs(statistic) >= abs(observed) - 1e-12:
            at_least_as_extreme += 1
        if statistic >= observed - 1e-12:
            at_least_as_large += 1
    # +1 correction keeps the p-value strictly positive (Phipson & Smyth).
    return RandomizationResult(
        observed_difference=round(observed, 8),
        p_value_two_sided=round((at_least_as_extreme + 1) / (iterations + 1), 8),
        p_value_one_sided=round((at_least_as_large + 1) / (iterations + 1), 8),
        iterations=iterations,
    )


# --------------------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------------------


@dataclass
class ComparisonDecision:
    superiority: str = "not_established"
    non_inferiority: str = "not_established"
    equivalence: str = "not_established"
    margin_pp: float = 0.0
    equivalence_margin_pp: float = 0.0
    margin_frozen: bool = False
    rationale: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "superiority": self.superiority,
            "non_inferiority": self.non_inferiority,
            "equivalence": self.equivalence,
            "non_inferiority_margin_pp": self.margin_pp,
            "equivalence_margin_pp": self.equivalence_margin_pp,
            "margin_frozen": self.margin_frozen,
            "rationale": self.rationale,
        }


def decide(
    bootstrap: BootstrapResult,
    *,
    non_inferiority_margin_pp: float,
    equivalence_margin_pp: float,
    margin_frozen: bool = False,
) -> ComparisonDecision:
    """Read superiority / non-inferiority / equivalence off the bootstrap interval.

    Conventions (all on the candidate-minus-baseline difference):

    * superiority      : the whole interval lies above 0.
    * non-inferiority  : the lower bound lies above ``-margin``.
    * equivalence      : the whole interval lies inside ``(-margin, +margin)`` (TOST).
    """
    decision = ComparisonDecision(
        margin_pp=non_inferiority_margin_pp,
        equivalence_margin_pp=equivalence_margin_pp,
        margin_frozen=margin_frozen,
    )
    low_pp = 100 * bootstrap.ci_low
    high_pp = 100 * bootstrap.ci_high

    if low_pp > 0:
        decision.superiority = "candidate_superior"
        decision.rationale.append(f"CI lower bound {low_pp:+.3f} pp > 0")
    elif high_pp < 0:
        decision.superiority = "baseline_superior"
        decision.rationale.append(f"CI upper bound {high_pp:+.3f} pp < 0")
    else:
        decision.rationale.append(f"CI [{low_pp:+.3f}, {high_pp:+.3f}] pp spans 0")

    if low_pp > -non_inferiority_margin_pp:
        decision.non_inferiority = "non_inferior"
        decision.rationale.append(f"CI lower bound {low_pp:+.3f} pp > -{non_inferiority_margin_pp} pp margin")
    else:
        decision.non_inferiority = "not_established"
        decision.rationale.append(f"CI lower bound {low_pp:+.3f} pp does not clear the -{non_inferiority_margin_pp} pp margin")

    if -equivalence_margin_pp < low_pp and high_pp < equivalence_margin_pp:
        decision.equivalence = "equivalent"
        decision.rationale.append(f"CI [{low_pp:+.3f}, {high_pp:+.3f}] pp lies inside ±{equivalence_margin_pp} pp")
    else:
        decision.equivalence = "not_established"

    if not margin_frozen:
        decision.rationale.append(
            "WARNING: the margin was not declared frozen. Freeze and justify it in the preregistration "
            "before opening TEST results, otherwise these decisions are exploratory only."
        )
    return decision


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni correction across a family of stratified tests."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    out: dict[str, dict[str, Any]] = {}
    previous_adjusted = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        adjusted = min(1.0, max(previous_adjusted, (total - rank) * p_value))
        previous_adjusted = adjusted
        out[name] = {
            "p_value": p_value,
            "adjusted_p_value": round(adjusted, 8),
            "rejected_at_alpha": adjusted < alpha,
            "rank": rank + 1,
            "family_size": total,
        }
    return out
