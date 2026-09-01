"""The power arithmetic decides which experiments are worth GPU time.

A wrong minimum-detectable-effect would either license an uninformative sweep or
forbid an affordable one, so the exact binomial machinery is checked against
values that can be verified by hand.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


power_module = _load("endpoint_power")


def test_exact_two_sided_p_matches_hand_computed_values() -> None:
    # 0 of 1 and 1 of 1 are the whole distribution: p = 1.
    assert power_module.exact_two_sided_p(0, 1) == pytest.approx(1.0)
    # 0 of 5: two-sided p = 2 * (1/32) = 0.0625.
    assert power_module.exact_two_sided_p(0, 5) == pytest.approx(0.0625)
    # 1 of 10: 2 * (1 + 10)/1024 = 0.021484375.
    assert power_module.exact_two_sided_p(1, 10) == pytest.approx(2 * 11 / 1024)
    # A perfectly balanced split can never be evidence against p=0.5.
    assert power_module.exact_two_sided_p(5, 10) == pytest.approx(1.0)


def test_exact_two_sided_p_is_symmetric() -> None:
    for trials in (7, 20, 133):
        for successes in range(trials + 1):
            assert power_module.exact_two_sided_p(successes, trials) == pytest.approx(
                power_module.exact_two_sided_p(trials - successes, trials)
            )


def test_exact_two_sided_p_reproduces_the_published_v1_headline() -> None:
    """54:79 discordant is the as-published ordered-baseline contrast, p = 0.0370."""

    assert power_module.exact_two_sided_p(54, 133) == pytest.approx(0.0370, abs=5e-5)
    # And the consistently-scored version, 54:78, is p = 0.0449.
    assert power_module.exact_two_sided_p(54, 132) == pytest.approx(0.0449, abs=5e-5)


def test_binomial_pmf_sums_to_one() -> None:
    for trials, probability in ((10, 0.5), (25, 0.63), (40, 0.2)):
        total = sum(power_module.binomial_pmf(k, trials, probability) for k in range(trials + 1))
        assert total == pytest.approx(1.0, abs=1e-12)


def test_binomial_pmf_survives_large_trials_without_overflow() -> None:
    """The naive 2**n formulation overflowed at n~1000; log space must not."""

    value = power_module.binomial_pmf(5000, 10000, 0.5)
    assert math.isfinite(value) and value > 0.0
    assert power_module.exact_two_sided_p(4900, 10000) > 0.0


def test_power_is_alpha_under_the_null_and_rises_with_effect() -> None:
    at_null = power_module.power(133, 0.5)
    assert at_null <= 0.05, "a test cannot reject more than alpha when the null holds"
    previous = at_null
    for probability in (0.55, 0.60, 0.65, 0.70, 0.80):
        current = power_module.power(133, probability)
        assert current > previous
        previous = current
    assert previous > 0.99


def test_minimum_detectable_effect_shrinks_with_more_records() -> None:
    rate = 133 / 2400
    previous = None
    for seeds in (1, 2, 3, 5, 10):
        detectable, _, _ = power_module.minimum_detectable_effect(800 * seeds, rate)
        if previous is not None:
            assert detectable < previous
        previous = detectable


def test_the_decision_relevant_numbers_hold() -> None:
    """These specific values gate B1. Pin them so a refactor cannot move them silently."""

    rate = 133 / 2400
    one_seed, _, _ = power_module.minimum_detectable_effect(800, rate)
    three_seed, _, _ = power_module.minimum_detectable_effect(2400, rate)
    assert one_seed == pytest.approx(0.0227, abs=5e-4)
    assert three_seed == pytest.approx(0.0139, abs=5e-4)
    # The observed deficit sits BELOW the three-seed MDE: it was detected with
    # roughly coin-flip power, which is why it must not be treated as firm.
    assert 0.0100 < three_seed
    assert power_module.power(132, 78 / 132) == pytest.approx(0.54, abs=0.03)


def test_required_records_inverts_minimum_detectable_effect() -> None:
    rate = 133 / 2400
    for effect in (0.02, 0.012, 0.008):
        records = power_module.required_records(effect, rate)
        detectable, _, _ = power_module.minimum_detectable_effect(records, rate)
        assert detectable <= effect
        smaller, _, _ = power_module.minimum_detectable_effect(records - 1, rate)
        assert smaller > effect, "required_records must return the SMALLEST sufficient n"
