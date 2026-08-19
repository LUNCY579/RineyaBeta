from __future__ import annotations

import pytest

from rineya_lab.analysis.stats import (
    binomial_cdf,
    decide,
    exact_mcnemar,
    holm_bonferroni,
    paired_bootstrap_ci,
    sign_flip_test,
)


def test_binomial_cdf_known_values():
    assert binomial_cdf(0, 1, 0.5) == pytest.approx(0.5)
    assert binomial_cdf(10, 10, 0.5) == pytest.approx(1.0)
    assert binomial_cdf(2, 10, 0.5) == pytest.approx(0.0546875)


def test_mcnemar_matches_hand_computed_exact_p():
    # b = 1 baseline-only, c = 9 candidate-only -> two-sided p = 2 * P(X <= 1 | n=10)
    baseline = [1] + [0] * 9 + [1] * 5
    candidate = [0] + [1] * 9 + [1] * 5
    result = exact_mcnemar(baseline, candidate)
    assert result.applicable and result.b == 1 and result.c == 9
    assert result.p_value == pytest.approx(2 * binomial_cdf(1, 10, 0.5))


def test_mcnemar_no_discordant_pairs():
    result = exact_mcnemar([1, 0, 1], [1, 0, 1])
    assert result.p_value == 1.0 and result.n_discordant == 0


def test_mcnemar_not_applicable_for_fractional_scores():
    result = exact_mcnemar([0.5, 1.0], [1.0, 1.0])
    assert not result.applicable and "fractional" in result.note


def test_bootstrap_recovers_a_known_difference():
    baseline = [0] * 50 + [1] * 50
    candidate = [1] * 100
    result = paired_bootstrap_ci(baseline, candidate, iterations=2000, seed=1)
    assert result.observed_difference == pytest.approx(0.5)
    assert result.ci_low > 0.3 and result.ci_high < 0.7


def test_bootstrap_is_deterministic():
    baseline = [0, 1] * 40
    candidate = [1, 1] * 40
    a = paired_bootstrap_ci(baseline, candidate, iterations=500, seed="fixed")
    b = paired_bootstrap_ci(baseline, candidate, iterations=500, seed="fixed")
    assert a.as_dict() == b.as_dict()


def test_bootstrap_of_identical_arms_brackets_zero():
    values = [1, 0, 1, 1, 0] * 20
    result = paired_bootstrap_ci(values, values, iterations=1000, seed=3)
    assert result.observed_difference == 0.0
    assert result.ci_low == 0.0 and result.ci_high == 0.0


def test_sign_flip_detects_a_real_difference():
    baseline = [0] * 60 + [1] * 10
    candidate = [1] * 60 + [1] * 10
    result = sign_flip_test(baseline, candidate, iterations=2000, seed=5)
    assert result.p_value_two_sided < 0.01


def test_sign_flip_p_value_is_never_zero():
    baseline = [0] * 100
    candidate = [1] * 100
    result = sign_flip_test(baseline, candidate, iterations=200, seed=7)
    assert result.p_value_two_sided > 0


def test_decision_superiority():
    boot = paired_bootstrap_ci([0] * 40 + [1] * 10, [1] * 50, iterations=1000, seed=9)
    decision = decide(boot, non_inferiority_margin_pp=2.0, equivalence_margin_pp=2.0, margin_frozen=True)
    assert decision.superiority == "candidate_superior"
    assert decision.non_inferiority == "non_inferior"
    assert decision.equivalence == "not_established"


def test_decision_equivalence_for_identical_arms():
    values = [1, 0] * 50
    boot = paired_bootstrap_ci(values, values, iterations=1000, seed=11)
    decision = decide(boot, non_inferiority_margin_pp=2.0, equivalence_margin_pp=2.0, margin_frozen=True)
    assert decision.equivalence == "equivalent"
    assert decision.superiority == "not_established"


def test_unfrozen_margin_emits_a_warning():
    values = [1, 0] * 20
    boot = paired_bootstrap_ci(values, values, iterations=500, seed=13)
    decision = decide(boot, non_inferiority_margin_pp=2.0, equivalence_margin_pp=2.0, margin_frozen=False)
    assert any("not declared frozen" in line for line in decision.rationale)


def test_baseline_superior_when_candidate_is_worse():
    boot = paired_bootstrap_ci([1] * 50, [0] * 40 + [1] * 10, iterations=1000, seed=17)
    decision = decide(boot, non_inferiority_margin_pp=2.0, equivalence_margin_pp=2.0, margin_frozen=True)
    assert decision.superiority == "baseline_superior"
    assert decision.non_inferiority == "not_established"


def test_holm_bonferroni_orders_and_adjusts():
    corrected = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.5}, alpha=0.05)
    assert corrected["a"]["adjusted_p_value"] == pytest.approx(0.003)
    assert corrected["a"]["rejected_at_alpha"]
    assert not corrected["c"]["rejected_at_alpha"]
    assert corrected["b"]["adjusted_p_value"] >= corrected["a"]["adjusted_p_value"]


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 0], [1], iterations=200)
    with pytest.raises(ValueError):
        exact_mcnemar([1, 0], [1])


def test_empty_comparison_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [], iterations=200)


def test_t_critical_matches_the_standard_table():
    """A wrong df index here silently widens every cross-seed interval."""
    from rineya_lab.analysis.multiseed import _t_critical

    assert _t_critical(0) is None          # one seed: no variance estimate exists
    assert _t_critical(1) == 12.706        # two seeds
    assert _t_critical(2) == 4.303         # three seeds
    assert _t_critical(3) == 3.182
    assert _t_critical(9) == 2.262
