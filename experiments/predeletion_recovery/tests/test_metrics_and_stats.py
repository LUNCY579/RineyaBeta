import pytest

from pdrr.metrics import handoff_completeness, recovery_ratio
from pdrr.stats import finalize_family, holm_adjust, paired_contrast, sign_flip_pvalue


def test_recovery_ratio_higher_is_better():
    assert recovery_ratio(1.0, 0.6, 0.8) == pytest.approx(0.5)
    assert recovery_ratio(1.0, 0.6, 1.1) > 1.0


def test_recovery_ratio_refuses_undefined_anchor():
    with pytest.raises(ValueError, match="undefined"):
        recovery_ratio(1.0, 1.0, 1.0)


def test_handoff_completeness():
    assert handoff_completeness(0.4, 0.1) == 0.75


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert all(0 <= value <= 1 for value in adjusted.values())
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_exact_sign_flip_null_for_balanced_differences():
    p = sign_flip_pvalue({"a": [1.0], "b": [-1.0]})
    assert p == 1.0


def test_confirmatory_family_requires_ci_and_holm_pass():
    contrasts = [
        paired_contrast("m__vs__a", {str(i): [0.1] for i in range(8)},
                        practical_threshold=0.01, bootstrap_samples=1000,
                        permutation_samples=1000, seed=1),
        paired_contrast("m__vs__b", {str(i): [0.08] for i in range(8)},
                        practical_threshold=0.01, bootstrap_samples=1000,
                        permutation_samples=1000, seed=2),
    ]
    final = finalize_family(contrasts)
    assert all(item.practical_pass for item in final)
    assert all(item.adjusted_p is not None for item in final)
