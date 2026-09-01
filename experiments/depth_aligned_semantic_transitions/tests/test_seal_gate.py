"""The sealed-TEST gate must hold on every path that can open a corpus file.

Two ways it did not, both found by audit and both fixed here:

1. `assert_split_allowed` compared `split.lower()` against "test" without
   stripping, so a split of " test " -- which `data_path_for_split` and
   `load_records` both tolerate -- returned early and reached the file open with
   the seal unchecked.
2. `dast.depth_cache.build_depth_cache` called `load_records` directly rather than
   through `training.py`, making it a second ungated load site.

These tests never touch a real sealed file: they assert the gate raises BEFORE any
path is opened, using configs whose paths do not exist.
"""

from __future__ import annotations

import pytest

from dast.config import DASTConfig, assert_split_allowed


def _sealed_config() -> DASTConfig:
    config = DASTConfig()
    config.experiment.allow_sealed_test = False
    config.experiment.preregistration_frozen = False
    config.experiment.preregistration_sha256 = ""
    return config


@pytest.mark.parametrize(
    "split",
    ["test", "TEST", "Test", " test", "test ", "  test  ", "\ttest\n", " TeSt "],
)
def test_every_spelling_of_test_is_refused(split: str) -> None:
    """Whitespace and case must not be a way around the seal."""

    with pytest.raises(PermissionError, match="sealed"):
        assert_split_allowed(_sealed_config(), split)


@pytest.mark.parametrize("split", ["dev", "train", "validation", "dev_test", "testing"])
def test_non_test_splits_are_allowed(split: str) -> None:
    """The gate must not become so broad that it blocks legitimate splits.

    'testing' and 'dev_test' merely CONTAIN 'test'; they are different splits and
    a substring match would wrongly refuse them.
    """

    assert_split_allowed(_sealed_config(), split)


def test_allow_flag_alone_is_not_enough() -> None:
    config = _sealed_config()
    config.experiment.allow_sealed_test = True
    with pytest.raises(PermissionError, match="preregistration_frozen"):
        assert_split_allowed(config, "test")


def test_frozen_flag_without_a_real_digest_is_not_enough() -> None:
    config = _sealed_config()
    config.experiment.allow_sealed_test = True
    config.experiment.preregistration_frozen = True
    config.experiment.preregistration_sha256 = "not-a-digest"
    with pytest.raises(PermissionError, match="preregistration_sha256"):
        assert_split_allowed(config, "test")


def test_build_depth_cache_refuses_the_sealed_split_before_reading_anything() -> None:
    """The cache builder is a load site and must check the seal itself.

    A PermissionError proves the gate fired first; any other error would mean the
    code got as far as resolving or opening a path.
    """

    from dast import depth_cache

    class _Unusable:
        """Any attribute access means the gate did not fire first."""

        def __getattr__(self, name: str):  # pragma: no cover - must not run
            raise AssertionError(f"model was touched before the seal check: {name!r}")

    config = _sealed_config()
    with pytest.raises(PermissionError, match="sealed"):
        depth_cache.build_depth_cache(
            _Unusable(), _Unusable(), config, layers=[1], split="test", progress=False
        )
    # And with the same padding that used to slip through.
    with pytest.raises(PermissionError, match="sealed"):
        depth_cache.build_depth_cache(
            _Unusable(), _Unusable(), config, layers=[1], split=" test ", progress=False
        )
