from pathlib import Path

import pytest

from dast.config import DASTConfig, ExperimentConfig, SemanticConfig, assert_split_allowed, load_config


def test_unknown_config_key_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("experiment:\n  arm: state_ordered\n  typo: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_config_extends_and_hash_are_deterministic(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "experiment:\n  arm: state_ordered\nsemantic:\n  state_weight: 1.0\noperators:\n  enabled: false\n",
        encoding="utf-8",
    )
    child.write_text("extends: base.yaml\nexperiment:\n  name: child\n", encoding="utf-8")
    first = load_config(child)
    second = load_config(child)
    assert first.experiment.name == "child"
    assert first.canonical_sha256() == second.canonical_sha256()


def test_baseline_requires_zero_auxiliary_weights():
    cfg = DASTConfig(
        experiment=ExperimentConfig(arm="cot_baseline"),
        semantic=SemanticConfig(state_weight=1.0, contrastive_weight=0.0),
    )
    with pytest.raises(ValueError, match="cot_baseline"):
        cfg.validate()


def test_test_split_is_sealed_by_default():
    with pytest.raises(PermissionError, match="TEST is sealed"):
        assert_split_allowed(DASTConfig(), "test")


def test_test_split_requires_frozen_hash():
    cfg = DASTConfig()
    cfg.experiment.allow_sealed_test = True
    cfg.experiment.preregistration_frozen = True
    cfg.experiment.preregistration_sha256 = "a" * 64
    assert_split_allowed(cfg, "test")


def test_feedback_and_gradient_checkpointing_are_incompatible():
    cfg = DASTConfig()
    cfg.model.feedback_enabled = True
    cfg.training.gradient_checkpointing = True
    with pytest.raises(ValueError, match="incompatible"):
        cfg.validate()
