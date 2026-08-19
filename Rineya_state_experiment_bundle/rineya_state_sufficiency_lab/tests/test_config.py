from __future__ import annotations

import pytest
import yaml

from rineya_lab.config import ConfigError, LabConfig, load_config, set_by_path


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_defaults_are_valid():
    cfg = load_config(None)
    assert cfg.experiment.condition == "FULL"
    assert cfg.data_view.view == "full_context"
    assert cfg.fingerprint() == LabConfig().fingerprint()


def test_extends_deep_merges(tmp_path):
    parent = write(tmp_path, "parent.yaml", {"experiment": {"name": "p", "seed": 1}, "lora": {"r": 8}})
    child = write(tmp_path, "child.yaml", {"extends": str(parent), "experiment": {"name": "c"}})
    cfg = load_config(child)
    assert cfg.experiment.name == "c"
    assert cfg.experiment.seed == 1       # inherited
    assert cfg.lora.r == 8                # inherited


def test_circular_extends_detected(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"extends": "b.yaml"}), encoding="utf-8")
    b.write_text(yaml.safe_dump({"extends": "a.yaml"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular"):
        load_config(a)


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(None, extra={"lora": {"rank": 8}})


def test_wrong_type_is_an_error():
    with pytest.raises(ConfigError, match="expected an int"):
        load_config(None, extra={"lora": {"r": "sixteen"}})


def test_dotted_override_uses_yaml_typing():
    cfg = load_config(None, ["training.epochs=2", "lora.target_modules=[q_proj]", "training.dry_run=true"])
    assert cfg.training.epochs == 2.0
    assert cfg.lora.target_modules == ["q_proj"]
    assert cfg.training.dry_run is True


def test_condition_and_view_must_agree():
    with pytest.raises(ConfigError, match="requires data_view.view"):
        load_config(None, extra={"experiment": {"condition": "SPLIT_STATEFUL"}, "data_view": {"view": "full_context"}})


def test_full_condition_rejects_order_ablation():
    with pytest.raises(ConfigError, match="order must stay 'shuffled'"):
        load_config(None, extra={"experiment": {"condition": "FULL"}, "data_view": {"order": "trajectory"}})


def test_matched_budget_requires_its_limit():
    with pytest.raises(ConfigError, match="max_optimizer_updates"):
        load_config(None, extra={"budget": {"mode": "matched_updates"}})


def test_quantization_requires_lora():
    with pytest.raises(ConfigError, match="QLoRA"):
        load_config(None, extra={"model": {"quantization": "4bit"}, "lora": {"enabled": False}})


def test_sampling_temperature_consistency():
    with pytest.raises(ConfigError, match="temperature"):
        load_config(None, extra={"evaluation": {"generation": {"do_sample": False, "temperature": 0.7}}})


def test_unknown_protocol_rejected():
    with pytest.raises(ConfigError, match="not one of"):
        load_config(None, extra={"evaluation": {"protocols": ["telepathy"]}})


def test_protocol_pair_format_validated():
    with pytest.raises(ConfigError, match="protocol pair"):
        load_config(None, extra={"comparison": {"protocol_pairs": ["full_direct"]}})


def test_set_by_path_creates_sections():
    data: dict = {}
    set_by_path(data, "a.b.c", "5")
    assert data == {"a": {"b": {"c": 5}}}


def test_fingerprint_changes_with_content():
    a = load_config(None)
    b = load_config(None, ["lora.r=32"])
    assert a.fingerprint() != b.fingerprint()
